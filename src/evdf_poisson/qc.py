"""Fit-quality checks that stay valid at low counts.

Two complementary statistics per epoch, both computed from the stored fit
parameters and the pixel grid:

* **Randomized PIT + KS test.** For a Poisson pixel with expected counts mu
  and observed counts C, the randomized probability integral transform
  u = F(C-1 | mu) + U * [F(C | mu) - F(C-1 | mu)] (Dunn & Smyth quantile
  residuals) is uniform on [0, 1] when the model is right, at any count
  level. A Kolmogorov-Smirnov test of the u values against the uniform
  distribution gives a p-value for the whole grid, and one for the strahl
  wedge alone. The uniform draw is seeded by the epoch time, so the result is
  reproducible.
* **Calibrated deviance.** The deviance D of every pixel has an exact Poisson
  expectation E[D | mu] and variance, computed here by truncated sums. The
  ratio R = sum(D) / sum(E[D]) and the score z = (sum(D) - sum(E[D])) /
  sqrt(sum(Var[D])) compare the fit with what Poisson noise alone would
  produce, which deviance/dof cannot do (its expectation depends on the
  count level).

Per epoch: ``qcEpoch``; per file from a runner CSV: ``qcFile``; whole run
directory from the command line::

    PYTHONPATH=src python -m evdf_poisson.qc --data-dir <dir> --run-dir <results dir> [--workers 8]

which writes ``<results dir>/qc/<stem>_qc.csv`` and ``qc_summary.csv``.
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kstest, poisson

from .epoch_io import loadEpoch
from .staged_fit import evaluateModel

STRAHL_WEDGE_HALF_ANGLE_DEG = 30.0
EXPECTED_COUNTS_FLOOR = 1e-12
RNG_SEED_MASK = 0xFFFFFFFF
MIN_WEDGE_PIXELS = 50
FREE_PARAMS_WITH_STRAHL = 13
FREE_PARAMS_WITHOUT_STRAHL = 8

# Flag rules used for the month-long runs (see the notebook 08 for the reasoning)
HARD_FLAG_MAX_DETECTED_PIXELS = 1000       # too few counts: information-poor, not a misfit
STRONG_FLAG_KS_P = 1e-3
STRONG_FLAG_DEVIANCE_RATIO = 1.1
WEDGE_FLAG_KS_P = 1e-3

QC_COLUMNS = ['epoch_index', 'epoch_ns', 'n_pixels', 'n_detected', 'ks_stat', 'ks_p', 'd_red', 'dev_ratio', 'dev_z',
              'ks_stat_wedge', 'ks_p_wedge', 'dev_ratio_wedge', 'dev_z_wedge']


def devianceTerms(observed, expected):
    """Poisson deviance contribution of every pixel."""
    terms = 2.0 * (expected - observed)
    hasCounts = observed > 0
    terms[hasCounts] += 2.0 * observed[hasCounts] * np.log(observed[hasCounts] / expected[hasCounts])
    return terms


def devianceMoments(expected):
    """Exact E[D] and Var[D] of every pixel by truncated Poisson sums over k."""
    maxCount = int(max(60, expected.max() + 10 * np.sqrt(expected.max()) + 10))
    kValues = np.arange(maxCount + 1, dtype=float)[:, None]
    pmf = poisson.pmf(kValues, expected[None, :])
    with np.errstate(divide='ignore', invalid='ignore'):
        logTerm = np.where(kValues > 0, kValues * np.log(kValues / expected[None, :]), 0.0)
    devianceTable = 2.0 * (expected[None, :] - kValues + logTerm)
    meanDeviance = np.sum(pmf * devianceTable, axis=0)
    meanSquare = np.sum(pmf * devianceTable ** 2, axis=0)
    return meanDeviance, np.maximum(meanSquare - meanDeviance ** 2, 0.0)


def randomizedPit(observed, expected, seed):
    """Randomized probability integral transform of Poisson counts (uniform under the model)."""
    rng = np.random.default_rng(int(seed) & RNG_SEED_MASK)
    lowCdf = poisson.cdf(observed - 1, expected)
    highCdf = poisson.cdf(observed, expected)
    return lowCdf + rng.uniform(size=observed.shape) * (highCdf - lowCdf)


def expectedCountsFromParams(fitParams, grid):
    model = evaluateModel(fitParams, grid)
    return np.maximum(model / grid['one_count_psd'], EXPECTED_COUNTS_FLOOR)


def qcEpoch(chunkPath, epochIndex, fitParams, strahlSign=0, numFreeParams=FREE_PARAMS_WITH_STRAHL):
    """QC statistics of one fitted epoch; ``fitParams`` is the 13-parameter dict of the fit."""
    grid, psd, counts, epochNs, _ = loadEpoch(chunkPath, epochIndex)
    valid = grid['valid']
    expected = expectedCountsFromParams(fitParams, grid)[valid]
    observed = counts[valid]
    pitValues = randomizedPit(observed, expected, epochNs)
    ksAll = kstest(pitValues, 'uniform')
    terms = devianceTerms(observed, expected)
    meanDeviance, varianceDeviance = devianceMoments(expected)
    result = {'epoch_index': int(epochIndex), 'epoch_ns': int(epochNs),
              'n_pixels': int(len(observed)), 'n_detected': int((observed > 0).sum()),
              'ks_stat': float(ksAll.statistic), 'ks_p': float(ksAll.pvalue),
              'd_red': float(terms.sum() / max(len(observed) - numFreeParams, 1)),
              'dev_ratio': float(terms.sum() / meanDeviance.sum()),
              'dev_z': float((terms.sum() - meanDeviance.sum()) / np.sqrt(varianceDeviance.sum())),
              'ks_stat_wedge': np.nan, 'ks_p_wedge': np.nan, 'dev_ratio_wedge': np.nan, 'dev_z_wedge': np.nan,
              'pit_values': pitValues}
    if strahlSign != 0:
        pitch = grid['pitch_angle'][valid]
        inWedge = pitch <= STRAHL_WEDGE_HALF_ANGLE_DEG if strahlSign > 0 else pitch >= 180.0 - STRAHL_WEDGE_HALF_ANGLE_DEG
        if inWedge.sum() >= MIN_WEDGE_PIXELS:
            ksWedge = kstest(pitValues[inWedge], 'uniform')
            wedgeDeviance = terms[inWedge].sum()
            result.update({'pit_values_wedge': pitValues[inWedge],
                           'ks_stat_wedge': float(ksWedge.statistic), 'ks_p_wedge': float(ksWedge.pvalue),
                           'dev_ratio_wedge': float(wedgeDeviance / meanDeviance[inWedge].sum()),
                           'dev_z_wedge': float((wedgeDeviance - meanDeviance[inWedge].sum())
                                                / np.sqrt(varianceDeviance[inWedge].sum()))})
    return result


def qcSummaryRow(chunkPath, row):
    """qcEpoch driven by one row of a runner CSV (needs fit_params_json, strahl_sign, strahl_tier)."""
    fitParams = json.loads(row['fit_params_json'])
    strahlSign = int(row['strahl_sign']) if pd.notna(row.get('strahl_sign')) else 0
    numFree = FREE_PARAMS_WITHOUT_STRAHL if row.get('strahl_tier') == 'quasi_iso' else FREE_PARAMS_WITH_STRAHL
    result = qcEpoch(chunkPath, int(row['epoch_index']), fitParams, strahlSign, numFree)
    return {key: result[key] for key in QC_COLUMNS}


def _qcSummaryRowTask(task):
    chunkPath, rowDict = task
    return qcSummaryRow(chunkPath, pd.Series(rowDict))


def qcFile(chunkPath, summaryFrame, workers=4):
    """QC table for every fitted row of ``summaryFrame`` that belongs to ``chunkPath``."""
    fitted = summaryFrame[summaryFrame['fit_params_json'].notna()]
    tasks = [(str(chunkPath), row.to_dict()) for _, row in fitted.iterrows()]
    if workers <= 1 or len(tasks) < 2:
        results = [_qcSummaryRowTask(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_qcSummaryRowTask, tasks, chunksize=4))
    return pd.DataFrame(results, columns=QC_COLUMNS).sort_values('epoch_index').reset_index(drop=True)


def flagQcTable(qcTable):
    """Add the flag columns used for month-long runs."""
    flagged = qcTable.copy()
    flagged['flag_hard'] = flagged['n_detected'] <= HARD_FLAG_MAX_DETECTED_PIXELS
    flagged['flag_strong'] = (flagged['ks_p'] < STRONG_FLAG_KS_P) & (flagged['dev_ratio'] > STRONG_FLAG_DEVIANCE_RATIO)
    flagged['flag_wedge'] = flagged['ks_p_wedge'] < WEDGE_FLAG_KS_P
    flagged['flag_soft'] = (flagged['ks_p'] < STRONG_FLAG_KS_P) & ~flagged['flag_strong']
    return flagged


def qcRunDirectory(dataDir, runDir, workers=4):
    """QC of every per-file CSV of a runner output directory -> <runDir>/qc/."""
    outputDir = Path(runDir) / 'qc'
    outputDir.mkdir(parents=True, exist_ok=True)
    written = []
    for csvPath in sorted(glob.glob(str(Path(runDir) / 'per_file' / '*_fit.csv'))):
        stem = Path(csvPath).name[:-len('_fit.csv')]
        chunkPath = Path(dataDir) / f'{stem}.h5'
        if not chunkPath.exists():
            print(f'{stem}: data file missing, skipped')
            continue
        table = flagQcTable(qcFile(chunkPath, pd.read_csv(csvPath), workers))
        outputPath = outputDir / f'{stem}_qc.csv'
        table.to_csv(outputPath, index=False)
        written.append(table.assign(file=stem))
        print(f'{stem}: {len(table)} epochs, strong {int(table["flag_strong"].sum())}, wedge {int(table["flag_wedge"].sum())}')
    if written:
        pd.concat(written, ignore_index=True).to_csv(outputDir / 'qc_summary.csv', index=False)
    return outputDir


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    qcRunDirectory(args.data_dir, args.run_dir, args.workers)


if __name__ == '__main__':
    main()
