"""Batch fitting of preprocessed SWA-EAS VDF files with the Poisson pipeline.

Every epoch of every input file goes through: PAD strahl detection -> seeds
-> staged fit (core -> halo -> strahl -> joint) with the chosen objective ->
physical parameters + fit-quality metrics. One CSV per input file is the
resume unit (a rerun skips files whose CSV already has one row per epoch);
``summary.csv`` merges them.

Break energies are read from ``<break_dir>/<file stem>_two_population_crossing.csv``
(src/compute_break_energy_crossing.py); epochs without an 'ok' row fall back
to the 7.7 T_c proxy inside the staged fit (column ``break_source`` tells which).

Usage, from the repository root:

    PYTHONPATH=src python -m evdf_poisson.runner \\
        --data-dir examples/example_data/poisson_2023-04-04 \\
        --out results/fitting/example_run_poisson \\
        --workers 4 [--objective poisson|chi2] [--plot-every 1]
        [--break-dir examples/example_data/poisson_2023-04-04] [--files NAME ...] [--no-resume]

    from evdf_poisson.runner import runFiles
    summary = runFiles('examples/example_data/poisson_2023-04-04', 'results/fitting/example_run_poisson')
"""

import argparse
import json
import logging
import multiprocessing
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .epoch_io import loadEpoch, countEpochs
from .seeds import seedFromData
from .pad_detection import detectStrahlFromPad
from .objectives import fitEpoch, OBJECTIVES
from .metrics import (poissonDeviancePerDof, reducedChiSquare,
                      componentMomentsFromParams)

log = logging.getLogger('evdf_poisson')

INPUT_FILE_GLOB = 'VDF_eas_forFitting_*.h5'
BREAK_FILE_SUFFIX = '_two_population_crossing.csv'
# A free parameter this close to its box edge (fraction of the box width) is
# reported as pinned: the box, not the data, decided its value.
PINNED_BOX_FRACTION = 1e-3
PROGRESS_EVERY_EPOCHS = 50

RESULT_COLUMNS = ['epoch_index', 'epoch_ns', 'time_iso', 'n_sc',
                  'n_c', 'u_par_c_cms', 'T_par_c_eV', 'T_perp_c_eV',
                  'n_h', 'u_par_h_cms', 'T_par_h_eV', 'T_perp_h_eV', 'kappa_h',
                  'n_b', 'u_par_b_cms', 'T_par_b_eV', 'T_perp_b_eV', 'kappa_b',
                  'n_total', 'deviance_per_dof', 'red_chi2', 'objective',
                  'strahl_sign', 'break_energy_eV', 'break_source',
                  'strahl_tier', 'pad_beta_par', 'pad_beta_anti',
                  'pad_paw_par', 'pad_paw_anti', 'pad_deficit',
                  'pinned_params', 'runtime_s', 'error', 'fit_params_json']


def setupLogging(outputDir):
    """File + console logging; the file lives inside the run's output dir."""
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler(Path(outputDir) / 'run.log'),
                  logging.StreamHandler()])
    # matplotlib reports font substitutions at INFO level; not run progress.
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def findPinnedParams(params, paramBounds):
    pinnedParams = []
    for name, (boxLow, boxHigh) in paramBounds.items():
        boxWidth = boxHigh - boxLow
        if boxWidth > 0 and min(params[name] - boxLow,
                                boxHigh - params[name]) < PINNED_BOX_FRACTION * boxWidth:
            pinnedParams.append(name)
    return pinnedParams


def fitOneEpoch(chunkPath, epochIndex, plotPath=None, breakEnergyEv=None,
                objective='poisson', freeHaloDrift=True):
    """Fit a single epoch; returns one result row (dict, see RESULT_COLUMNS).

    breakEnergyEv: per-epoch crossing break energy (eV); None falls back to
    the 7.7 T_c|| proxy inside the staged fit.
    freeHaloDrift: False pins the halo parallel drift at 0 (default: free).
    """
    row = {'epoch_index': epochIndex, 'objective': objective}
    startTime = time.perf_counter()
    try:
        grid, psd, counts, epochNs, scDensity = loadEpoch(chunkPath, epochIndex)
        row['epoch_ns'] = int(epochNs)
        row['time_iso'] = str(pd.Timestamp(int(epochNs)))
        row['n_sc'] = float(scDensity)
        if not grid['valid'].any():
            row['error'] = ('no_velocity_grid'
                            if not np.isfinite(grid['v_par']).any() else 'no_data')
            row['runtime_s'] = time.perf_counter() - startTime
            return row

        strahlDetection = detectStrahlFromPad(chunkPath, epochIndex,
                                              breakEnergyEv=breakEnergyEv)
        seedValues = seedFromData(psd, counts, grid,
                                  strahlDetection=strahlDetection,
                                  scDensity=scDensity,
                                  breakEnergyEv=breakEnergyEv)
        params, modelPsd, diagnostics = fitEpoch(psd, counts, grid, seedValues,
                                                 objective=objective,
                                                 freeHaloDrift=freeHaloDrift)
        numFreeParams = diagnostics['free_params']
        moments = componentMomentsFromParams(params)
        row.update({'n_c': moments['n_c'], 'u_par_c_cms': moments['u_par_c'],
                    'T_par_c_eV': moments['T_par_c_eV'],
                    'T_perp_c_eV': moments['T_perp_c_eV'],
                    'n_h': moments['n_h'], 'u_par_h_cms': moments['u_par_h'],
                    'T_par_h_eV': moments['T_par_h_eV'],
                    'T_perp_h_eV': moments['T_perp_h_eV'], 'kappa_h': moments['kappa_h'],
                    'n_b': moments['n_b'], 'u_par_b_cms': moments['u_par_b'],
                    'T_par_b_eV': moments['T_par_b_eV'],
                    'T_perp_b_eV': moments['T_perp_b_eV'], 'kappa_b': moments['kappa_b'],
                    'n_total': moments['n_c'] + moments['n_h'] + moments['n_b'],
                    'deviance_per_dof': poissonDeviancePerDof(
                        modelPsd, counts, grid, numFreeParams),
                    'red_chi2': reducedChiSquare(modelPsd, psd, counts, grid,
                                                 numFreeParams),
                    'strahl_sign': diagnostics['strahl_sign'],
                    'break_energy_eV': diagnostics['break_energy_eV'],
                    'break_source': diagnostics['break_source'],
                    'strahl_tier': strahlDetection['class'],
                    'pad_beta_par': strahlDetection['beta_par'],
                    'pad_beta_anti': strahlDetection['beta_anti'],
                    'pad_paw_par': strahlDetection['paw_par'],
                    'pad_paw_anti': strahlDetection['paw_anti'],
                    'pad_deficit': strahlDetection['deficit'],
                    'pinned_params': ';'.join(
                        findPinnedParams(params, diagnostics['param_bounds'])),
                    'fit_params_json': json.dumps(
                        {key: float(value) for key, value in params.items()}),
                    'error': ''})
        if plotPath is not None:
            import matplotlib
            matplotlib.use('Agg')
            from .plotting import plotPoissonFitEpoch
            plotPoissonFitEpoch(grid, psd, counts, params, diagnostics,
                                row['deviance_per_dof'], row['time_iso'], plotPath)
    except Exception:
        row['error'] = traceback.format_exc(limit=3).replace('\n', ' | ')
    row['runtime_s'] = time.perf_counter() - startTime
    return row


def loadBreakEnergies(breakDir, chunkPath):
    """{epoch_id: break_E_eV} from the file's crossing CSV ('ok' rows only);
    empty when the CSV is missing (every epoch then uses the proxy break)."""
    if breakDir is None:
        return {}
    csvPath = Path(breakDir) / (Path(chunkPath).stem + BREAK_FILE_SUFFIX)
    if not csvPath.exists():
        log.warning(f'no break-energy file {csvPath.name}: using the 7.7 T_c proxy')
        return {}
    frame = pd.read_csv(csvPath)
    okRows = frame[(frame['status'] == 'ok') & np.isfinite(frame['break_E_eV'])]
    return dict(zip(okRows['epoch_id'].astype(int), okRows['break_E_eV'].astype(float)))


def resultCsvPath(outputDir, chunkPath):
    return Path(outputDir) / 'per_file' / f'{Path(chunkPath).stem}_fit.csv'


def isFileComplete(csvPath, numEpochs):
    if not csvPath.exists():
        return False
    try:
        return len(pd.read_csv(csvPath)) == numEpochs
    except Exception:
        return False


def processFile(chunkPath, outputDir, numWorkers, objective='poisson',
                plotEvery=0, breakByEpoch=None, freeHaloDrift=True):
    """Fit every epoch of one file in a process pool; writes its CSV.
    Returns (numRows, numFailed)."""
    numEpochs = countEpochs(chunkPath)
    csvPath = resultCsvPath(outputDir, chunkPath)
    csvPath.parent.mkdir(parents=True, exist_ok=True)
    breakByEpoch = breakByEpoch or {}

    def plotPathFor(epochIndex):
        if not plotEvery or epochIndex % plotEvery != 0:
            return None
        plotDir = Path(outputDir) / 'plots' / Path(chunkPath).stem
        plotDir.mkdir(parents=True, exist_ok=True)
        return str(plotDir / f'epoch{epochIndex:04d}.png')

    rows = []
    fileStart = time.perf_counter()
    # Fresh (spawned) workers import the package themselves, so the one-thread
    # BLAS setting in evdf_poisson/__init__ applies even when the caller (a
    # notebook, say) had numpy loaded long before; same behaviour on every OS.
    spawnContext = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=numWorkers, mp_context=spawnContext) as pool:
        futures = {pool.submit(fitOneEpoch, chunkPath, epochIndex,
                               plotPathFor(epochIndex),
                               breakByEpoch.get(epochIndex), objective,
                               freeHaloDrift): epochIndex
                   for epochIndex in range(numEpochs)}
        for finishedCount, future in enumerate(as_completed(futures), 1):
            epochIndex = futures[future]
            try:
                rows.append(future.result())
            except Exception as workerError:
                rows.append({'epoch_index': epochIndex, 'objective': objective,
                             'error': f'worker: {workerError}', 'runtime_s': np.nan})
            if finishedCount % PROGRESS_EVERY_EPOCHS == 0 or finishedCount == numEpochs:
                elapsedMinutes = (time.perf_counter() - fileStart) / 60
                log.info(f'    {finishedCount}/{numEpochs} epochs ({elapsedMinutes:.1f} min)')

    frame = pd.DataFrame(rows).sort_values('epoch_index')
    frame = frame.reindex(columns=RESULT_COLUMNS)
    frame.to_csv(csvPath, index=False)
    numFailed = int((frame['error'].fillna('') != '').sum())
    return len(frame), numFailed


def mergeResultCsvs(outputDir):
    """Concatenate the per-file CSVs into <outputDir>/summary.csv; returns the frame."""
    outputDir = Path(outputDir)
    csvPaths = sorted((outputDir / 'per_file').glob('*_fit.csv'))
    merged = pd.concat([pd.read_csv(path) for path in csvPaths], ignore_index=True)
    merged.to_csv(outputDir / 'summary.csv', index=False)
    log.info(f'merged {len(csvPaths)} file CSVs -> summary.csv ({len(merged)} rows)')
    return merged


def runFiles(dataDir, outputDir, fileNames=None, breakDir=None, workers=4,
             objective='poisson', plotEvery=0, resume=True, freeHaloDrift=True):
    """Fit all (or the named) VDF files in dataDir; returns the merged summary frame.

    breakDir defaults to dataDir (the crossing CSVs sit next to the h5 files).
    With resume=True files whose CSV is already complete are skipped.
    freeHaloDrift=False pins the halo parallel drift at 0 (default: free)."""
    if objective not in OBJECTIVES:
        raise ValueError(f'objective must be one of {OBJECTIVES}, got {objective!r}')
    dataDir = Path(dataDir)
    outputDir = Path(outputDir)
    outputDir.mkdir(parents=True, exist_ok=True)
    if not log.handlers and not logging.getLogger().handlers:
        setupLogging(outputDir)
    breakDir = dataDir if breakDir is None else Path(breakDir)

    if fileNames is None:
        chunkPaths = sorted(dataDir.glob(INPUT_FILE_GLOB))
    else:
        chunkPaths = [dataDir / Path(name).name for name in fileNames]
    if not chunkPaths:
        raise FileNotFoundError(f'no {INPUT_FILE_GLOB} files in {dataDir}')
    log.info(f'{len(chunkPaths)} files, {workers} workers, objective={objective}, '
             f'halo drift {"free" if freeHaloDrift else "fixed 0"} -> {outputDir}')

    for fileNumber, chunkPath in enumerate(chunkPaths, 1):
        numEpochs = countEpochs(chunkPath)
        csvPath = resultCsvPath(outputDir, chunkPath)
        if resume and isFileComplete(csvPath, numEpochs):
            log.info(f'[{fileNumber}/{len(chunkPaths)}] {chunkPath.name} already complete, skipping')
            continue
        log.info(f'[{fileNumber}/{len(chunkPaths)}] {chunkPath.name} ({numEpochs} epochs)')
        numRows, numFailed = processFile(
            str(chunkPath), outputDir, workers, objective=objective,
            plotEvery=plotEvery, breakByEpoch=loadBreakEnergies(breakDir, chunkPath),
            freeHaloDrift=freeHaloDrift)
        log.info(f'    done: {numRows} rows, {numFailed} failed -> {csvPath.name}')

    return mergeResultCsvs(outputDir)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-dir', required=True, help='directory with the VDF h5 files')
    parser.add_argument('--out', required=True, help='output directory of this run')
    parser.add_argument('--break-dir', default=None,
                        help='directory with <stem>_two_population_crossing.csv '
                             '(default: --data-dir)')
    parser.add_argument('--files', nargs='*', default=None,
                        help='fit only these file names (default: all)')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--objective', choices=OBJECTIVES, default='poisson')
    parser.add_argument('--plot-every', type=int, default=0,
                        help='save the 4-panel diagnostic figure every N epochs (0 = off)')
    parser.add_argument('--no-resume', action='store_true',
                        help='refit files whose CSV already exists')
    parser.add_argument('--fix-halo-drift', action='store_true',
                        help='pin the halo parallel drift u_par_h at 0 (default: free)')
    args = parser.parse_args()
    runFiles(args.data_dir, args.out, fileNames=args.files, breakDir=args.break_dir,
             workers=args.workers, objective=args.objective,
             plotEvery=args.plot_every, resume=not args.no_resume,
             freeHaloDrift=not args.fix_halo_drift)


if __name__ == '__main__':
    main()
