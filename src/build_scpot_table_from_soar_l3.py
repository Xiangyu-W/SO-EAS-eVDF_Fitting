"""Build a spacecraft-potential table for ``preprocess_main(scpot_file=...)``
from the SOAR RPW L3 ``rpw-bia-scpot`` daily files.

Why this exists: with ``shift_SCPOT=True`` the preprocessor fetches the CDAWeb
10-second SCPOT product, which ends on 2023-01-01. SOAR keeps publishing the
full-cadence (about 3 ms) L3 daily files beyond that date, so for later windows
this script reduces them to a 20 s median series in the two-column format the
preprocessor already accepts (``time, scpotV``; extra columns are ignored).

Usage (repository root):

    PYTHONPATH=src python src/build_scpot_table_from_soar_l3.py \\
        --start 2023-04-04 --end 2023-04-05 --download \\
        --out examples/example_data/scpot_rpw_l3_20230404_20230405_20s.csv

``--end`` is the last day to include. Daily files start a couple of minutes
after midnight and run into the next day, so the day before ``--start`` is read
as well when it is present, and the table carries ``--margin-minutes`` on both
sides of the requested window.
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

SCPOT_PRODUCT = 'rpw-bia-scpot'
DEFAULT_BIN_SECONDS = 20
DEFAULT_MIN_QUALITY_FLAG = 3   # RPW QUALITY_FLAG: 3 = nominal, lower = degraded
DEFAULT_MARGIN_MINUTES = 30
MAD_TO_SIGMA = 1.4826


def dailyFilePattern(day):
    return f'solo_L3_{SCPOT_PRODUCT}_{day:%Y%m%d}_V*.cdf'


def discoverDailyFiles(startDay, endDay, cdfDir):
    """Newest version of every daily file from the day before startDay to endDay."""
    files = []
    for day in pd.date_range(startDay - pd.Timedelta(days=1), endDay, freq='D'):
        candidates = sorted(glob.glob(os.path.join(cdfDir, dailyFilePattern(day))))
        if candidates:
            files.append(candidates[-1])
    return files


def downloadDailyFiles(startDay, endDay, cdfDir):
    """Fetch the daily files through sunpy-soar (imported lazily: only needed here)."""
    from sunpy.net import Fido, attrs as soarAttrs
    import sunpy_soar  # noqa: F401  registers the SOAR client

    queryStart = (startDay - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    queryEnd = (endDay + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    result = Fido.search(soarAttrs.Time(queryStart, queryEnd), soarAttrs.Level(3),
                         soarAttrs.soar.Product(SCPOT_PRODUCT))
    os.makedirs(cdfDir, exist_ok=True)
    fetched = Fido.fetch(result, path=os.path.join(cdfDir, '{file}'), progress=False)
    if fetched.errors:
        raise RuntimeError(f'SOAR download failed for: {fetched.errors}')
    return list(fetched)


def readDailyPotential(filePath, minQualityFlag):
    """Full-cadence SCPOT of one daily file with fill values and low-quality samples removed."""
    import cdflib
    cdf = cdflib.CDF(filePath)
    potential = np.asarray(cdf.varget('SCPOT'), dtype=float)
    qualityFlag = np.asarray(cdf.varget('QUALITY_FLAG'))
    fillValue = cdf.varattsget('SCPOT').get('FILLVAL', -1e31)
    isValid = np.isfinite(potential) & (potential != fillValue) & (qualityFlag >= minQualityFlag)
    epoch = cdflib.cdfepoch.to_datetime(cdf.varget('Epoch'))
    frame = pd.DataFrame({'time': epoch[isValid], 'scpotV': potential[isValid],
                          'qualityFlag': qualityFlag[isValid]})
    print(f'{os.path.basename(filePath)}: {isValid.sum()}/{len(potential)} samples kept '
          f'(QUALITY_FLAG >= {minQualityFlag}); generated {cdf.globalattsget().get("Generation_date", ["?"])[0]}')
    return frame


def binMedian(frame, binSeconds):
    """Median per bin, labelled at the bin centre, with a robust spread and the sample count."""
    indexed = frame.set_index('time').sort_index()
    grouped = indexed['scpotV'].resample(f'{binSeconds}s')
    table = pd.DataFrame({
        'scpotV': grouped.median(),
        'scpotSigmaV': grouped.apply(lambda values: MAD_TO_SIGMA * np.median(np.abs(values - np.median(values)))
                                     if len(values) else np.nan),
        'nSamples': grouped.count(),
        'qualityFlagMin': indexed['qualityFlag'].resample(f'{binSeconds}s').min(),
    })
    table = table[table['nSamples'] > 0]
    table.index = table.index + pd.Timedelta(seconds=binSeconds / 2)
    table.index.name = 'time'
    return table.reset_index()


def buildTable(startDay, endDay, cdfDir, binSeconds, minQualityFlag, marginMinutes):
    files = discoverDailyFiles(startDay, endDay, cdfDir)
    if not files:
        raise FileNotFoundError(f'no {SCPOT_PRODUCT} daily files for {startDay:%Y-%m-%d}..{endDay:%Y-%m-%d} in {cdfDir}')
    frames = [readDailyPotential(path, minQualityFlag) for path in files]
    table = binMedian(pd.concat(frames, ignore_index=True), binSeconds)
    windowStart = startDay - pd.Timedelta(minutes=marginMinutes)
    windowEnd = endDay + pd.Timedelta(days=1, minutes=marginMinutes)
    table = table[(table['time'] >= windowStart) & (table['time'] < windowEnd)].reset_index(drop=True)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--start', required=True, help='first day, YYYY-MM-DD')
    parser.add_argument('--end', required=True, help='last day (inclusive), YYYY-MM-DD')
    parser.add_argument('--cdf-dir', default=str(Path(__file__).resolve().parents[1] / 'data' / 'rpw_l3_scpot'))
    parser.add_argument('--out', required=True, help='CSV path for the table')
    parser.add_argument('--download', action='store_true', help='fetch the daily files from SOAR first')
    parser.add_argument('--bin-seconds', type=int, default=DEFAULT_BIN_SECONDS)
    parser.add_argument('--min-quality-flag', type=int, default=DEFAULT_MIN_QUALITY_FLAG)
    parser.add_argument('--margin-minutes', type=int, default=DEFAULT_MARGIN_MINUTES)
    args = parser.parse_args()

    startDay = pd.Timestamp(args.start).normalize()
    endDay = pd.Timestamp(args.end).normalize()
    if args.download:
        downloadDailyFiles(startDay, endDay, args.cdf_dir)
    table = buildTable(startDay, endDay, args.cdf_dir, args.bin_seconds, args.min_quality_flag, args.margin_minutes)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    table.to_csv(args.out, index=False, float_format='%.4f', date_format='%Y-%m-%dT%H:%M:%S')
    coverage = table['nSamples'].gt(0).mean()
    print(f'wrote {len(table)} rows to {args.out}: {table["time"].iloc[0]} .. {table["time"].iloc[-1]}, '
          f'median {table["scpotV"].median():.2f} V, p5/p95 {table["scpotV"].quantile(0.05):.2f}/{table["scpotV"].quantile(0.95):.2f} V, '
          f'bins with data {coverage:.1%}')


if __name__ == '__main__':
    main()
