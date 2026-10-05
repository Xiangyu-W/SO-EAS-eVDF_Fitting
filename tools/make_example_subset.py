"""Cut a small example subset out of a full preprocessed VDF file.

Extracts epochs [epoch_begin, epoch_end) from a 6-hour ``VDF_eas_forFitting_*.h5``
file and writes them with the pipeline's own HDF5 writer, so the structure is
identical by construction; file attributes (scpot source, saturation mask) are
copied. Run ``compute_break_energy_crossing.py`` on the output directory
afterwards to produce the matching crossing CSV.

    python tools/make_example_subset.py --source <chunk.h5> --epoch-begin 33 --epoch-end 63 \\
        --out examples/example_data/poisson_2023-04-04
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from data_io import load_data                      # noqa: E402
from preprocess_data_for_fit_hdf5 import save_data_toHDF5   # noqa: E402

STATIC_KEYS = {'pitch_angles', 'one_count_psd_in',
               'instrument_energy_eas1_eV', 'instrument_energy_eas2_eV'}   # every other array is time-indexed on axis 0


def sliceTimeAxis(value, epochCount, epochBegin, epochEnd):
    if value is None:
        return None
    assert value.shape[0] == epochCount, f'unexpected leading dimension {value.shape}'
    return value[epochBegin:epochEnd]


def cutSubset(sourcePath, epochBegin, epochEnd, outputDir):
    data = load_data(sourcePath.parent, sourcePath.name)
    epochCount = len(data['time_EAS'])
    if not 0 <= epochBegin < epochEnd <= epochCount:
        raise ValueError(f'epoch range [{epochBegin}, {epochEnd}) outside 0..{epochCount}')

    subset = {}
    for key, value in data.items():
        if isinstance(value, dict):
            subset[key] = {subKey: sliceTimeAxis(subValue, epochCount, epochBegin, epochEnd)
                           for subKey, subValue in value.items()}
        elif value is None or key in STATIC_KEYS:
            subset[key] = value
        else:
            subset[key] = sliceTimeAxis(value, epochCount, epochBegin, epochEnd)

    firstTime = pd.Timestamp(subset['time_EAS'][0])
    lastTime = pd.Timestamp(subset['time_EAS'][-1])
    stem = f'VDF_eas_forFitting_{firstTime:%Y%m%d%H%M%S}_{lastTime:%Y%m%d%H%M%S}_1min'
    outputDir.mkdir(parents=True, exist_ok=True)
    outputPath = outputDir / f'{stem}.h5'
    save_data_toHDF5(subset, str(outputPath))

    with h5py.File(sourcePath, 'r') as source, h5py.File(outputPath, 'a') as target:
        for name, value in source.attrs.items():
            target.attrs[name] = value
        for key in STATIC_KEYS & set(source) & set(target):
            for name, value in source[key].attrs.items():
                target[key].attrs[name] = value
        target.attrs['source_file'] = sourcePath.name
        target.attrs['source_epochs'] = f'{epochBegin}-{epochEnd - 1}'

    reloaded = load_data(outputDir, outputPath.name)
    assert set(reloaded) == set(data), 'key set mismatch'
    assert np.array_equal(reloaded['time_EAS'], data['time_EAS'][epochBegin:epochEnd])
    assert np.array_equal(reloaded['pa_vec_in'], data['pa_vec_in'][epochBegin:epochEnd], equal_nan=True)
    sizeMb = outputPath.stat().st_size / 1024 ** 2
    print(f'wrote {outputPath} ({sizeMb:.1f} MB): epochs {epochBegin}-{epochEnd - 1}, {firstTime} .. {lastTime}')
    return outputPath


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--epoch-begin', required=True, type=int)
    parser.add_argument('--epoch-end', required=True, type=int, help='exclusive')
    parser.add_argument('--out', required=True, type=Path, help='output directory')
    args = parser.parse_args()
    cutSubset(args.source, args.epoch_begin, args.epoch_end, args.out)


if __name__ == '__main__':
    main()
