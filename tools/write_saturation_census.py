"""Per-epoch census of the L1 saturation blocks masked at preprocessing.

Reads ``n_saturated_pixels`` from every preprocessed file of a directory and
writes ``saturation_census.csv`` (epoch_ns, time_iso, file, n_saturated_pixels)
next to a fit run, so notebooks can mark those epochs without the 4.8 GB of data.

    python tools/write_saturation_census.py --data-dir data/processed/2023-04-04_05 \\
        --out results/fitting/example_days_2023-04-04_05/saturation_census.csv
"""
import argparse
import glob
import os

import h5py
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(glob.glob(os.path.join(args.data_dir, 'VDF_eas_forFitting_*.h5'))):
        with h5py.File(path, 'r') as h5:
            epochNs = h5['time_EAS'][:].astype('int64')
            saturated = h5['n_saturated_pixels'][:] if 'n_saturated_pixels' in h5 else [0] * len(epochNs)
        rows.extend({'epoch_ns': int(ns), 'time_iso': str(pd.Timestamp(int(ns))), 'file': os.path.basename(path),
                     'n_saturated_pixels': int(count)} for ns, count in zip(epochNs, saturated))
    table = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    table.to_csv(args.out, index=False)
    affected = int((table['n_saturated_pixels'] > 0).sum())
    print(f'{len(table)} epochs, {affected} with masked saturation blocks -> {args.out}')


if __name__ == '__main__':
    main()
