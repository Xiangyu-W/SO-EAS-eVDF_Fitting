"""Run the three-component eVDF fit on the bundled example dataset.

Fits 11 one-minute epochs of Solar Orbiter SWA-EAS data
(2022-03-02 12:15-12:25 UT) shipped in examples/example_data/. Takes a few
minutes
on 4 cores; increase n_jobs below if you have more. Fits are deterministic:
rerunning (with any n_jobs) reproduces the same parameters exactly.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

import pandas as pd

from main_fitting import main2

RUN_NAME = 'example_run'
N_JOBS = 4

if __name__ == '__main__':
    main2(run_name=RUN_NAME, n_jobs=N_JOBS)  # all other defaults point at examples/example_data/

    run_dir = REPO_ROOT / 'results' / 'fitting' / RUN_NAME
    summary = pd.read_csv(run_dir / 'data' / 'summary.csv')
    successCount = (summary['status'] == 'success').sum()

    print(f'\n{successCount}/{len(summary)} epochs fitted successfully')
    print(summary[['epoch_id', 'time_stamp', 'status', 'redChiSqr']].to_string(index=False))
    print(f"\nPlots:       {run_dir / 'plots'}")
    print(f"Fit results: {run_dir / 'data' / 'chunks'}")
    print(f"Summary:     {run_dir / 'data' / 'summary.csv'}")
