"""Fit the bundled example dataset with the Poisson pipeline.

Fits the 30 one-minute epochs of Solar Orbiter SWA-EAS data (2023-04-04
00:50-01:19 UT) in examples/example_data/poisson_2023-04-04/ and writes the
results to results/fitting/example_run_poisson/ (summary.csv, one CSV per
input file and a four-panel diagnostic figure per epoch). Takes about 1.5
minutes on 4 cores.
The fits are deterministic: a rerun reproduces the same parameters exactly.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from evdf_poisson.runner import runFiles

EXAMPLE_DIR = REPO_ROOT / 'examples' / 'example_data' / 'poisson_2023-04-04'
OUTPUT_DIR = REPO_ROOT / 'results' / 'fitting' / 'example_run_poisson'
N_WORKERS = 4

if __name__ == '__main__':
    summary = runFiles(EXAMPLE_DIR, OUTPUT_DIR, workers=N_WORKERS, plotEvery=1, resume=False)

    fittedCount = int((summary['error'].fillna('') == '').sum())
    print(f'\n{fittedCount}/{len(summary)} epochs fitted')
    print(summary[['epoch_index', 'time_iso', 'strahl_tier', 'n_c', 'T_par_c_eV', 'n_h',
                   'kappa_h', 'n_b', 'deviance_per_dof']].round(3).to_string(index=False))
    print(f"\nSummary: {OUTPUT_DIR / 'summary.csv'}")
    print(f"Figures: {OUTPUT_DIR / 'plots'}")
