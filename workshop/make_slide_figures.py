"""Produce the slide figures of the S3Park 2026 session into workshop/figures/.

Deterministic: everything is drawn from the bundled example subset, the
geometry file and the reference fit CSVs. Run from the repository root:

    PYTHONPATH=src python workshop/make_slide_figures.py
"""
import json
import os
import shutil
import sys
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name, '1')

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import poisson

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from eas_geometry import HEAD_SLICES, easAzimuthElevation, loadHeadGeometry, pixelUnitVectorsSrf   # noqa: E402
from evdf_poisson import componentMomentsFromParams, detectStrahlFromPad, fitEpoch, loadEpoch, seedFromData   # noqa: E402
from evdf_poisson.plotting import plotFitComparison, plotMeasuredVdf, plotPadSpectrogram, wingMask   # noqa: E402

CALIBRATION_DIR = REPO_ROOT / 'examples' / 'example_data'
EXAMPLE_DIR = CALIBRATION_DIR / 'poisson_2023-04-04'
EXAMPLE_FILE = EXAMPLE_DIR / 'VDF_eas_forFitting_20230404005008_20230404011908_1min.h5'
RESULTS_DIR = REPO_ROOT / 'results' / 'fitting' / 'example_run_poisson'
DAYS_DIR = REPO_ROOT / 'results' / 'fitting' / 'example_days_2023-04-04_05'
FIGURE_DIR = REPO_ROOT / 'workshop' / 'figures'
EPOCH = 20
DPI = 160


def save(fig, name):
    fig.savefig(FIGURE_DIR / name, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print('wrote', FIGURE_DIR / name)


def figureHeadsSrf(magneticField):
    geometry = loadHeadGeometry(CALIBRATION_DIR / 'eas_head_geometry.npz')
    azimuth, elevation = easAzimuthElevation(pixelUnitVectorsSrf(geometry))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for headName, marker, color in (('EAS1', 'x', 'blue'), ('EAS2', '+', 'red')):
        block = HEAD_SLICES[headName]
        ax.scatter(azimuth[block], elevation[block], s=16, marker=marker, color=color, alpha=0.75, label=headName)
    for vector, marker, label in ((magneticField, '*', 'along +B (pitch angle 0)'), (-magneticField, 'P', 'along -B (pitch angle 180)'),
                                  ((-1.0, 0.0, 0.0), 'v', 'moving away from the Sun (-X)'), ((1.0, 0.0, 0.0), '^', 'moving towards the Sun (+X)')):
        markerAzimuth, markerElevation = easAzimuthElevation(np.asarray(vector, float))
        ax.plot(markerAzimuth, markerElevation, marker, color='k', markersize=12, markerfacecolor='yellow', label=label)
    ax.set_xlim(-186, 186); ax.set_ylim(-92, 92)
    ax.set_xticks(np.arange(-180, 181, 30)); ax.set_yticks(np.arange(-90, 91, 15))
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6); ax.legend(fontsize=8, loc='upper center', ncol=3)
    ax.set_xlabel('EAS azimuth [deg]  (0 = moving along -X, away from the Sun; clockwise positive)'); ax.set_ylabel('elevation [deg]')
    ax.set_title('SWA-EAS two-head field of view in SRF, 2023-04-04 01:10 UT (velocity directions; Owen et al. 2020 convention)')
    save(fig, 'eas_heads_srf.png')


def figureOneCountAndZeros(grid, counts):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    rows = [row for row in range(64) if grid['valid'][row].any()]
    rowEnergy = [np.median(grid['energy_eV'][row][grid['valid'][row]]) for row in rows]
    rowOneCount = [np.median(grid['one_count_psd'][row][grid['valid'][row]]) for row in rows]
    axes[0].loglog(rowEnergy, rowOneCount, 'o-', color='0.3', markersize=3)
    axes[0].set_xlabel('energy [eV]'); axes[0].set_ylabel(r'one-count level $g$ [s$^3$ cm$^{-6}$]'); axes[0].grid(alpha=0.3)
    axes[0].set_title('PSD represented by a single count')
    for wingName, label, color in (('par', 'parallel cone', '#2a78d6'), ('anti', 'anti-parallel cone', '#e34948'), ('perp', 'perpendicular cone', '0.3')):
        cone = wingMask(grid, wingName)
        energies = np.array([np.median(grid['energy_eV'][row][cone[row]]) if cone[row].any() else np.nan for row in range(64)])
        zeroFraction = np.array([np.mean(counts[row][cone[row]] == 0) if cone[row].any() else np.nan for row in range(64)])
        axes[1].semilogx(energies, zeroFraction, 'o-', markersize=3, color=color, label=label)
    axes[1].set_xlabel('energy [eV]'); axes[1].set_ylabel('fraction of pixels with zero counts'); axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=9); axes[1].set_title('most pixels above 100 eV are empty')
    fig.tight_layout(); save(fig, 'one_count_and_zero_fractions.png')


def figureExpectedDeviance():
    def expectedDeviancePerPixel(mu, maxCount=400):
        k = np.arange(maxCount + 1)[:, None]; muRow = np.asarray(mu)[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            logTerm = np.where(k > 0, k * np.log(k / muRow), 0.0)
        return np.sum(poisson.pmf(k, muRow) * 2.0 * (muRow - k + logTerm), axis=0)
    mu = np.logspace(-3, 2, 120)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.semilogx(mu, expectedDeviancePerPixel(mu), color='0.2', linewidth=2)
    ax.axhline(1, color='#d62728', linestyle='--', label='chi-square expectation')
    ax.axvspan(0.01, 1, color='0.92', label='typical suprathermal EAS pixel')
    ax.set_xlabel(r'expected counts $\mu$ in a pixel'); ax.set_ylabel(r'E[$D$] per pixel for a perfect model')
    ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=9)
    ax.set_title('why deviance/dof is not 1 for a good fit'); save(fig, 'expected_deviance.png')


def figureTwoDays():
    if not (DAYS_DIR / 'summary.csv').exists():
        print('two-day summary missing, skipping two_days_parameters.png'); return
    days = pd.read_csv(DAYS_DIR / 'summary.csv'); days['time'] = pd.to_datetime(days['time_iso'])
    days = days[days['error'].fillna('') == ''].sort_values('time')
    hasStrahl = days['n_b'] > 0
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(days['time'], days['n_sc'], 'k-', linewidth=0.8, label='RPW $n_{sc}$')
    axes[0].plot(days['time'], days['n_c'], '-', color='#e34948', linewidth=0.8, label='core')
    axes[0].plot(days['time'], days['n_h'], '-', color='#eda100', linewidth=0.8, label='halo')
    axes[0].plot(days['time'][hasStrahl], days['n_b'][hasStrahl], '.', color='#2a78d6', markersize=2, label='strahl')
    axes[0].set_yscale('log'); axes[0].set_ylabel('density [cm$^{-3}$]')
    axes[1].plot(days['time'], days['T_par_c_eV'], '-', color='#e34948', linewidth=0.8, label=r'core $T_\parallel$')
    axes[1].plot(days['time'], days['T_par_h_eV'], '-', color='#eda100', linewidth=0.8, label=r'halo $T_\parallel$')
    axes[1].plot(days['time'][hasStrahl], days['T_par_b_eV'][hasStrahl], '.', color='#2a78d6', markersize=2, label=r'strahl $T_\parallel$')
    axes[1].set_yscale('log'); axes[1].set_ylabel('temperature [eV]')
    axes[2].plot(days['time'], days['break_energy_eV'], '-', color='0.3', linewidth=0.8, label='break energy')
    axes[2].plot(days['time'], 7.7 * days['T_par_c_eV'], '--', color='#e34948', linewidth=0.8, label=r'7.7 $\times$ core $T_\parallel$')
    axes[2].set_ylabel('energy [eV]')
    tierCode = days['strahl_tier'].map({'anti_strong': -2, 'anti_weak': -1, 'quasi_iso': 0, 'par_weak': 1, 'par_strong': 2, 'both': 3})
    axes[3].plot(days['time'], tierCode, '.', color='0.3', markersize=2)
    axes[3].set_yticks([-2, -1, 0, 1, 2]); axes[3].set_yticklabels(['anti strong', 'anti weak', 'none', 'par weak', 'par strong']); axes[3].set_ylabel('PAD verdict')
    for ax in axes[:3]: ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9)
    for ax in axes: ax.grid(alpha=0.3)
    fig.autofmt_xdate(); fig.tight_layout(); save(fig, 'two_days_parameters.png')


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    grid, psd, counts, epochNs, scDensity = loadEpoch(EXAMPLE_FILE, EPOCH)
    epochTime = str(pd.Timestamp(int(epochNs)))
    with h5py.File(EXAMPLE_FILE, 'r') as h5:
        magneticField = h5['B_avg'][EPOCH]
    reference = pd.read_csv(RESULTS_DIR / 'summary.csv').set_index('epoch_index')
    fitParams = json.loads(reference.loc[EPOCH, 'fit_params_json'])
    breakEnergyEv = float(reference.loc[EPOCH, 'break_energy_eV'])

    figureHeadsSrf(magneticField)
    figureOneCountAndZeros(grid, counts)
    save(plotMeasuredVdf(grid, psd, counts, params=fitParams, breakEnergyEv=breakEnergyEv, timeLabel=epochTime), 'measured_vdf_components.png')
    ax = plotPadSpectrogram(EXAMPLE_FILE, EPOCH); ax.set_title(f'pitch-angle distribution, {epochTime}'); save(ax.figure, 'pad_spectrogram.png')
    figureExpectedDeviance()
    shutil.copy(RESULTS_DIR / 'plots' / EXAMPLE_FILE.stem / f'epoch{EPOCH:04d}.png', FIGURE_DIR / 'four_panel_epoch20.png')
    print('copied four_panel_epoch20.png')

    detection = detectStrahlFromPad(EXAMPLE_FILE, EPOCH, breakEnergyEv)
    seeds = seedFromData(psd, counts, grid, strahlDetection=detection, scDensity=scDensity, breakEnergyEv=breakEnergyEv)
    fits = {}
    for objective, label in (('poisson', 'Poisson (all pixels)'), ('chi2', 'chi2 (counts > 0)')):
        params, _, _ = fitEpoch(psd, counts, grid, seeds, objective=objective)
        fits[label] = params
    save(plotFitComparison(grid, psd, counts, fits, breakEnergyEv=breakEnergyEv, timeLabel=epochTime), 'poisson_vs_chi2_epoch20.png')
    moments = pd.DataFrame({label: componentMomentsFromParams(params) for label, params in fits.items()})
    moments.to_csv(FIGURE_DIR / 'poisson_vs_chi2_epoch20.csv')
    figureTwoDays()


if __name__ == '__main__':
    main()
