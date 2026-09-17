"""Generate the example notebooks of SO-EAS-eVDF_Fitting (poisson branch).

    python build_notebooks.py [00 01 ...]     # default: all defined
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from notebook_builder import Notebook

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / 'examples'

SETUP_CELL = '''import os
import sys
from pathlib import Path

# Single-threaded linear algebra, set before numpy is imported: the fits are
# many small least-squares problems, and this makes the numbers in this
# notebook identical to those of the batch runner (which spawns 1-thread workers).
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name, '1')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Repository root = the first parent directory that contains src/evdf_poisson
REPO_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'src' / 'evdf_poisson').is_dir())
sys.path.insert(0, str(REPO_ROOT / 'src'))

CALIBRATION_DIR = REPO_ROOT / 'examples' / 'example_data'          # EAS geometric factors, head geometry, scpot table
EXAMPLE_DIR = CALIBRATION_DIR / 'poisson_2023-04-04'                # the 30-epoch example subset
EXAMPLE_FILE = EXAMPLE_DIR / 'VDF_eas_forFitting_20230404005008_20230404011908_1min.h5'
RESULTS_DIR = REPO_ROOT / 'results' / 'fitting' / 'example_run_poisson'       # reference fit of the subset
DAYS_DIR = REPO_ROOT / 'results' / 'fitting' / 'example_days_2023-04-04_05'   # reference fit of the two full days
GEOMETRY_FILE = CALIBRATION_DIR / 'eas_head_geometry.npz'
print('repository:', REPO_ROOT)'''


# ----------------------------------------------------------------------- 00
def build00():
    nb = Notebook()
    nb.md('''# 00 - Quickstart: fit the example data

Fit the bundled example and look at the result: 30 one-minute epochs of Solar Orbiter
SWA-EAS data, 2023-04-04 00:50-01:19 UT, 0.33 au. No network needed.

**Runtime:** about 1.5 minutes on 4 workers.

To understand what happens inside, continue with notebook 01 and the ones after it.''')
    nb.code(SETUP_CELL)
    nb.md('''## 1. Fit every epoch of the example file

`runFiles` runs the whole pipeline on every `VDF_eas_forFitting_*.h5` file of a directory:

1. strahl detection from the pitch-angle distribution (PAD),
2. initial guesses from the data,
3. staged Poisson fit: core -> halo -> strahl -> joint,
4. physical parameters and fit-quality metrics.

It writes one CSV per file, a merged `summary.csv` and, with `plotEvery=1`, a diagnostic
figure per epoch. The break energies are read from the `*_two_population_crossing.csv`
file next to the data (notebook 03).''')
    nb.code('''from evdf_poisson.runner import runFiles

summary = runFiles(EXAMPLE_DIR, RESULTS_DIR, workers=4, plotEvery=1, resume=False)''')
    nb.md('''## 2. The fitted parameters

One row per epoch; densities in cm$^{-3}$, temperatures in eV, drifts in cm/s.

- `deviance_per_dof`: Poisson goodness of fit over all pixels, zero counts included.
- `red_chi2`: classical reduced chi-square, detected pixels only.
- `strahl_tier`: the PAD verdict.
- `pinned_params`: parameters that ended on the edge of their allowed range.

Notebook 06 explains every column.''')
    nb.code('''COLUMNS = ['time_iso', 'n_sc', 'n_c', 'T_par_c_eV', 'T_perp_c_eV',
           'n_h', 'u_par_h_cms', 'T_par_h_eV', 'kappa_h', 'n_b', 'T_par_b_eV', 'T_perp_b_eV',
           'deviance_per_dof', 'red_chi2', 'strahl_tier', 'pinned_params']
print(f"{(summary['error'].fillna('') == '').sum()}/{len(summary)} epochs fitted")
summary[COLUMNS].round(3)''')
    nb.md('''## 3. One diagnostic figure

- **Top:** cuts parallel and perpendicular to **B**. Points are channel means over all
  pixels of the cone (zeros included) with Poisson error bars. Curves: core (red),
  halo (orange), strahl (blue), sum (black).
- **Bottom left:** the measured 2D distribution with the model contours.
- **Bottom right:** the Poisson deviance pull of every pixel (red = data above model,
  blue = below).''')
    nb.code('''from IPython.display import Image

EPOCH = 20
figurePath = RESULTS_DIR / 'plots' / EXAMPLE_FILE.stem / f'epoch{EPOCH:04d}.png'
Image(filename=str(figurePath), width=900)''')
    nb.md('''## Where to go next

- `01_eas_instrument_and_data.ipynb` - the instrument, its frames, what the data are
- `02_download_and_preprocess.ipynb` - from SOAR files to the fitting-ready file (pre-executed)
- `03_pad_strahl_and_break_energy.ipynb` - the two inputs measured before the fit
- `04_models_and_initial_guesses.ipynb` - the three models, their initial guesses and ranges
- `05_poisson_fit_step_by_step.ipynb` - the four stages and the goodness of fit, one epoch
- `06_batch_fit_and_two_days.ipynb` - many epochs, the output files, two full days
- `07_poisson_vs_chi2_low_counts.ipynb` - why Poisson and not chi-square
- `08_fit_quality_qc.ipynb` - quality flags that work at low counts''')
    return nb


# ----------------------------------------------------------------------- 01
def build01():
    nb = Notebook()
    nb.md('''# 01 - The EAS instrument and its data

Everything the fit works with, shown on one real epoch:

0. the two sensor heads and their frames
1. counts
2. from counts to phase space density (PSD): the one-count level
3. magnetic field, pitch angle and the $(v_\\parallel, v_\\perp)$ grid
4. core, halo and strahl
5. the pitch-angle distribution (PAD)
6. spacecraft potential and the low-energy cut
7. the break energy
8. why zero-count pixels are data

**Runtime:** under a minute. Nothing is fitted here; the model curves come from the
reference run in `results/fitting/example_run_poisson/` (notebook 00 regenerates it).''')
    nb.code(SETUP_CELL)
    add01Geometry(nb)
    nb.md('''## 1. What EAS measures: counts

At each of the 64 energy steps, every pixel counts the electrons that arrive during the
accumulation time. The preprocessing (notebook 02) merges both heads into one grid of
**64 energies x 1024 pixels** per epoch and stores for every pixel its pitch angle, PSD,
$v_\\parallel$, $v_\\perp$ and the raw **counts**.

`loadEpoch` reads one epoch in CGS units.''')
    nb.code('''from evdf_poisson import loadEpoch

EPOCH = 20
grid, psd, counts, epochNs, scDensity = loadEpoch(EXAMPLE_FILE, EPOCH)
epochTime = pd.Timestamp(int(epochNs))
print('epoch time:', epochTime)
print('grid arrays:', {key: value.shape for key, value in grid.items()})
print('valid pixels (finite velocity, above the low-energy cut):', int(grid['valid'].sum()))
print('RPW electron density n_sc = %.1f cm^-3' % scDensity)''')
    nb.md('''Counts are small integers, and most of the grid registered **nothing**: above 100 eV,
three quarters or more of the pixels have zero counts. This is the regime where the
fitting objective matters (notebook 07).''')
    nb.code('''ENERGY_BANDS_EV = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 1000), (1000, 6000)]
rows = []
for low, high in ENERGY_BANDS_EV:
    inBand = grid['valid'] & (grid['energy_eV'] >= low) & (grid['energy_eV'] < high)
    bandCounts = counts[inBand]
    rows.append({'band_eV': f'{low}-{high}', 'pixels': inBand.sum(),
                 'zero_count_fraction': np.mean(bandCounts == 0),
                 'median_counts': np.median(bandCounts), 'max_counts': bandCounts.max()})
pd.DataFrame(rows).round(2)''')
    nb.md('''## 2. From counts to PSD: the one-count level

A pixel that registered $C$ counts has the phase space density

$$ f = C \\cdot g, \\qquad g = \\frac{m_e^2}{2\\,\\Delta t\\, E^2\\, G(\\text{el},\\text{az})\\, Q(E)} $$

with $\\Delta t$ the accumulation time, $G$ the geometric factor of the pixel and $Q$ the
detector efficiency.

$g$ is the **one-count level**: the PSD that a single count represents in that pixel. It
belongs to the instrument, not the plasma, and exists for every pixel, empty ones
included. That is what lets the Poisson fit use zero-count pixels: a model PSD becomes
an expected number of counts, $\\mu = f_\\text{model} / g$, to compare with the observed $C$.

`loadEpoch` recovers $g$ as PSD / counts where counts > 0 (exact) and uses the median of
the energy row for empty pixels.''')
    nb.code('''detected = grid['valid'] & (counts > 0)
print('PSD / counts equals the stored one-count level on every detected pixel:',
      np.allclose(psd[detected] / counts[detected], grid['one_count_psd'][detected], rtol=1e-9))

rowsWithData = [row for row in range(64) if grid['valid'][row].any()]
rowEnergy = [np.median(grid['energy_eV'][row][grid['valid'][row]]) for row in rowsWithData]
rowOneCount = [np.median(grid['one_count_psd'][row][grid['valid'][row]]) for row in rowsWithData]
fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(rowEnergy, rowOneCount, 'o-', color='0.3', markersize=3)
ax.set_xlabel('energy [eV]'); ax.set_ylabel(r'one-count level $g$ [s$^3$ cm$^{-6}$]')
ax.set_title('PSD of a single count, median over the pixels of each energy step')
ax.grid(alpha=0.3)''')
    nb.md('''## 3. Magnetic field, pitch angle, the gyrotropic grid

Electrons gyrate around **B**, so their distribution is close to symmetric about it. The
preprocessing rotates every pixel into the field-aligned frame (MAG averaged over the
epoch) and describes it by the **pitch angle** $\\alpha$ between the electron velocity
and **B**:

$$ v_\\parallel = v\\cos\\alpha, \\qquad v_\\perp = v\\sin\\alpha . $$

$\\alpha = 0$ is along **B**, 180 deg against it. Below: the pixels in the
$(v_\\parallel, v_\\perp)$ half-plane, coloured by pitch angle. Every energy step is a
semicircle.''')
    nb.code('''import h5py

with h5py.File(EXAMPLE_FILE, 'r') as h5:
    magneticField = h5['B_avg'][EPOCH]
    protonVelocity = h5['u'][EPOCH]
print('B (SRF, nT):', np.round(magneticField, 2), '   proton bulk velocity (SRF, km/s):', np.round(protonVelocity, 1))

valid = grid['valid']
fig, ax = plt.subplots(figsize=(7, 3.8))
scatterHandle = ax.scatter(grid['v_par'][valid] / 1e9, grid['v_perp'][valid] / 1e9,
                           c=grid['pitch_angle'][valid], s=2, cmap='twilight_shifted', vmin=0, vmax=180)
ax.set_aspect('equal'); ax.set_xlabel(r'$v_\\parallel$ [$10^9$ cm/s]'); ax.set_ylabel(r'$v_\\perp$ [$10^9$ cm/s]')
fig.colorbar(scatterHandle, ax=ax, label='pitch angle [deg]')
ax.set_title('pixel positions of one epoch');''')
    nb.md('''`plotMeasuredVdf` is the standard view of these notebooks:

- **left:** the parallel cut, channel means of the pixels within 15 deg of **B**
  ($v_\\parallel > 0$) and within 15 deg of $-$**B** ($v_\\parallel < 0$);
- **middle:** the perpendicular cut (pitch angles 75-105 deg);
- **right:** the binned 2D distribution.

Zero-count pixels enter the means as zeros; the error bars are Poisson. The bump at
negative $v_\\parallel$ is the **strahl**, a beam streaming away from the Sun along the
field.''')
    nb.code('''from evdf_poisson.plotting import plotMeasuredVdf

fig = plotMeasuredVdf(grid, psd, counts, timeLabel=str(epochTime))''')
    nb.md('''## 4. The three populations

The standard description (e.g. Stverak et al. 2009, Owen et al. 2022):

| component | model | parameters | typical share |
|---|---|---|---|
| **core** | bi-Maxwellian | $n_c$, $u_{\\parallel c}$, $T_{\\parallel c}$, $T_{\\perp c}$ | > 90 % of the density, below ~60-100 eV |
| **halo** | bi-kappa | $n_h$, $u_{\\parallel h}$, $T_{\\parallel h}$, $T_{\\perp h}$, $\\kappa_h$ | a few %, the near-isotropic high-energy tail |
| **strahl** | drifting, truncated bi-kappa | $n_b$, $u_{\\parallel b}$, $T_{\\parallel b}$, $T_{\\perp b}$, $\\kappa_b$ | ~1 %, a narrow beam along $\\pm$**B** |

Bi-Maxwellian:

$$ f_c(v_\\parallel, v_\\perp) = \\frac{n_c}{\\pi^{3/2} v_{th\\parallel} v_{th\\perp}^2}
   \\exp\\!\\left[-\\frac{(v_\\parallel - u_\\parallel)^2}{v_{th\\parallel}^2} - \\frac{v_\\perp^2}{v_{th\\perp}^2}\\right],
   \\qquad v_{th} = \\sqrt{2 k_B T / m_e}. $$

Bi-kappa: the exponential is replaced by
$\\left[1 + \\frac{(v_\\parallel - u_\\parallel)^2}{(\\kappa - 3/2) v_{th\\parallel}^2} + \\frac{v_\\perp^2}{(\\kappa - 3/2) v_{th\\perp}^2}\\right]^{-\\kappa - 1}$
times a $\\kappa$-dependent normalisation. Small $\\kappa$ is a hard power-law tail;
$\\kappa \\to \\infty$ recovers the Maxwellian. With the $(\\kappa - 3/2)$ convention, $T$
means the same for every component.

Strahl: a bi-kappa centred at the drift $u_{\\parallel b}$ whose parallel width is
compressed by $\\sqrt{5}$ on the side facing the core, so the beam does not leak into
the core.

The functions are in `evdf_poisson/models.py` (details in notebook 04). Below they are
drawn with the reference fit of this epoch.''')
    nb.code('''import json
from evdf_poisson import componentMomentsFromParams

referenceSummary = pd.read_csv(RESULTS_DIR / 'summary.csv').set_index('epoch_index')
fitParams = json.loads(referenceSummary.loc[EPOCH, 'fit_params_json'])
breakEnergyEv = float(referenceSummary.loc[EPOCH, 'break_energy_eV'])

fig = plotMeasuredVdf(grid, psd, counts, params=fitParams, breakEnergyEv=breakEnergyEv,
                      timeLabel=str(epochTime))
physical = componentMomentsFromParams(fitParams)
pd.Series(physical).round(3).to_frame('reference fit')''')
    nb.md('''## 5. The pitch-angle distribution (PAD)

Averaging the PSD over pitch-angle bins at every energy gives the PAD. Normalised per
energy step, it shows where the electrons of that energy go:

- the core is nearly isotropic (flat in pitch angle);
- the strahl is a bright band near 0 or 180 deg at energies above the core. Here it
  sits near 180 deg: the field points sunward and the strahl streams anti-sunward.

The strahl side and width are measured from the PAD before the fit (notebook 03),
because the fit needs to know where to put the beam.''')
    nb.code('''from evdf_poisson.plotting import plotPadSpectrogram

ax = plotPadSpectrogram(EXAMPLE_FILE, EPOCH)
ax.set_title(f'pitch-angle distribution, {epochTime}');''')
    nb.md('''## 6. Spacecraft potential and the low-energy cut

- The spacecraft charges a few volts positive, so every electron arrives with that
  extra energy. RPW measures the potential $\\phi_{sc}$; the preprocessing subtracts it
  from the energy of every measurement (`scpot_aligned` in the file).
- Below about 14 eV in the spacecraft frame the signal is dominated by **secondary and
  photo-electrons** from the spacecraft itself. The fit ignores energies below
  $14\\,\\text{eV} - \\phi_{sc}$ (the `valid` mask of the grid).''')
    nb.code('''from evdf_poisson.epoch_io import SECONDARY_ELECTRON_CUT_EV, PSD_TO_CGS, KM_TO_CM
from evdf_poisson.models import ELECTRON_MASS_G, ERG_PER_EV

with h5py.File(EXAMPLE_FILE, 'r') as h5:
    spacecraftPotential = float(h5['scpot_aligned'][EPOCH])
    rawPixels = h5['pa_vec_in'][EPOCH]
lowEnergyCut = SECONDARY_ELECTRON_CUT_EV - spacecraftPotential
print(f'spacecraft potential {spacecraftPotential:.2f} V -> low-energy cut {lowEnergyCut:.1f} eV')

# perpendicular-cone channel means of the raw file, including energies below the cut
rawSpeed = np.sqrt((rawPixels[:, :, 2] * KM_TO_CM) ** 2 + (rawPixels[:, :, 3] * KM_TO_CM) ** 2)
rawEnergy = 0.5 * ELECTRON_MASS_G * rawSpeed ** 2 / ERG_PER_EV
perpCone = (rawPixels[:, :, 0] > 75) & (rawPixels[:, :, 0] < 105) & np.isfinite(rawEnergy)
channelEnergy = np.array([np.nanmedian(rawEnergy[row][perpCone[row]]) if perpCone[row].any() else np.nan for row in range(64)])
channelMean = np.array([np.nanmean(rawPixels[row, :, 1][perpCone[row]]) * PSD_TO_CGS if perpCone[row].any() else np.nan for row in range(64)])

fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(channelEnergy, channelMean, 's', color='0.3', markersize=4, label='perpendicular cone, channel mean')
ax.axvline(lowEnergyCut, color='#d62728', linestyle='--', label=f'low-energy cut {lowEnergyCut:.1f} eV')
ax.set_xlabel('energy [eV] (spacecraft potential removed)'); ax.set_ylabel(r'PSD [s$^3$ cm$^{-6}$]')
ax.set_xlim(2, 3000); ax.legend(); ax.grid(alpha=0.3)
ax.set_title('the rise below the cut is spacecraft secondaries, not solar wind');''')
    nb.md('''## 7. The break energy

The **break energy** separates the core from the suprathermal electrons (halo and
strahl). It is found before the fit, from the perpendicular cone only (no strahl there):
a Maxwellian core and a power-law tail are fitted jointly to the cone spectrum, and the
break is where the two contribute equally (notebook 03).

The fit uses it to select the pixels of the core stage (below the break) and of the halo
stage (above), and to scale the strahl detection band. The dotted lines in the cuts
above mark it. For this epoch: about 139 eV, roughly 8 times the core temperature.''')
    nb.code('''breakTable = pd.read_csv(EXAMPLE_DIR / (EXAMPLE_FILE.stem + '_two_population_crossing.csv'))
breakTable[['epoch_id', 'timestamp', 'scpot_V', 'break_E_eV', 'core_T_eV', 'tail_index', 'status']].round(2)''')
    nb.md('''## 8. Why zero-count pixels are data

Above 200 eV, 9 pixels in 10 registered nothing (table of section 1).

- A fit that keeps only pixels with counts > 0 sees only the upward fluctuations of a
  faint tail, and is pushed up.
- A Poisson likelihood gives a zero-count pixel with expected counts $\\mu$ the
  probability $e^{-\\mu}$, so empty pixels keep pulling the model down.

This is the central idea of the pipeline (notebook 07). Below: the fraction of
zero-count pixels per energy step in the parallel, anti-parallel and perpendicular
cones.''')
    nb.code('''from evdf_poisson.plotting import wingMask

fig, ax = plt.subplots(figsize=(6, 4))
for wingName, label, color in (('par', 'parallel cone (pitch < 15 deg)', '#2a78d6'),
                               ('anti', 'anti-parallel cone (pitch > 165 deg)', '#e34948'),
                               ('perp', 'perpendicular cone (90 +/- 15 deg)', '0.3')):
    cone = wingMask(grid, wingName)
    energies = np.array([np.median(grid['energy_eV'][row][cone[row]]) if cone[row].any() else np.nan for row in range(64)])
    zeroFraction = np.array([np.mean(counts[row][cone[row]] == 0) if cone[row].any() else np.nan for row in range(64)])
    ax.semilogx(energies, zeroFraction, 'o-', markersize=3, color=color, label=label)
ax.set_xlabel('energy [eV]'); ax.set_ylabel('fraction of pixels with zero counts')
ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=9);''')
    nb.md('''## Summary

- EAS delivers **counts** on a 64 x 1024 grid per minute. The one-count level $g$
  converts them to PSD and is known for every pixel.
- MAG gives the field direction; the fit works in the gyrotropic
  $(v_\\parallel, v_\\perp)$ plane.
- Three components: Maxwellian **core**, kappa **halo**, field-aligned **strahl**.
- Two inputs are measured before the fit: the strahl side and width (PAD) and the
  **break energy** (notebook 03).
- Above ~100 eV most pixels are zeros. That is data, and the Poisson objective uses
  it (notebooks 05 and 07).''')
    return nb


# ----------------------------------------------------------------------- 03
def build03():
    nb = Notebook()
    nb.md('''# 03 - Before the fit: break energy and strahl verdict

Two quantities are measured **before** any component is fitted, because the staged fit
uses them to assign pixels to stages:

1. the **break energy** between core and suprathermal electrons (two-population
   crossing);
2. the **strahl side, strength and width**, from the pitch-angle distribution (PAD).

Both are cheap. The break energies are stored next to the data in
`<file stem>_two_population_crossing.csv`; the PAD verdict goes into the fit output.

**Runtime:** under a minute.''')
    nb.code(SETUP_CELL)
    nb.md('''## 1. Break energy: the two-population crossing

1. Take the **perpendicular cone** (pitch angles 90 $\\pm$ 20 deg), where there is no
   strahl.
2. Average the PSD of *all* its pixels per energy step, zeros included. Averaging only
   the detected pixels would bias the faint channels upward.
3. Fit a Maxwellian core and a power-law tail **jointly** to that spectrum:

$$ f(E) = A_c\\, e^{-E/kT_c} + A_t\\, E^{-p} . $$

4. The break energy is where the two terms are equal, the standard core/halo
   definition (Stverak et al. 2009).

Fitting both terms at once avoids choosing a "core window" by hand. Only channels
resolved at 3 sigma or better enter, and the spectrum starts at
$12\\,\\text{eV} - \\phi_{sc}$ (spacecraft secondaries below that). Epochs without a
crossing get a `status` other than `ok` and no break energy.

The functions are in `src/break_energy_methods.py`. Below they are called one by one on
a single epoch.''')
    nb.code('''import break_energy_methods as bem
from data_io import load_data, convert_units
from evdf_poisson.plotting import plotBreakEnergyCrossing

EPOCH = 20
CONE_HALF_WIDTH_DEG = 20.0
data = convert_units(load_data(EXAMPLE_DIR, EXAMPLE_FILE.name))   # the whole file, CGS units
spacecraftPotential = float(data['scpot_aligned'][EPOCH])
lowCutEv = bem.BASELINE_LOW_ENERGY_EV - spacecraftPotential

profile, profileStatus = bem.buildConeProfile(EPOCH, data, CONE_HALF_WIDTH_DEG)
channels = bem.aggregateByEnergyChannel(profile)
crossing = bem.findTwoPopulationCrossing(channels, minEnergyEv=lowCutEv)

print(f'cone pixels: {len(profile)}   channels: {len(channels)}   reliable: {int(channels["is_reliable"].sum())}')
print(f"status: {crossing['status']}   break = {crossing['break_E_eV']:.1f} eV   "
      f"core T = {crossing['core_T_eV']:.1f} eV   tail index = {crossing['tail_index']:.2f}")
ax = plotBreakEnergyCrossing(crossing, lowCutEv=lowCutEv)''')
    nb.md('''The channel table behind the figure. `detection_fraction` is the share of cone pixels
with at least one count. It drops below 10 % above ~300 eV, which is why the cone mean
includes the zeros.''')
    nb.code('''shown = channels[channels['E_eV'] >= lowCutEv].iloc[::3]
shown[['E_eV', 'log_psd', 'n_pixels', 'total_count', 'detection_fraction', 'signal_to_noise', 'is_reliable']].round(3)''')
    nb.md('''### Whole files: `compute_break_energy_crossing.py`

The script (or `runCrossingPipeline`) does this for every epoch of every file in a
directory and writes `<file stem>_two_population_crossing.csv`. The fitting runner finds
it next to the data (or in `--break-dir`). Rerunning it here reproduces the file shipped
in `examples/example_data/`.''')
    nb.code('''from compute_break_energy_crossing import runCrossingPipeline

runCrossingPipeline(EXAMPLE_DIR, EXAMPLE_DIR, workers=1, overwrite=True, writeCombined=False)
breakTable = pd.read_csv(EXAMPLE_DIR / (EXAMPLE_FILE.stem + '_two_population_crossing.csv'), parse_dates=['timestamp'])
breakTable[['epoch_id', 'timestamp', 'scpot_V', 'low_cut_eV', 'break_E_eV', 'core_T_eV', 'tail_index', 'status']].round(2)''')
    nb.code('''fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(breakTable['timestamp'], breakTable['break_E_eV'], 'o-', color='0.2', label='break energy')
ax.plot(breakTable['timestamp'], 7.7 * breakTable['core_T_eV'], 's--', color='#e34948', label=r'7.7 $\\times$ core $kT$ (proxy used when no crossing exists)')
ax.set_ylabel('energy [eV]'); ax.legend(fontsize=9); ax.grid(alpha=0.3); fig.autofmt_xdate()''')
    nb.md('''## 2. Strahl detection from the PAD

The fit must know on which side of **B** the strahl is, how strong and how wide, and
must not invent one where there is none. All of that comes from the PAD:

1. **Band mean.** Average the PAD over $[\\max(61, 0.8\\,E_\\text{break}),\\ 1000]$ eV. The
   lower edge follows the break so the near-isotropic core does not dilute the beam
   contrast.
2. **Model.** Fit a background plus two Gaussians centred at 0 and 180 deg
   (Owen et al. 2022):
   $P(\\alpha) = P_B + P_0\\, e^{-\\alpha^2 / 4W_0^2} + P_{180}\\, e^{-(\\alpha-180)^2 / 4W_{180}^2}$.
3. **Contrast and width.** For each side, the contrast against the perpendicular
   plateau, $\\beta_\\text{side} = \\langle P\\rangle_\\text{side} / \\langle P\\rangle_{65-115^\\circ} - 1$,
   and the beam width $\\text{PAW} = 2\\sqrt{2\\ln 2}\\, W$ (full width at half maximum).

The background $P_B$ is bounded below at 0.9 times the band mean. For a bright beam the
fitted plateau therefore sits above the perpendicular data, and $\\beta$ is a
conservative contrast.''')
    nb.code('''from evdf_poisson.pad_detection import fitBandMeanPad, detectStrahlFromPad
from evdf_poisson.plotting import plotPadSpectrogram, plotPadBandFit

breakEnergyEv = float(breakTable.loc[EPOCH, 'break_E_eV'])
padFit = fitBandMeanPad(EXAMPLE_FILE, EPOCH, breakEnergyEv)
detection = detectStrahlFromPad(EXAMPLE_FILE, EPOCH, breakEnergyEv)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
plotPadSpectrogram(EXAMPLE_FILE, EPOCH, ax=axes[0], bandEv=padFit['band_ev'])
plotPadBandFit(padFit, detection, ax=axes[1])
fig.tight_layout()
print({name: round(param.value, 3) for name, param in padFit['params'].items()})
detection''')
    nb.md('''### The two-tier verdict

| condition (per side) | tier | what the fit does |
|---|---|---|
| $\\beta > 2$ and PAW $\\le 60^\\circ$ | **strong** | strahl component on that side, narrow perpendicular initial width |
| $0.6 < \\beta \\le 2$ and PAW $\\le 60^\\circ$ | **weak** | strahl component on that side, broad perpendicular initial width |
| otherwise | none | no strahl component (`quasi_iso`: 9 free parameters instead of 14) |
| $\\beta < -0.5$ | *deficit* flag | recorded only; a sunward depletion, not fitted |

- If both sides qualify, the larger $\\beta$ wins (`both`).
- $\\beta > 2$ is the "meaningful strahl" threshold of Owen et al. (2022). The weak tier
  keeps scattered, broadened beams that the fit still has to account for.
- The 60 deg cap rejects fits that are not beams.

`seedFromData` receives the dictionary above: `sign` (+1 parallel, -1 anti-parallel,
0 none), `tier`, and the width `paw_par` / `paw_anti` that anchors the strahl's
perpendicular thermal speed.''')
    nb.code('''verdicts = pd.DataFrame([{'epoch': epochIndex,
                          **detectStrahlFromPad(EXAMPLE_FILE, epochIndex, float(breakTable.loc[epochIndex, 'break_E_eV']))}
                         for epochIndex in range(len(breakTable))]).set_index('epoch')
verdicts.round(2)''')
    nb.md('''## 3. What the fit does with the two numbers

- **Core stage:** pixels with $E \\le E_\\text{break}$, all pitch angles.
- **Halo stage:** pixels with $E > E_\\text{break}$, outside both 30 deg cones.
- **Strahl stage:** the 30 deg cone on the detected side, all energies, only for a
  `strong` or `weak` verdict. The PAD width sets the initial perpendicular width.
- The strahl drift is searched in $[0.3, 3] \\times v(E_\\text{break})$.

Notebook 05 runs these stages one at a time.''')
    return nb


# ----------------------------------------------------------------------- 05
STAGE_TABLES = {
    'A': '''| parameter | initial value | allowed range |
|---|---|---|
| $n_c$ | $0.8\\, n_{sc}$ (RPW density) | $[0,\\ 2\\, n_{sc}]$ |
| $u_{\\parallel c}$ | 0 | $\\pm 1.5\\times10^8$ cm/s (free sign) |
| $v_{th\\parallel c}$, $v_{th\\perp c}$ | from the peak PSD, $(n / \\pi^{3/2} f_\\text{peak})^{1/3}$ | $[10^7,\\ 8\\times10^8]$ cm/s |''',
    'B': '''| parameter | initial value | allowed range |
|---|---|---|
| $n_h$ | $0.03\\, n_c^\\text{fit}$ | $[10^{-5},\\ 0.2] \\times n_c^\\text{fit}$ |
| $u_{\\parallel h}$ | 0 | $\\pm 0.2\\, v_{th\\perp h}^\\text{seed}$ |
| $v_{th\\parallel h}$, $v_{th\\perp h}$ | $\\sqrt{5}\\, v_{th c}^\\text{fit}$ ($T_h = 5\\,T_c$) | $\\ge 2 \\times$ the fitted core width |
| $\\kappa_h$ | 5 | $[2,\\ 12]$ |''',
    'C': '''| parameter | initial value | allowed range |
|---|---|---|
| $n_b$ | $0.01\\, n_{sc}$ | $[0,\\ 0.5\\, n_{sc}]$ |
| $u_{\\parallel b}$ | sign $\\times\\ 0.9\\, v_\\text{break}$ | sign $\\times [0.3,\\ 3]\\, v_\\text{break}$ |
| $v_{th\\parallel b}$ | $0.5\\, v_\\text{break}$ | $[2\\times10^7,\\ 1.5\\times10^9]$ cm/s |
| $v_{th\\perp b}$ | $\\tan(\\text{PAW}/2)\\,\\lvert u_{\\parallel b}\\rvert$ | tied to $u_{\\parallel b}$ during this stage |
| $\\kappa_b$ | 5 | $[1.7,\\ 20]$ |''',
}


def build05():
    nb = Notebook()
    nb.md('''# 05 - The Poisson fit, one stage at a time

One epoch through the whole fit: inputs -> initial guesses -> stage A (core) ->
B (halo) -> C (strahl) -> D (joint) -> physical parameters. Every stage is drawn with
the pixels it used.

**Runtime:** 1-2 minutes (the fit takes about 10 s; the rest is plotting).''')
    nb.code(SETUP_CELL)
    nb.md('''## 0. Inputs and initial guesses

Per epoch the fit needs the pixel grid (`loadEpoch`), the break energy (crossing CSV)
and the PAD verdict (`detectStrahlFromPad`); see notebooks 01 and 03. `seedFromData`
turns them into initial guesses (notebook 04):

- **core:** density and widths from the moments of the pixels below 70 eV;
- **halo:** 3 % of the core density, five times hotter, at rest ($u_{\\parallel h} = 0$);
- **strahl:** on the side the PAD found, perpendicular width anchored to the PAD width.''')
    nb.code('''from evdf_poisson import loadEpoch, detectStrahlFromPad, seedFromData, componentMomentsFromParams
from evdf_poisson.models import temperatureFromThermalSpeed

EPOCH = 20
grid, psd, counts, epochNs, scDensity = loadEpoch(EXAMPLE_FILE, EPOCH)
epochTime = pd.Timestamp(int(epochNs))
breakTable = pd.read_csv(EXAMPLE_DIR / (EXAMPLE_FILE.stem + '_two_population_crossing.csv'))
breakEnergyEv = float(breakTable.loc[EPOCH, 'break_E_eV'])
detection = detectStrahlFromPad(EXAMPLE_FILE, EPOCH, breakEnergyEv)
seeds = seedFromData(psd, counts, grid, strahlDetection=detection, scDensity=scDensity, breakEnergyEv=breakEnergyEv)

print(f'{epochTime}   break {breakEnergyEv:.1f} eV   PAD verdict {detection["class"]} (sign {detection["sign"]})')
seedTable = {key: value for key, value in seeds.items() if not key.startswith('v_th')}
seedTable.update({f'T from {key} [eV]': temperatureFromThermalSpeed(value) for key, value in seeds.items() if key.startswith('v_th')})
pd.Series(seedTable).map(lambda value: f'{value:.4g}' if isinstance(value, float) else value).to_frame('seed')''')
    nb.md('''## 1. The objective: Poisson deviance

Pixel $i$ registered $C_i$ counts. A model PSD $f_i$ predicts $\\mu_i = f_i / g_i$ counts
($g_i$ = one-count level, notebook 01). The fit minimises the Poisson **deviance**,
equivalent to maximising the likelihood $\\sum_i \\left[C_i \\ln \\mu_i - \\mu_i\\right]$ and
written as a sum of squared residuals so that a bounded least-squares solver can be
used (lmfit / `scipy.optimize.least_squares`):

$$ r_i = \\operatorname{sign}(C_i - \\mu_i)\\sqrt{2\\left[\\mu_i - C_i + C_i \\ln (C_i/\\mu_i)\\right]},
\\qquad D = \\sum_i r_i^2 . $$

- A zero-count pixel contributes $r_i = \\sqrt{2\\mu_i}$: the model pays for predicting
  counts that were not seen. A chi-square fit on detected pixels throws this term away
  (notebook 07).
- `deviance_per_dof` is $D$ divided by (pixels minus free parameters).

## 2. Why four stages

Released all at once, the 14 parameters do not converge to the physical answer
(notebook 04, section 4). Each component is first fitted where it dominates, the
others frozen; then all are released together, inside ranges anchored to those results.

`staged_fit.runStagedFit` runs the stages, `fitEpoch` adds the objective, and the
callback below records a snapshot after each stage.''')
    nb.code('''from evdf_poisson import fitEpoch
from evdf_poisson.plotting import plotMeasuredVdf

snapshots = {}
def recordStage(stageName, stageParams, pixelMask, freeNames):
    snapshots[stageName] = {'params': stageParams, 'mask': pixelMask, 'free': freeNames}

params, modelPsd, diagnostics = fitEpoch(psd, counts, grid, seeds, objective='poisson', onStageComplete=recordStage)
for stageName, snapshot in snapshots.items():
    print(f'{stageName:9s} pixels {int(snapshot["mask"].sum()):6d}   free parameters: {snapshot["free"]}')

def componentsOnly(stageParams, keep):
    """Copy of a parameter dict with the components not in `keep` switched off (for drawing)."""
    drawn = dict(stageParams)
    if 'halo' not in keep: drawn['n_h'] = 0.0
    if 'strahl' not in keep: drawn['n_b'] = 0.0
    return drawn''')
    nb.md('''## Stage A: the core

- **Pixels:** every valid pixel with $E \\le E_\\text{break}$, all pitch angles (halo and
  strahl contribute little there).
- **Free:** the 4 core parameters. Halo and strahl are switched off.

''' + STAGE_TABLES['A'] + '''

The drift sign is free. Forcing it away from the strahl side pins $u_{\\parallel c}$ at
zero under the Poisson objective and displaces the whole core.''')
    nb.code('''fig = plotMeasuredVdf(grid, psd, counts, componentsOnly(snapshots['A_core']['params'], ['core']),
                      breakEnergyEv=breakEnergyEv, highlightMask=snapshots['A_core']['mask'],
                      highlightLabel='stage A pixels', timeLabel='after stage A: core only')''')
    nb.md('''## Stage B: the halo

- **Pixels:** $E > E_\\text{break}$, **outside both 30 deg cones** around $\\pm$**B** (no
  strahl, no sunward deficit).
- **Free:** the 5 halo parameters: density, drift, two widths, $\\kappa$. The core is
  frozen at its stage-A value.
- Initial values and ranges are rebuilt from the *fitted* core:

''' + STAGE_TABLES['B'] + '''

What the ranges are for:

- **widths $\\ge$ 2x the core width:** the halo cannot become a second core;
- **$\\kappa_h \\ge 2$:** it cannot become a spike under the core peak (a bi-kappa's
  central amplitude grows as $(\\kappa - 3/2)^{-3/2}$, and stage B never sees the centre);
- **$|u_{\\parallel h}| \\le 0.2$ halo widths:** the halo may counter-drift against the
  strahl (zero-current condition) but cannot slide onto an undetected strahl wing.
  `fitEpoch(..., freeHaloDrift=False)` (runner: `--fix-halo-drift`) pins the drift at
  zero.''')
    nb.code('''fig = plotMeasuredVdf(grid, psd, counts, componentsOnly(snapshots['B_halo']['params'], ['core', 'halo']),
                      breakEnergyEv=breakEnergyEv, highlightMask=snapshots['B_halo']['mask'],
                      highlightLabel='stage B pixels', timeLabel='after stage B: core + halo')''')
    nb.md('''## Stage C: the strahl

Runs only if the PAD found a beam.

- **Pixels:** the 30 deg cone on the strahl side, **all** energies. The frozen core and
  halo bridge the thermal range, so the whole cone profile constrains the beam.
- **Free:** density, drift, parallel width and $\\kappa$. The perpendicular width
  follows the drift through the PAD width,
  $v_{th\\perp b} = \\tan(\\text{PAW}/2)\\,|u_{\\parallel b}|$.

''' + STAGE_TABLES['C'] + '''

The lower drift limit keeps the strahl suprathermal. Without it the component
degenerates into a second drifting core.''')
    nb.code('''fig = plotMeasuredVdf(grid, psd, counts, snapshots['C_strahl']['params'],
                      breakEnergyEv=breakEnergyEv, highlightMask=snapshots['C_strahl']['mask'],
                      highlightLabel='stage C pixels', timeLabel='after stage C: core + halo + strahl')''')
    nb.md('''## Stage D: joint refinement

- **Pixels:** all valid pixels.
- **Free:** all 14 parameters (9 without a strahl), each inside its range from the
  previous stages.
- The strahl drift is locked to $\\pm 20\\,\\%$ of its stage-C value; its perpendicular
  width may move within $[0.3, 3]$ times its anchor.

This is the result the pipeline reports. The four-panel figure is the standard
diagnostic, also written by the batch runner: cuts, 2D distribution with contours,
deviance pull per pixel.''')
    nb.code('''from evdf_poisson import poissonDeviancePerDof, reducedChiSquare
from evdf_poisson.plotting import plotPoissonFitEpoch

deviancePerDof = poissonDeviancePerDof(modelPsd, counts, grid, diagnostics['free_params'])
redChi2 = reducedChiSquare(modelPsd, psd, counts, grid, diagnostics['free_params'])
print(f'deviance/dof = {deviancePerDof:.3f}   reduced chi2 (detected pixels) = {redChi2:.3f}   '
      f'free parameters = {diagnostics["free_params"]}   break source = {diagnostics["break_source"]}')
fig = plotPoissonFitEpoch(grid, psd, counts, params, diagnostics, deviancePerDof, str(epochTime))''')
    nb.md('''## How good is the fit?

- **`deviance_per_dof`** = $D$ / (pixels minus free parameters). **Not** expected to be
  1: the expected deviance of a pixel is well below 1 when its expected counts $\\mu$ are
  small, as in most EAS pixels. The value tracks how bright the plasma is, so compare
  epochs of similar brightness.
- **Calibrated ratio** $R = D / \\sum_i \\mathrm{E}[D_i \\mid \\mu_i]$: the deviance relative
  to what a perfect model would give on this grid. $R \\approx 1$: as good as Poisson
  noise allows. $R$ well above 1: structure the model does not describe. Notebook 08
  builds quality flags on it.
- **`red_chi2`**: classical reduced chi-square on detected pixels
  ($\\sigma = f/\\sqrt{C}$). For comparison with the literature only; it cannot see the
  zero-count pixels (notebook 07).
- **Pull map** (bottom right of the four-panel figure): *where* the deviance comes
  from. Red/blue speckle is noise. A coherent patch of one colour is a model
  deficiency: a sunward deficit, a strahl broader than a bi-kappa, secondaries at the
  low-energy edge.''')
    nb.code('''from scipy.stats import poisson

def expectedDeviancePerPixel(expectedCounts, maxCount=400):
    """E[D | mu] of a Poisson pixel: sum over k of P(k | mu) * d(k, mu); maxCount must exceed the largest mu by far."""
    k = np.arange(maxCount + 1)[:, None]
    mu = np.asarray(expectedCounts)[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        logTerm = np.where(k > 0, k * np.log(k / mu), 0.0)
    deviance = 2.0 * (mu - k + logTerm)
    return np.sum(poisson.pmf(k, mu) * deviance, axis=0)

muCurve = np.logspace(-3, 2, 120)
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.semilogx(muCurve, expectedDeviancePerPixel(muCurve), color='0.2')
ax.axhline(1, color='#d62728', linestyle='--', label='chi-square expectation (1 per pixel)')
ax.set_xlabel(r'expected counts $\\mu$ in a pixel'); ax.set_ylabel(r'E[$D$] per pixel')
ax.set_title('what a perfect model scores on a pixel with mu expected counts'); ax.grid(alpha=0.3, which='both'); ax.legend()

valid = grid['valid']
expectedCounts = modelPsd[valid] / grid['one_count_psd'][valid]
observedDeviance = deviancePerDof * (valid.sum() - diagnostics['free_params'])
calibratedRatio = observedDeviance / expectedDeviancePerPixel(expectedCounts).sum()
print(f'this epoch: deviance/dof = {deviancePerDof:.3f}   median expected counts per pixel = {np.median(expectedCounts):.3f}   '
      f'calibrated ratio R = {calibratedRatio:.3f}   reduced chi2 (detected pixels) = {redChi2:.3f}')''')
    nb.md('''## 3. Reading the result

The fit parameters are thermal speeds in cm/s; `componentMomentsFromParams` converts
them to temperatures in eV. The table follows each quantity through the stages: the
core barely moves after stage A, halo and strahl settle in their own stages, and the
joint stage adjusts everything by a few per cent.''')
    nb.code('''stageMoments = {stageName: componentMomentsFromParams(snapshot['params']) for stageName, snapshot in snapshots.items()}
stageTable = pd.DataFrame(stageMoments)
stageTable.loc[['n_h', 'u_par_h', 'T_par_h_eV', 'T_perp_h_eV', 'kappa_h'], 'A_core'] = np.nan   # not fitted yet
stageTable.loc[['n_b', 'u_par_b', 'T_par_b_eV', 'T_perp_b_eV', 'kappa_b'], ['A_core', 'B_halo']] = np.nan
stageTable.map(lambda value: '' if np.isnan(value) else f'{value:.4g}')''')
    nb.md('''**Pinned parameters.** A free parameter that ends within 0.1 % of the edge of its
range was decided by the range, not by the data. The batch runner lists them in
`pinned_params`. Two are expected:

- **halo widths on their lower limit** (twice the core width): the deviance is flat
  along the halo width;
- **strahl drift on the lower edge of its $\\pm 20\\,\\%$ lock**: the joint stage pulls the
  beam down-energy to patch the core-strahl transition, and the lock decides how far.

The table marks the pinned parameters of this epoch.''')
    nb.code('''from evdf_poisson.runner import findPinnedParams

pinned = findPinnedParams(params, diagnostics['param_bounds'])
boxTable = pd.DataFrame({name: {'value': f'{params[name]:.4g}', 'box low': f'{low:.4g}', 'box high': f'{high:.4g}',
                                'pinned': 'yes' if name in pinned else ''}
                         for name, (low, high) in diagnostics['param_bounds'].items()}).T
boxTable''')
    nb.md('''## 4. The same in one call

`fitOneEpoch` does all of the above (load, PAD, initial guesses, fit, metrics) and
returns the row the batch runner writes to CSV. Its numbers are identical to the
step-by-step ones and to the reference `summary.csv`.''')
    nb.code('''import json
from evdf_poisson import fitOneEpoch

row = fitOneEpoch(EXAMPLE_FILE, EPOCH, breakEnergyEv=breakEnergyEv)
reference = pd.read_csv(RESULTS_DIR / 'summary.csv').set_index('epoch_index').loc[EPOCH]
print('identical to the step-by-step fit:', json.loads(row['fit_params_json']) == {k: float(v) for k, v in params.items()})
print('identical to the reference summary.csv:', json.loads(row['fit_params_json']) == json.loads(reference['fit_params_json']))
pd.Series(row).drop('fit_params_json')''')
    return nb


# ----------------------------------------------------------------------- 06
def build06():
    nb = Notebook()
    nb.md('''# 06 - Batch fitting, the output files, two full days

How to run the pipeline on whole files, what it writes, and how to turn the CSV into
parameter time series.

**Runtime:** a few seconds if notebook 00 already fitted the example, about 1.5 minutes
otherwise.''')
    nb.code(SETUP_CELL)
    nb.md('''## 1. Running the batch fit

`runFiles(dataDir, outputDir, ...)` fits every `VDF_eas_forFitting_*.h5` file in
`dataDir` (or those listed in `fileNames`). Each file is fitted in a process pool and
written to `per_file/<stem>_fit.csv`; at the end everything is merged into
`summary.csv`.

| argument | default | meaning |
|---|---|---|
| `breakDir` | `dataDir` | where `<stem>_two_population_crossing.csv` lives (notebook 03) |
| `workers` | 4 | parallel processes (one epoch each; one BLAS thread per process) |
| `objective` | `'poisson'` | `'poisson'` or `'chi2'` (notebook 07) |
| `plotEvery` | 0 | save the four-panel figure of every N-th epoch to `plots/<stem>/` |
| `resume` | `True` | skip files whose CSV already has one row per epoch |
| `freeHaloDrift` | `True` | halo parallel drift free within $\\pm 0.2$ halo widths; `False` (`--fix-halo-drift`) pins it at 0 |

From a terminal, at the repository root:

```
PYTHONPATH=src python -m evdf_poisson.runner --data-dir examples/example_data \\
    --out results/fitting/example_run_poisson --workers 4 --plot-every 1
```

With `resume`, rerunning the same command after an interruption only fits what is
missing. To refit, use a fresh output directory or `resume=False` / `--no-resume`.''')
    nb.code('''from evdf_poisson.runner import runFiles

summary = runFiles(EXAMPLE_DIR, RESULTS_DIR, workers=4, plotEvery=1)   # resumes if notebook 00 already ran
sorted(path.relative_to(RESULTS_DIR).as_posix() for path in RESULTS_DIR.rglob('*') if path.is_file())[:8]''')
    nb.md('''## 2. The columns of `summary.csv`

| column | meaning |
|---|---|
| `epoch_index`, `epoch_ns`, `time_iso` | epoch index within the file, time as ns since 1970 and as text |
| `n_sc` | independent electron density from RPW (cm$^{-3}$), used to bound the core and strahl densities |
| `n_c`, `u_par_c_cms`, `T_par_c_eV`, `T_perp_c_eV` | core density, parallel drift (cm/s, spacecraft frame along **B**), temperatures |
| `n_h`, `u_par_h_cms`, `T_par_h_eV`, `T_perp_h_eV`, `kappa_h` | halo density, parallel drift (0 when the run used `freeHaloDrift=False`), temperatures, kappa |
| `n_b`, `u_par_b_cms`, `T_par_b_eV`, `T_perp_b_eV`, `kappa_b` | strahl; `n_b` = 0 and NaN temperatures when no strahl was fitted |
| `n_total` | $n_c + n_h + n_b$ |
| `deviance_per_dof` | Poisson deviance over all valid pixels / (pixels - free parameters) |
| `red_chi2` | classical reduced chi-square on the detected (counts > 0) pixels with $\\sigma = f/\\sqrt{C}$ |
| `objective` | `poisson` or `chi2` |
| `strahl_sign`, `strahl_tier` | PAD verdict: +1 / -1 / 0 and `par_strong`, `anti_weak`, `quasi_iso`, ... |
| `break_energy_eV`, `break_source` | break used by the fit; `crossing` (from the CSV) or `proxy_7.7T` (fallback) |
| `pad_beta_par`, `pad_beta_anti`, `pad_paw_par`, `pad_paw_anti`, `pad_deficit` | PAD contrasts, widths (deg) and deficit flag |
| `pinned_params` | free parameters that ended on the edge of their range (`;`-separated) |
| `runtime_s`, `error` | seconds per epoch; non-empty `error` = the epoch failed (row kept) |
| `fit_params_json` | all 14 raw parameters (CGS) as JSON, enough to redraw the model |

EAS densities can differ from RPW by a calibration factor that depends on the
instrument state. Treat `n_sc` as the reference for absolute densities.''')
    nb.code('''summary['time'] = pd.to_datetime(summary['time_iso'])
summary[['time_iso', 'n_sc', 'n_total', 'n_c', 'n_h', 'n_b', 'strahl_tier', 'break_source', 'deviance_per_dof', 'pinned_params', 'runtime_s']].round(3)''')
    nb.md('''## 3. Parameter time series

The same plotting code works for thirty minutes or for a month (section 6 applies it
to two days). Strahl quantities are masked where no strahl was fitted.''')
    nb.code('''hasStrahl = summary['n_b'] > 0
fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)

axes[0].plot(summary['time'], summary['n_sc'], 'k--', label='RPW $n_{sc}$')
axes[0].plot(summary['time'], summary['n_c'], 'o-', color='#e34948', label='core')
axes[0].plot(summary['time'], summary['n_h'], 'o-', color='#eda100', label='halo')
axes[0].plot(summary['time'][hasStrahl], summary['n_b'][hasStrahl], 'o-', color='#2a78d6', label='strahl')
axes[0].set_yscale('log'); axes[0].set_ylabel('density [cm$^{-3}$]')

axes[1].plot(summary['time'], summary['T_par_c_eV'], 'o-', color='#e34948', label=r'core $T_\\parallel$')
axes[1].plot(summary['time'], summary['T_perp_c_eV'], 's--', color='#e34948', label=r'core $T_\\perp$')
axes[1].plot(summary['time'], summary['T_par_h_eV'], 'o-', color='#eda100', label=r'halo $T_\\parallel$')
axes[1].plot(summary['time'][hasStrahl], summary['T_par_b_eV'][hasStrahl], 'o-', color='#2a78d6', label=r'strahl $T_\\parallel$')
axes[1].set_ylabel('temperature [eV]')

axes[2].plot(summary['time'], summary['kappa_h'], 'o-', color='#eda100', label=r'halo $\\kappa$')
axes[2].plot(summary['time'][hasStrahl], summary['kappa_b'][hasStrahl], 'o-', color='#2a78d6', label=r'strahl $\\kappa$')
axes[2].set_ylabel(r'$\\kappa$')

axes[3].plot(summary['time'][hasStrahl], summary['u_par_b_cms'][hasStrahl] / 1e8, 'o-', color='#2a78d6', label=r'strahl drift $u_{\\parallel b}$')
axes[3].plot(summary['time'], summary['u_par_c_cms'] / 1e8, 'o-', color='#e34948', label=r'core drift $u_{\\parallel c}$')
axes[3].plot(summary['time'], summary['u_par_h_cms'] / 1e8, 'o-', color='#eda100', label=r'halo drift $u_{\\parallel h}$')
axes[3].set_ylabel(r'drift [$10^8$ cm/s]')
for ax in axes:
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9); ax.grid(alpha=0.3)
fig.autofmt_xdate(); fig.tight_layout()''')
    nb.md('''## 4. Fit quality and pinned parameters

- `deviance_per_dof` is the number to compare between epochs. It is **not** expected to
  be 1: at low expected counts a perfect model has a deviance well below 1 per pixel,
  so the typical value depends on how bright the plasma is. Look for outliers within a
  data set, not for a fixed threshold (notebook 08 has a calibrated version).
- `red_chi2` is for comparison with the chi-square literature only.
- `pinned_params` lists the parameters decided by their range. Frequent and harmless:
  `v_th_perp_h` on its lower limit (the halo width is degenerate with its density; the
  limit picks the physical corner), `u_par_b` on the lower edge of its +/-20 % lock,
  `kappa_b` at its ceiling (weakly constrained). A pinned `n_c`, `v_th_par_c` or `n_h`
  deserves a look at the figure.''')
    nb.code('''fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(summary['time'], summary['deviance_per_dof'], 'o-', color='0.2', label='deviance / dof (all pixels)')
axes[0].plot(summary['time'], summary['red_chi2'], 's--', color='0.6', label=r'reduced $\\chi^2$ (detected pixels)')
axes[0].set_ylim(0, 1.5); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
pinnedCounts = summary['pinned_params'].fillna('').str.split(';').explode().value_counts().drop('', errors='ignore')
axes[1].barh(pinnedCounts.index, pinnedCounts.values, color='0.5'); axes[1].set_xlabel('epochs pinned (of %d)' % len(summary))
fig.autofmt_xdate(); fig.tight_layout()''')
    nb.md('''## 5. Single epochs, figures, your own data

- **One epoch with its figure.** Python: `fitOneEpoch(file, epochIndex,
  plotPath='epoch.png', breakEnergyEv=...)`. Terminal:
  `PYTHONPATH=src python -m evdf_poisson.plotting --chunk <h5> --epoch 20 --break-energy 139`.
- **Redraw a stored fit.** `fit_params_json` holds everything needed (notebook 01 does
  exactly that).
- **Your own interval.** Three commands, detailed in notebook 02:
  `preprocess_main(start_time, end_time, psd_source='soar', shift_SCPOT=True, scpot_file=..., output_dir=D)`,
  then `python src/compute_break_energy_crossing.py --vdf-dir D --output-dir D`,
  then `python -m evdf_poisson.runner --data-dir D --out results/fitting/<name> --workers N`.
  Keep the file-name pattern: the runner derives the break-energy file name from it.
- **Cost.** A six-hour file (~360 epochs) fits in about 10 minutes on 16 workers, with
  a few hundred MB per worker.''')
    nb.md('''## 6. Two full days: 2023-04-04 and 2023-04-05

The example subset is half an hour out of a two-day run: 2 850 one-minute epochs in
eight six-hour files, fitted with the same code in about 12 minutes on 32 cores. The
preprocessed files (4.8 GB) are a release asset (see `workshop/README.md`). The fit
results ship in `results/fitting/example_days_2023-04-04_05/summary.csv`, so this
section runs offline.''')
    nb.code('''days = pd.read_csv(DAYS_DIR / 'summary.csv')
days['time'] = pd.to_datetime(days['time_iso'])
days = days[days['error'].fillna('') == ''].sort_values('time').reset_index(drop=True)
print(f'{len(days)} epochs, {days["time"].iloc[0]} .. {days["time"].iloc[-1]}')
print('strahl verdicts:', days['strahl_tier'].value_counts(normalize=True).round(3).to_dict())
print('break source:', days['break_source'].value_counts().to_dict())
print('deviance/dof: median %.3f, 5-95 %% %.3f-%.3f' % (days['deviance_per_dof'].median(), days['deviance_per_dof'].quantile(0.05), days['deviance_per_dof'].quantile(0.95)))''')
    nb.code('''hasStrahl = days['n_b'] > 0
fig, axes = plt.subplots(5, 1, figsize=(11, 14), sharex=True)
axes[0].plot(days['time'], days['n_sc'], 'k-', linewidth=0.8, label='RPW $n_{sc}$')
axes[0].plot(days['time'], days['n_c'], '-', color='#e34948', linewidth=0.8, label='core')
axes[0].plot(days['time'], days['n_h'], '-', color='#eda100', linewidth=0.8, label='halo')
axes[0].plot(days['time'][hasStrahl], days['n_b'][hasStrahl], '.', color='#2a78d6', markersize=2, label='strahl')
axes[0].set_yscale('log'); axes[0].set_ylabel('density [cm$^{-3}$]')
axes[1].plot(days['time'], days['T_par_c_eV'], '-', color='#e34948', linewidth=0.8, label=r'core $T_\\parallel$')
axes[1].plot(days['time'], days['T_perp_c_eV'], '-', color='#f5a3a3', linewidth=0.8, label=r'core $T_\\perp$')
axes[1].plot(days['time'], days['T_par_h_eV'], '-', color='#eda100', linewidth=0.8, label=r'halo $T_\\parallel$')
axes[1].plot(days['time'][hasStrahl], days['T_par_b_eV'][hasStrahl], '.', color='#2a78d6', markersize=2, label=r'strahl $T_\\parallel$')
axes[1].set_yscale('log'); axes[1].set_ylabel('temperature [eV]')
axes[2].plot(days['time'], days['kappa_h'], '.', color='#eda100', markersize=2, label=r'halo $\\kappa$')
axes[2].plot(days['time'][hasStrahl], days['kappa_b'][hasStrahl], '.', color='#2a78d6', markersize=2, label=r'strahl $\\kappa$')
axes[2].set_ylabel(r'$\\kappa$')
axes[3].plot(days['time'], days['break_energy_eV'], '-', color='0.3', linewidth=0.8, label='break energy')
axes[3].plot(days['time'], 7.7 * days['T_par_c_eV'], '--', color='#e34948', linewidth=0.8, label=r'7.7 $\\times$ core $T_\\parallel$')
axes[3].set_ylabel('energy [eV]')
tierCode = days['strahl_tier'].map({'anti_strong': -2, 'anti_weak': -1, 'quasi_iso': 0, 'par_weak': 1, 'par_strong': 2, 'both': 3})
axes[4].plot(days['time'], tierCode, '.', color='0.3', markersize=2)
axes[4].set_yticks([-2, -1, 0, 1, 2, 3]); axes[4].set_yticklabels(['anti strong', 'anti weak', 'none', 'par weak', 'par strong', 'both'])
axes[4].set_ylabel('PAD verdict')
for ax in axes[:4]:
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9)
for ax in axes:
    ax.grid(alpha=0.3)
fig.autofmt_xdate(); fig.tight_layout()''')
    nb.md('''Two days of near-perihelion solar wind (0.33 au):

- a strong anti-parallel strahl almost throughout;
- a break energy that follows the core temperature;
- an EAS density below the RPW one by a slowly changing calibration factor (compare
  `n_c` with `n_sc`, and treat `n_sc` as the absolute reference).

### Fit quality over two days

`deviance_per_dof` against the fitted core density: the deviance rises with brightness
for the reason given in notebook 05, not because bright epochs fit worse. Epochs with
masked L1 saturation blocks (`n_saturated_pixels` in the files) are marked; their fits
are ordinary. Without the mask, the ten epochs of 04-05 with a block at a suprathermal
energy step would have deviance/dof in the hundreds.''')
    nb.code('''censusPath = DAYS_DIR / 'saturation_census.csv'
census = pd.read_csv(censusPath) if censusPath.exists() else None
if census is not None:
    days = days.merge(census[['epoch_ns', 'n_saturated_pixels']], on='epoch_ns', how='left')
    days['n_saturated_pixels'] = days['n_saturated_pixels'].fillna(0).astype(int)
    print(f"epochs with masked saturation blocks: {(days['n_saturated_pixels'] > 0).sum()} of {len(days)}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(days['n_c'], days['deviance_per_dof'], s=3, color='0.4', alpha=0.5, label='all epochs')
if census is not None:
    masked = days['n_saturated_pixels'] > 0
    axes[0].scatter(days['n_c'][masked], days['deviance_per_dof'][masked], s=14, color='#d62728', label='saturation block masked')
axes[0].set_xscale('log'); axes[0].set_xlabel('fitted core density [cm$^{-3}$]'); axes[0].set_ylabel('deviance / dof')
axes[0].set_title('correlation %.2f with log n_c' % np.corrcoef(np.log(days['n_c']), days['deviance_per_dof'])[0, 1])
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
pinnedCounts = days['pinned_params'].fillna('').str.split(';').explode().value_counts().drop('', errors='ignore')
axes[1].barh(pinnedCounts.index, pinnedCounts.values / len(days), color='0.5')
axes[1].set_xlabel('fraction of epochs pinned'); axes[1].set_title('pinned parameters over two days'); axes[1].grid(alpha=0.3, axis='x')
fig.tight_layout()''')
    return nb


# ----------------------------------------------------------------------- 07
def build07():
    nb = Notebook()
    nb.md('''# 07 - Poisson likelihood versus chi-square at low counts

The pipeline minimises the **Poisson deviance** over every pixel, zeros included. Most
published eVDF fits minimise a **chi-square over the detected pixels** (counts > 0),
usually in log space, where a zero PSD has no logarithm.

This notebook compares the two on real data and on synthetic data with a known truth.
Model, stages, initial guesses and parameter ranges are identical, so the objective is
the only difference (`fitEpoch(..., objective='poisson' | 'chi2')`).

**Runtime:** about 4 minutes (two fits of one epoch, a chi-square batch run of the 30
epochs, 24 synthetic fits on 4 workers).''')
    nb.code(SETUP_CELL)
    nb.md('''## 1. The same epoch, two objectives

Both fits get identical inputs. `chi2` minimises
$\\sum_{C_i > 0} [\\log_{10} f_\\text{model}(v_i) - \\log_{10} f_i]^2$;
`poisson` minimises the deviance of notebook 05 over all valid pixels.''')
    nb.code('''from evdf_poisson import loadEpoch, detectStrahlFromPad, seedFromData, fitEpoch
from evdf_poisson import poissonDeviancePerDof, reducedChiSquare, componentMomentsFromParams
from evdf_poisson.plotting import plotFitComparison

EPOCH = 20
grid, psd, counts, epochNs, scDensity = loadEpoch(EXAMPLE_FILE, EPOCH)
epochTime = pd.Timestamp(int(epochNs))
breakTable = pd.read_csv(EXAMPLE_DIR / (EXAMPLE_FILE.stem + '_two_population_crossing.csv'))
breakEnergyEv = float(breakTable.loc[EPOCH, 'break_E_eV'])
detection = detectStrahlFromPad(EXAMPLE_FILE, EPOCH, breakEnergyEv)
seeds = seedFromData(psd, counts, grid, strahlDetection=detection, scDensity=scDensity, breakEnergyEv=breakEnergyEv)

fits = {}
for objective, label in (('poisson', 'Poisson (all pixels)'), ('chi2', 'chi2 (counts > 0)')):
    params, modelPsd, diagnostics = fitEpoch(psd, counts, grid, seeds, objective=objective)
    fits[label] = {'params': params,
                   'deviance_per_dof': poissonDeviancePerDof(modelPsd, counts, grid, diagnostics['free_params']),
                   'red_chi2': reducedChiSquare(modelPsd, psd, counts, grid, diagnostics['free_params'])}

fig = plotFitComparison(grid, psd, counts, {label: fit['params'] for label, fit in fits.items()},
                        breakEnergyEv=breakEnergyEv, timeLabel=str(epochTime))''')
    nb.code('''comparison = pd.DataFrame({label: {**componentMomentsFromParams(fit['params']),
                                   'deviance_per_dof': fit['deviance_per_dof'], 'red_chi2': fit['red_chi2']}
                           for label, fit in fits.items()})
comparison['chi2 / Poisson'] = comparison.iloc[:, 1] / comparison.iloc[:, 0]
comparison.map(lambda value: f'{value:.4g}')''')
    nb.md('''What to look at:

- **The tails.** Above ~300 eV the chi-square total (red) runs above the channel means
  in every cut, by up to an order of magnitude at 1 keV. Its pull map is blue at high
  energy: it predicts counts that were not seen. The Poisson total (black) follows the
  means.
- **The halo.** Chi-square makes it three times hotter and ten times denser, with
  $\\kappa_h$ halved: the only way to pass through the detected pixels of a faint tail.
- **The goodness of fit.** Chi-square has the *better* reduced chi-square (the quantity
  it minimised, on the pixels it looked at) and the *worse* deviance. Reduced
  chi-square on detected pixels cannot see the problem.''')
    nb.md('''## 2. All thirty epochs

Is this systematic? The ratio chi-square / Poisson of every parameter, epoch by epoch
(`runFiles(..., objective='chi2')`):''')
    nb.code('''from evdf_poisson.runner import runFiles

CHI2_DIR = REPO_ROOT / 'results' / 'fitting' / 'example_run_chi2'
poissonSummary = pd.read_csv(RESULTS_DIR / 'summary.csv').set_index('epoch_index')
chi2Summary = runFiles(EXAMPLE_DIR, CHI2_DIR, workers=4, objective='chi2').set_index('epoch_index')

PARAMETERS = ['n_c', 'T_par_c_eV', 'n_h', 'T_par_h_eV', 'kappa_h', 'n_b', 'T_par_b_eV', 'T_perp_b_eV', 'kappa_b']
ratio = chi2Summary[PARAMETERS] / poissonSummary[PARAMETERS]
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.axhline(1, color='0.5', linewidth=1)
for column in PARAMETERS:
    ax.plot([column] * len(ratio), ratio[column], 'o', color='#d62728', alpha=0.6)
ax.set_yscale('log'); ax.set_ylabel('chi2 fit / Poisson fit'); ax.grid(alpha=0.3, which='both')
ax.set_title('30 epochs: parameter ratio chi-square / Poisson')
pd.DataFrame({'median ratio chi2/Poisson': ratio.median(),
              'Poisson median': poissonSummary[PARAMETERS].median(),
              'chi2 median': chi2Summary[PARAMETERS].median()}).round(3)''')
    nb.code('''fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, column, title in ((axes[0], 'deviance_per_dof', 'Poisson deviance / dof (all pixels)'),
                          (axes[1], 'red_chi2', r'reduced $\\chi^2$ (detected pixels only)')):
    ax.plot(poissonSummary.index, poissonSummary[column], 'o-', color='black', label='Poisson fit')
    ax.plot(chi2Summary.index, chi2Summary[column], 's-', color='#d62728', label='chi2 fit')
    ax.set_title(title); ax.set_xlabel('epoch'); ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()''')
    nb.md('''On every epoch chi-square wins on its own metric and loses on the deviance. The halo
changes *qualitatively*: ten times the density, three times the temperature, $\\kappa_h$
halved. The strahl comes out 1.5-2 times hotter with a much harder tail
($\\kappa_b$ ~3 instead of ~20).''')
    nb.md('''## 3. Known truth: synthetic data at decreasing count level

Real data have no truth, so we make some:

1. take the Poisson fit of this epoch as the truth;
2. evaluate it on the real instrument grid and convert to expected counts with the real
   one-count levels;
3. draw Poisson counts and fit them with both objectives (always the same draw for
   both).

The **brightness** factor scales the expected counts, as a shorter accumulation time
or a smaller geometric factor would: 1 = as observed, 0.3 and 0.1 = fainter. The truth
model is the fit model itself, so any deviation is caused by the objective alone.''')
    nb.code('''from evdf_poisson.synthetic import runSyntheticBenchmark

truthParams = fits['Poisson (all pixels)']['params']
benchmark = runSyntheticBenchmark(EXAMPLE_FILE, EPOCH, truthParams, detection, breakEnergyEv,
                                  brightnessLevels=(1.0, 0.3, 0.1), numRealisations=4, workers=4)
benchmark.groupby(['brightness', 'objective'])[['total_counts', 'zero_count_fraction', 'deviance_per_dof', 'red_chi2']].mean().round(3)''')
    nb.code('''PARAMETERS = ['n_c', 'T_par_c_eV', 'n_h', 'T_par_h_eV', 'kappa_h', 'n_b', 'T_par_b_eV', 'T_perp_b_eV']
STYLE = {'poisson': ('black', 'o', 'Poisson'), 'chi2': ('#d62728', 's', 'chi2')}
fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharex=True)
for ax, name in zip(axes.ravel(), PARAMETERS):
    for objective, (color, marker, label) in STYLE.items():
        rows = benchmark[benchmark['objective'] == objective]
        jitter = (0.93 if objective == 'poisson' else 1.07)
        ax.plot(rows['brightness'] * jitter, rows[f'fit_{name}'] / rows[f'truth_{name}'],
                marker, color=color, alpha=0.7, linestyle='none', label=label)
    ax.axhline(1, color='0.5', linewidth=1)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_title(name); ax.grid(alpha=0.3, which='both')
    ax.set_xticks([1.0, 0.3, 0.1]); ax.set_xticklabels(['1', '0.3', '0.1'])
for ax in axes[1]: ax.set_xlabel('brightness (count level)')
for ax in axes[:, 0]: ax.set_ylabel('fitted / truth')
axes[0, 0].legend()
fig.suptitle('synthetic truth on the real grid: recovered / true parameter versus count level')
fig.tight_layout()''')
    nb.md('''Reading the panels:

- **Core:** recovered by both objectives at every count level (it lives where counts
  are high).
- **Halo and strahl, Poisson:** recovered at all count levels, with a scatter that
  grows as the counts drop.
- **Halo and strahl, chi-square:** biased already at the observed brightness, and more
  so as the counts drop: halo too hot and too dense, $\\kappa_h$ too low, strahl too
  wide. Its own reduced chi-square stays near or below 1 throughout.''')
    nb.md('''## 4. Why: the survivor bias of detected pixels

A pixel that expects $\\mu$ counts registers zero with probability $e^{-\\mu}$. Averaging
only the pixels with $C > 0$ estimates not $\\mu$ but

$$ \\mathrm{E}[C \\mid C > 0] = \\frac{\\mu}{1 - e^{-\\mu}} . $$

For $\\mu = 0.1$ that is 1.05, ten times the truth. A fit that sees only the detected
pixels is pulled towards this censored mean wherever the expected counts are below
about one, which in EAS data is the whole suprathermal range. The Poisson likelihood
keeps the $e^{-\\mu}$ of the empty pixels and is unbiased at any count level (Cash 1979;
Humphrey et al. 2009; Nicolaou et al. 2024).''')
    nb.code('''expectedCounts = np.logspace(-2, 1, 200)
survivorMean = expectedCounts / (1 - np.exp(-expectedCounts))
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.loglog(expectedCounts, survivorMean / expectedCounts, color='#d62728')
ax.set_xlabel(r'expected counts $\\mu$ in a pixel'); ax.set_ylabel(r'E[$C \\mid C>0$] / $\\mu$')
ax.set_title('bias of the mean over detected pixels'); ax.grid(alpha=0.3, which='both')
ax.axvspan(0.01, 1, color='0.9', label='typical suprathermal EAS pixel'); ax.legend();''')
    nb.md('''## 5. Conclusions

- Same model, stages and ranges: the objective alone changes the halo density by an
  order of magnitude, its temperature by a factor 3 and $\\kappa_h$ from ~5.3 to ~2.8
  on this data.
- Reduced chi-square on detected pixels does not diagnose the problem; the Poisson
  deviance on all pixels does.
- On synthetic data with a known truth, the Poisson fit is unbiased down to a tenth of
  the observed count level. The chi-square fit is biased at every level, increasingly
  so as counts drop.
- This is why the pipeline uses the Poisson deviance in every stage and reports
  `deviance_per_dof` as its goodness of fit.''')
    return nb


# ----------------------------------------------------------------------- 02
def build02():
    nb = Notebook()
    nb.md('''# 02 - Download and preprocessing: from SOAR files to the fitting-ready file

**This notebook ships pre-executed.** Running it needs the raw EAS files (830 MB per six
hours), the SOAR RPW potential files (170 MB per day) and a network connection for MAG,
PAS and RPW density (CDAWeb). Read the outputs; run it when you preprocess your own
interval. The other notebooks start from the bundled example and need none of this.

**Runtime when executed:** about 3 minutes per six-hour segment, downloads aside.

## What the preprocessing does

`src/preprocess_data_for_fit_hdf5.py` writes six-hour
`VDF_eas_forFitting_<start>_<end>_1min.h5` files:

1. read the **L2 PSD** of both heads and the **L1 counts** (PSD / counts is the
   one-count level of notebook 01);
2. mask L1 telemetry blocks at the counter **saturation value** 65 535 (set to NaN;
   census in `n_saturated_pixels`);
3. fetch **MAG**, **PAS** and the **RPW electron density** through `cdasws`;
4. subtract the **spacecraft potential** from the energies: automatic before 2023, from
   a table afterwards (section 2);
5. rotate every pixel into the field-aligned frame: pitch angle, $v_\\parallel$,
   $v_\\perp$;
6. merge the heads into one 64 x 1024 grid, build the 18-bin PAD, resample to 1 minute,
   write compressed.

The EAS geometric factors and efficiencies ship in `examples/example_data/`.''')
    nb.code(SETUP_CELL)
    nb.md('''## 1. The EAS files from SOAR

`download_L2_PSD_from_SOAR.py` queries the Solar Orbiter Archive through `sunpy-soar`,
day by day, for the L2 `swa-eas{1,2}-nm3d-psd` and L1 `swa-eas{1,2}-nm3d` products and
keeps the newest version of every file. Files go to `data/vdfs/` and `data/L1_Count/`
(both ignored by git).

Do not mix V01 and V03 files in one run: their energy tables have different dimensions.

The cell lists what is on disk for the first six hours of 2023-04-04. Set
`DOWNLOAD_IF_MISSING = True` to fetch the four files (about 830 MB).''')
    nb.code('''from download_L2_PSD_from_SOAR import (discoverLocalL2Psd, discoverLocalL1Count,
                                       downloadSoL2PsdNm3d, downloadSoL1CountNm3d)

RAW_DIR = REPO_ROOT / 'data'
SEGMENT = ['2023-04-04 00:00:00', '2023-04-04 06:00:00']
DOWNLOAD_IF_MISSING = False

eas1PsdFiles, eas2PsdFiles = discoverLocalL2Psd(SEGMENT, downloadDir=str(RAW_DIR / 'vdfs'))
eas1CountFiles, eas2CountFiles = discoverLocalL1Count(SEGMENT, downloadDir=str(RAW_DIR / 'L1_Count'))
if DOWNLOAD_IF_MISSING and not (eas1PsdFiles and eas2PsdFiles):
    eas1PsdFiles, eas2PsdFiles = downloadSoL2PsdNm3d(SEGMENT, downloadDir=str(RAW_DIR / 'vdfs'))
if DOWNLOAD_IF_MISSING and not (eas1CountFiles and eas2CountFiles):
    eas1CountFiles, eas2CountFiles = downloadSoL1CountNm3d(SEGMENT, downloadDir=str(RAW_DIR / 'L1_Count'))

rawFiles = [*eas1PsdFiles, *eas2PsdFiles, *eas1CountFiles, *eas2CountFiles]
hasRawFiles = len(rawFiles) == 4
for path in rawFiles:
    print(f'{Path(path).name:70s} {Path(path).stat().st_size / 1e6:6.0f} MB')
if not hasRawFiles:
    print('raw files missing: the preprocessing cells below are skipped (set DOWNLOAD_IF_MISSING = True to fetch them)')''')
    nb.md('''## 2. The spacecraft potential

The potential measured by RPW is subtracted from the electron energies.

- **Before 2023** the preprocessor fetches the CDAWeb 10-second product by itself.
- **From 2023 on** that product no longer exists. `src/build_scpot_table_from_soar_l3.py`
  downloads the SOAR RPW L3 `rpw-bia-scpot` daily files (`--download`), keeps the
  samples with `QUALITY_FLAG` 3 and reduces them to 20 s medians, in the two-column
  format (`time, scpotV`) that the preprocessor takes as `scpot_file`:

```
PYTHONPATH=src python src/build_scpot_table_from_soar_l3.py --start 2023-04-04 --end 2023-04-05 \\
    --download --out examples/example_data/scpot_rpw_l3_20230404_20230405_20s.csv
```

The table for the two example days ships with the repository and is plotted below
(rebuilt from the daily CDFs when they are present).''')
    nb.code('''from build_scpot_table_from_soar_l3 import buildTable, discoverDailyFiles

SCPOT_TABLE = CALIBRATION_DIR / 'scpot_rpw_l3_20230404_20230405_20s.csv'
SCPOT_CDF_DIR = RAW_DIR / 'rpw_l3_scpot'
startDay, endDay = pd.Timestamp('2023-04-04'), pd.Timestamp('2023-04-05')
if discoverDailyFiles(startDay, endDay, str(SCPOT_CDF_DIR)):
    scpotTable = buildTable(startDay, endDay, str(SCPOT_CDF_DIR), binSeconds=20, minQualityFlag=3, marginMinutes=30)
    print('rebuilt from the daily CDFs; identical to the shipped table:',
          np.allclose(scpotTable['scpotV'].to_numpy(), pd.read_csv(SCPOT_TABLE)['scpotV'].to_numpy(), atol=5e-5))
else:
    scpotTable = pd.read_csv(SCPOT_TABLE, parse_dates=['time'])
    print('daily CDFs not on disk; using the shipped table')

fig, ax = plt.subplots(figsize=(10, 3.2))
ax.fill_between(scpotTable['time'], scpotTable['scpotV'] - scpotTable['scpotSigmaV'], scpotTable['scpotV'] + scpotTable['scpotSigmaV'], color='0.8', label='spread within 20 s')
ax.plot(scpotTable['time'], scpotTable['scpotV'], color='0.2', linewidth=0.7, label='20 s median')
ax.set_ylabel('spacecraft potential [V]'); ax.grid(alpha=0.3); ax.legend(fontsize=9); fig.autofmt_xdate()
print(f"{len(scpotTable)} rows, {scpotTable['time'].iloc[0]} .. {scpotTable['time'].iloc[-1]}, median {scpotTable['scpotV'].median():.2f} V")''')
    nb.md('''## 3. Run the preprocessing

`preprocess_main` takes the file lists, or a time range with `psd_source='soar'`
(download) or `'local'` (files on disk), plus the potential table and the output
directory. `compute_break_energy` would chain the break-energy step; it is off here so
that section 5 can show it separately.''')
    nb.code('''import time
from preprocess_data_for_fit_hdf5 import preprocess_main

PROCESSED_DIR = REPO_ROOT / 'data' / 'processed' / 'demo_2023-04-04'
if hasRawFiles:
    startTime = time.time()
    processedPaths = preprocess_main(
        eas1_files=eas1PsdFiles, eas2_files=eas2PsdFiles,
        count_eas1_files=eas1CountFiles, count_eas2_files=eas2CountFiles,
        download_dir=str(RAW_DIR), output_dir=str(PROCESSED_DIR),
        shift_SCPOT=True, scpot_file=str(SCPOT_TABLE),
        time_resolution='1min', compute_break_energy=False, psd_source='local',
    )
    print(f'\\nfinished in {(time.time() - startTime) / 60:.1f} min ->', processedPaths)''')
    nb.md('''## 4. What the file contains

One time-indexed dataset per quantity:

- `pa_vec_in`: the pixel grid the fit works on (pitch angle, PSD, $v_\\parallel$,
  $v_\\perp$, counts per pixel);
- `pad_in`: the pitch-angle distribution;
- `n_saturated_pixels`: the saturation census of step 2;
- the rest: ancillary measurements per epoch.

The attributes record where the potential came from and what was masked.''')
    nb.code('''import h5py

if hasRawFiles:
    processedFile = sorted(PROCESSED_DIR.glob('VDF_eas_forFitting_*.h5'))[0]
    with h5py.File(processedFile, 'r') as h5:
        def describe(name, item):
            if isinstance(item, h5py.Dataset):
                print(f'{name:28s} {str(item.shape):22s} {item.dtype}')
        h5.visititems(describe)
        times = pd.to_datetime(h5['time_EAS'][()])
        saturated = h5['n_saturated_pixels'][()]
        print('\\nattributes:', dict(h5.attrs))
    print(f'\\n{processedFile.name}: {len(times)} epochs, {times[0]} to {times[-1]}, '
          f'{processedFile.stat().st_size / 1e6:.0f} MB on disk')
    print(f'epochs with masked saturation blocks: {int((saturated > 0).sum())} '
          f'(pixels: {int(saturated.sum())})')''')
    nb.md('''## 5. Break energies for the new file

The crossing method (notebook 03) runs on the whole directory and writes
`<stem>_two_population_crossing.csv` next to the data, where the fitting runner looks
for it (`--break-dir` to point elsewhere).''')
    nb.code('''from compute_break_energy_crossing import runCrossingPipeline

if hasRawFiles:
    runCrossingPipeline(PROCESSED_DIR, PROCESSED_DIR, workers=4, writeCombined=False, overwrite=True)
    breakTable = pd.read_csv(PROCESSED_DIR / (processedFile.stem + '_two_population_crossing.csv'))
    print(breakTable['status'].value_counts().to_dict())
    print(breakTable[['epoch_id', 'timestamp', 'scpot_V', 'break_E_eV', 'core_T_eV', 'status']].iloc[68:73].round(2).to_string(index=False))''')
    nb.md('''## 6. Fit one epoch and compare with the bundled example

The bundled subset is epochs 50-79 of this segment, so the fit of the
demonstration epoch (01:10:08 UT, epoch 20 of the subset, epoch 70 of the segment)
must reproduce the reference values exactly. The archives occasionally reprocess MAG,
PAS or RPW; if they have, the ratios move away from 1 by the corresponding amount.''')
    nb.code('''from evdf_poisson import fitOneEpoch

if hasRawFiles:
    reference = pd.read_csv(RESULTS_DIR / 'summary.csv')
    referenceRow = reference[reference['time_iso'] == '2023-04-04 01:10:08'].iloc[0]
    newEpoch = int(np.flatnonzero(times == pd.Timestamp('2023-04-04 01:10:08'))[0])
    newBreak = float(breakTable.loc[newEpoch, 'break_E_eV'])
    newRow = fitOneEpoch(processedFile, newEpoch, breakEnergyEv=newBreak)
    columns = ['n_c', 'T_par_c_eV', 'T_perp_c_eV', 'n_h', 'T_par_h_eV', 'kappa_h', 'n_b', 'T_par_b_eV', 'deviance_per_dof', 'break_energy_eV']
    comparison = pd.DataFrame({'bundled example': referenceRow[columns],
                               'new preprocessing': pd.Series(newRow)[columns]}).astype(float)
    comparison['ratio'] = comparison['new preprocessing'] / comparison['bundled example']
    display(comparison.round(6))''')
    nb.md('''## 7. Your own interval

Three commands, from the repository root with `PYTHONPATH=src`:

1. preprocess:
   `preprocess_main(start_time, end_time, psd_source='soar', shift_SCPOT=True, scpot_file=..., output_dir=D)`,
   or edit the `__main__` block of `src/preprocess_data_for_fit_hdf5.py`. Leave
   `scpot_file` out for intervals before 2023;
2. `python src/compute_break_energy_crossing.py --vdf-dir D --output-dir D`;
3. `python -m evdf_poisson.runner --data-dir D --out results/fitting/<name> --workers N`.

Practical notes:

- keep a segment within one SOAR file version and within MAG coverage (epochs without
  MAG are dropped);
- six hours at 1-minute cadence: ~650 MB of RAM, a ~600 MB file, ~360 epochs; a day is
  four files;
- fitting takes ~6 s per epoch and core: a day is about 40 minutes on 4 workers,
  5 minutes on 32.''')
    return nb


# ===== APPENDED: instrument geometry section of 01 and the models/seeds notebook 04
def add01Geometry(nb):
    nb.md('''## 0. The instrument: two heads, two frames

EAS sits at the end of the boom, in the shadow of the heat shield. Its two identical
top-hat analysers, **EAS1** and **EAS2**, are mounted at right angles and together see
almost the whole sky. Each head has

- 16 **elevation** bins (about $\\pm$45 deg),
- 32 **azimuth** anodes (11.25 deg each),
- 64 logarithmic **energy** steps, about 1 eV to 5 keV,

so one epoch holds 2 x 16 x 32 x 64 = 65 536 measurements.

Pixel angles are defined in each head's own frame. One fixed rotation matrix per head
(`EAS1_TO_SRF`, `EAS2_TO_SRF`, from the L2 files) takes them to the **spacecraft
reference frame (SRF)**: $+X$ to the Sun, $+Z$ roughly solar north. MAG and PAS come in
SRF too, so the field-aligned grid of section 3 is built there. The tables are in
`examples/example_data/eas_head_geometry.npz` (written by `src/eas_geometry.py`).

The map shows the **velocity direction of the electrons** each pixel measures (the
pixel looks the opposite way), in the convention of Owen et al. (2020, A&A 642, A16,
Fig. 3): azimuth 0 = moving along $-X_\\mathrm{SRF}$, away from the Sun, clockwise
positive ($+Y_\\mathrm{SRF}$ at +90); elevation = latitude of the velocity.''')
    nb.code('''import h5py
from eas_geometry import (loadHeadGeometry, pixelUnitVectorsSrf, pixelSolidAngles,
                          pixelPitchAngles, easAzimuthElevation, HEAD_SLICES)

geometry = loadHeadGeometry(GEOMETRY_FILE)
for headName in ('EAS1', 'EAS2'):
    head = geometry[headName]
    print(f"{headName}: elevation {head['elevation_deg'].min():.1f}..{head['elevation_deg'].max():.1f} deg "
          f"({len(head['elevation_deg'])} bins), azimuth {len(head['azimuth_deg'])} anodes, "
          f"energy {head['energy_eV'].min():.1f}..{head['energy_eV'].max():.0f} eV ({len(head['energy_eV'])} steps)")
    print(f"{headName}_TO_SRF =\\n{np.round(head['to_srf'], 3)}")

unitVectors = pixelUnitVectorsSrf(geometry)          # (1024, 3) velocity directions, in the pixel order of the files
pixelAzimuth, pixelElevation = easAzimuthElevation(unitVectors)
print(f'sky covered by the two heads: {pixelSolidAngles(geometry).sum() / (4 * np.pi):.0%} of 4 pi sr')

EPOCH = 20
with h5py.File(EXAMPLE_FILE, 'r') as h5:
    magneticField = h5['B_avg'][EPOCH]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [2.4, 1]})
ax = axes[0]
for headName, marker, color in (('EAS1', 'x', 'blue'), ('EAS2', '+', 'red')):
    block = HEAD_SLICES[headName]
    ax.scatter(pixelAzimuth[block], pixelElevation[block], s=14, marker=marker, color=color, alpha=0.75, label=headName)
for vector, marker, label in ((magneticField, '*', 'along +B (pitch angle 0)'),
                              (-magneticField, 'P', 'along -B (pitch angle 180)'),
                              ((-1.0, 0.0, 0.0), 'v', 'moving away from the Sun (-X)'),
                              ((1.0, 0.0, 0.0), '^', 'moving towards the Sun (+X)')):
    azimuth, elevation = easAzimuthElevation(np.asarray(vector, float))
    ax.plot(azimuth, elevation, marker, color='k', markersize=11, markerfacecolor='yellow', label=label)
ax.set_xlim(-186, 186); ax.set_ylim(-92, 92)
ax.set_xticks(np.arange(-180, 181, 30)); ax.set_yticks(np.arange(-90, 91, 15))
ax.set_xlabel('EAS azimuth [deg]  (0 = moving along -X, away from the Sun; clockwise positive)')
ax.set_ylabel('elevation [deg]')
ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6); ax.legend(fontsize=8, loc='upper center', ncol=3)
ax.set_title('SWA-EAS two-head field of view in SRF (velocity directions; cf. Owen et al. 2020, Fig. 3)')
ax = axes[1]
for headName, color in (('EAS1', '#2a78d6'), ('EAS2', '#e34948')):
    head = geometry[headName]
    ax.errorbar(np.arange(64), head['energy_eV'], yerr=[head['energy_delta_lower_eV'], head['energy_delta_upper_eV']],
                fmt='.', color=color, label=headName, markersize=3)
ax.set_yscale('log'); ax.set_xlabel('energy step'); ax.set_ylabel('energy [eV]')
ax.grid(alpha=0.3, which='both'); ax.legend(); ax.set_title('energy table (bars = bin width)')
fig.tight_layout()''')
    nb.md('''**Check.** The pitch angle of a pixel is the angle between its velocity direction and
**B**. Recomputing it from the geometry file and the field of the epoch reproduces the
values stored in the preprocessed file, so the conventions above (velocity direction,
head rotation, pixel order) are the ones the pipeline uses.''')
    nb.code('''with h5py.File(EXAMPLE_FILE, 'r') as h5:
    storedPitchAngle = h5['pa_vec_in'][EPOCH, 30, :, 0]      # energy step 30, all 1024 pixels
recomputedPitchAngle = pixelPitchAngles(unitVectors, magneticField)
finite = np.isfinite(storedPitchAngle)
print(f'max |stored - recomputed| pitch angle over {int(finite.sum())} pixels: '
      f'{np.max(np.abs(storedPitchAngle[finite] - recomputedPitchAngle[finite])):.1e} deg')''')


# ----------------------------------------------------------------------- 04
def build04():
    nb = Notebook()
    nb.md('''# 04 - The component models and the initial guesses

What the fit adjusts: three analytic components with 14 parameters, their initial
guesses and their allowed ranges. Nothing is fitted here; the curves use the reference
fit of one epoch (`results/fitting/example_run_poisson/`, notebook 00).

**Runtime:** under a minute.''')
    nb.code(SETUP_CELL)
    nb.md('''## 1. The three models

All three are gyrotropic functions of $(v_\\parallel, v_\\perp)$, the velocity along and
across **B** in the spacecraft frame. For every component
$v_{th} = \\sqrt{2 k_B T / m_e}$, so a temperature always means the same thing.

**Core: bi-Maxwellian** (4 parameters: $n_c$, $u_{\\parallel c}$, $T_{\\parallel c}$, $T_{\\perp c}$)

$$ f_c = \\frac{n_c}{\\pi^{3/2} v_{th\\parallel} v_{th\\perp}^2}
   \\exp\\!\\left[-\\frac{(v_\\parallel - u_{\\parallel c})^2}{v_{th\\parallel}^2} - \\frac{v_\\perp^2}{v_{th\\perp}^2}\\right] $$

**Halo: bi-kappa** (5 parameters: $n_h$, $u_{\\parallel h}$, $T_{\\parallel h}$, $T_{\\perp h}$, $\\kappa_h$)

$$ f_h = \\frac{n_h}{\\pi^{3/2} v_{th\\parallel} v_{th\\perp}^2}\\,
   \\frac{\\Gamma(\\kappa + 1)}{(\\kappa - 3/2)^{3/2}\\,\\Gamma(\\kappa - 1/2)}
   \\left[1 + \\frac{(v_\\parallel - u_{\\parallel h})^2}{(\\kappa - 3/2) v_{th\\parallel}^2} + \\frac{v_\\perp^2}{(\\kappa - 3/2) v_{th\\perp}^2}\\right]^{-\\kappa - 1} $$

The $(\\kappa - 3/2)$ convention makes $T$ the actual second moment. Small $\\kappa$ is a
hard power-law tail; $\\kappa \\to \\infty$ is a Maxwellian. The halo drift
$u_{\\parallel h}$ is free but kept small (section 3).

**Strahl: drifting, truncated bi-kappa** (5 parameters: $n_b$, $u_{\\parallel b}$,
$T_{\\parallel b}$, $T_{\\perp b}$, $\\kappa_b$): a bi-kappa centred at the drift
$u_{\\parallel b}$ whose parallel width is divided by $\\sqrt{D}$, $D = 5$, on the side
facing the core, so the beam cannot leak into the core.

Code: `evdf_poisson/models.py`.''')
    nb.code('''import json
from evdf_poisson.models import (biMaxwellian, biKappa, strahlTruncatedBiKappa, coreHaloStrahlModel,
                                 temperatureFromThermalSpeed, ELECTRON_MASS_G, ERG_PER_EV, STRAHL_TRUNCATION_D)
from evdf_poisson.staged_fit import ALL_PARAM_NAMES

EPOCH = 20
reference = pd.read_csv(RESULTS_DIR / 'summary.csv').set_index('epoch_index')
fitParams = json.loads(reference.loc[EPOCH, 'fit_params_json'])
breakEnergyEv = float(reference.loc[EPOCH, 'break_energy_eV'])
breakSpeed = np.sqrt(2 * breakEnergyEv * ERG_PER_EV / ELECTRON_MASS_G)
print(f'epoch {EPOCH} ({reference.loc[EPOCH, "time_iso"]}): break {breakEnergyEv:.1f} eV = {breakSpeed / 1e8:.2f} x 10^8 cm/s')
parameterTable = pd.Series({name: fitParams[name] for name in ALL_PARAM_NAMES}).to_frame('reference fit (CGS)')
parameterTable['as temperature [eV]'] = [temperatureFromThermalSpeed(fitParams[name]) if name.startswith('v_th') else np.nan for name in ALL_PARAM_NAMES]
parameterTable.map(lambda value: '' if isinstance(value, float) and np.isnan(value) else f'{value:.4g}')''')
    nb.code('''vAxis = np.linspace(-2.5e9, 2.5e9, 1201)
zeros = np.zeros_like(vAxis)
components = {
    'core (bi-Maxwellian)': (biMaxwellian(vAxis, zeros, fitParams['n_c'], fitParams['u_par_c'], fitParams['v_th_par_c'], fitParams['v_th_perp_c']),
                             biMaxwellian(zeros, vAxis, fitParams['n_c'], fitParams['u_par_c'], fitParams['v_th_par_c'], fitParams['v_th_perp_c']), '#e34948'),
    'halo (bi-kappa)': (biKappa(vAxis, zeros, fitParams['n_h'], 0.0, fitParams['v_th_par_h'], fitParams['v_th_perp_h'], fitParams['kappa_h']),
                        biKappa(zeros, vAxis, fitParams['n_h'], 0.0, fitParams['v_th_par_h'], fitParams['v_th_perp_h'], fitParams['kappa_h']), '#eda100'),
    'strahl (truncated bi-kappa)': (strahlTruncatedBiKappa(vAxis, zeros, fitParams['n_b'], fitParams['u_par_b'], fitParams['v_th_par_b'], fitParams['v_th_perp_b'], fitParams['kappa_b']),
                                    strahlTruncatedBiKappa(zeros, vAxis, fitParams['n_b'], fitParams['u_par_b'], fitParams['v_th_par_b'], fitParams['v_th_perp_b'], fitParams['kappa_b']), '#2a78d6'),
}
totalPar = coreHaloStrahlModel(vAxis, zeros, fitParams)
totalPerp = coreHaloStrahlModel(zeros, vAxis, fitParams)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
for label, (alongPar, alongPerp, color) in components.items():
    axes[0].semilogy(vAxis / 1e9, alongPar, color=color, label=label)
    axes[1].semilogy(vAxis / 1e9, alongPerp, color=color, label=label)
axes[0].semilogy(vAxis / 1e9, totalPar, 'k', linewidth=1.2, label='sum')
axes[1].semilogy(vAxis / 1e9, totalPerp, 'k', linewidth=1.2, label='sum')
for ax, title in zip(axes, (r'along $\\mathbf{B}$ ($v_\\perp = 0$)', r'across $\\mathbf{B}$ ($v_\\parallel = 0$)')):
    for sign in (-1, 1):
        ax.axvline(sign * breakSpeed / 1e9, color='0.5', linestyle=':', label='break speed' if sign > 0 else None)
    ax.set_ylim(1e-32, 1e-22); ax.set_xlabel(r'$v$ [$10^9$ cm/s]'); ax.set_ylabel(r'$f$ [s$^3$ cm$^{-6}$]')
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle('the three components of the reference fit, epoch %d' % EPOCH); fig.tight_layout()''')
    nb.md('''What each parameter does, and why they cannot all be fitted at once:''')
    nb.code('''haloWidth, haloDensity = fitParams['v_th_par_h'], fitParams['n_h']
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for kappa, color in ((2.0, '#d62728'), (3.0, '#eda100'), (5.0, '#2a78d6'), (12.0, '0.45')):
    axes[0].semilogy(vAxis / 1e9, biKappa(vAxis, zeros, haloDensity, 0.0, haloWidth, haloWidth, kappa), color=color, label=f'kappa = {kappa:g}')
axes[0].semilogy(vAxis / 1e9, biMaxwellian(vAxis, zeros, haloDensity, 0.0, haloWidth, haloWidth), 'k--', label='Maxwellian (kappa -> inf)')
axes[0].set_title('bi-kappa: same n and T, different kappa'); axes[0].set_ylim(1e-33, 1e-25)

beam = dict(n=fitParams['n_b'], u_par=fitParams['u_par_b'], v_th_par=fitParams['v_th_par_b'], v_th_perp=fitParams['v_th_perp_b'], kappa=fitParams['kappa_b'])
axes[1].semilogy(vAxis / 1e9, biKappa(vAxis, zeros, **beam), color='0.5', linestyle='--', label='plain bi-kappa at the strahl drift')
axes[1].semilogy(vAxis / 1e9, strahlTruncatedBiKappa(vAxis, zeros, **beam), color='#2a78d6', label=f'truncated (D = {STRAHL_TRUNCATION_D}): narrower on the core side')
axes[1].semilogy(vAxis / 1e9, components['core (bi-Maxwellian)'][0], color='#e34948', alpha=0.5, label='core, for scale')
axes[1].set_title('strahl: the truncation keeps the beam out of the core'); axes[1].set_ylim(1e-33, 1e-23)
for ax in axes:
    ax.set_xlabel(r'$v_\\parallel$ [$10^9$ cm/s]'); ax.set_ylabel(r'$f$ [s$^3$ cm$^{-6}$]'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout()''')
    nb.md('''Two things to take from these curves:

- **A low-$\\kappa$ halo and a Maxwellian core overlap almost everywhere.** Halo density
  and width trade against each other, and against the core, wherever the tail is faint.
  Unconstrained, the halo becomes a second, slightly hotter core.
- **A strahl with a free width can imitate a drifting core** unless its drift is kept
  suprathermal.

This is why the fit is staged and every parameter has a range (sections 3 and 4).

## 2. Initial guesses from the data

`seedFromData` builds the initial guesses from the measured grid, the RPW density, the
break energy and the PAD verdict (notebook 03):

| component | initial guess | from |
|---|---|---|
| core | $n_c = 0.8\\, n_{sc}$ (RPW); drift and widths from the weighted moments of the detected pixels below 70 eV, widths rescaled so that the peak PSD matches $n / (\\pi^{3/2} v_{th\\parallel} v_{th\\perp}^2)$ | data |
| halo | $n_h = 0.03\\, n_c$, $T_h = 5\\, T_c$, $\\kappa_h = 5$, $u_{\\parallel h} = 0$ (free, starts at rest) | typical solar-wind ratios |
| strahl | side from the PAD verdict, $n_b = 0.01\\, n_c$, drift and width from the break speed, $\\kappa_b = 5$, perpendicular width from the PAD width | PAD + break |

Stages B and C re-anchor the halo and strahl guesses on the *fitted* core and the break
speed (section 3), so the numbers below only start stage A.''')
    nb.code('''from evdf_poisson import loadEpoch, detectStrahlFromPad, seedFromData
from evdf_poisson.plotting import plotMeasuredVdf

grid, psd, counts, epochNs, scDensity = loadEpoch(EXAMPLE_FILE, EPOCH)
detection = detectStrahlFromPad(EXAMPLE_FILE, EPOCH, breakEnergyEv)
seeds = seedFromData(psd, counts, grid, strahlDetection=detection, scDensity=scDensity, breakEnergyEv=breakEnergyEv)
seedParams = {name: float(seeds[name]) for name in ALL_PARAM_NAMES}
print(f"PAD verdict: {detection['class']} (sign {detection['sign']}, width {detection['paw_anti' if detection['sign'] < 0 else 'paw_par']:.0f} deg)   RPW density {scDensity:.1f} cm^-3")

comparison = pd.DataFrame({'seed': seedParams, 'reference fit': {name: fitParams[name] for name in ALL_PARAM_NAMES}})
comparison['fit / seed'] = (comparison['reference fit'] / comparison['seed']).replace([np.inf, -np.inf], np.nan)   # u_par_h starts at 0
comparison.map(lambda value: '' if np.isnan(value) else f'{value:.3g}')''')
    nb.md('''The initial guesses drawn over the data, with the reference fit for comparison. The
core starts at 0.8 times the RPW density, which is above what EAS measures on these
days (calibration factor, notebook 06), so stage A lowers it. Halo and strahl are
placeholders that stages B and C replace.''')
    nb.code('''fig = plotMeasuredVdf(grid, psd, counts, params=seedParams, breakEnergyEv=breakEnergyEv,
                      timeLabel='initial guesses, no fitting yet')
fig = plotMeasuredVdf(grid, psd, counts, params={name: fitParams[name] for name in ALL_PARAM_NAMES}, breakEnergyEv=breakEnergyEv,
                      timeLabel='reference fit')''')
    nb.md('''## 3. The parameter ranges

Every free parameter moves inside a range anchored to a measured quantity (the RPW
density, the fitted core width, the break speed, the PAD width), so the ranges scale
with the plasma. Stage by stage (notebook 05 runs them):

**Stage A, core** (all pixels below the break)

''' + STAGE_TABLES['A'] + '''

**Stage B, halo** (above the break, outside both 30 deg cones; core frozen)

''' + STAGE_TABLES['B'] + '''

**Stage C, strahl** (the 30 deg cone on the detected side; core and halo frozen)

''' + STAGE_TABLES['C'] + '''

**Stage D, joint**: all 14 parameters free inside these ranges. The strahl drift is
locked to $\\pm 20\\,\\%$ of its stage-C value, and its perpendicular width may move
within $[0.3, 3]$ times the PAD anchor.

The numbers are constants of `evdf_poisson/staged_fit.py`:''')
    nb.code('''import evdf_poisson.staged_fit as stagedFit

constants = {name: getattr(stagedFit, name) for name in dir(stagedFit)
             if name.isupper() and not name.endswith('NAMES')}
pd.Series(constants).to_frame('value')''')
    nb.md('''## 4. Why the fit is staged

Released all at once on a distribution that is 90 % core, the 14 parameters do not
converge to the physical answer: the halo turns into a second, slightly hotter core and
the strahl into a drifting copy of the core (section 1). So:

1. each component is first fitted on the pixels where it dominates, the others frozen;
2. its ranges are anchored to the previous result;
3. only then is everything released together, inside those ranges.

Notebook 05 runs the four stages on this epoch. Notebook 07 shows that the objective
(Poisson versus chi-square) matters as much as the staging.

## Summary

- Core = bi-Maxwellian, halo = bi-kappa, strahl = drifting truncated bi-kappa;
  14 parameters, thermal speeds with $v_{th} = \\sqrt{2 k_B T / m_e}$.
- Initial guesses come from the data: moments below 70 eV, RPW density, break energy,
  PAD verdict.
- Ranges are anchored to measured quantities and re-derived stage by stage.''')
    return nb


# ----------------------------------------------------------------------- 08
def build08():
    nb = Notebook()
    nb.md("""# 08 - Fit-quality control that works at low counts

The expectation of `deviance_per_dof` depends on how bright the plasma is (notebook
05), so a fixed threshold on it flags brightness, not misfits. This notebook computes
two statistics whose expectation does not depend on the count level, for the 30
example epochs, and shows how to turn them into flags for a long run.

**Runtime:** about half a minute.""")
    nb.code(SETUP_CELL)
    nb.md("""## 1. Two statistics

**Randomized PIT and KS test.** For a Poisson pixel with expected counts $\\mu$ and
observed counts $C$, the randomized probability integral transform

$$ u = F(C - 1 \\mid \\mu) + U \\,[F(C \\mid \\mu) - F(C - 1 \\mid \\mu)], \\qquad U \\sim \\mathcal{U}(0, 1), $$

($F$ = Poisson CDF) is uniform when the model is right, at any count level, zeros
included. A Kolmogorov-Smirnov test of all the $u$ against the uniform distribution
gives one p-value per epoch, and one for the strahl wedge alone. The uniform draws are
seeded by the epoch time, so the numbers are reproducible.

**Calibrated deviance.** Under the Poisson model the deviance of a pixel has an exact
expectation $\\mathrm{E}[D \\mid \\mu]$ and variance. Summed over the grid:
$R = \\sum D / \\sum \\mathrm{E}[D]$ and $z = (\\sum D - \\sum \\mathrm{E}[D]) / \\sqrt{\\sum \\mathrm{Var}[D]}$.
Poisson noise alone gives $R = 1$, $z \\approx 0$.

Code: `evdf_poisson/qc.py`. `qcEpoch` handles one epoch, `qcFile` every row of a runner
CSV, and `python -m evdf_poisson.qc --data-dir <dir> --run-dir <results>` a whole run
(writes `<results>/qc/`).""")
    nb.code("""import json
from evdf_poisson.qc import qcEpoch, qcFile, flagQcTable

EPOCH = 20
summary = pd.read_csv(RESULTS_DIR / 'summary.csv')
row = summary.set_index('epoch_index').loc[EPOCH]
result = qcEpoch(EXAMPLE_FILE, EPOCH, json.loads(row['fit_params_json']), strahlSign=int(row['strahl_sign']))
print({key: round(value, 4) for key, value in result.items() if isinstance(value, float)})

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, values, title in ((axes[0], result['pit_values'], 'all valid pixels'),
                          (axes[1], result['pit_values_wedge'], 'strahl wedge (30 deg cone)')):
    ax.hist(values, bins=40, range=(0, 1), color='0.55', density=True)
    ax.axhline(1, color='#d62728', linestyle='--', label='perfect model')
    ax.set_xlabel('randomized PIT'); ax.set_ylabel('density'); ax.set_title(f'{title}: {len(values)} pixels'); ax.legend(fontsize=9)
fig.tight_layout()""")
    nb.md("""Reading the PIT histogram:

- **flat:** the counts are consistent with the model;
- **surplus near 0:** fewer counts than predicted (model too high there); **near 1:**
  the opposite;
- **U shape:** the data scatter more than the model allows (over-dispersion); a
  **hump:** less.

## 2. The 30 epochs""")
    nb.code("""qcTable = flagQcTable(qcFile(EXAMPLE_FILE, summary, workers=4))
qcTable[['epoch_index', 'n_pixels', 'n_detected', 'ks_p', 'd_red', 'dev_ratio', 'dev_z',
         'ks_p_wedge', 'dev_ratio_wedge', 'flag_hard', 'flag_strong', 'flag_wedge']].round(4)""")
    nb.code("""fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
axes[0].plot(summary['n_c'], qcTable['d_red'], 'o', color='0.4')
axes[0].set_xlabel('fitted core density [cm$^{-3}$]'); axes[0].set_ylabel('deviance / dof'); axes[0].set_title('depends on brightness')
axes[1].plot(summary['n_c'], qcTable['dev_ratio'], 'o', color='0.2', label='all pixels')
axes[1].plot(summary['n_c'], qcTable['dev_ratio_wedge'], 's', color='#2a78d6', label='strahl wedge')
axes[1].axhline(1, color='#d62728', linestyle='--'); axes[1].set_xlabel('fitted core density [cm$^{-3}$]')
axes[1].set_ylabel('R = D / E[D]'); axes[1].set_title('calibrated: 1 = Poisson noise only'); axes[1].legend(fontsize=9)
axes[2].semilogy(qcTable['epoch_index'], np.maximum(qcTable['ks_p'], 1e-300), 'o', color='0.2', label='all pixels')
axes[2].semilogy(qcTable['epoch_index'], np.maximum(qcTable['ks_p_wedge'], 1e-300), 's', color='#2a78d6', label='strahl wedge')
axes[2].axhline(1e-3, color='#d62728', linestyle='--', label='p = 0.001'); axes[2].set_xlabel('epoch'); axes[2].set_ylabel('KS p-value')
axes[2].set_title('KS-PIT'); axes[2].legend(fontsize=9)
for ax in axes: ax.grid(alpha=0.3)
fig.tight_layout()""")
    nb.md("""## 3. From statistics to flags

The rules used on month-long runs at 1 au (January 2022, 30 000 epochs):

| flag | rule | meaning |
|---|---|---|
| hard | detected pixels $\\le 1000$ | too few counts: information-poor, not a misfit |
| strong | KS $p < 10^{-3}$ **and** $R > 1.1$ | large and significant misfit over the whole grid |
| wedge | KS $p_\\text{wedge} < 10^{-3}$ | misfit inside the strahl cone (matters for strahl science only) |
| soft | KS $p < 10^{-3}$ but $R \\le 1.1$ | significant but small; recorded, not excluded |

On that month 0.8 % of the epochs were *strong* and 1.3 % *wedge*. On the 30 example
epochs the strong rule fires on most and the wedge rule on nearly all. The reason is
the count level: at 0.33 au the grid holds 35 000 valid pixels with up to ~15 counts at
the peak, so the tests resolve the limits of the model itself, not fluctuations. A
single truncated bi-kappa cannot reproduce both the shoulder and the tail of the
strahl; the wedge $R$ of 1.3-1.8 says exactly that.

Two consequences:

- Thresholds transfer between data sets only where the model is adequate at that count
  level. Calibrate the flags on the run's own distribution of $R$ (for example median
  plus three robust standard deviations) and treat $R$ as a *ranking* of epochs.
- Keep the wedge statistics separate from the whole-grid ones, and use them only when
  the science is about the strahl.""")
    nb.code("""ratio = qcTable['dev_ratio']
robustSigma = 1.4826 * np.median(np.abs(ratio - ratio.median()))
threshold = ratio.median() + 3 * robustSigma
outliers = qcTable.loc[ratio > threshold, ['epoch_index', 'dev_ratio', 'dev_ratio_wedge', 'ks_p']]
print(f'R: median {ratio.median():.3f}, robust sigma {robustSigma:.3f}, run-relative threshold {threshold:.3f}')
print(f'epochs above the run-relative threshold: {len(outliers)} of {len(qcTable)}')
outliers.round(4)""")
    nb.md("""## Summary

- The randomized PIT + KS test and the calibrated deviance ($R$, $z$) are valid at any
  count level; `deviance_per_dof` is not comparable across brightness.
- Run them on a whole result directory with
  `python -m evdf_poisson.qc --data-dir <dir> --run-dir <results>`.
- Flags need calibrating on each data set: a statistically significant misfit is not
  necessarily a bad fit when the counts are high enough to resolve the model's own
  limits.""")
    return nb


BUILDERS = {'00': ('00_quickstart_poisson.ipynb', build00),
            '01': ('01_eas_instrument_and_data.ipynb', build01),
            '02': ('02_download_and_preprocess.ipynb', build02),
            '03': ('03_pad_strahl_and_break_energy.ipynb', build03),
            '04': ('04_models_and_initial_guesses.ipynb', build04),
            '05': ('05_poisson_fit_step_by_step.ipynb', build05),
            '06': ('06_batch_fit_and_two_days.ipynb', build06),
            '07': ('07_poisson_vs_chi2_low_counts.ipynb', build07),
            '08': ('08_fit_quality_qc.ipynb', build08)}


def main(selected):
    for key in selected or BUILDERS:
        fileName, builder = BUILDERS[key]
        path = builder().write(EXAMPLES / fileName)
        print('wrote', path)


if __name__ == '__main__':
    main(sys.argv[1:])

