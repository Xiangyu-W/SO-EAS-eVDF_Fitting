# SO-EAS-eVDF_Fitting

[![DOI](https://img.shields.io/badge/DOI-10.3847%2F1538--4357%2Fae3c7b-blue)](https://doi.org/10.3847/1538-4357/ae3c7b)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Python toolkit for fitting the three-component electron velocity distribution function (eVDF) measured by the Solar Wind Analyser -- Electron Analyser System (SWA-EAS) onboard Solar Orbiter.

---

## Overview

Solar wind electrons consist of three major populations: the **core**, **halo**, and **strahl**. Their velocity distribution functions (VDFs) encode information about the solar corona and the transport processes that shape the heliosphere. Accurate decomposition of the measured electron VDF into these three components is essential for deriving physical parameters such as density, temperature, drift velocity, and the kappa index of each population.

This repository provides the fitting pipeline used in:

> **Xiangyu Wu**, Christopher J. Owen, Jesse Coburn, Georgios Nicolaou, Daniel Verscharen, Jingting Liu, Charalambos Ioannou, Hao Ran, Yeimy J. Rivera, and Stephanie L. Yardley (2026). *Correlation between Electron Temperature and Ion Charge-state Ratios in the Solar Wind at ~0.5 au.* The Astrophysical Journal, 1000, 13. [DOI: 10.3847/1538-4357/ae3c7b](https://doi.org/10.3847/1538-4357/ae3c7b)

The code processes Solar Orbiter SWA-EAS 3D electron VDF data and decomposes the resulting distributions into core, halo, and strahl components through a multi-step fitting procedure with adaptive initial conditions and iterative refinement.

---

## Quickstart

Requires Python 3.10. In a fresh virtual environment (conda or venv):

```bash
git clone https://github.com/Xiangyu-W/SO-EAS-eVDF_Fitting.git
cd SO-EAS-eVDF_Fitting
pip install -r requirements.txt

python examples/run_example.py
```

This fits the bundled example dataset (11 one-minute epochs of SWA-EAS data from 2022-03-02 12:15--12:25 UT, all with a clear strahl; see [examples/example_data/README.md](examples/example_data/README.md)) with 4 parallel workers and writes to `results/fitting/example_run/`:

- `plots/` -- per-epoch figures: 1D parallel/perpendicular VDF cuts with the fitted components, and 2D VDF contours of data vs. model
- `data/chunks/FitResult_*.pkl` -- full fit results (lmfit parameters per epoch)
- `data/summary.csv` -- per-epoch summary: fit status, reduced chi-square, fitted parameters

All 11 epochs should fit successfully with reduced chi-square around 0.8--1.1. The reference outputs of this run ship with the repository in the same location for comparison (your run regenerates the large `FitResult_*.pkl`, overwrites the bundled plots, and appends to the bundled `summary.csv`; the fitted values are identical). Fits are deterministic: each epoch's random-restart optimizer is seeded from the epoch timestamp, so reruns reproduce identical parameters regardless of the worker count. Edit `N_JOBS` in the example script to match your CPU. See [docs/TUTORIAL.md](docs/TUTORIAL.md) for the full pipeline, including preprocessing your own time intervals from raw Solar Orbiter data.

---

## Method Summary

The fitting pipeline follows a three-phase approach, employing globally optimized fitting to robustly decompose the electron VDF into its constituent populations. In Phase 2, each component is fitted individually using the **AMPGO** (Adaptive Memory Programming for Global Optimization; [Lasdon et al. 2010](https://doi.org/10.1016/j.cor.2009.11.006)) algorithm with L-BFGS-B as the local solver. AMPGO is well-suited for the individual component fitting because the non-linear parameter space of each distribution model contains multiple local minima that local optimizers alone may fail to escape. The globally optimized results from Phase 2 then serve as well-constrained initial values for the final combined fit in Phase 3.

### Phase 1 -- Preprocessing and Strahl Identification

- **Pitch-Angle Distribution (PAD) fitting:** The electron VDF is reorganized into pitch-angle bins using the local magnetic field direction from MAG. The PAD is averaged over a selected energy range and fitted with a double-Gaussian model ([Owen et al. 2022](https://doi.org/10.3390/universe8100509)) to determine the strahl direction and angular width.
- **Break energy detection:** Pixels within a cone perpendicular to the magnetic field (90 deg +/- 20 deg pitch angle) are ordered by perpendicular velocity, and an ordered binary split on smoothed log-PSD features (value, first and second derivative) identifies the energy boundary between the thermal core and suprathermal populations (method lineage: [Bakrania et al. 2020](https://doi.org/10.1051/0004-6361/202037840); [Abraham et al. 2022](https://doi.org/10.3847/1538-4357/ac6605)).
- **Spacecraft potential correction:** Electron energies are corrected by subtracting the floating spacecraft potential measured by the RPW instrument.

### Phase 2 -- Sequential Component Fitting

Each component is fitted independently to provide robust initial estimates:

1. **Core** (bi-Maxwellian): Fitted below the break energy at all pitch angles. Four free parameters: density, parallel drift velocity, parallel and perpendicular thermal velocities.
2. **Halo** (modified bi-kappa with flat-top suppression, [Stverak et al. 2009](https://doi.org/10.1029/2008JA013883)): Fitted above the break energy outside the strahl cone. Six free parameters: density, parallel drift velocity, parallel and perpendicular thermal velocities, kappa index, and flat-top width.
3. **Strahl** (truncated bi-kappa, [Stverak et al. 2009](https://doi.org/10.1029/2008JA013883)): Fitted within the strahl cone above the break energy. Five free parameters: density, parallel drift velocity, parallel and perpendicular thermal velocities, and kappa index. A truncation parameter suppresses the low-energy portion where the core dominates.

Each component is fitted using the **AMPGO global optimizer** with adaptive iterative parameter refinement to ensure convergence across the non-convex parameter landscape of each distribution model.

### Phase 3 -- Final Combined Fit

All parameters from Phase 2 are used as initial values for a simultaneous fit of the full model (core + halo + strahl) to the measured VDF. The goodness of fit is assessed using the reduced chi-square statistic for the overall fit and for each individual component.

---

## Code Structure

```
.
├── examples/
│   ├── run_example.py                        # Quickstart: fit the bundled example dataset
│   └── example_data/                         # 11-epoch sample input (HDF5 + break energy) and EAS calibration data
├── results/fitting/example_run/              # Reference outputs of run_example.py (plots, summary)
├── docs/
│   └── TUTORIAL.md                           # Usage guide for the full pipeline
└── src/
    ├── preprocess_data_for_fit_hdf5.py       # Step 1: raw CDF -> fitting-ready HDF5 (+ break energy)
    ├── compute_break_energy_cone_features.py # Step 1b: standalone break energy detection (CLI)
    ├── main_fitting.py                       # Step 2: orchestrates the three-component fitting
    ├── pad_fitting.py                        # Pitch-angle distribution fitting
    ├── Core_fitting.py                       # Core component fitting (bi-Maxwellian)
    ├── Halo_fitting.py                       # Halo component fitting (modified bi-kappa)
    ├── Beam_fitting.py                       # Strahl/beam component fitting (truncated bi-kappa)
    ├── Overall_fitting.py                    # Final combined fit of all components
    ├── fit_functions.py                      # Analytical VDF model definitions
    ├── Analysis_functions.py                 # Post-fit analysis (chi-square, moments, temperatures)
    ├── Plotting.py                           # Visualization (1D cuts, 2D VDF contour plots)
    ├── utils.py                              # Utility functions (unit conversion, energy/velocity tools)
    ├── data_io.py                            # Data I/O and unit conversion
    ├── constant.py                           # Physical constants and fitting parameter bounds
    ├── download_EAS_Data.py                  # Solar Orbiter EAS data download utilities
    └── cda_download.py                       # CDAWeb data access for ancillary measurements
```

---

## Data

This project uses in situ measurements from Solar Orbiter, available from the [Solar Orbiter Archive (SOAR)](https://soar.esac.esa.int/soar/):

- **SWA-EAS**: 3D electron VDF (Level-1 Counts and Level-2 PSD, normal mode)
- **SWA-PAS**: Solar wind proton bulk velocity
- **MAG**: Vector magnetic field (for VDF rotation to field-aligned frame)
- **RPW**: Floating spacecraft potential (for electron energy correction)

MAG, PAS, and RPW data are fetched automatically through [cdasws](https://pypi.org/project/cdasws/) during preprocessing. Geometric factors and quantum efficiencies for the EAS instrument (used during preprocessing to convert counts to PSD and to compute the VDF uncertainty) ship with this repository under `examples/example_data/`. See [docs/TUTORIAL.md](docs/TUTORIAL.md) for how to obtain and preprocess the EAS CDF files for your own time intervals.

---

## Dependencies

Listed in [requirements.txt](requirements.txt) (Python 3.10): NumPy, SciPy, pandas, h5py, Matplotlib, [lmfit](https://lmfit.github.io/lmfit-py/), joblib, tqdm, num2tex, Astropy, xarray, cdasws, requests, BeautifulSoup4, sunpy, and sunpy-soar.

Install them into a fresh environment:

```bash
# conda
conda create -n evdf python=3.10
conda activate evdf
pip install -r requirements.txt
```

```bash
# or venv + pip
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Citation

If you use this code or method in your research, please cite:

```bibtex
@article{Wu2026,
    author  = {Wu, Xiangyu and Owen, Christopher J. and Coburn, Jesse and Nicolaou, Georgios and Verscharen, Daniel and Liu, Jingting and Ioannou, Charalambos and Ran, Hao and Rivera, Yeimy J. and Yardley, Stephanie L.},
    title   = {Correlation between Electron Temperature and Ion Charge-state Ratios in the Solar Wind at ~0.5 au},
    journal = {The Astrophysical Journal},
    volume  = {1000},
    number  = {1},
    pages   = {13},
    year    = {2026},
    doi     = {10.3847/1538-4357/ae3c7b}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
