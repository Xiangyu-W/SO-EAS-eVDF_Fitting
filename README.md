# SO-EAS-eVDF_Fitting

[![DOI](https://img.shields.io/badge/DOI-10.3847%2F1538--4357%2Fae3c7b-blue)](https://doi.org/10.3847/1538-4357/ae3c7b)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Fit the **core**, **halo** and **strahl** of the electron velocity distribution functions (eVDFs) measured by the Solar Wind Analyser -- Electron Analyser System (SWA-EAS) on Solar Orbiter, and get the density, temperature, drift and kappa index of each population.

---

<!-- ## Two fitting methods

| method | objective | status | where |
|---|---|---|---|
| **Poisson fit** | Poisson likelihood on every pixel, zero counts included | **current, recommended** | this README, `src/evdf_poisson/` |
| **Reduced chi-square fit** | least squares on pixels with counts >= 2, AMPGO global optimiser | previous version, kept unchanged; used in [Wu et al. (2026)](https://doi.org/10.3847/1538-4357/ae3c7b) | [README_chi2_legacy.md](README_chi2_legacy.md), `src/main_fitting.py` |

Both share the same preprocessing and the same three-component model.

**Why Poisson.** Above ~100 eV most EAS pixels register zero counts. A chi-square fit on the detected pixels sees only the upward fluctuations of a faint tail. On the example data it returns a halo ten times denser and three times hotter than the Poisson fit of the same epochs. On synthetic data with a known truth, the Poisson fit stays unbiased down to a tenth of the observed count level ([notebook 07](examples/07_poisson_vs_chi2_low_counts.ipynb)). -->

> **S3Park 2026 hands-on session (18 September 2026):** start at [workshop/README.md](workshop/README.md). The tag `s3park-2026` is the version presented.

---

## Quickstart

Python 3.10, in a fresh conda or venv environment:

```bash
git clone https://github.com/Xiangyu-W/SO-EAS-eVDF_Fitting.git
cd SO-EAS-eVDF_Fitting
pip install -r requirements.txt

python examples/run_example_poisson.py
```

This fits the bundled example (30 one-minute epochs, 2023-04-04 00:50--01:19 UT at 0.33 au, 27 of them with a strong anti-parallel strahl; see [examples/example_data/README.md](examples/example_data/README.md)) on 4 workers in about 1.5 minutes. It writes to `results/fitting/example_run_poisson/`:

- `summary.csv`: one row per epoch with the parameters of core, halo and strahl, the strahl verdict and the fit-quality metrics;
- `per_file/*.csv`: the same, one file per input file (the resume unit);
- `plots/<file>/epochNNNN.png`: a four-panel figure per epoch (parallel and perpendicular cuts with the fitted components, 2D distribution with model contours, Poisson pull per pixel).

The reference outputs ship at the same location. The fit is deterministic: a rerun with the pinned package versions reproduces them exactly.

---

## Tutorial notebooks

Open the notebooks in `examples/` in order (`jupyter lab` from the repository root). They run offline on the bundled data, except 02, and all ship with their outputs.

| notebook | content | runtime |
|---|---|---|
| `00_quickstart_poisson` | run the whole pipeline on 30 epochs, look at one result | 1.5 min |
| `01_eas_instrument_and_data` | the instrument, counts and phase space density, pitch angles, the three populations, why zero counts are data | 1 min |
| `02_download_and_preprocess` | from SOAR files to the fitting-ready HDF5 file (needs raw data and a network; pre-executed) | demo |
| `03_pad_strahl_and_break_energy` | the two inputs measured before the fit: break energy and strahl verdict | 1 min |
| `04_models_and_initial_guesses` | the three models, their initial guesses and allowed ranges | 1 min |
| `05_poisson_fit_step_by_step` | the Poisson objective and the four fit stages on one epoch; reading the goodness of fit | 1 min |
| `06_batch_fit_and_two_days` | fitting whole files, every output column, time series, two full days | 1 min |
| `07_poisson_vs_chi2_low_counts` | Poisson versus chi-square on real data and on synthetic data with a known truth | 4 min |
| `08_fit_quality_qc` | fit-quality statistics that work at any count level, and how to flag a long run | 1 min |

Reference: [docs/poisson_pipeline.md](docs/poisson_pipeline.md) lists every initial guess, allowed range and threshold. [docs/index.md](docs/index.md) is the documentation entry page.

---

## How the Poisson fit works

Per epoch (one minute of data, 64 energies x 1024 pixels):

1. **Inputs.** The pixel grid in the field-aligned frame, the one-count phase space density of every pixel (the exact counts <-> PSD conversion), the spacecraft potential and the electron density from RPW.
2. **Break energy.** A Maxwellian core and a power-law tail are fitted jointly to the spectrum perpendicular to **B**, zeros included. The break is where the two cross.
3. **Strahl verdict.** The pitch-angle distribution above the break is fitted with a background plus two Gaussians ([Owen et al. 2022](https://doi.org/10.3390/universe8100509)). The contrast of each side gives a *strong*, *weak* or *no-strahl* verdict and the beam width.
4. **Staged fit**, Poisson deviance in every stage:
   - **A** core (bi-Maxwellian) on the pixels below the break;
   - **B** halo (bi-kappa with a small free drift) above the break, outside the strahl cones;
   - **C** strahl (drifting, truncated bi-kappa) inside the cone on the detected side;
   - **D** all 14 parameters together, inside ranges anchored to the previous stages.
5. **Outputs.** Physical parameters, deviance per degree of freedom, the classical reduced chi-square for comparison, the parameters that ended on the edge of their range, and the diagnostic figure.

The stages and ranges exist because the likelihood is flat along several directions (halo density versus width, large kappa); the ranges pick the physical corner. The reasons behind each choice are in [docs/poisson_pipeline.md](docs/poisson_pipeline.md).

---

## Your own interval

Three commands from the repository root, with `PYTHONPATH=src` (details in [notebook 02](examples/02_download_and_preprocess.ipynb)):

```bash
# 1. preprocess: SOAR L2 PSD + L1 counts -> VDF_eas_forFitting_*.h5 (six hours each)
python src/preprocess_data_for_fit_hdf5.py        # after editing its __main__ block
# 2. break energies
python src/compute_break_energy_crossing.py --vdf-dir <dir> --output-dir <dir>
# 3. fit
python -m evdf_poisson.runner --data-dir <dir> --out results/fitting/<name> --workers 8
```

From 2023 on, build the spacecraft-potential table first with `src/build_scpot_table_from_soar_l3.py` and pass it as `scpot_file`. Fitting takes about 6 s per epoch and core: one day is about 40 minutes on 4 workers, 5 minutes on 32.

---

## Repository layout

```
.
├── README.md                     # this file (Poisson fit)
├── README_chi2_legacy.md         # README of the previous reduced chi-square version
├── examples/
│   ├── 00 ... 08 *.ipynb         # tutorial notebooks
│   ├── run_example_poisson.py    # quickstart, Poisson fit
│   ├── run_example.py            # quickstart, previous chi-square fit
│   └── example_data/             # example epochs, break energies, EAS calibration and geometry, potential table
├── src/
│   ├── evdf_poisson/             # the Poisson pipeline (CLI: python -m evdf_poisson.runner)
│   ├── preprocess_data_for_fit_hdf5.py, download_L2_PSD_from_SOAR.py,
│   │   build_scpot_table_from_soar_l3.py, eas_geometry.py      # shared preprocessing
│   ├── compute_break_energy_crossing.py, break_energy_methods.py
│   └── main_fitting.py, *_fitting.py, ...                      # previous chi-square pipeline
├── results/fitting/              # reference outputs: example_run_poisson, example_days_2023-04-04_05, example_run
├── docs/                         # index.md, poisson_pipeline.md, TUTORIAL.md (chi-square pipeline guide)
├── tools/                        # notebook sources (build_notebooks.py) and example-subset tools
└── workshop/                     # S3Park 2026 session material and the data download script
```

---

## Data

In situ measurements from Solar Orbiter, from the [Solar Orbiter Archive (SOAR)](https://soar.esac.esa.int/soar/):

- **SWA-EAS**: 3D electron VDF, Level-1 counts and Level-2 PSD, normal mode (`src/download_L2_PSD_from_SOAR.py`);
- **MAG**: magnetic field, for the field-aligned frame;
- **SWA-PAS**: proton bulk velocity and density;
- **RPW**: spacecraft potential and electron density.

MAG, PAS and the RPW density are fetched through [cdasws](https://pypi.org/project/cdasws/) during preprocessing. So is the RPW potential before 2023; from 2023 on it comes from the SOAR RPW L3 daily files. The EAS geometric factors and efficiencies ship under `examples/example_data/`.

The preprocessed files of the two full example days (2023-04-04 and 2023-04-05, 4.8 GB) are too large to distribute; their reference fit ships in `results/fitting/example_days_2023-04-04_05/summary.csv`. To rebuild the files, see [workshop/README.md](workshop/README.md).

---

## Dependencies

Listed in [requirements.txt](requirements.txt) (Python 3.10). The fit itself needs NumPy, SciPy, pandas, h5py, Matplotlib and [lmfit](https://lmfit.github.io/lmfit-py/); the rest serves the preprocessing and the SOAR downloads.

```bash
conda create -n evdf python=3.10 && conda activate evdf    # or: python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Citation

If you use this code, please cite the paper below and state the repository version (tag or commit) you used. The paper describes the three-component model and the reduced chi-square version of the fit.

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

Released under the [MIT License](LICENSE).
