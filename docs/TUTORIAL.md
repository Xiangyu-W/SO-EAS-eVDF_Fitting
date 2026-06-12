# Tutorial — SO-EAS eVDF Fitting Pipeline

This guide covers how to run the pipeline. For the scientific background and
the fitting method, see the [README](../README.md).

## Pipeline overview

```
Raw data (SOAR CDF)
  │  EAS L2 PSD + L1 Counts (local files or auto-download)
  │  MAG / PAS / RPW SCPOT (fetched automatically via cdasws)
  ▼
┌──────────────────────────────────────────────────────┐
│ Step 1  preprocess_data_for_fit_hdf5.py              │
│   preprocessing → VDF_eas_forFitting_*.h5            │
│   └─ chained → break energy detection (cone split)  │
└──────────────────────────────────────────────────────┘
  │  data/processed/<run>/*.h5
  │  results/break_energy/<run>/*_cone_feature_split.pkl/.csv
  ▼
┌──────────────────────────────────────────────────────┐
│ Step 2  main_fitting.py                              │
│   core / halo / strahl three-component fit (AMPGO)   │
└──────────────────────────────────────────────────────┘
  │  results/fitting/<run_name>/
  │    plots/, data/chunks/, summary.csv, run_config.json
  ▼
Your analysis
```

Step 1 runs **preprocessing and break energy detection in one command**; break
energy can also be recomputed separately with
`compute_break_energy_cone_features.py` (Step 1b).

If you just want to try the fitting, you can skip Step 1 entirely: the
repository ships a preprocessed example dataset (see Quickstart below).

---

## Quickstart: fitting the example data

```bash
pip install -r requirements.txt
python examples/run_example.py
```

This runs `main_fitting.main2()` with its defaults, which point at the
11-epoch sample in `examples/example_data/`. Useful knobs (pass them to `main2()` or
edit `examples/run_example.py`):

| Argument | Default | Meaning |
|---|---|---|
| `n_jobs` | 4 | Parallel workers (joblib). Set to your physical core count. |
| `run_name` | `'example_run'` | Output goes to `results/fitting/<run_name>/`. |
| `fit_beg`, `fit_end` | `None`, `None` | Epoch index range to fit; `None` = the whole file. |
| `cond_final_plot` | `True` | Save the per-epoch diagnostic figures (turn off for speed). |
| `overwrite_existing_chunks` | `False` | `False`: skip files whose result chunk already exists (resume). |
| `pickle_dir`, `pickle_files` | `examples/example_data/` | Preprocessed input directory and file list. |
| `break_E_dir` | `examples/example_data/` | Directory holding the `*_cone_feature_split.pkl` files. |

**Resume behavior:** rerunning with the same `run_name` skips any input file
whose chunk file already exists. To force a refit, change `run_name` or set
`overwrite_existing_chunks=True`.

**Reproducibility:** the AMPGO global optimizer uses random restarts, so each
epoch's fit is seeded deterministically from the epoch timestamp. Rerunning
the same data returns bit-identical parameters, independent of `n_jobs` or of
whether the epoch sits in a file subset. To reproduce the published results,
install the pinned versions from `requirements.txt` in a fresh environment.

A console log of every run is mirrored to `logs/fitting_<run_name>.log`.

---

## Interpreting the outputs

```
results/fitting/<run_name>/
  run_config.json                 # full configuration of this run
  plots/<time_segment>/           # per-epoch figures (2 PNGs per epoch)
  data/chunks/FitResult_<time_segment>[_epochXXXX-YYYY].pkl
  data/summary.csv                # one row per epoch (including failed ones)
```

### Figures

- `Final_fit_ID<i>_(<timestamp>).png` — 1D cuts of the VDF parallel and
  perpendicular to the magnetic field, with the fitted core/halo/strahl
  curves, plus the pitch-angle distribution diagnostics.
- `FittedVDF_2D_ID<i>_(<timestamp>).png` — 2D gyrotropic VDF
  (v_parallel, v_perpendicular): measured data vs. the fitted model, with the
  fitted strahl contours overlaid.

### summary.csv

One row per epoch. Key columns: `status` (`success`/`failed`), `epoch_id`,
`time_stamp`, `redChiSqr` / `redChiSqr_beam` / `redChiSqr_halo`,
strahl temperatures `T_para_b` / `T_antiPara_b` (eV), the per-component energy
ranges (`component_energy_json`), and all fitted parameter values
(`final_n_c`, `final_v_th_par_c`, ..., `final_kappa_h`) with standard errors.

### Result chunks (pickle)

Each input file produces one `FitResult_*.pkl` holding a dict with, per epoch:
`r_vdf` (the final lmfit result — inspect with
`r_vdf.params.pretty_print()`), `fit_vdf` (model evaluated on the velocity
grid), `initial_core/halo/beamPar/beamAntiPar` (Phase 2 results),
`redChiSqr*`, `T_para_b`/`T_antiPara_b`, `component_energy`, and
`failed_epoch_id`.

```bash
python -c "
import pickle
r = pickle.load(open('results/fitting/example_run/data/chunks/FitResult_20220302121509_20220302122509.pkl','rb'))
r['r_vdf'][0].params.pretty_print()"
```

Fitted parameter naming: `*_c` core, `*_h` halo, `*_b_par` / `*_b_anti_par`
strahl parallel/anti-parallel to the magnetic field (`n` density, `u_par`
drift, `v_th_par`/`v_th_perp` thermal velocities in cm/s, `kappa` index).

---

## Full pipeline: processing your own time interval

### Step 0 — Get the raw data

You need, for each interval, four CDF files from
[SOAR](https://soar.esac.esa.int/soar/):

- `solo_L2_swa-eas1-nm3d-psd_*.cdf` and `solo_L2_swa-eas2-nm3d-psd_*.cdf`
  (L2 electron phase space density, normal mode)
- `solo_L1_swa-eas1-NM3D_*.cdf` and `solo_L1_swa-eas2-NM3D_*.cdf`
  (L1 counts, used for uncertainty estimates)

Place them under `data/vdfs/` (L2) and `data/L1_Count/` (L1). Helpers:
`src/download_EAS_Data.py` downloads L2 PSD files for a time range; if you
leave the count file arguments as `None`, preprocessing auto-downloads the L1
files from SOAR via sunpy-soar. MAG, PAS, and RPW data are always fetched
automatically through cdasws — no manual download needed.

**Calibration files:** count-to-PSD conversion and the VDF uncertainty
need the EAS geometric factors and quantum efficiencies (four files:
`EAS1/2_AGFs_*.txt` and `EAS1/2_Flight_QuantumEfficiencies_*.line`). They
ship with this repository in `examples/example_data/` and are read from
there automatically.

### Step 1 — Preprocess + break energy (one command)

Edit the `__main__` block at the bottom of
`src/preprocess_data_for_fit_hdf5.py` (it contains a documented one-segment
template), then:

```bash
python src/preprocess_data_for_fit_hdf5.py
```

Or call it from your own script/notebook:

```python
from preprocess_data_for_fit_hdf5 import preprocess_main

processed_file_paths = preprocess_main(
    eas1_files=eas1Files,               # list of EAS1 L2 PSD CDF paths
    eas2_files=eas2Files,               # list of EAS2 L2 PSD CDF paths
    count_eas1_files=countEAS1Files,    # L1 count CDFs (None = auto-download)
    count_eas2_files=countEAS2Files,
    output_dir='data/processed/my_run/',
    shift_SCPOT=True,                   # correct electron energies with RPW SCPOT (recommended)
    time_resolution='1min',
    # ---- break energy hook (on by default) ----
    compute_break_energy=True,          # False = preprocessing only
    break_energy_output_dir=None,       # default: results/break_energy/<output_dir name>/
    cone_half_width_deg=None,           # default 20 deg (pixels within 90±20 deg)
    break_energy_workers=None,          # default min(32, CPU count)
)
```

Behavior:

- Data are chunked into ~6-hour segments; each produces one
  `VDF_eas_forFitting_<start>_<end>_<timeRes>.h5` (HDF5, gzip-compressed).
- After preprocessing, break energy is computed for every produced `.h5`,
  plus a combined summary (pkl + csv + overview figure).
- Break energy output defaults to `results/break_energy/<run name>/`,
  mirroring the preprocessing output directory name.

### Step 1b — Recompute break energy separately (optional)

`src/compute_break_energy_cone_features.py` is a standalone CLI; it accepts
both `.h5` and `.pkl` inputs.

```bash
# everything in the default input dir (data/processed/)
python src/compute_break_energy_cone_features.py

# specific directory and files (glob supported)
python src/compute_break_energy_cone_features.py \
    --pickle-dir data/processed/my_run \
    --pickle-files 'VDF_eas_forFitting_2022030*.h5' \
    --output-dir results/break_energy/my_run

# quick tests: first 20 epochs per file / specific epochs
python src/compute_break_energy_cone_features.py --max-epochs 20
python src/compute_break_energy_cone_features.py --epoch-indices 30,97,180,300

# re-combine existing per-file results without recomputing
python src/compute_break_energy_cone_features.py --combine-existing \
    --output-dir results/break_energy/my_run
```

| Option | Default | Meaning |
|---|---|---|
| `--pickle-dir` | `data/processed` | input directory |
| `--pickle-files` | all `VDF_eas_forFitting_*.pkl/.h5` | file names/paths/globs |
| `--cone-half-width-deg` | 20 | cone half-width around 90 deg pitch angle |
| `--features` | `v_perp,log_psd,deriv,second_deriv` | features used by the ordered split |
| `--workers` | min(32, CPU) | parallel workers per file |
| `--max-epochs` / `--epoch-indices` | — | for testing; mutually exclusive |
| `--output-dir` | `results/break_energy` | output directory |
| `--combined-output-name` | `combined_cone_feature_split` | combined output file name |
| `--skip-combine` / `--combine-existing` | — | skip combining / only combine existing results |

Result columns (`*_cone_feature_split.pkl/.csv`):

| Column | Meaning |
|---|---|
| `epoch_id` | epoch index within the file (starts at 0 per file) |
| `combined_epoch_id` | combined output only: global index after sorting by time |
| `timestamp` | epoch time |
| `break_v` / `break_E_eV` | break velocity (cm/s) / energy (eV); NaN if not detected |
| `n_points` / `PA_bins` | pixels used / pitch-angle bins involved |
| `status` | `ok` / `all_nan` / `no_cone_pixels` / `too_few_points` / `no_transition` / `error` |
| `features` / `cone_half_width_deg` / `input_file` | provenance of the run configuration |

Method: pixels within 90 deg +/- cone half-width, energies from 12 eV
(SCPOT-corrected) to 1000 eV with count >= 1, are ordered by perpendicular
velocity; log(PSD) is Savitzky-Golay filtered to obtain first and second
derivatives, and a single ordered split minimizing the two-segment variance of
the standardized features marks the core/suprathermal boundary.

### Step 2 — Fit your own files

Call `main2()` with your paths (or edit the defaults in
`src/main_fitting.py`):

```python
import sys; sys.path.insert(0, 'src')
from main_fitting import main2

main2(
    pickle_dir='data/processed/my_run/',
    pickle_files=['VDF_eas_forFitting_20220302060509_20220302120509_1min.h5'],
    break_E_dir='results/break_energy/my_run/',
    run_name='my_fit_run',
    n_jobs=16,
)
```

The break energy file name is derived automatically as
`<input file stem>_cone_feature_split.pkl` — you never list it manually, but
Step 1/1b must have been run on the same input files.

Per epoch the pipeline runs: PAD fit (strahl direction/width) → energy-range
partition at the break energy → core (bi-Maxwellian) → halo (modified
bi-kappa) → strahl (truncated bi-kappa) → final combined fit, all using AMPGO
global optimization. The break energy is taken from the cone-split results
after filtering to the 35–140 eV window and Savitzky-Golay smoothing
(`break_E_range` in `src/constant.py`).

---

## Tuning knobs (`src/constant.py`)

| Constant | Default | Meaning |
|---|---|---|
| `break_E_range` | `[35, 140]` eV | accepted break energy window; epochs outside use the smoothed neighbor value |
| `low_pad_val`, `high_pad_val` | 61, 1000 eV | energy range averaged for the PAD strahl detection |
| `beam_ratio_cond` | 1.2 | PAD peak/background ratio above which a strahl is declared |
| `deficit_ratio_cond` | 0.5 | threshold for detecting a sunward deficit |
| `low_c_val` | 14 eV | low-energy cutoff of the core fit (shifted by SCPOT at runtime) |
| `CORE_PARAMS`, `HALO_PARAMS`, ... | — | initial values and bounds of the fit parameters |

---

## Troubleshooting

- **Failed epochs** are recorded in `summary.csv` with `status='failed'` and
  in the chunk's `failed_epoch_id` — the run continues past them.
- **All break energies filtered out**: if no epoch of a file has
  `break_E_eV` inside `break_E_range`, the fitting cannot proceed for that
  file. Check the `status` column of the cone-split output and the overview
  PNG in `results/break_energy/<run>/`.
- **Memory**: a full 6-hour, 1-min resolution input file is ~650 MB on disk
  and is loaded fully per file; with many parallel workers, make sure the
  machine has a few GB of RAM headroom.
- **Wrong/missing file names**: input files must keep the
  `VDF_eas_forFitting_<14-digit start>_<14-digit end>_<res>.h5` pattern — the
  fitting derives output folder names and the break-energy file name from it.
