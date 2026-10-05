# S3Park 2026 hands-on: fitting Solar Orbiter EAS electron distributions

*Newcastle, Friday 18 September 2026, 20 minutes.* This page is the
take-home guide for the session: how to set up, which notebooks to run in
which order, where the example data are, and how to run the pipeline on an
interval of your own. The tag `s3park-2026` freezes the repository as it was
presented; `main` moves on.

## 1. Set up (5 minutes, before or after the session)

Python 3.10 in a fresh environment:

```bash
git clone --branch s3park-2026 https://github.com/Xiangyu-W/SO-EAS-eVDF_Fitting.git
cd SO-EAS-eVDF_Fitting
python3.10 -m venv .venv && source .venv/bin/activate     # or: conda create -n evdf python=3.10
pip install -r requirements.txt
python examples/run_example_poisson.py                    # 30 epochs, ~1.5 min on 4 cores
jupyter lab                                                # then open examples/
```

The quickstart fits the bundled 30-epoch example and must print
`30/30 epochs fitted`; its `results/fitting/example_run_poisson/summary.csv`
is identical to the one shipped with the repository.

## 2. The notebooks, in order

| # | notebook | what you learn | runs offline | minutes |
|---|---|---|---|---|
| 00 | `00_quickstart_poisson` | the whole pipeline on 30 epochs, one diagnostic figure | yes | 1.5 |
| 01 | `01_eas_instrument_and_data` | the two EAS heads and the spacecraft frame (SRF); counts, PSD and the one-count level; pitch angles; core, halo, strahl; the PAD; spacecraft potential; break energy; why zero counts are data | yes | 1 |
| 02 | `02_download_and_preprocess` | SOAR L2/L1 files, the spacecraft-potential table from the SOAR RPW L3 product, the preprocessing call, what the HDF5 contains | no (pre-executed) | 4 |
| 03 | `03_pad_strahl_and_break_energy` | the two-population crossing break energy; the two-tier strahl verdict from the PAD | yes | 1 |
| 04 | `04_models_and_initial_guesses` | bi-Maxwellian, bi-kappa and truncated bi-kappa; where the seeds come from; the parameter boxes; why the fit is staged | yes | 1 |
| 05 | `05_poisson_fit_step_by_step` | the Poisson deviance; stages A (core), B (halo), C (strahl), D (joint) on one epoch; deviance/dof, calibrated ratio, pull map, pinned parameters | yes | 1 |
| 06 | `06_batch_fit_and_two_days` | the batch runner and its CSV; parameter time series; the two full days 2023-04-04/05 | yes | 1 |
| 07 | `07_poisson_vs_chi2_low_counts` | the same fit with a chi-square objective on real and synthetic data: the censoring bias | yes | 4 |
| 08 | `08_fit_quality_qc` | fit-quality statistics that work at any count level (randomized PIT + KS, calibrated deviance) and how to flag a long run | yes | 1 |

The session shows 00, 02, 03, 05 and 06; 01 and 04 are the reading behind
the slides; 07 is the argument for the objective; 08 is for whoever runs
the pipeline on a long interval.

## 3. Example data

- **In the repository**: `examples/example_data/poisson_2023-04-04/`, 30
  one-minute epochs of 2023-04-04 00:50-01:19 UT (0.33 au, strong
  anti-parallel strahl in 27 of them), with its break energies. See
  `examples/example_data/README.md` for provenance.
- **The two full days** 2023-04-04 and 2023-04-05 (2 849 epochs): the
  reference fit ships in
  `results/fitting/example_days_2023-04-04_05/summary.csv`. The eight
  six-hour preprocessed files (about 600 MB each, 4.8 GB in total) are too
  large to distribute; rebuild them from the SOAR files:

```bash
# main block of the preprocessor: these two days. Set psd_source='soar' there to
# download the raw EAS files first (about 830 MB per six hours).
PYTHONPATH=src python src/preprocess_data_for_fit_hdf5.py
PYTHONPATH=src python src/compute_break_energy_crossing.py --vdf-dir data/processed/2023-04-04_05 \
    --output-dir data/processed/2023-04-04_05
PYTHONPATH=src python -m evdf_poisson.runner --data-dir data/processed/2023-04-04_05 \
    --out results/fitting/my_two_days --workers 8 --plot-every 60
```

About 6 s per epoch and core: one six-hour file is 10 minutes on 4 workers,
a day 40 minutes, the two days 12 minutes on 32 workers. `--resume` (the
default) continues an interrupted run.

## 4. Your own interval, in three commands

```bash
export PYTHONPATH=src
# 1. spacecraft potential (intervals from 2023 on; earlier ones are fetched automatically)
python src/build_scpot_table_from_soar_l3.py --start 2023-04-04 --end 2023-04-05 --download \
    --out examples/example_data/scpot_rpw_l3_20230404_20230405_20s.csv
# 2. preprocess: edit the __main__ block of src/preprocess_data_for_fit_hdf5.py (dates, psd_source='soar'
#    to download, scpot_file, output_dir), then
python src/preprocess_data_for_fit_hdf5.py
# 3. break energies and the fit
python src/compute_break_energy_crossing.py --vdf-dir data/processed/<name> --output-dir data/processed/<name>
python -m evdf_poisson.runner --data-dir data/processed/<name> --out results/fitting/<name> --workers 8
```

Notebook 02 walks through step 2 on one six-hour segment. Practicalities:
keep a run within one SOAR file version and inside MAG coverage; six hours
at 1-minute cadence need ~650 MB of RAM and give a ~600 MB file; the SOAR
downloads are ~3.3 GB per day.

## 5. Things to keep in mind

- `deviance_per_dof` is not a reduced chi-square: its expectation depends on
  the count level, so compare it between epochs of similar brightness or use
  the calibrated ratio of notebook 05.
- The RPW density `n_sc` is the absolute reference; EAS densities carry a
  calibration factor that varies with the instrument state.
- The strahl drift is locked to +/-20 % of its stage-C value in the joint
  stage; `pinned_params` lists it (lower edge) in most epochs, and that is
  expected.
- From 2023 on, the spacecraft potential must be supplied as a table (the
  CDAWeb 10-second product ends on 2023-01-01; SOAR publishes the L3 daily
  files).
- Questions after the session: xiangyu.wu.23@ucl.ac.uk, or open an issue.
