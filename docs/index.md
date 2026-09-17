# SO-EAS eVDF Fitting: start here

This repository decomposes Solar Orbiter SWA-EAS electron velocity
distribution functions into **core**, **halo** and **strahl** components. Two
pipelines are included:

| pipeline | package | objective | use it for |
|---|---|---|---|
| **Poisson** (recommended) | `src/evdf_poisson/` | Poisson deviance on all pixels, zero counts included | new analyses; halo and strahl parameters at low counts |
| legacy chi-square | `src/main_fitting.py` and the `*_fitting.py` modules | unweighted least squares on pixels with counts $\ge$ 2, AMPGO global optimiser | reproducing Wu et al. (2026) |

## If you have ten minutes

```
pip install -r requirements.txt
python examples/run_example_poisson.py
```

fits the bundled 30-epoch example (2023-04-04 00:50-01:19 UT) and writes
`results/fitting/example_run_poisson/`.

## If you want to understand what it does

Open the notebooks in `examples/` in order. Every notebook except 02 runs on
the bundled example data without a network connection, and all ship with
their outputs.

| notebook | what it teaches | minutes |
|---|---|---|
| `00_quickstart_poisson` | run everything, look at a result | 1.5 |
| `01_eas_instrument_and_data` | the two heads and the spacecraft frame, counts and PSD, pitch angles, the three populations, the PAD, spacecraft potential, break energy, why zero counts are data | 1 |
| `02_download_and_preprocess` | SOAR files and the spacecraft-potential table to the fitting-ready HDF5 (needs data and network; pre-executed) | demo |
| `03_pad_strahl_and_break_energy` | the two quantities determined before the fit | 1 |
| `04_models_and_initial_guesses` | the three analytic components, the seeds, the parameter boxes | 1 |
| `05_poisson_fit_step_by_step` | the objective, the four stages on one epoch, how to read the goodness of fit | 1 |
| `06_batch_fit_and_two_days` | fitting whole files, reading the CSV, parameter time series, two full days | 1 |
| `07_poisson_vs_chi2_low_counts` | why Poisson: real data and synthetic truth at decreasing count level | 4 |
| `08_fit_quality_qc` | randomized PIT + KS test and the calibrated deviance; flags for long runs | 1 |

## Reference pages

- [The Poisson fitting pipeline](poisson_pipeline.md): model, objective,
  every seed and allowed range, output flags, design notes.
- [Legacy chi-square pipeline tutorial](TUTORIAL.md): the published pipeline,
  break energy with the cone-split method. Its original README is kept as
  [README_chi2_legacy.md](../README_chi2_legacy.md).
- [S3Park 2026 hands-on session](../workshop/README.md): the 20-minute
  session, the two-day example data release, how to run your own interval.

## Your own data

1. Preprocess the interval (`src/preprocess_data_for_fit_hdf5.py`, notebook
   02) into `VDF_eas_forFitting_*.h5` files. From 2023 on, build the
   spacecraft-potential table first (`src/build_scpot_table_from_soar_l3.py`)
   and pass it as `scpot_file`.
2. Break energies: `python src/compute_break_energy_crossing.py --vdf-dir <dir> --output-dir <dir>`.
3. Fit: `PYTHONPATH=src python -m evdf_poisson.runner --data-dir <dir> --out <results dir> --workers 8`.
