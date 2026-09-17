# S3Park 2026 workshop release: two days of preprocessed SWA-EAS data

Preprocessed Solar Orbiter SWA-EAS electron VDF data for **2023-04-04 and
2023-04-05** (0.33 au), fitting-ready for the Poisson pipeline of this
repository at tag `s3park-2026`.

Assets:

- `VDF_eas_forFitting_<start>_<end>_1min.h5` (8 files, six hours each,
  ~600 MB): 64 energies x 1024 pixels per one-minute epoch with pitch angle,
  PSD, v_par, v_perp and counts; the pitch-angle distribution; MAG (SRF), PAS,
  RPW density; the spacecraft potential; the census of masked L1 saturation
  blocks. Produced by `src/preprocess_data_for_fit_hdf5.py` from SOAR L2 PSD
  V03 and L1 counts V03, with the spacecraft potential from the SOAR RPW L3
  `rpw-bia-scpot` daily files (20 s medians, QUALITY_FLAG 3).
- `VDF_eas_forFitting_<start>_<end>_1min_two_population_crossing.csv` (8 files):
  break energies of every epoch, `src/compute_break_energy_crossing.py`.
- `reference_fit_2023-04-04_05.tar.gz`: the reference Poisson fit of all
  2 850 epochs (`summary.csv`, per-file CSVs, the run log, and the four-panel
  figure of every tenth epoch), produced with
  `python -m evdf_poisson.runner --workers 32 --plot-every 10`.
- `SHA256SUMS`: checksums of everything above.

Download with `python workshop/download_example_days.py` (verifies the
checksums, selects days or single files) and fit with
`PYTHONPATH=src python -m evdf_poisson.runner --data-dir <dir> --out <dir> --workers N`.
