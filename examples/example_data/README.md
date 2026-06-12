# Example data

A small, fitting-ready sample so the pipeline can be tried without
downloading anything:

| File | Content |
|---|---|
| `VDF_eas_forFitting_20220302121509_20220302122509_1min.h5` | 11 one-minute epochs of preprocessed SWA-EAS electron VDF data (~20 MB) |
| `VDF_eas_forFitting_..._cone_feature_split.pkl` | break energy results (input to the fitting) |
| `VDF_eas_forFitting_..._cone_feature_split.csv` | same content as the pkl, human-readable |
| `energy_deltas.npz` | EAS energy bin widths, used by the fitting for the strahl moment calculation |
| `EAS1/2_AGFs_*.txt`, `EAS1/2_Flight_QuantumEfficiencies_*.line` | EAS geometric factors and quantum efficiencies, used by the preprocessing to convert counts to PSD and compute the VDF uncertainty |

**Provenance.** Solar Orbiter SWA-EAS (EAS1+EAS2) normal-mode 3D
measurements, 2022-03-02 12:15:09–12:25:09 UT, at ~0.55 au — an interval
with a clear anti-parallel strahl in every epoch. Produced by
`src/preprocess_data_for_fit_hdf5.py` from the SOAR L2 PSD and L1 count CDFs
with spacecraft-potential correction enabled (`shift_SCPOT=True`, RPW), MAG
field-aligned rotation, 1-minute resampling; break energies from the
cone-feature ordered split (20 deg cone half-width). The HDF5 file contains
epochs 10–20 cut from the full 6-hour segment (12:05–18:05 UT), with the
exact structure documented in [docs/TUTORIAL.md](../../docs/TUTORIAL.md).

The break-energy files keep the **full 6-hour series** (361 rows, all
`status='ok'`): the fitting smooths the break-energy series over all rows
before the nearest-timestamp lookup, so shipping the complete series makes
the example fits bit-identical to a run on the full segment. The fits are
also deterministic — each epoch's optimizer is seeded from the epoch
timestamp.

Run `python examples/run_example.py` from the repository root to fit it.
