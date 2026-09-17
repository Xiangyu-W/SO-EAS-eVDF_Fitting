# Slides outline (8 slides, ~7 minutes)

Figures are produced by `workshop/make_slide_figures.py` into `workshop/figures/`.

1. **Title and outcomes.** Fitting core, halo and strahl in SWA-EAS
   distributions with a Poisson likelihood. Take home: the repository at tag
   `s3park-2026`, eight notebooks, two days of preprocessed data.
2. **The instrument and its frames.** Two heads, 16 x 32 x 64; head frame ->
   SRF matrix; figure `eas_heads_srf.png` (velocity-direction sky map with
   +B, -B, sunward); one epoch = 64 x 1024 grid. (nb01 §0)
3. **From counts to phase space density.** f = C g; g is known for every
   pixel, also the empty ones; figure `one_count_and_zero_fractions.png`
   (one-count level vs energy; zero-count fraction per energy). (nb01 §1-2, §8)
4. **Three components.** Figure `measured_vdf_components.png` (parallel and
   perpendicular cuts with the fitted core, halo, strahl) and
   `pad_spectrogram.png`; the break energy from the perpendicular cone.
   (nb01 §4-5, nb03 §1)
5. **Objective and goodness of fit.** The deviance formula with its
   zero-count term; the expected deviance per pixel vs expected counts
   (`expected_deviance.png`); the four-panel figure with the pull map
   (`four_panel_epoch20.png`); R, pinned parameters. (nb05)
6. **The pipeline in three commands.** preprocess -> crossing -> runner;
   the release download script; runtimes per epoch and per day.
7. **Two days of results.** Figure `two_days_parameters.png` (densities,
   temperatures, verdicts, break energy); deviance vs brightness. (nb06 §6)
8. **Caveats and contact.** Density calibration vs RPW; the strahl drift
   lock; potential tables from 2023 on; Poisson vs chi-square in one figure
   (`poisson_vs_chi2_epoch20.png`, nb07).
