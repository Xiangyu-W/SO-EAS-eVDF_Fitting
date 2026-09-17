# Session script: 20 minutes

Setup before the session: VS Code remote on the server, repository at tag
`s3park-2026`, terminal with `PYTHONPATH=src`, every notebook open with its
saved outputs and a restarted kernel, `results/fitting/live_demo/` removed,
the GitHub release page in a browser tab, HTML exports of all notebooks
(`workshop/html/`) as the fallback if Jupyter misbehaves.

| clock | what | material | mode |
|---|---|---|---|
| 0:00 | Goal: core, halo and strahl parameters from EAS with a Poisson likelihood. What you take home: repository tag, notebooks, two days of data. | slide 1 | slide |
| 1:00 | EAS: two heads, 16 elevations x 32 azimuths x 64 energies; head frame -> SRF matrices; the sky map of velocity directions with +B, -B, sunward. One epoch = one 64 x 1024 grid. | slide 2 (nb01 §0) | slide |
| 3:00 | Counts -> PSD through the one-count level g; zero-count fractions by energy (3/4 of the pixels above 100 eV, 99 % above 400 eV). | slide 3 (nb01 §1-2, §8) | slide |
| 4:30 | Three components in the field-aligned frame; the PAD shows the strahl; the break energy separates core and suprathermals. | slide 4 (nb01 §4-5, nb03) | slide |
| 5:30 | Poisson deviance versus chi-square: the zero-count term. Goodness of fit: deviance/dof depends on brightness, calibrated ratio R, pull map, pinned parameters. | slide 5 (nb05) | slide |
| 7:00 | **Live**: repository tree, `workshop/README.md`; run notebook 00 cell 1 with `workers=32` into `live_demo` (~15 s); the parameter table; one four-panel figure. | nb00 | live |
| 8:00 | Notebook 02 scrolled: the four SOAR files, the spacecraft-potential table and its plot, the `preprocess_main` call, the HDF5 layout, the saturation census. | nb02 | pre-executed |
| 9:30 | Notebook 03: crossing on one epoch (figure), **live** rerun of the crossing on the 30 epochs (3 s), the PAD band fit and the two-tier verdict. | nb03 | one cell live |
| 11:00 | Notebook 04 in one screen: the three curves, the seed table over the data, the box table. | nb04 | pre-executed |
| 12:00 | **Live**: notebook 05 from the seeds cell: `fitEpoch` with the stage callback (~10 s), stage A/B/C/D figures, the four-panel figure, the goodness-of-fit cell, the pinned table, identity with the reference. | nb05 | live, cell by cell |
| 15:30 | Notebook 06 §6: two days of parameters, verdicts and break energy; deviance against brightness; the masked epochs. The three commands for your own interval. | nb06 | pre-executed |
| 17:00 | Notebook 07: the same epoch under both objectives (figure and ratio table); synthetic brightness panels. | nb07 §1, §3 | pre-executed |
| 18:00 | Wrap-up: tag, release download script, the caveats list in `workshop/README.md`. Questions. | slides 6-8 | slide |

Running late: at 11:00 skip notebook 04; at 15:30 skip notebook 07.

Pre-flight on the day: `uptime` (load), `nproc`, `rm -rf results/fitting/live_demo`,
`git status` clean at the tag, browser tabs open, font size up.
