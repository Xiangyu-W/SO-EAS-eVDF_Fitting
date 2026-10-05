# The Poisson fitting pipeline

This page is the reference for what `evdf_poisson` does to one epoch. The
example notebooks show the same steps on real data; this page keeps the
numbers (seeds, allowed ranges, thresholds) in one place. Code:
`src/evdf_poisson/` (`staged_fit.py` for the stages and boxes, `models.py`
for the functions, `seeds.py` for the initial guesses, `pad_detection.py`
for the strahl verdict, `objectives.py` for the residuals).

## Model

- **core**: bi-Maxwellian, parameters $n_c$, $u_{\parallel c}$, $v_{th\parallel c}$, $v_{th\perp c}$
- **halo**: bi-kappa, parameters $n_h$, $u_{\parallel h}$, $v_{th\parallel h}$, $v_{th\perp h}$, $\kappa_h$
- **strahl**: drifting bi-kappa, truncated on the side facing the core (parallel
  width divided by $\sqrt{D}$, $D = 5$), parameters $n_b$, $u_{\parallel b}$,
  $v_{th\parallel b}$, $v_{th\perp b}$, $\kappa_b$

Thermal speeds use the $(\kappa - 3/2)$ convention, $v_{th} = \sqrt{2 k_B T / m_e}$,
so a temperature means the same thing for every component. Velocities are in
cm/s, densities in cm$^{-3}$, PSD in s$^3$ cm$^{-6}$ (CGS) throughout the code.

## Objective

Every stage minimises the **Poisson deviance** over its pixel subset, zero
counts included: for pixel $i$ with $C_i$ counts and expected counts
$\mu_i = f_\text{model}(v_i) / g_i$ ($g_i$ = one-count PSD of the pixel),

$$ r_i = \operatorname{sign}(C_i - \mu_i)\sqrt{2\,[\mu_i - C_i + C_i \ln(C_i/\mu_i)]}, \qquad D = \sum_i r_i^2, $$

solved with `lmfit` / `scipy.optimize.least_squares` (trust-region reflective,
box constraints). `objective='chi2'` replaces the residual by
$\log_{10} f_\text{model} - \log_{10} f$ on pixels with $C_i > 0$ (the
literature baseline; kept for comparison, see notebook 07).

The one-count PSD is $g = m_e^2 / (2\,\Delta t\, E^2\, G\, Q)$. The
preprocessing evaluates it from the geometric-factor and efficiency tables and
writes the static `one_count_psd_in` field, exact for every pixel; energy
steps without a single count therefore enter the likelihood as zero-count
pixels. Files without the field fall back to a per-epoch recovery from
PSD / counts (exact where counts > 0, energy-row median for dark pixels,
energy steps without any count excluded).

## Inputs determined before the fit

| input | method | where |
|---|---|---|
| break energy $E_\text{break}$ | two-population crossing on the perpendicular cone (90 $\pm$ 20 deg): joint Maxwellian core + power-law tail, break at the crossing; low cut $12\,\text{eV} - \phi_{sc}$ | `compute_break_energy_crossing.py`, `break_energy_methods.py` |
| strahl side, tier, width | band-mean PAD over $[\max(61, 0.8 E_\text{break}), 1000]$ eV, double-Gaussian fit, two-tier verdict (see below) | `pad_detection.py` |
| low-energy cut | $14\,\text{eV} - \phi_{sc}$ (spacecraft secondaries below) | `epoch_io.py` |
| $n_{sc}$ | RPW electron density from the file, boxes the core and strahl densities | `epoch_io.py` |

If no crossing break exists for an epoch, the fit falls back to
$E_\text{break} = 7.7\,k T_{c\parallel}$ clipped to [40, 300] eV, recomputed
after stage A (`break_source = proxy_7.7T` in the output).

### Two-tier strahl verdict

From the fitted band-mean PAD model, per side: contrast
$\beta = \langle P \rangle_\text{side} / \langle P \rangle_{65-115^\circ} - 1$
(side = 0-45 deg or 135-180 deg) and width PAW $= 2\sqrt{2\ln 2}\,W$.

| condition | tier | consequence |
|---|---|---|
| $\beta > 2$, PAW $\le 60^\circ$ | strong | strahl fitted on that side |
| $0.6 < \beta \le 2$, PAW $\le 60^\circ$ | weak | strahl fitted, broad perpendicular seed |
| neither side | none (`quasi_iso`) | no strahl component, 8 free parameters |
| $\beta < -0.5$ | deficit flag | recorded (`pad_deficit`), not fitted |

Both sides qualifying gives class `both`; the larger $\beta$ decides the sign.

## Stages

### Stage A: core

Pixels: valid, $E \le E_\text{break}$, all pitch angles. Free: the four core
parameters.

| parameter | initial | range |
|---|---|---|
| $n_c$ | $0.8\,n_{sc}$ | $[0,\ 2\,n_{sc}]$ (no $n_{sc}$: peak-PSD seed $\times [0.1, 10]$) |
| $u_{\parallel c}$ | 0 | $\pm 1.5\times10^8$ cm/s, free sign |
| $v_{th\parallel c}$, $v_{th\perp c}$ | $(0.8\,n_{sc} / \pi^{3/2} f_\text{peak})^{1/3}$ | $[10^7,\ 8\times10^8]$ cm/s |

### Stage B: halo

Pixels: valid, $E > E_\text{break}$, outside both 30 deg cones around
$\pm\mathbf{B}$. Free: the four halo parameters; core frozen. Seeds and
boxes are rebuilt from the **fitted** core.

| parameter | initial | range |
|---|---|---|
| $n_h$ | $0.03\,n_c^\text{fit}$ | $[10^{-5},\ 0.2]\times n_c^\text{fit}$ |
| $v_{th\parallel h}$ | $\max(\sqrt5\, v_{th\parallel c}^\text{fit},\ 1.05 \times \text{floor})$ | $[2\, v_{th\parallel c}^\text{fit},\ 4\sqrt5\, v_{th\parallel c}^\text{fit}]$ |
| $v_{th\perp h}$ | same, anchored on $v_{th\perp c}^\text{fit}$ | $[2\, v_{th\perp c}^\text{fit},\ 4\sqrt5\, v_{th\parallel c}^\text{fit}]$ |
| $\kappa_h$ | 5 | $[2.0,\ 12]$ |
| $u_{\parallel h}$ | 0 | $\pm 0.2\, v_{th\perp h}^\text{seed}$ (`freeHaloDrift=False` pins it at 0) |

### Stage C: strahl (only with a strong or weak verdict)

Pixels: the 30 deg cone on the strahl side, all valid energies (at least 20
pixels, otherwise the stage is skipped). Free: $n_b$, $u_{\parallel b}$,
$v_{th\parallel b}$, $\kappa_b$; $v_{th\perp b}$ is tied to the drift. Core
and halo frozen.

| parameter | initial | range |
|---|---|---|
| $n_b$ | $0.01\,n_{sc}$ ($0.01\,n_c$ without $n_{sc}$) | $[0,\ 0.5\,n_{sc}]$ ($[0,\ 0.1\,n_c]$) |
| $u_{\parallel b}$ | sign $\times\ 0.9\, v_\text{break}$ | sign $\times [0.3,\ 3.0]\, v_\text{break}$ |
| $v_{th\parallel b}$ | $0.5\, v_\text{break}$ | $[2\times10^7,\ 1.5\times10^9]$ cm/s |
| $v_{th\perp b}$ | $\tan(\text{clip}(\text{PAW}/2,\ 5^\circ, 45^\circ))\ \lvert u_{\parallel b} \rvert$ | expression, not free |
| $\kappa_b$ | 5 | $[1.7,\ 20]$ |

with $v_\text{break} = \sqrt{2 E_\text{break} / m_e}$.

### Stage C to D

| parameter | change |
|---|---|
| $u_{\parallel b}$ | box tightened to the stage-C value $\pm 20\,\%$ |
| $v_{th\perp b}$ | expression released, re-anchored to $\tan(\text{half angle})\,\lvert u_C \rvert$, box $[0.3,\ 3.0] \times$ anchor |

### Stage D: joint

Pixels: all valid. Free: 14 parameters (9 without strahl, $n_b$ locked at
0) inside the boxes above.

## Output quality flags

- `deviance_per_dof`: $D$ over all valid pixels divided by (pixels minus
  free parameters). Not expected to be near 1 for a good fit: at low
  expected counts a perfect model has a deviance well below 1 per pixel.
  Compare epochs within a data set.
- `red_chi2`: classical reduced chi-square on detected pixels with
  $\sigma = f / \sqrt{C}$; reported for comparison with the literature only.
- `pinned_params`: free parameters within 0.1 % of a box edge. Common and
  harmless: `v_th_perp_h` on its floor, `u_par_b` on the lower edge of its
  $\pm 20\,\%$ lock (the joint stage pulls the beam down-energy; the lock
  sets how far),
  `kappa_b` at 20. A pinned core parameter or `n_h` deserves a look at the
  figure.

## Design notes (what the tests behind these choices found)

- **Free-sign core drift.** Clamping $u_{\parallel c}$ away from the strahl
  side (a legacy convention) pins it at zero under the Poisson objective and
  displaces the whole core; the anti-strahl wing is then over-predicted by
  2-4 and the strahl inflates to compensate.
- **Halo boxes anchored on the fitted core, not on the seeds.** With
  seed-anchored boxes the 0.2 $n_c$ ceiling and the 2 $\times$ width floor were
  ineffective (halo densities of 21 % of the core, widths 1.4 $\times$ core).
- **$\kappa_h \ge 2$.** The central amplitude of a bi-kappa grows as
  $(\kappa - 3/2)^{-3/2}$; at 1.6 the halo becomes a spike hidden under the
  core peak that stage B (which only sees $E > E_\text{break}$) cannot
  detect, tripling $n_h$ at no cost in deviance. At 2.0 the enhancement is
  under 3 and the spike is gone.
- **Strahl drift floor.** Without it the strahl degenerates into a second
  drifting core on real data ($u_b \sim 0.6$-1.7 $v_{th c}$, $n_b$ up to
  0.5 $n_c$, $\kappa_b$ to the Maxwellian limit).
- **Stage C on the whole cone, not only above the break.** With a band
  starting at 0.8 $E_\text{break}$, hot-core epochs (break near 180 eV) hid
  the 70-140 eV transition from the strahl stage, the drift landed too
  high and the parallel cut showed a dip between core and strahl.
- **Truncation $D = 5$ rather than 10.** $D$ only shapes the core/strahl
  junction (deviance flat, parameters within 10 %); at 10 the beam's inner
  side falls too fast and leaves a flat step at the handover.
- **The deviance is flat along several directions** ($n_h$ versus width,
  large $\kappa$, $D$). The boxes exist to choose the physical corner of
  those ridges, not to improve the fit; that is why each is anchored to a
  measured quantity.
- **A symmetric halo cannot represent a sunward deficit** (anti-strahl wing
  eroded, $\beta < -0.5$): the model over-predicts that wing by 2-4. The
  deficit is flagged in the output, not fitted.
