"""Common data-driven initial guesses shared by all fitting methods,
so that method differences reflect objective/model choices, not seeding."""

import numpy as np

from .models import temperatureFromThermalSpeed, thermalSpeedFromTemperature

CORE_BAND_MAX_EV = 70.0
STRAHL_DETECTION_BAND_EV = (100.0, 600.0)
STRAHL_FLUX_RATIO_THRESHOLD = 1.6
STRAHL_CONE_HALF_ANGLE_DEG = 30.0


def seedFromData(psd, counts, grid, strahlDetection=None, scDensity=None,
                 breakEnergyEv=None):
    """Moment-based seeds for core, halo and strahl from the observed PSD.

    strahlDetection: optional dict from pad_detection.detectStrahlFromPad.
    When given (and not detector_failed), its sign/tier replace the legacy
    100-600 eV cone flux-ratio detector below: a weak tier broadens the
    strahl width seed, tier 'none' marks the epoch for a no-strahl fit.
    Without it the legacy detector runs and behavior is unchanged
    (strahl_tier None = always fit the strahl component).
    """
    valid = grid['valid'] & (counts > 0)
    v_par = grid['v_par']
    v_perp = grid['v_perp']
    energy = grid['energy_eV']

    coreBand = valid & (energy <= CORE_BAND_MAX_EV)
    weights = psd[coreBand]
    u_par_seed = np.sum(weights * v_par[coreBand]) / np.sum(weights)
    varPar = np.sum(weights * (v_par[coreBand] - u_par_seed) ** 2) / np.sum(weights)
    varPerp = np.sum(weights * v_perp[coreBand] ** 2) / np.sum(weights) / 2

    vthParSeed = max(np.sqrt(2 * varPar), 1e8)
    vthPerpSeed = max(np.sqrt(2 * varPerp), 1e8)

    # Peak-PSD density: f(0) ~ n / (pi^1.5 vth_par vth_perp^2) for a Maxwellian
    peakPsd = np.nanmax(psd[coreBand]) if coreBand.any() else 1e-26
    coreDensitySeed = peakPsd * np.pi ** 1.5 * vthParSeed * vthPerpSeed ** 2

    # Strahl detection: parallel vs antiparallel mean PSD in the suprathermal band
    lowBand, highBand = STRAHL_DETECTION_BAND_EV
    suprathermal = valid & (energy >= lowBand) & (energy <= highBand)
    pitch = grid['pitch_angle']
    parallelCone = suprathermal & (pitch <= STRAHL_CONE_HALF_ANGLE_DEG)
    antiParallelCone = suprathermal & (pitch >= 180 - STRAHL_CONE_HALF_ANGLE_DEG)

    def meanPsdInCone(coneMask):
        return np.nanmean(psd[coneMask]) if coneMask.sum() >= 5 else np.nan

    parallelPsd = meanPsdInCone(parallelCone)
    antiParallelPsd = meanPsdInCone(antiParallelCone)
    strahlSign = 0
    if np.isfinite(parallelPsd) and np.isfinite(antiParallelPsd) and antiParallelPsd > 0:
        ratio = parallelPsd / antiParallelPsd
        if ratio > STRAHL_FLUX_RATIO_THRESHOLD:
            strahlSign = 1
        elif ratio < 1 / STRAHL_FLUX_RATIO_THRESHOLD:
            strahlSign = -1
    elif np.isfinite(parallelPsd) and not np.isfinite(antiParallelPsd):
        strahlSign = 1

    coreTemperature = temperatureFromThermalSpeed(vthParSeed)
    haloVthSeed = thermalSpeedFromTemperature(coreTemperature * 5)

    strahlTier = None
    strahlPawDeg = np.nan
    perpWidthFactor = 0.4
    if strahlDetection is not None and strahlDetection['class'] != 'detector_failed':
        strahlSign = strahlDetection['sign']
        strahlTier = strahlDetection['tier']
        if strahlTier != 'none':
            # Beam-side measured PAD width (FWHM, deg) anchors v_th_perp_b in
            # buildParameters (overall-fit parity).
            strahlPawDeg = (strahlDetection['paw_par'] if strahlSign > 0
                            else strahlDetection['paw_anti'])
        if strahlTier == 'weak':
            # Scattered/broadened beam: seed the perpendicular width broad so
            # the strahl stage starts near the observed 40-60 deg PAD widths.
            perpWidthFactor = 0.8

    return {'n_c': coreDensitySeed, 'u_par_c': u_par_seed,
            'v_th_par_c': vthParSeed, 'v_th_perp_c': vthPerpSeed,
            'core_peak_psd': float(peakPsd),
            'break_E_eV': np.nan if breakEnergyEv is None else float(breakEnergyEv),
            'n_sc': np.nan if scDensity is None else float(scDensity),
            'n_h': 0.03 * coreDensitySeed, 'u_par_h': 0.0,
            'v_th_par_h': haloVthSeed, 'v_th_perp_h': haloVthSeed, 'kappa_h': 5.0,
            'strahl_sign': strahlSign,
            'strahl_tier': strahlTier,
            'strahl_paw_deg': strahlPawDeg,
            'n_b': 0.01 * coreDensitySeed if strahlSign != 0 else 1e-4 * coreDensitySeed,
            'u_par_b': (strahlSign if strahlSign != 0 else 1) * 2.5 * vthParSeed,
            'v_th_par_b': haloVthSeed, 'v_th_perp_b': perpWidthFactor * haloVthSeed,
            'kappa_b': 5.0}
