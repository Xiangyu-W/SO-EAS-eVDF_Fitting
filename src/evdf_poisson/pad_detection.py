"""Two-tier pitch-angle-distribution (PAD) strahl detection.

Chain: band-mean PAD over [max(61, 0.8*break), 1000] eV -> double-Gaussian
fit (Owen et al. 2022 form: background + one Gaussian at 0 deg + one at
180 deg) -> two-tier criterion per side (beta > 2 strong, 0.6 < beta <= 2
weak, pitch-angle width <= 60 deg; beta < -0.5 records a sunward deficit).

The break-scaled lower edge matters: the band mean is PSD-weighted toward its
lowest channels, so on high-break windows (0.3 au, break ~185 eV) a fixed
61 eV edge lets the near-isotropic core dilute the strahl contrast to
quasi-isotropic. Without a break energy the edge stays at the 61 eV floor.

The classification only needs PSD ratios, so the raw (unconverted) ``pad_in``
stored in the HDF5 file is used directly.
"""

import warnings

import h5py
import numpy as np

from constant import low_pad_val, high_pad_val
from utils import determineSWA_energyIdx
from pad_fitting import pad_fit_function, pad_model, evaluate_pad_conditions_two_tier

TIER_NAMES = {0: 'none', 1: 'weak', 2: 'strong'}
# Overall-pipeline parity (evaluateStrahlConeFromBreakBand).
PAD_BAND_BREAK_FRACTION = 0.8
PAD_PSD_COLUMN = 0
FIT_CURVE_PITCH_DEG = np.linspace(0.0, 180.0, 181)

DETECTOR_FAILED = {'class': 'detector_failed', 'sign': 0, 'tier': 'none',
                   'beta_par': np.nan, 'beta_anti': np.nan,
                   'paw_par': np.nan, 'paw_anti': np.nan, 'deficit': ''}


def padBandEdgesEv(breakEnergyEv=None):
    """(low, high) energy edges of the PAD band used for strahl detection."""
    lowEdgeEv = float(low_pad_val)
    if breakEnergyEv is not None and np.isfinite(breakEnergyEv):
        lowEdgeEv = max(lowEdgeEv, PAD_BAND_BREAK_FRACTION * float(breakEnergyEv))
    return lowEdgeEv, float(high_pad_val)


def fitBandMeanPad(chunkPath, epochIndex, breakEnergyEv=None):
    """Band-mean PAD of one epoch and its double-Gaussian fit.

    Returns a dict: mean_pad (per pitch-angle bin), pitch_centers, pitch_angles
    (bin edges), band_ev, params (lmfit Parameters of the fitted model),
    fit_alpha / fit_curve (the fitted model on a dense pitch-angle grid).
    Raises on an empty band."""
    with h5py.File(chunkPath, 'r') as h5:
        pad = h5['pad_in'][epochIndex]
        pitchAngles = h5['pitch_angles'][:]
        swaEnergyAll = h5['swa_energy']
        swaEnergy = (swaEnergyAll[epochIndex] if swaEnergyAll.ndim == 2
                     else swaEnergyAll[:])
    pitchCenters = np.diff(pitchAngles) / 2 + pitchAngles[:-1]
    bandEv = padBandEdgesEv(breakEnergyEv)
    lowIdx, highIdx = determineSWA_energyIdx(bandEv[0], bandEv[1], swaEnergy)
    bandSlice = pad[highIdx:lowIdx, :, PAD_PSD_COLUMN]
    if not np.isfinite(bandSlice).any():
        raise ValueError('empty PAD band')
    meanPad = np.nanmean(bandSlice, axis=0)
    with warnings.catch_warnings():
        # lmfit's uncertainty propagation warns on zero-error parameters of
        # the PAD model; the verdict only uses the fitted values.
        warnings.simplefilter('ignore', UserWarning)
        params, _ = pad_fit_function(meanPad, pitchCenters, pitchAngles)
    return {'mean_pad': meanPad, 'pitch_centers': pitchCenters,
            'pitch_angles': pitchAngles, 'band_ev': bandEv, 'params': params,
            'fit_alpha': FIT_CURVE_PITCH_DEG,
            'fit_curve': pad_model.eval(params, alpha=FIT_CURVE_PITCH_DEG)}


def classifyPadVerdict(verdict):
    """Two-tier per-side verdict -> detection class, strahl sign and tier."""
    parTier, antiTier = verdict['par_tier'], verdict['anti_tier']
    if parTier and antiTier:
        detectionClass = 'both'
        sign = 1 if verdict['beta_par'] >= verdict['beta_anti'] else -1
        tier = TIER_NAMES[max(parTier, antiTier)]
    elif parTier:
        detectionClass = f'par_{TIER_NAMES[parTier]}'
        sign, tier = 1, TIER_NAMES[parTier]
    elif antiTier:
        detectionClass = f'anti_{TIER_NAMES[antiTier]}'
        sign, tier = -1, TIER_NAMES[antiTier]
    else:
        detectionClass, sign, tier = 'quasi_iso', 0, 'none'
    return {'class': detectionClass, 'sign': sign, 'tier': tier,
            'beta_par': verdict['beta_par'], 'beta_anti': verdict['beta_anti'],
            'paw_par': verdict['paw_par_deg'],
            'paw_anti': verdict['paw_anti_deg'], 'deficit': verdict['deficit']}


def detectStrahlFromPad(chunkPath, epochIndex, breakEnergyEv=None):
    """Classify one epoch's PAD. Returns a dict:

    class:    par_strong | par_weak | anti_strong | anti_weak | both |
              quasi_iso | detector_failed
    sign:     +1 (parallel) / -1 (anti-parallel) / 0 (no strahl)
    tier:     strong | weak | none   (for 'both': the stronger side's tier)
    beta_par, beta_anti, paw_par, paw_anti, deficit ('', par, anti, both)
    """
    try:
        padFit = fitBandMeanPad(chunkPath, epochIndex, breakEnergyEv)
        verdict = evaluate_pad_conditions_two_tier(padFit['params'],
                                                   padFit['pitch_centers'])
    except Exception:
        return dict(DETECTOR_FAILED)
    return classifyPadVerdict(verdict)
