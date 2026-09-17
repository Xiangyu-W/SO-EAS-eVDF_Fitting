"""Read one epoch of a preprocessed ``VDF_eas_forFitting_*.h5`` file into the
gyrotropic (v_par, v_perp) pixel grid the fit works on.

The file stores, per epoch, ``pa_vec_in`` of shape (64 energies, 1024 detector
pixels, 5) with columns [pitch angle (deg), PSD (file units), v_par (km/s),
v_perp (km/s), counts]. The one-count PSD ``g`` (PSD registered by a single
count in that pixel) is the exact PSD<->counts conversion the Poisson
objective needs for EVERY pixel, zero-count ones included.
"""

import warnings

import h5py
import numpy as np

from .models import ELECTRON_MASS_G, ERG_PER_EV

PSD_TO_CGS = 1e-30          # file PSD units -> s^3 cm^-6
KM_TO_CM = 1e5
LOW_ENERGY_CUT_EV = 8.0     # absolute floor of the fitted energy range
# Spacecraft secondary/photo-electrons dominate below ~14 eV in the spacecraft
# frame; in the potential-corrected frame that is 14 - scpot (legacy constant.low_c_val).
SECONDARY_ELECTRON_CUT_EV = 14.0

PA_VEC_COLUMNS = {'pitch_angle': 0, 'psd': 1, 'v_par': 2, 'v_perp': 3, 'counts': 4}


def countEpochs(chunkPath):
    with h5py.File(chunkPath, 'r') as h5:
        return int(h5['pa_vec_in'].shape[0])


def readEpochTimes(chunkPath):
    """Epoch times as int64 nanoseconds since 1970-01-01 UTC."""
    with h5py.File(chunkPath, 'r') as h5:
        return h5['time_EAS'][()]


def recoverOneCountPsd(paVec):
    """One-count PSD field from a single epoch's pa_vec (file units).

    PSD / counts is exactly g wherever counts > 0 (the preprocessing built PSD
    by multiplying counts with g). Pixels with zero counts in this epoch take
    the median g of their energy row -- an approximation for those dark pixels
    only; files carrying a static ``one_count_psd_in`` field avoid it."""
    psd = paVec[:, :, PA_VEC_COLUMNS['psd']]
    counts = paVec[:, :, PA_VEC_COLUMNS['counts']]
    with np.errstate(invalid='ignore', divide='ignore'), warnings.catch_warnings():
        # Energy rows without any detected pixel give an all-NaN median: they
        # stay NaN and are excluded by the valid mask, so the warning is noise.
        warnings.simplefilter('ignore', RuntimeWarning)
        oneCountPsd = np.where(counts > 0, psd / counts, np.nan)
        rowMedian = np.nanmedian(oneCountPsd, axis=1)
    return np.where(np.isfinite(oneCountPsd), oneCountPsd, rowMedian[:, None])


def loadEpoch(chunkPath, epochIndex):
    """Return (grid, psd, counts, epochNs, scDensity) for one epoch, in CGS.

    grid: dict with v_par, v_perp (cm/s), pitch_angle (deg), one_count_psd,
    energy_eV and the boolean ``valid`` mask (finite velocities, defined
    one-count PSD, energy above the low-energy cut).
    scDensity: independent electron density from RPW (cm^-3), NaN if absent.
    """
    with h5py.File(chunkPath, 'r') as h5:
        paVec = h5['pa_vec_in'][epochIndex]
        if 'one_count_psd_in' in h5:
            oneCountPsd = h5['one_count_psd_in'][()]
        else:
            oneCountPsd = recoverOneCountPsd(paVec)
        epochNs = h5['time_EAS'][epochIndex]
        scDensity = h5['n_sc'][epochIndex]
        scPotential = h5['scpot_aligned'][epochIndex]

    v_par = paVec[:, :, PA_VEC_COLUMNS['v_par']] * KM_TO_CM
    v_perp = paVec[:, :, PA_VEC_COLUMNS['v_perp']] * KM_TO_CM
    psd = paVec[:, :, PA_VEC_COLUMNS['psd']] * PSD_TO_CGS
    counts = paVec[:, :, PA_VEC_COLUMNS['counts']]
    oneCount = oneCountPsd * PSD_TO_CGS
    speed = np.sqrt(v_par ** 2 + v_perp ** 2)
    energy_eV = 0.5 * ELECTRON_MASS_G * speed ** 2 / ERG_PER_EV

    lowEnergyCut = LOW_ENERGY_CUT_EV
    if np.isfinite(scPotential):
        lowEnergyCut = max(LOW_ENERGY_CUT_EV, SECONDARY_ELECTRON_CUT_EV - float(scPotential))
    validMask = (np.isfinite(v_par) & np.isfinite(v_perp) & np.isfinite(oneCount)
                 & np.isfinite(psd) & (oneCount > 0) & (energy_eV >= lowEnergyCut))
    grid = {'v_par': v_par, 'v_perp': v_perp,
            'pitch_angle': paVec[:, :, PA_VEC_COLUMNS['pitch_angle']],
            'one_count_psd': oneCount, 'energy_eV': energy_eV, 'valid': validMask}
    return grid, np.nan_to_num(psd), np.nan_to_num(counts), epochNs, scDensity


loadRealEpoch = loadEpoch
