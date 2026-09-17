"""Fit-quality metrics and parameter-to-physics conversion."""

import numpy as np

from .models import temperatureFromThermalSpeed

EXPECTED_COUNTS_FLOOR = 1e-12


def poissonDeviancePerDof(modelPsd, counts, grid, numFreeParams):
    """Exact Poisson deviance of a model PSD field against the observed counts,
    evaluated on ALL valid pixels (count >= 0), divided by (pixels - free params).

    The same yardstick for every objective. It is NOT expected to be ~1 for a
    good fit: at low expected counts the deviance of a perfect model is well
    below 1 per pixel, so compare epochs within a data set rather than against
    a fixed threshold (see the QC notebook for the calibrated version)."""
    valid = grid['valid']
    expectedCounts = np.maximum(modelPsd[valid] / grid['one_count_psd'][valid],
                                EXPECTED_COUNTS_FLOOR)
    observed = counts[valid]
    withCounts = observed > 0
    devianceTerms = np.empty_like(expectedCounts)
    devianceTerms[~withCounts] = expectedCounts[~withCounts]
    devianceTerms[withCounts] = (expectedCounts[withCounts] - observed[withCounts]
                                 + observed[withCounts]
                                 * np.log(observed[withCounts] / expectedCounts[withCounts]))
    deviance = 2 * np.sum(devianceTerms)
    dof = valid.sum() - numFreeParams
    return deviance / dof


def reducedChiSquare(modelPsd, psd, counts, grid, numFreeParams):
    """Classical reduced chi-square on the DETECTED pixels only (counts > 0),
    with the Poisson uncertainty sigma_f = f / sqrt(counts) of each pixel.

    This is the goodness-of-fit number the chi-square pipeline family reports.
    It only sees pixels that registered at least one count, so it says nothing
    about how the model behaves where the data are zeros -- which is exactly
    where a censored fit goes wrong."""
    detected = grid['valid'] & (counts > 0)
    sigma = psd[detected] / np.sqrt(counts[detected])
    residuals = (modelPsd[detected] - psd[detected]) / sigma
    dof = detected.sum() - numFreeParams
    return float(np.sum(residuals ** 2) / dof)


def sliceDataMeans(psd, counts, grid, sliceMask):
    """count>=0 channel means per energy row with the Poisson uncertainty of the mean.

    Zero-count pixels enter as zeros: this is the unbiased estimate of f in
    the slice, unlike the mean over counts>0 pixels which is censored upward."""
    energies, means, errors = [], [], []
    for rowIndex in range(sliceMask.shape[0]):
        rowMask = sliceMask[rowIndex]
        numPixels = rowMask.sum()
        if numPixels < 2:
            continue
        rowCounts = counts[rowIndex][rowMask]
        totalCounts = rowCounts.sum()
        if totalCounts <= 0:
            continue
        meanPsd = np.sum(psd[rowIndex][rowMask]) / numPixels
        meanOneCount = np.mean(grid['one_count_psd'][rowIndex][rowMask])
        energies.append(np.median(grid['energy_eV'][rowIndex][rowMask]))
        means.append(meanPsd)
        errors.append(np.sqrt(totalCounts) * meanOneCount / numPixels)
    order = np.argsort(energies)
    return (np.asarray(energies)[order], np.asarray(means)[order],
            np.asarray(errors)[order])


def componentMomentsFromParams(params):
    """Physical summary (densities cm^-3, temperatures eV, drift cm/s) from fit params."""
    moments = {}
    moments['n_c'] = params.get('n_c', np.nan)
    moments['u_par_c'] = params.get('u_par_c', np.nan)
    moments['T_par_c_eV'] = temperatureFromThermalSpeed(params['v_th_par_c']) \
        if 'v_th_par_c' in params else np.nan
    moments['T_perp_c_eV'] = temperatureFromThermalSpeed(params['v_th_perp_c']) \
        if 'v_th_perp_c' in params else np.nan
    if 'n_h' in params:
        moments['n_h'] = params['n_h']
        moments['u_par_h'] = params.get('u_par_h', 0.0)
        moments['T_par_h_eV'] = temperatureFromThermalSpeed(params['v_th_par_h'])
        moments['T_perp_h_eV'] = temperatureFromThermalSpeed(params['v_th_perp_h'])
        moments['kappa_h'] = params.get('kappa_h', np.nan)
    if 'n_b' in params:
        moments['n_b'] = params['n_b']
        moments['u_par_b'] = params.get('u_par_b', np.nan)
        moments['T_par_b_eV'] = temperatureFromThermalSpeed(params['v_th_par_b'])
        moments['T_perp_b_eV'] = temperatureFromThermalSpeed(params['v_th_perp_b'])
        moments['kappa_b'] = params.get('kappa_b', np.nan)
    return moments


def relativeError(recovered, truth):
    if truth == 0 or not np.isfinite(truth) or not np.isfinite(recovered):
        return np.nan
    return (recovered - truth) / abs(truth)
