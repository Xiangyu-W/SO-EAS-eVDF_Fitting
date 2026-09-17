"""Fit objectives: exact Poisson deviance (the default) and the log-space
chi-square baseline.

Both run the same four stages (core -> halo -> strahl -> joint) with the same
seeds and parameter boxes (staged_fit.runStagedFit); only the residual vector
handed to the least-squares minimiser differs:

* ``poisson`` -- signed square-root Poisson deviance residuals on ALL valid
  pixels, zero counts included. The instrument one-count PSD field converts
  the model PSD into expected counts, so a zero-count pixel contributes
  sqrt(2 * expected) and keeps pulling the model down where nothing was
  detected. Maximum-likelihood for Poisson counts (Ioannou-style PMLE).
* ``chi2`` -- residual log10(model) - log10(data) on pixels with counts > 0
  only (the log of a zero PSD is undefined). The literature baseline
  (Stverak 2009, Bercic 2020, the legacy pipeline family). Kept so the
  censoring bias at low counts can be demonstrated: dropping the zero-count
  pixels keeps only the upward-fluctuated survivors of a faint tail.
"""

import numpy as np

from .models import coreHaloStrahlModel
from .staged_fit import runStagedFit, evaluateModel, MODEL_PSD_FLOOR

EXPECTED_COUNTS_FLOOR = 1e-12
OBJECTIVES = ('poisson', 'chi2')


def buildPoissonResidual(counts, grid):
    """Residual builder for the Poisson deviance objective."""
    v_par = grid['v_par']
    v_perp = grid['v_perp']
    oneCount = grid['one_count_psd']

    def buildResidual(mask):
        maskedVpar = v_par[mask]
        maskedVperp = v_perp[mask]
        maskedOneCount = oneCount[mask]
        observed = counts[mask]
        hasCounts = observed > 0

        def residual(params):
            paramsDict = {name: params[name].value for name in params}
            model = coreHaloStrahlModel(maskedVpar, maskedVperp, paramsDict)
            expected = np.maximum(model / maskedOneCount, EXPECTED_COUNTS_FLOOR)
            devianceTerm = expected - observed
            devianceTerm[hasCounts] += observed[hasCounts] * np.log(
                observed[hasCounts] / expected[hasCounts])
            return np.sign(observed - expected) * np.sqrt(
                2 * np.maximum(devianceTerm, 0))

        return residual

    return buildResidual


def buildLogChiSquareResidual(psd, counts, grid):
    """Residual builder for the log-space chi-square objective (counts > 0 only)."""
    v_par = grid['v_par']
    v_perp = grid['v_perp']

    def buildResidual(mask):
        fitMask = mask & (counts > 0)
        logData = np.log10(psd[fitMask])
        maskedVpar = v_par[fitMask]
        maskedVperp = v_perp[fitMask]

        def residual(params):
            paramsDict = {name: params[name].value for name in params}
            model = coreHaloStrahlModel(maskedVpar, maskedVperp, paramsDict)
            return np.log10(np.maximum(model, MODEL_PSD_FLOOR)) - logData

        return residual

    return buildResidual


def fitEpoch(psd, counts, grid, seedValues, objective='poisson', onStageComplete=None,
             freeHaloDrift=True):
    """Staged three-component fit of one epoch.

    Returns (params, modelPsd, diagnostics): the fitted parameter dict (CGS),
    the model PSD evaluated on the full pixel grid, and the stage diagnostics
    (break energy and its source, strahl sign/tier, effective stage-D boxes,
    number of free parameters, objective name).

    onStageComplete(stageName, paramsDict, pixelMask, freeNames) receives a
    snapshot after each stage (see staged_fit.runStagedFit).
    freeHaloDrift=False pins the halo parallel drift u_par_h at 0 (default: free).
    """
    if objective == 'poisson':
        buildResidual = buildPoissonResidual(counts, grid)
    elif objective == 'chi2':
        buildResidual = buildLogChiSquareResidual(psd, counts, grid)
    else:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    paramsDict, diagnostics = runStagedFit(psd, counts, grid, seedValues, buildResidual,
                                           onStageComplete=onStageComplete,
                                           freeHaloDrift=freeHaloDrift)
    diagnostics['objective'] = objective
    return paramsDict, evaluateModel(paramsDict, grid), diagnostics


def fitPoissonMle(psd, counts, grid, seedValues):
    return fitEpoch(psd, counts, grid, seedValues, objective='poisson')


def fitStagedLogChiSquare(psd, counts, grid, seedValues):
    return fitEpoch(psd, counts, grid, seedValues, objective='chi2')
