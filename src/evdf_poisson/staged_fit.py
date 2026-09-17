"""Staged core -> halo -> strahl -> joint fit: seeds, parameter boxes and the
four stages shared by every objective.

The Poisson and chi-square fits (objectives.py) run exactly these stages with
the same seeds and boxes; only the residual definition differs.
"""

import numpy as np
import lmfit

from .models import coreHaloStrahlModel
from .seeds import STRAHL_CONE_HALF_ANGLE_DEG

BREAK_ENERGY_CORE_TEMPERATURE_FACTOR = 7.7  # findTwoPopulationCrossing calibration
BREAK_ENERGY_LIMITS_EV = (40.0, 300.0)
# Overall-fit core stage settings (Core_fitting.set_core_initial_params):
# density seeded at 0.8 n_sc in a [0, 2 n_sc] box, drift within +-2e8 cm/s
# sign-restricted away from the strahl side, widths seeded by peak-PSD
# inversion inside the absolute 8e8 cm/s production box.
CORE_DENSITY_SEED_FRACTION = 0.8
CORE_DENSITY_BOX_FACTOR = 2.0
# Free-sign core drift: production's anti-strahl sign clamp pins u_par_c at 0
# under the Poisson objective and displaces the whole core (2022-01-01 e16:
# anti wing over-predicted 2-4x, strahl inflated into a tail bump). The fit
# only needs mild drifts (e16: +2e7), so the box is a gentle +-1.5e8 cm/s.
CORE_DRIFT_BOUND_CMS = 1.5e8
CORE_WIDTH_BOUNDS_CMS = (1e7, 8e8)
MODEL_PSD_FLOOR = 1e-40
# Production convention (Beam_fitting u_const_range): |u_par_b| >= 0.6 * v(E_break),
# the strahl drift may not fall much below the core/suprathermal break velocity.
STRAHL_DRIFT_FLOOR_BREAK_FRACTION = 0.6
# Overall-fit parity (used when the PAD detector supplies a beam width):
# stage-C drift box in break speeds, stage-D drift box fraction around the
# stage-C value, perp-width box around the PAW-derived anchor, and the
# half-angle clip applied to the measured PAW/2 (test_two_stage_fit values).
# Lower edge stays at the production suprathermal floor 0.6 (overall stage-1
# opens down to 0.3 but is protected by its external beam seed; under the
# count-dominated Poisson cone objective the drift dives to the bottom edge
# and the stage-D lock then freezes the mistake — 2022-01-09 v2 run).
# Stage-D box widened from +-5% to +-20% on 2026-09-16 (user decision).
STRAHL_DRIFT_SEARCH_BREAK_FRACTIONS = (0.6, 3.0)
STRAHL_JOINT_DRIFT_BOX_FRACTION = 0.20
STRAHL_PERP_WIDTH_BOX = (0.3, 3.0)
STRAHL_HALF_ANGLE_CLIP_DEG = (5.0, 45.0)
# Stage C is skipped (strahl left at its seed) below this many cone pixels.
MIN_STRAHL_STAGE_PIXELS = 20
# Overall stage-1 parity mode for stage C (runStrahlStage1 fallback seeds and
# boxes): the wedge is fitted at ALL valid energies with core+halo frozen, the
# perp width is expr-tied to tan(halfAngle)*|u_par_b| during the stage, and the
# generic seeds below replace the data-derived strahl seeds.
STAGE_C_OVERALL_SEED_DENSITY_FRACTION = 0.01
STAGE_C_OVERALL_SEED_DRIFT_BREAK_FRACTION = 0.9
STAGE_C_OVERALL_SEED_WIDTH_BREAK_FRACTION = 0.5
STAGE_C_OVERALL_SEED_KAPPA = 5.0
STAGE_C_OVERALL_DRIFT_BREAK_FRACTIONS = (0.3, 3.0)
STAGE_C_OVERALL_WIDTH_BOUNDS_CMS = (2e7, 1.5e9)

# Halo parallel drift (free by default; freeHaloDrift=False pins it at 0):
# seeded at 0 and boxed to a fraction of the seed halo perp width. The halo
# may counter-drift against the strahl to keep zero current (Stverak et al.
# 2009), so it may move but not far. At 0.5 the halo could slide onto an
# undetected anti-parallel strahl wing (2023-04-04 example epoch 26), so the
# box is 0.2.
HALO_DRIFT_BOUND_IN_WIDTHS = 0.2
HALO_DRIFT_PARAM_NAME = 'u_par_h'

CORE_PARAM_NAMES = ['n_c', 'u_par_c', 'v_th_par_c', 'v_th_perp_c']
HALO_PARAM_NAMES = ['n_h', HALO_DRIFT_PARAM_NAME, 'v_th_par_h', 'v_th_perp_h', 'kappa_h']
STRAHL_PARAM_NAMES = ['n_b', 'u_par_b', 'v_th_par_b', 'v_th_perp_b', 'kappa_b']
ALL_PARAM_NAMES = CORE_PARAM_NAMES + HALO_PARAM_NAMES + STRAHL_PARAM_NAMES


def buildParameters(seedValues, freeHaloDrift=True):
    params = lmfit.Parameters()

    # Core: overall-fit settings when the independent density is available
    # (n = 0.8 n_sc in [0, 2 n_sc], widths by peak-PSD inversion in the
    # absolute production box, free-sign drift in a gentle +-1.5e8 box);
    # legacy relative boxes otherwise (RPW gaps).
    inputDensity = seedValues.get('n_sc', np.nan)
    hasInputDensity = np.isfinite(inputDensity) and inputDensity > 0
    peakPsd = seedValues.get('core_peak_psd', np.nan)
    if hasInputDensity:
        coreDensity = CORE_DENSITY_SEED_FRACTION * inputDensity
        params.add('n_c', value=coreDensity, min=0.0,
                   max=CORE_DENSITY_BOX_FACTOR * inputDensity)
    else:
        coreDensity = seedValues['n_c']
        params.add('n_c', value=coreDensity, min=0.1 * coreDensity,
                   max=10 * coreDensity)
    if hasInputDensity and np.isfinite(peakPsd) and peakPsd > 0:
        vthCore = float(np.clip((coreDensity / (np.pi ** 1.5 * peakPsd)) ** (1 / 3),
                                *CORE_WIDTH_BOUNDS_CMS))
        vthPerpSeed = vthCore
    else:
        vthCore = seedValues['v_th_par_c']
        vthPerpSeed = seedValues['v_th_perp_c']
    params.add('u_par_c', value=0.0,
               min=-CORE_DRIFT_BOUND_CMS, max=CORE_DRIFT_BOUND_CMS)
    params.add('v_th_par_c', value=vthCore, min=CORE_WIDTH_BOUNDS_CMS[0],
               max=CORE_WIDTH_BOUNDS_CMS[1])
    params.add('v_th_perp_c', value=vthPerpSeed, min=CORE_WIDTH_BOUNDS_CMS[0],
               max=CORE_WIDTH_BOUNDS_CMS[1])

    # Production settled-recipe halo constraints: width >= 2x core width and a
    # 0.2 n_c ceiling — otherwise the halo degenerates into a second core
    # (real-data mode: n_h ~ n_c with T_h ~ 1.4 T_c). These placeholder boxes
    # are rebuilt from the stage-A fitted core by _applyHaloSeedsFromCore.
    haloVth = seedValues['v_th_par_h']
    haloWidthFloor = 2.0 * vthCore
    params.add('n_h', value=seedValues['n_h'], min=1e-5 * coreDensity, max=0.2 * coreDensity)
    params.add('v_th_par_h', value=max(haloVth, 1.05 * haloWidthFloor),
               min=haloWidthFloor, max=4 * haloVth)
    params.add('v_th_perp_h', value=max(seedValues['v_th_perp_h'], 1.05 * haloWidthFloor),
               min=haloWidthFloor, max=4 * haloVth)
    params.add('kappa_h', value=seedValues['kappa_h'], min=KAPPA_H_BOUNDS[0],
               max=KAPPA_H_BOUNDS[1])
    if freeHaloDrift:
        haloDriftBound = HALO_DRIFT_BOUND_IN_WIDTHS * haloVth
        params.add(HALO_DRIFT_PARAM_NAME, value=0.0,
                   min=-haloDriftBound, max=haloDriftBound)

    # Strahl must stay suprathermal: without a drift floor the component
    # degenerates into a second drifting core on real data (2022-01 finding:
    # u_b ~ 0.6-1.7 vth_c, n_b up to 0.5 n_c, kappa_b -> Maxwellian).
    from .models import temperatureFromThermalSpeed, thermalSpeedFromTemperature
    breakEnergy = np.clip(
        BREAK_ENERGY_CORE_TEMPERATURE_FACTOR * temperatureFromThermalSpeed(vthCore),
        *BREAK_ENERGY_LIMITS_EV)
    beamDrift = seedValues['u_par_b']
    driftSign = 1 if beamDrift >= 0 else -1
    breakSpeed = thermalSpeedFromTemperature(breakEnergy)
    pawDeg = seedValues.get('strahl_paw_deg', np.nan)
    if np.isfinite(pawDeg):
        # Overall-fit parity, stage-C box: |u| in [0.3, 3.0] v(E_break);
        # runStagedFit tightens it to +-20% after stage C.
        lowFraction, highFraction = STRAHL_DRIFT_SEARCH_BREAK_FRACTIONS
        driftBounds = sorted((driftSign * lowFraction * breakSpeed,
                              driftSign * highFraction * breakSpeed))
        driftValue = driftSign * np.clip(abs(beamDrift),
                                         1.05 * lowFraction * breakSpeed,
                                         0.95 * highFraction * breakSpeed)
    else:
        # Legacy box (no PAD width available): suprathermal drift floor.
        driftFloor = STRAHL_DRIFT_FLOOR_BREAK_FRACTION * breakSpeed
        driftBounds = sorted((driftSign * driftFloor,
                              driftSign * max(3 * abs(beamDrift), 2 * driftFloor)))
        driftValue = driftSign * max(abs(beamDrift), 1.05 * driftFloor)

    # n_b ceiling mirrors the overall joint fit: 0.5x the independent RPW
    # density when available, 0.1 n_c otherwise; floor 0 so the joint stage
    # may drop the component entirely.
    inputDensity = seedValues.get('n_sc', np.nan)
    strahlDensityCeiling = (0.5 * inputDensity
                            if np.isfinite(inputDensity) and inputDensity > 0
                            else 0.1 * coreDensity)
    params.add('n_b', value=min(seedValues['n_b'], 0.9 * strahlDensityCeiling),
               min=0.0, max=strahlDensityCeiling)
    params.add('u_par_b', value=driftValue,
               min=driftBounds[0], max=driftBounds[1])
    params.add('v_th_par_b', value=seedValues['v_th_par_b'],
               min=0.2 * haloVth, max=4 * haloVth)
    if np.isfinite(pawDeg):
        # Overall-fit parity: the perp width is anchored to the measured PAD
        # width, v_th_perp_b ~ tan(PAW/2) * |u_par_b| within a 0.3-3.0x box.
        anchor = strahlPerpAnchor(pawDeg, driftValue)
        params.add('v_th_perp_b', value=anchor,
                   min=STRAHL_PERP_WIDTH_BOX[0] * anchor,
                   max=STRAHL_PERP_WIDTH_BOX[1] * anchor)
    else:
        params.add('v_th_perp_b', value=seedValues['v_th_perp_b'],
                   min=0.05 * haloVth, max=4 * haloVth)
    # kappa_b box matches the overall fit's strahl stages (1.7-20).
    params.add('kappa_b', value=seedValues['kappa_b'], min=1.7, max=20)
    return params


# kappa_h floor 2.0, not the production 1.6: the bi-kappa central amplitude
# grows as (kappa-1.5)^-1.5, so a 1.6 halo is a 30x spike at v=0 hidden under
# the core (stage B fits only E > break and never sees it; 2022-01-01 e5:
# n_h tripled to 17% of n_c at zero deviance cost). At 2.0 the enhancement
# is ~3x and the spike is gone.
KAPPA_H_BOUNDS = (2.0, 12.0)
HALO_SEED_TEMPERATURE_RATIO = 5.0
HALO_SEED_DENSITY_FRACTION = 0.03
HALO_WIDTH_FLOOR_CORE_FACTOR = 2.0
HALO_DENSITY_CEILING_CORE_FRACTION = 0.2


def _applyHaloSeedsFromCore(params):
    """Re-anchor the halo seeds and boxes on the stage-A fitted core.

    buildParameters can only anchor them on the pre-fit seed estimates; the
    settled-recipe constraints (width >= 2x core, n_h <= 0.2 n_c) are meant
    against the measured core, so they are rebuilt here once stage A is done."""
    from .models import temperatureFromThermalSpeed, thermalSpeedFromTemperature
    coreDensityFit = params['n_c'].value
    vthParFit = params['v_th_par_c'].value
    vthPerpFit = params['v_th_perp_c'].value
    haloVth = thermalSpeedFromTemperature(
        HALO_SEED_TEMPERATURE_RATIO * temperatureFromThermalSpeed(vthParFit))
    parFloor = HALO_WIDTH_FLOOR_CORE_FACTOR * vthParFit
    perpFloor = HALO_WIDTH_FLOOR_CORE_FACTOR * vthPerpFit
    params['n_h'].set(value=HALO_SEED_DENSITY_FRACTION * coreDensityFit,
                      min=1e-5 * coreDensityFit,
                      max=HALO_DENSITY_CEILING_CORE_FRACTION * coreDensityFit)
    params['v_th_par_h'].set(value=max(haloVth, 1.05 * parFloor), min=parFloor,
                             max=max(4 * haloVth, 1.2 * parFloor))
    params['v_th_perp_h'].set(value=max(haloVth, 1.05 * perpFloor), min=perpFloor,
                              max=max(4 * haloVth, 1.2 * perpFloor))
    if HALO_DRIFT_PARAM_NAME in params:
        haloDriftBound = HALO_DRIFT_BOUND_IN_WIDTHS * params['v_th_perp_h'].value
        params[HALO_DRIFT_PARAM_NAME].set(value=0.0, min=-haloDriftBound,
                                          max=haloDriftBound)


def strahlPerpAnchor(pawDeg, driftValue):
    """PAW-derived perpendicular width: tan(half-angle) * |drift|, with the
    half-angle = measured PAW/2 clipped to the production 5-45 deg range."""
    halfAngle = np.radians(np.clip(pawDeg / 2.0, *STRAHL_HALF_ANGLE_CLIP_DEG))
    return float(np.tan(halfAngle) * abs(driftValue))


def paramsToDict(params):
    """Parameter values in ALL_PARAM_NAMES order (u_par_h absent when pinned at 0)."""
    return {name: params[name].value for name in ALL_PARAM_NAMES if name in params}


def _setVary(params, freeNames):
    for name in params:
        params[name].vary = name in freeNames


def _minimise(params, freeNames, residualFunction):
    _setVary(params, freeNames)
    result = lmfit.minimize(residualFunction, params, method='least_squares')
    for name in params:
        params[name].value = result.params[name].value
    return result


def _notifyStage(onStageComplete, stageName, params, pixelMask, freeNames):
    if onStageComplete is not None:
        onStageComplete(stageName, paramsToDict(params), pixelMask.copy(), list(freeNames))


def _applyOverallStage1Seeds(params, seedValues, breakEnergy, strahlSign, pawDeg):
    """Replace the data-derived strahl seeds/boxes with runStrahlStage1's
    fallback settings before stage C (overall stage-1 parity)."""
    from .models import thermalSpeedFromTemperature
    breakSpeed = thermalSpeedFromTemperature(breakEnergy)

    coreDensity = params['n_c'].value
    inputDensity = seedValues.get('n_sc', np.nan)
    densityBase = (inputDensity if np.isfinite(inputDensity) and inputDensity > 0
                   else coreDensity)
    params['n_b'].set(value=min(STAGE_C_OVERALL_SEED_DENSITY_FRACTION * densityBase,
                                0.9 * params['n_b'].max))

    lowFraction, highFraction = STAGE_C_OVERALL_DRIFT_BREAK_FRACTIONS
    driftLow, driftHigh = sorted((strahlSign * lowFraction * breakSpeed,
                                  strahlSign * highFraction * breakSpeed))
    params['u_par_b'].set(
        value=strahlSign * STAGE_C_OVERALL_SEED_DRIFT_BREAK_FRACTION * breakSpeed,
        min=driftLow, max=driftHigh)

    widthLow, widthHigh = STAGE_C_OVERALL_WIDTH_BOUNDS_CMS
    params['v_th_par_b'].set(
        value=np.clip(STAGE_C_OVERALL_SEED_WIDTH_BREAK_FRACTION * breakSpeed,
                      widthLow, widthHigh),
        min=widthLow, max=widthHigh)
    params['kappa_b'].set(value=STAGE_C_OVERALL_SEED_KAPPA)

    if np.isfinite(pawDeg):
        halfAngle = np.radians(np.clip(pawDeg / 2.0, *STRAHL_HALF_ANGLE_CLIP_DEG))
        coneTangent = float(np.tan(halfAngle))
        params['v_th_perp_b'].set(expr=f'{coneTangent:.8e}*abs(u_par_b)')


def runStagedFit(psd, counts, grid, seedValues, buildResidual,
                 onStageComplete=None, freeHaloDrift=True):
    """Run the four stages A (core) -> B (halo) -> C (strahl) -> D (joint).

    freeHaloDrift (default True) adds the halo parallel drift u_par_h (seed 0,
    box +-HALO_DRIFT_BOUND_IN_WIDTHS x the seed halo perp width) to stages B
    and D; False pins it at 0.

    buildResidual(mask) -> function(params) -> residual vector on that mask;
    it defines the objective (see objectives.py) and is the only thing that
    differs between the Poisson and chi-square fits.

    Stage C fits the strahl-side cone at ALL valid energies with core and halo
    frozen (the frozen components bridge the thermal range), from generic
    seeds (u = 0.9 v_break in a [0.3, 3] v_break box, width = 0.5 v_break in
    the absolute [2e7, 1.5e9] box, n_b = 0.01 n_sc, kappa 5); the perp width
    is expr-tied to tan(halfAngle)*|u_par_b| during the stage.

    Break energy: seedValues['break_E_eV'] (per-epoch crossing value) when
    finite — used as-is for every stage, never updated; otherwise the
    7.7 T_c|| proxy, recomputed after stage A.

    onStageComplete(stageName, paramsDict, pixelMask, freeNames), if given, is
    called after each stage ('A_core', 'B_halo', 'C_strahl', 'D_joint') with a
    snapshot of the parameters, the pixels that stage fitted and the names of
    the parameters it varied -- for step-by-step inspection.

    Returns (paramsDict, diagnostics).
    """
    params = buildParameters(seedValues, freeHaloDrift=freeHaloDrift)
    haloNames = [name for name in HALO_PARAM_NAMES
                 if freeHaloDrift or name != HALO_DRIFT_PARAM_NAME]
    valid = grid['valid']
    energy = grid['energy_eV']
    pitch = grid['pitch_angle']
    # strahl_tier from the PAD detector: 'none' -> fit without a strahl
    # component; None (legacy callers without PAD detection) keeps the old
    # always-fit behavior.
    strahlTier = seedValues.get('strahl_tier')
    hasStrahl = strahlTier != 'none'
    strahlSign = seedValues['strahl_sign'] if seedValues['strahl_sign'] != 0 else 1
    strahlCone = (pitch <= STRAHL_CONE_HALF_ANGLE_DEG) if strahlSign > 0 \
        else (pitch >= 180 - STRAHL_CONE_HALF_ANGLE_DEG)
    antiStrahlCone = (pitch >= 180 - STRAHL_CONE_HALF_ANGLE_DEG) if strahlSign > 0 \
        else (pitch <= STRAHL_CONE_HALF_ANGLE_DEG)

    from .models import temperatureFromThermalSpeed
    externalBreak = seedValues.get('break_E_eV', np.nan)
    useExternalBreak = np.isfinite(externalBreak) and externalBreak > 0
    if useExternalBreak:
        breakEnergy = float(externalBreak)
    else:
        breakEnergy = np.clip(
            BREAK_ENERGY_CORE_TEMPERATURE_FACTOR
            * temperatureFromThermalSpeed(seedValues['v_th_par_c']),
            *BREAK_ENERGY_LIMITS_EV)

    # Stage A: core on the thermal band
    coreMask = valid & (energy <= breakEnergy)
    _minimise(params, CORE_PARAM_NAMES, buildResidual(coreMask))
    _notifyStage(onStageComplete, 'A_core', params, coreMask, CORE_PARAM_NAMES)
    if not useExternalBreak:
        breakEnergy = np.clip(
            BREAK_ENERGY_CORE_TEMPERATURE_FACTOR
            * temperatureFromThermalSpeed(params['v_th_par_c'].value),
            *BREAK_ENERGY_LIMITS_EV)

    # Stage B: halo above the break, outside both strahl cones, seeds and
    # boxes re-anchored on the stage-A fitted core.
    _applyHaloSeedsFromCore(params)
    haloMask = valid & (energy > breakEnergy) & ~strahlCone & ~antiStrahlCone
    _minimise(params, haloNames, buildResidual(haloMask))
    _notifyStage(onStageComplete, 'B_halo', params, haloMask, haloNames)

    if hasStrahl:
        # Stage C: strahl with core+halo frozen. With a measured PAD width the
        # perp width stays tied to its anchor (overall stage-1 parity: 4 free
        # strahl params, perp width follows tan(PAW/2)*|u|).
        pawDeg = seedValues.get('strahl_paw_deg', np.nan)
        usePawAnchor = np.isfinite(pawDeg)
        stageCNames = ([name for name in STRAHL_PARAM_NAMES
                        if name != 'v_th_perp_b'] if usePawAnchor
                       else STRAHL_PARAM_NAMES)
        strahlMask = valid & strahlCone
        _applyOverallStage1Seeds(params, seedValues, breakEnergy,
                                 strahlSign, pawDeg)
        if strahlMask.sum() >= MIN_STRAHL_STAGE_PIXELS:
            _minimise(params, stageCNames, buildResidual(strahlMask))
            _notifyStage(onStageComplete, 'C_strahl', params, strahlMask, stageCNames)
            if usePawAnchor:
                params['v_th_perp_b'].set(expr='')
                # Overall joint parity: drift box tightens to +-20% around the
                # stage-C value and the perp width re-anchors to it.
                stageCDrift = params['u_par_b'].value
                driftLow, driftHigh = sorted(
                    (stageCDrift * (1 - STRAHL_JOINT_DRIFT_BOX_FRACTION),
                     stageCDrift * (1 + STRAHL_JOINT_DRIFT_BOX_FRACTION)))
                params['u_par_b'].set(min=driftLow, max=driftHigh)
                anchor = strahlPerpAnchor(pawDeg, stageCDrift)
                params['v_th_perp_b'].set(
                    value=anchor, min=STRAHL_PERP_WIDTH_BOX[0] * anchor,
                    max=STRAHL_PERP_WIDTH_BOX[1] * anchor)
        elif usePawAnchor:
            # Stage C skipped for lack of pixels: drop the expr tie so the
            # joint stage can vary the perp width normally.
            params['v_th_perp_b'].set(expr='')
        jointNames = CORE_PARAM_NAMES + haloNames + STRAHL_PARAM_NAMES
    else:
        # No strahl detected by the PAD two-tier criterion: lock the
        # component at exactly zero and fit core + halo only.
        params['n_b'].value = 0.0
        jointNames = CORE_PARAM_NAMES + haloNames

    # Stage D: joint refinement
    jointResult = _minimise(params, jointNames, buildResidual(valid))
    _notifyStage(onStageComplete, 'D_joint', params, valid, jointNames)

    diagnostics = {'break_energy_eV': float(breakEnergy),
                   'break_source': 'crossing' if useExternalBreak else 'proxy_7.7T',
                   # Effective boxes of the stage-D free parameters (halo boxes
                   # re-anchored on the fitted core, strahl drift locked +-20%):
                   # the seed-time buildParameters boxes are stale for these.
                   'param_bounds': {name: (float(params[name].min),
                                           float(params[name].max))
                                    for name in jointNames},
                   'strahl_sign': int(strahlSign) if hasStrahl else 0,
                   'strahl_tier': strahlTier,
                   'free_params': len(jointNames),
                   'joint_nfev': int(jointResult.nfev),
                   'joint_success': bool(jointResult.success)}
    return paramsToDict(params), diagnostics


def evaluateModel(paramsDict, grid):
    model = coreHaloStrahlModel(grid['v_par'], grid['v_perp'], paramsDict)
    return np.maximum(model, MODEL_PSD_FLOOR)
