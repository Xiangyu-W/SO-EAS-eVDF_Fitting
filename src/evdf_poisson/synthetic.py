"""Synthetic EAS eVDFs on a real instrument grid with the exact EAS noise model:

    counts ~ Poisson(truth_psd / one_count_psd),   observed_psd = counts * one_count_psd.

Used to test objectives against a known truth: the same grid, the same
one-count field and the same zero-count censoring as the real data, at any
brightness (count level) you choose.
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .epoch_io import loadEpoch
from .models import (coreHaloStrahlModel, syntheticTruthModel,
                     thermalSpeedFromTemperature)
from .seeds import seedFromData
from .objectives import fitEpoch
from .metrics import poissonDeviancePerDof, reducedChiSquare, componentMomentsFromParams

DEFAULT_BRIGHTNESS_LEVELS = (1.0, 0.3, 0.1)
DEFAULT_OBJECTIVES = ('poisson', 'chi2')


def loadInstrumentGrid(chunkPath, epochIndex=0):
    """Velocity grid + one-count PSD field of one real epoch (CGS)."""
    grid, _, _, _, _ = loadEpoch(chunkPath, epochIndex)
    return grid


def truthParamsFromPhysical(row):
    """Physical description (densities cm^-3, temperatures eV, drifts cm/s)
    -> model parameter dict (CGS thermal speeds)."""
    return {'n_c': row['n_c'], 'u_par_c': row['u_par_c'],
            'v_th_par_c': thermalSpeedFromTemperature(row['T_par_c_eV']),
            'v_th_perp_c': thermalSpeedFromTemperature(row['T_perp_c_eV']),
            'n_h': row['n_h'],
            'v_th_par_h': thermalSpeedFromTemperature(row['T_par_h_eV']),
            'v_th_perp_h': thermalSpeedFromTemperature(row['T_perp_h_eV']),
            'kappa_h': row['kappa_h'],
            'n_b': row['n_b'], 'u_par_b': row['u_par_b'],
            'v_th_par_b': thermalSpeedFromTemperature(row['T_par_b_eV']),
            'v_th_perp_b': thermalSpeedFromTemperature(row['T_perp_b_eV']),
            'kappa_b': row['kappa_b']}


def realiseSyntheticVdf(truthParams, grid, rng, brightness=1.0,
                        truthModel=coreHaloStrahlModel):
    """Truth PSD -> expected counts (scaled by ``brightness``) -> Poisson draw
    -> observed PSD (CGS).

    brightness < 1 emulates a fainter plasma / shorter exposure: the truth PSD
    is unchanged, only the expected counts shrink, so the same distribution is
    sampled at a lower count level. truthModel defaults to the fit model
    itself (no model mismatch); syntheticTruthModel (non-truncated strahl) is
    the deliberately mismatched alternative."""
    truthPsd = truthModel(grid['v_par'], grid['v_perp'], truthParams)
    expectedCounts = np.where(grid['valid'],
                              brightness * truthPsd / grid['one_count_psd'], 0.0)
    observedCounts = rng.poisson(expectedCounts).astype(float)
    observedPsd = observedCounts * grid['one_count_psd'] / brightness
    return {'truth_params': dict(truthParams), 'truth_psd': truthPsd,
            'expected_counts': expectedCounts,
            'counts': observedCounts, 'psd': observedPsd,
            'one_count_psd': grid['one_count_psd'] / brightness}


def gridWithBrightness(grid, brightness):
    """Copy of ``grid`` whose one-count PSD reflects the brightness scaling,
    so a fit of a synthetic realisation converts PSD <-> counts consistently."""
    scaled = dict(grid)
    scaled['one_count_psd'] = grid['one_count_psd'] / brightness
    return scaled


def _fitSyntheticCase(task):
    """Worker: one synthetic realisation fitted with one objective -> result row.

    The random draw depends on (baseSeed, brightness, realisation) only, so
    every objective fits exactly the same synthetic counts (paired test)."""
    (chunkPath, epochIndex, truthParams, brightness, realisation, objective,
     strahlDetection, breakEnergyEv, baseSeed, truthModelName) = task
    grid = loadInstrumentGrid(chunkPath, epochIndex)
    rng = np.random.default_rng([baseSeed, int(round(brightness * 1e6)), realisation])
    truthModel = syntheticTruthModel if truthModelName == 'mismatched' else coreHaloStrahlModel
    sample = realiseSyntheticVdf(truthParams, grid, rng, brightness, truthModel=truthModel)
    scaledGrid = gridWithBrightness(grid, brightness)
    truthDensity = truthParams['n_c'] + truthParams['n_h'] + truthParams.get('n_b', 0.0)
    seeds = seedFromData(sample['psd'], sample['counts'], scaledGrid,
                         strahlDetection=strahlDetection, scDensity=truthDensity,
                         breakEnergyEv=breakEnergyEv)
    params, modelPsd, diagnostics = fitEpoch(sample['psd'], sample['counts'], scaledGrid,
                                             seeds, objective=objective)
    row = {'brightness': brightness, 'realisation': realisation, 'objective': objective,
           'total_counts': float(sample['counts'][scaledGrid['valid']].sum()),
           'zero_count_fraction': float(np.mean(sample['counts'][scaledGrid['valid']] == 0)),
           'deviance_per_dof': poissonDeviancePerDof(modelPsd, sample['counts'], scaledGrid,
                                                     diagnostics['free_params']),
           'red_chi2': reducedChiSquare(modelPsd, sample['psd'], sample['counts'],
                                        scaledGrid, diagnostics['free_params'])}
    row.update({f'fit_{name}': value
                for name, value in componentMomentsFromParams(params).items()})
    return row


def runSyntheticBenchmark(chunkPath, epochIndex, truthParams, strahlDetection,
                          breakEnergyEv, brightnessLevels=DEFAULT_BRIGHTNESS_LEVELS,
                          numRealisations=4, objectives=DEFAULT_OBJECTIVES,
                          workers=4, baseSeed=2026, truthModelName='same'):
    """Fit Poisson realisations of a known truth at several count levels with
    every objective; returns one row per (brightness, realisation, objective)
    with the recovered physical parameters (fit_*) next to the truth (truth_*).

    truthModelName 'same' draws the counts from the fit model itself (no model
    mismatch, so differences come from the objective alone); 'mismatched'
    uses syntheticTruthModel (non-truncated strahl)."""
    tasks = [(str(chunkPath), epochIndex, dict(truthParams), float(brightness), realisation,
              objective, dict(strahlDetection), breakEnergyEv, baseSeed, truthModelName)
             for brightness in brightnessLevels
             for realisation in range(numRealisations)
             for objective in objectives]
    spawnContext = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks)),
                             mp_context=spawnContext) as pool:
        rows = list(pool.map(_fitSyntheticCase, tasks))
    frame = pd.DataFrame(rows)
    for name, value in componentMomentsFromParams(truthParams).items():
        frame[f'truth_{name}'] = value
    return frame
