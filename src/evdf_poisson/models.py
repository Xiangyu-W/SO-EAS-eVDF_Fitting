"""Analytic eVDF component models in CGS units (v in cm/s, f in s^3/cm^6, n in cm^-3).

Conventions follow the legacy pipeline (Core_fitting / Halo_fitting / Beam_fitting):
thermal speeds use the (kappa - 3/2) convention so T_eV maps to v_th independent of kappa.
"""

import numpy as np
from scipy.special import gamma

ELECTRON_MASS_G = 9.1093837e-28
ERG_PER_EV = 1.602176634e-12
# The legacy pipeline uses D = 10; 5 removes the flat step at the core/strahl
# junction (the D=10 anti-beam side falls too fast to bridge the handover) at
# zero deviance cost (2026-08 truncation scan over D in {10..3}).
STRAHL_TRUNCATION_D = 5


def thermalSpeedFromTemperature(temperature_eV):
    """v_th = sqrt(2 k T / m) in cm/s from T in eV."""
    return np.sqrt(2 * temperature_eV * ERG_PER_EV / ELECTRON_MASS_G)


def temperatureFromThermalSpeed(v_th):
    return ELECTRON_MASS_G * v_th ** 2 / (2 * ERG_PER_EV)


def biMaxwellian(v_par, v_perp, n, u_par, v_th_par, v_th_perp):
    prefactor = n / (np.pi ** 1.5 * v_th_par * v_th_perp ** 2)
    return prefactor * np.exp(-((v_par - u_par) ** 2) / v_th_par ** 2
                              - (v_perp ** 2) / v_th_perp ** 2)


def biKappa(v_par, v_perp, n, u_par, v_th_par, v_th_perp, kappa):
    prefactor = n / (np.pi ** 1.5 * v_th_par * v_th_perp ** 2)
    gammaRatio = gamma(kappa + 1) / (gamma(kappa - 0.5) * (kappa - 1.5) ** 1.5)
    bracket = (1 + ((v_par - u_par) ** 2) / (v_th_par ** 2 * (kappa - 1.5))
               + (v_perp ** 2) / (v_th_perp ** 2 * (kappa - 1.5)))
    return prefactor * gammaRatio * bracket ** (-kappa - 1)


def strahlTruncatedBiKappa(v_par, v_perp, n, u_par, v_th_par, v_th_perp, kappa):
    """Legacy combined_beam form: non-truncated on the beam side of u_par,
    parallel width compressed by sqrt(D) on the opposite side (D = STRAHL_TRUNCATION_D)."""
    truncationD = STRAHL_TRUNCATION_D
    prefactor = (2 * np.sqrt(truncationD) / np.sqrt(truncationD + 1)) * n \
        / (np.pi ** 1.5 * v_th_par * v_th_perp ** 2)
    gammaRatio = gamma(kappa + 1) / (gamma(kappa - 0.5) * (kappa - 1.5) ** 1.5)

    def bracketPower(parallelCoeff):
        bracket = (1 + parallelCoeff * ((v_par - u_par) ** 2) / (v_th_par ** 2 * (kappa - 1.5))
                   + (v_perp ** 2) / (v_th_perp ** 2 * (kappa - 1.5)))
        return bracket ** (-kappa - 1)

    openSide = bracketPower(1.0)
    truncatedSide = bracketPower(truncationD)
    if u_par < 0:
        return prefactor * gammaRatio * np.where(v_par < u_par, openSide, truncatedSide)
    return prefactor * gammaRatio * np.where(v_par >= u_par, openSide, truncatedSide)


def coreHaloStrahlModel(v_par, v_perp, params):
    """Full fit model: bi-Maxwellian core + bi-kappa halo + truncated bi-kappa strahl.

    params: dict with keys n_c,u_par_c,v_th_par_c,v_th_perp_c,
    n_h,v_th_par_h,v_th_perp_h,kappa_h, n_b,u_par_b,v_th_par_b,v_th_perp_b,kappa_b,
    optionally u_par_h (the halo parallel drift; 0 when absent, i.e. for a fit
    run with freeHaloDrift=False).
    """
    total = biMaxwellian(v_par, v_perp, params['n_c'], params['u_par_c'],
                         params['v_th_par_c'], params['v_th_perp_c'])
    total = total + biKappa(v_par, v_perp, params['n_h'], params.get('u_par_h', 0.0),
                            params['v_th_par_h'], params['v_th_perp_h'], params['kappa_h'])
    if params.get('n_b', 0) > 0:
        total = total + strahlTruncatedBiKappa(
            v_par, v_perp, params['n_b'], params['u_par_b'],
            params['v_th_par_b'], params['v_th_perp_b'], params['kappa_b'])
    return total


def syntheticTruthModel(v_par, v_perp, params):
    """Ground-truth generator model, deliberately different from the fit model:
    bi-Maxwellian core + bi-kappa halo + NON-truncated drifting bi-kappa strahl."""
    total = biMaxwellian(v_par, v_perp, params['n_c'], params['u_par_c'],
                         params['v_th_par_c'], params['v_th_perp_c'])
    total = total + biKappa(v_par, v_perp, params['n_h'], 0.0,
                            params['v_th_par_h'], params['v_th_perp_h'], params['kappa_h'])
    if params.get('n_b', 0) > 0:
        total = total + biKappa(v_par, v_perp, params['n_b'], params['u_par_b'],
                                params['v_th_par_b'], params['v_th_perp_b'], params['kappa_b'])
    return total
