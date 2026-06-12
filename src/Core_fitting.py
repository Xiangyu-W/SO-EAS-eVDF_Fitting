# src/corehalo_fitting.py
import numpy as np
from utils import eV_to_vel
from constant import PixelCountThreshold_core
import lmfit

stderr_fact = 0.05 # Estimate the error of fitted parameters if their std is 0
# kw_ampgo = {'local_opts':{'maxiter':2000, 'ftol':1e-27},
#                         'totaliter':50,
#                         'tabulistsize': 8,
#                         'eps1': 1e-27, 'eps2': 0.01, 'glbtol': 1e-28, 
#                         'disp':False}
# kw_ampgo = {'disp':True}

def set_core_initial_params(par_strahl_cond, anti_par_strahl_cond, n_in, corePSD_max):
    """
    Set initial values for core and halo parameters based on conditions.
    Returns updated core_init, halo_init, core_const, halo_const.
    """
    n_core_ratio = 0.8
    core_init = [n_in * n_core_ratio, 0, 10**8, 10**8]
    core_const = [[0, n_in*2], [-2e8, 2e8], [-8e8, 8e8], [-8e8, 8e8]] # n, u_par(-3,3eV), v_th_par(-35,35eV), v_th_perp

    # Compute v_thermal_core_init based on corePSD_max
    v_th_core_init = ((core_init[0]/((np.pi**(3/2))*corePSD_max ))**(1/3))
    core_init[2], core_init[3] = v_th_core_init, v_th_core_init

    # Set core velocity based on beam location
    if not anti_par_strahl_cond and not par_strahl_cond:
        core_init[1] = 0.0 # u_par
    elif not anti_par_strahl_cond and par_strahl_cond:
        core_init[1] = -1 * eV_to_vel(2)
        core_const[1] = [-2e8, 0]
    elif anti_par_strahl_cond and not par_strahl_cond:
        core_init[1] = eV_to_vel(2)
        core_const[1] = [0, 2e8]
    else:  # anti_par_strahl_cond and par_strahl_cond both True
        core_init[1] = 0.0

    return core_init, core_const

def perform_core_fit(pa_vec, low_c, high_c, core_init, core_const, v_par_mesh, v_perp_mesh):
    """
    Perform the core fitting using the specified energy ranges and initial conditions.
    pa_vec is indexed as pa_vec[energy, bin, [PA, PSD, v_par, v_perp, counts]].
    """
    energy_slice = slice(high_c, low_c) 

    # Extract only points with counts >= 0
    countMask_c = np.where(pa_vec[energy_slice, :, 4] >= PixelCountThreshold_core)
    psd_data = pa_vec[energy_slice, :, 1][countMask_c]
    v_par_data = pa_vec[energy_slice, :, 2][countMask_c]
    v_perp_data = pa_vec[energy_slice, :, 3][countMask_c]


    ''' set initial core guess '''
    p_core = lmfit.Parameters()
    p_core.add('n', core_init[0], vary = True, min = core_const[0][0], max = core_const[0][1])
    p_core.add('u_par', core_init[1], vary = True, min = core_const[1][0], max = core_const[1][1])
    p_core.add('v_th_par', core_init[2], vary = True, min = core_const[2][0], max = core_const[2][1])
    p_core.add('v_th_perp', core_init[3], vary = True, min = core_const[3][0], max = core_const[3][1])

    p_list = list(p_core.valuesdict().keys())


    try :
        r_core = core.fit(psd_data, p_core, method = 'leastsq', v_par = v_par_data, v_perp = v_perp_data, max_nfev=5000) #, fit_kws=kw_ampgo
    except ValueError :
        print('Core fitting failed.')
        r_core = None

    def fit_update(p_core, r_core, vary_list, psd_data, v_par, v_perp, core_const) :

        p_core.add('n', r_core.params['n'].value, vary = vary_list[0], min = core_const[0][0], max = core_const[0][1])
        p_core.add('u_par', r_core.params['u_par'].value, vary = vary_list[1], min = core_const[1][0], max = core_const[1][1])
        p_core.add('v_th_par', r_core.params['v_th_par'].value, vary = vary_list[2], min = core_const[2][0], max = core_const[2][1])
        p_core.add('v_th_perp', r_core.params['v_th_perp'].value, vary = vary_list[3], min = core_const[3][0], max = core_const[3][1])

        try :
            r_core = core.fit(psd_data, p_core, method = 'leastsq', v_par = v_par, v_perp = v_perp, max_nfev=5000)
        except ValueError :
            print('Core updating failed.')
            r_core = None

        return r_core

    vary_list = [[True, True, False, False], 
                 [True, False, True, False], 
                 [True, False, False, True], 
                 [True, True, True, True]]

    for i in range(len(vary_list)) :
        if r_core is not None :
            r_core = fit_update(p_core, r_core, vary_list[i], psd_data, v_par_data, v_perp_data, core_const)

    if r_core is not None :
        fit_core = core.eval(r_core.params, v_par = v_par_mesh, v_perp = v_perp_mesh)
        ''' if no errors were generated or if stderr is zero '''
        if r_core.errorbars == False :
            for i in range(len(p_list)) :
                r_core.params[p_list[i]].stderr = stderr_fact*np.abs(r_core.params[p_list[i]].value)

        for i in range(len(p_list)) :
            if r_core.params[p_list[i]].stderr == 0 :
                r_core.params[p_list[i]].stderr = stderr_fact*np.abs(r_core.params[p_list[i]].value)
    else :
        fit_core = core.eval(p_core, v_par = v_par_mesh, v_perp = v_perp_mesh)

    return r_core, fit_core

# Model definition
def bi_Max(v_par, v_perp, n, u_par, v_th_par, v_th_perp) :
    """
    Produces a VDF following a bi-Maxwellian distribution. All in CGS units.

    Parameters
    ----------
    v_par : array
        Velocity in parallel direction.
    v_perp : array
        Velocity in the perpependicular direction.
    n : float
        The plasma density.
    u_par : array
        Bulk velocity in the parallel direction.
    u_perp : array
        Bulk velocity in the perpendicular direction.
    v_th_par : float
        Thermal velocity in parallel direction.
    v_th_perp : float
        Thermal velocity in perpendicular direction.

    Returns
    -------
    f : array
        The Velocity distribution function VDF.

    """

    c_par = v_par - u_par
    c_perp = v_perp

    denominator = (np.pi ** (3/2)) * v_th_par * (v_th_perp**2)
    term1 = n / denominator
    exponent = (c_par/v_th_par)**2 + (c_perp/v_th_perp)**2

    f = term1 * np.exp(- exponent)

    return f

core = lmfit.Model(bi_Max, independent_vars = ['v_par', 'v_perp'], \
    param_names = ['n', 'u_par', 'v_th_par', 'v_th_perp'])

# %%
if __name__ == '__main__':
    core_maxPSD = np.nanmax(pa_vec[high_c:low_c,:,1])
    core_init, core_const= set_core_initial_params(par_strahl_cond, anti_par_strahl_cond, n_in, core_maxPSD)
    r_core, fit_core = perform_core_fit(pa_vec, low_c, high_c, core_init, core_const, v_par_mesh, v_perp_mesh)