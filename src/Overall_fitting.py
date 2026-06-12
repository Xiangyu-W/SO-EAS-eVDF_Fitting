import numpy as np
import lmfit
import copy
from constant import stderr_fact, D_CONST, PixelCountThreshold_all
from scipy.special import gamma
from Beam_fitting import beam, beam_nonTruncated, beam_truncated
from Halo_fitting import bi_kappa_flattop_stepan
from Core_fitting import bi_Max

# stderr_fact = 0.05  # Similar error estimation factor as before
# kw_ampgo = {'local_opts':{'maxiter':5000, 'ftol':1e-30},
#                     'totaliter':100, 'maxiter':20, 
#                     'tabulistsize': 20,
#                     'eps1': 2e-29, 'eps2': 0.1, 'glbtol': 1e-30, 
#                     'disp':False}
# kw_ampgo = {'local_opts':{'maxiter':1000, 'ftol':1e-26},'totaliter':100,'tabulistsize': 8, 'eps1': 2e-26, 'eps2': 0.1, 'glbtol': 1e-26}
kw_ampgo = {'disp':False}
# kw_ampgo = {'local_opts':{'maxiter':2000, 'ftol':1e-8},
#                         'totaliter':50,
#                         'tabulistsize': 8,
#                         'eps1': 1e-6, 'eps2': 0.2, 'glbtol': 1e-8, 
#                         'disp':False}

def param_range(param, stderr_fact=stderr_fact):
    """
    Return [min_val, max_val] for a parameter based on its value and stderr.
    If stderr is None or zero, use a fraction of the value as a fallback.
    """
    val = param.value
    err = param.stderr if param.stderr and param.stderr != 0 and param.stderr >1e-3 else abs(val)*stderr_fact
    return [val - err, val + err]

def update_model_params(source_result, target_model, param_maps, max_percent_diff=0.2):
    """
    Update model parameters based on fitting results.
    Use fitted bounds if they're within max_percent_diff of the value,
    otherwise use standard constraints.
    
    Parameters:
    -----------
    source_result : ModelResult
        The source of parameter values (typically the overall fit result)
    target_model : ModelResult
        The model to update with new parameter values
    param_maps : dict
        Mapping between source and target parameter names: {target_param: source_param}
    max_percent_diff : float, default=0.2
        Maximum allowed percentage difference between bounds and value
    """
    # Standard constraint ranges as fallback
    std_constraints = {
        'n': [0.8, 1.2],            # ±20% for density
        'u_par': [0.8, 1.2],        # ±20% for velocity
        'v_th_par': [0.8, 1.2],     # ±20% for thermal velocity
        'v_th_perp': [0.8, 1.2],    # ±20% for thermal velocity
        'kappa': [0.8, 1.2]         # ±20% for kappa
    }
    
    # For any parameter not in the above list
    default_constraint = [0.8, 1.2]  # ±20% as default
    
    # Update each parameter in the target model
    for target_param, source_param in param_maps.items():
        if source_param in source_result.params:
            # Only update if the parameter exists in both models
            if target_param in target_model.params:
                # Get the fitted value
                fitted_value = source_result.params[source_param].value
                target_model.params[target_param].value = fitted_value
                
                # Check if fitted bounds exist and are within reasonable range
                has_reasonable_bounds = False
                
                if (source_result.params[source_param].min is not None and 
                    source_result.params[source_param].max is not None):
                    
                    fitted_min = source_result.params[source_param].min
                    fitted_max = source_result.params[source_param].max
                    
                    if fitted_min != fitted_max:
                        # Calculate percentage differences
                        diff_percent_of_min = abs(fitted_min - fitted_value) / abs(fitted_value) if fitted_value != 0 else float('inf')
                        diff_percent_of_max = abs(fitted_max - fitted_value) / abs(fitted_value) if fitted_value != 0 else float('inf')
                        
                    # Check if both bounds are within reasonable range
                    if diff_percent_of_min <= max_percent_diff and diff_percent_of_max <= max_percent_diff:
                        has_reasonable_bounds = True
                
                # Set bounds based on our check
                if has_reasonable_bounds:
                    # Use the fitted bounds
                    target_model.params[target_param].min = source_result.params[source_param].min
                    target_model.params[target_param].max = source_result.params[source_param].max
                    print(f"  Using fitted bounds for {target_param}: "
                          f"[{target_model.params[target_param].min:.3e}, "
                          f"{target_model.params[target_param].max:.3e}]")
                else:
                    # Use standard constraints
                    constraint = std_constraints.get(target_param, default_constraint)
                    target_model.params[target_param].min = fitted_value * constraint[0]
                    target_model.params[target_param].max = fitted_value * constraint[1]
                    print(f"  Using standard bounds for {target_param}: "
                          f"[{target_model.params[target_param].min:.3e}, "
                          f"{target_model.params[target_param].max:.3e}]")

def fit_non_beam_vdf(psd_data, psd_unc, v_par, v_perp, n, r_core, r_halo, v_par_mesh, v_perp_mesh):
    """
    Final fitting scenario with no beam components.
    """
    # Create constraints from core and halo fits
    # Extract core parameter ranges
    n_c_range = param_range(r_core.params['n'])
    u_par_c_range = param_range(r_core.params['u_par'])
    v_th_par_c_range = param_range(r_core.params['v_th_par'])
    v_th_perp_c_range = param_range(r_core.params['v_th_perp'])

    # Extract halo parameter ranges
    n_h_range = param_range(r_halo.params['n'])
    u_par_h_range = param_range(r_halo.params['u_par'])
    v_th_par_h_range = param_range(r_halo.params['v_th_par'])
    v_th_perp_h_range = param_range(r_halo.params['v_th_perp'])
    kappa_h_range = param_range(r_halo.params['kappa'])
    # delta_range = param_range(r_halo.params['delta'])
    
    # Beam is absent, set dummy constraints
    n_b_range = [0, 1]
    u_par_b_range = [0, 1]
    v_th_par_b_range = [0, 1]
    v_th_perp_b_range = [0, 1]
    kappa_b_range = [0, 1]
    n_sc_range = [None, None]

    vdf_const = [n_c_range, u_par_c_range, v_th_par_c_range, v_th_perp_c_range,
                 n_b_range, u_par_b_range, v_th_par_b_range, v_th_perp_b_range, kappa_b_range,
                 n_h_range, u_par_h_range, v_th_par_h_range, v_th_perp_h_range, kappa_h_range, n_sc_range]

    p_vdf = lmfit.Parameters()
    p_vdf.add('n_c', r_core.params['n'], vary=True, min=n_c_range[0], max=n_c_range[1])
    p_vdf.add('u_par_c', r_core.params['u_par'], vary=True, min=u_par_c_range[0], max=u_par_c_range[1])
    p_vdf.add('v_th_par_c', r_core.params['v_th_par'], vary=True, min=v_th_par_c_range[0], max=v_th_par_c_range[1])
    p_vdf.add('v_th_perp_c', r_core.params['v_th_perp'], vary=True, min=v_th_perp_c_range[0], max=v_th_perp_c_range[1])
    # No beam
    p_vdf.add('n_b', 0.0, vary=False, min=0, max=1)
    p_vdf.add('u_par_b', 0.0, vary=False, min=0, max=1)
    p_vdf.add('v_th_par_b', 1.0, vary=False, min=0, max=1)
    p_vdf.add('v_th_perp_b', 1.0, vary=False, min=0, max=1)
    p_vdf.add('kappa_b', 1.0, vary=False, min=0, max=1)
    # Halo
    p_vdf.add('n_h', r_halo.params['n'], vary=True, min=n_h_range[0], max=n_h_range[1])
    p_vdf.add('u_par_h', r_halo.params['u_par'], vary=True, min=u_par_h_range[0], max=u_par_h_range[1], expr='(u_par_c*n_c)/n_h')
    p_vdf.add('v_th_par_h', r_halo.params['v_th_par'], vary=True, min=v_th_par_h_range[0], max=v_th_par_h_range[1])
    p_vdf.add('v_th_perp_h', r_halo.params['v_th_perp'], vary=True, min=v_th_perp_h_range[0], max=v_th_perp_h_range[1])
    p_vdf.add('kappa_h', r_halo.params['kappa'], vary=True, min=kappa_h_range[0], max=kappa_h_range[1])
    # p_vdf.add('delta', r_halo.params['delta'], vary=True, min=5, max=12)
    p_vdf.add('n_sc', n, vary=False, min=None, max=None)

    try:
        r_vdf = vdf_1beam.fit(psd_data, p_vdf, method='ampgo', v_par=v_par, v_perp=v_perp) # leastsq, max_nfev=300
    except (ValueError, TypeError):
        r_vdf = None
        print('VDF_noBeam final fitting failed.')

    if r_vdf is not None :
        # print(r_vdf.chisqr)
        fit_vdf = vdf_1beam.eval(r_vdf.params, v_par = v_par_mesh, v_perp = v_perp_mesh)
    else:
        fit_vdf = vdf_1beam.eval(p_vdf, v_par = v_par_mesh, v_perp = v_perp_mesh)
        # r_vdf.params = p_vdf

    return r_vdf, fit_vdf

def fit_one_beam_vdf(psd_data, psd_unc, v_par, v_perp, n, r_core, r_halo, r_beam, v_par_mesh, v_perp_mesh):
    """
    Final fitting scenario with one beam component.
    """
    # Create constraints from core and halo fits
    # Extract core parameter ranges
    n_c_range = param_range(r_core.params['n'])
    u_par_c_range = param_range(r_core.params['u_par'])
    v_th_par_c_range = param_range(r_core.params['v_th_par'])
    v_th_perp_c_range = param_range(r_core.params['v_th_perp'])

    # Extract halo parameter ranges
    n_h_range = param_range(r_halo.params['n'])
    u_par_h_range = param_range(r_halo.params['u_par'])
    v_th_par_h_range = param_range(r_halo.params['v_th_par'])
    v_th_perp_h_range = param_range(r_halo.params['v_th_perp'])
    kappa_h_range = param_range(r_halo.params['kappa'])
    # delta_range = param_range(r_halo.params['delta'])

    # Beam
    if r_beam is not None:
        n_b_range = param_range(r_beam.params['n'])
        u_par_b_range = param_range(r_beam.params['u_par'])
        v_th_par_b_range = param_range(r_beam.params['v_th_par'])
        v_th_perp_b_range = param_range(r_beam.params['v_th_perp'])
        kappa_b_range = param_range(r_beam.params['kappa'])
    else:
        n_b_range = [0, 1]
        u_par_b_range = [0, 1]
        v_th_par_b_range = [0, 1]
        v_th_perp_b_range = [0, 1]
        kappa_b_range = [0, 1]
    n_sc_range = [None, None]

    # psd_unc_safe = np.copy(psd_unc)
    # psd_unc_safe[psd_unc_safe == 0] = 0.1*np.min(psd_unc[psd_unc > 0]) if np.any(psd_unc > 0) else 1e-10
    # weight_set = 1.0/psd_unc_safe # np.mean(psd_data)#np.median(psd_uncertainty)

    vdf_const = [n_c_range, u_par_c_range, v_th_par_c_range, v_th_perp_c_range,
                 n_b_range, u_par_b_range, v_th_par_b_range, v_th_perp_b_range, kappa_b_range,
                 n_h_range, u_par_h_range, v_th_par_h_range, v_th_perp_h_range, kappa_h_range, n_sc_range]
    
    p_vdf = lmfit.Parameters()
    p_vdf.add('n_c', r_core.params['n'], vary=True, min=n_c_range[0], max=n_c_range[1])
    p_vdf.add('u_par_c', r_core.params['u_par'], vary=True, min=u_par_c_range[0], max=u_par_c_range[1])
    p_vdf.add('v_th_par_c', r_core.params['v_th_par'], vary=True, min=v_th_par_c_range[0], max=v_th_par_c_range[1])
    p_vdf.add('v_th_perp_c', r_core.params['v_th_perp'], vary=True, min=v_th_perp_c_range[0], max=v_th_perp_c_range[1])
    # Beam
    if r_beam is not None :
        p_vdf.add('n_b', r_beam.params['n'], vary = True, min = n_b_range[0], max = n_b_range[1])
        p_vdf.add('u_par_b', r_beam.params['u_par'], vary = True, min = u_par_b_range[0], max = u_par_b_range[1])
        p_vdf.add('v_th_par_b', r_beam.params['v_th_par'], vary = True, min = v_th_par_b_range[0], max = v_th_par_b_range[1])
        p_vdf.add('v_th_perp_b', r_beam.params['v_th_perp'], vary = True, min = v_th_perp_b_range[0], max = v_th_perp_b_range[1])
        p_vdf.add('kappa_b', r_beam.params['kappa'], vary = True, min = kappa_b_range[0], max = kappa_b_range[1])
    else :
        p_vdf.add('n_b', 0.0, vary=False, min=0, max=1)
        p_vdf.add('u_par_b', 0.0, vary=False, min=0, max=1)
        p_vdf.add('v_th_par_b', 1.0, vary=False, min=0, max=1)
        p_vdf.add('v_th_perp_b', 1.0, vary=False, min=0, max=1)
        p_vdf.add('kappa_b', 3.0, vary=False, min=0, max=1)
    # Halo
    p_vdf.add('n_h', r_halo.params['n'], vary=True, min=n_h_range[0], max=n_h_range[1])
    p_vdf.add('u_par_h', r_halo.params['u_par'], vary=True, min=u_par_h_range[0], max=u_par_h_range[1], expr='(u_par_c*n_c)/n_h')
    p_vdf.add('v_th_par_h', r_halo.params['v_th_par'], vary=True, min=v_th_par_h_range[0], max=v_th_par_h_range[1])
    p_vdf.add('v_th_perp_h', r_halo.params['v_th_perp'], vary=True, min=v_th_perp_h_range[0], max=v_th_perp_h_range[1])
    p_vdf.add('kappa_h', r_halo.params['kappa'], vary=True, min=kappa_h_range[0], max=kappa_h_range[1])
    # p_vdf.add('delta', r_halo.params['delta'], vary=True, min=5, max=12)
    p_vdf.add('n_sc', n, vary=False, min=None, max=None)

    try :
        r_vdf = vdf_1beam.fit(psd_data, p_vdf, method = 'ampgo', v_par = v_par, v_perp = v_perp,
                              fit_kws=kw_ampgo, max_nfev=50000) #, fit_kws=kw_ampgo,basinhopping, dual_annealing, ampgo , max_nfev=500
    except (ValueError, TypeError) :
        r_vdf = None
        print('VDF_oneBeam final fitting failed.')
    
    if r_vdf is not None :
        fit_vdf = vdf_1beam.eval(r_vdf.params, v_par = v_par_mesh, v_perp = v_perp_mesh)
    else:
        fit_vdf = vdf_1beam.eval(p_vdf, v_par = v_par_mesh, v_perp = v_perp_mesh)
        # r_vdf.params = p_vdf


    return r_vdf, fit_vdf

def fit_two_beam_vdf(psd_data, psd_unc, v_par, v_perp, n, r_core, r_halo, r_beam_par, r_beam_anti_par, v_par_mesh, v_perp_mesh):
    """
    Final fitting scenario with two beam components.
    """
    # Extract core parameter ranges
    n_c_range = param_range(r_core.params['n'])
    u_par_c_range = param_range(r_core.params['u_par'])
    v_th_par_c_range = param_range(r_core.params['v_th_par'])
    v_th_perp_c_range = param_range(r_core.params['v_th_perp'])

    # Extract halo parameter ranges
    n_h_range = param_range(r_halo.params['n'])
    u_par_h_range = param_range(r_halo.params['u_par'])
    v_th_par_h_range = param_range(r_halo.params['v_th_par'])
    v_th_perp_h_range = param_range(r_halo.params['v_th_perp'])
    kappa_h_range = param_range(r_halo.params['kappa'])
    # delta_range = param_range(r_halo.params['delta'])
    

    # Set 2 beams
    if r_beam_par is not None:
        n_b_par_range = param_range(r_beam_par.params['n'])
        u_par_b_par_range = param_range(r_beam_par.params['u_par'])
        v_th_par_b_par_range = param_range(r_beam_par.params['v_th_par'])
        v_th_perp_b_par_range = param_range(r_beam_par.params['v_th_perp'])
        kappa_b_par_range = param_range(r_beam_par.params['kappa'])
    else:
        n_b_par_range = [0, 1]
        u_par_b_par_range = [0, 1]
        v_th_par_b_par_range = [0, 1]
        v_th_perp_b_par_range = [0, 1]
        kappa_b_par_range = [0, 1]
    if r_beam_anti_par is not None:
        n_b_anti_par_range = param_range(r_beam_anti_par.params['n'])
        u_par_b_anti_par_range = param_range(r_beam_anti_par.params['u_par'])
        v_th_par_b_anti_par_range = param_range(r_beam_anti_par.params['v_th_par'])
        v_th_perp_b_anti_par_range = param_range(r_beam_anti_par.params['v_th_perp'])
        kappa_b_anti_par_range = param_range(r_beam_anti_par.params['kappa'])
    else:
        n_b_anti_par_range = [0, 1]
        u_par_b_anti_par_range = [0, 1]
        v_th_par_b_anti_par_range = [0, 1]
        v_th_perp_b_anti_par_range = [0, 1]
        kappa_b_anti_par_range = [0, 1]
    n_sc_range = [None, None]

    # psd_unc_safe = np.copy(psd_unc)
    # psd_unc_safe[psd_unc_safe == 0] = 0.1*np.min(psd_unc[psd_unc > 0]) if np.any(psd_unc > 0) else 1e-10
    # weight_set = 1.0/psd_unc_safe # np.mean(psd_data)#np.median(psd_uncertainty)

    vdf_const = [n_c_range, u_par_c_range, v_th_par_c_range, v_th_perp_c_range,
                 n_b_par_range, u_par_b_par_range, v_th_par_b_par_range, v_th_perp_b_par_range, kappa_b_par_range,
                 n_b_anti_par_range, u_par_b_anti_par_range, v_th_par_b_anti_par_range, v_th_perp_b_anti_par_range, kappa_b_anti_par_range,
                 n_h_range, u_par_h_range, v_th_par_h_range, v_th_perp_h_range, kappa_h_range, n_sc_range]
    
    p_vdf = lmfit.Parameters()
    p_vdf.add('n_c', r_core.params['n'], vary=True, min=n_c_range[0], max=n_c_range[1])
    p_vdf.add('u_par_c', r_core.params['u_par'], vary=True, min=u_par_c_range[0], max=u_par_c_range[1])
    p_vdf.add('v_th_par_c', r_core.params['v_th_par'], vary=True, min=v_th_par_c_range[0], max=v_th_par_c_range[1])
    p_vdf.add('v_th_perp_c', r_core.params['v_th_perp'], vary=True, min=v_th_perp_c_range[0], max=v_th_perp_c_range[1])
    # Parallel Beam
    if r_beam_par is not None :
        p_vdf.add('n_b_par', r_beam_par.params['n'], vary = True, min = n_b_par_range[0], max = n_b_par_range[1])
        p_vdf.add('u_par_b_par', r_beam_par.params['u_par'], vary = True, min = u_par_b_par_range[0], max = u_par_b_par_range[1])
        p_vdf.add('v_th_par_b_par', r_beam_par.params['v_th_par'], vary = True, min = v_th_par_b_par_range[0], max = v_th_par_b_par_range[1])
        p_vdf.add('v_th_perp_b_par', r_beam_par.params['v_th_perp'], vary = True, min = v_th_perp_b_par_range[0], max = v_th_perp_b_par_range[1])
        p_vdf.add('kappa_b_par', r_beam_par.params['kappa'], vary = True, min = kappa_b_par_range[0], max = kappa_b_par_range[1])
    else :
        p_vdf.add('n_b_par', 0.0, vary=False, min=0, max=1)
        p_vdf.add('u_par_b_par', 0.0, vary=False, min=0, max=1)
        p_vdf.add('v_th_par_b_par', 1.0, vary=False, min=0, max=1)
        p_vdf.add('v_th_perp_b_par', 1.0, vary=False, min=0, max=1)
        p_vdf.add('kappa_b_par', 10.0, vary=False, min=0, max=1)
    # AntiParallel Beam
    if r_beam_anti_par is not None :
        p_vdf.add('n_b_anti_par', r_beam_anti_par.params['n'], vary = True, min = n_b_anti_par_range[0], max = n_b_anti_par_range[1])
        p_vdf.add('u_par_b_anti_par', r_beam_anti_par.params['u_par'], vary = True, min = u_par_b_anti_par_range[0], max = u_par_b_anti_par_range[1])
        p_vdf.add('v_th_par_b_anti_par', r_beam_anti_par.params['v_th_par'], vary = True, min = v_th_par_b_anti_par_range[0], max = v_th_par_b_anti_par_range[1])
        p_vdf.add('v_th_perp_b_anti_par', r_beam_anti_par.params['v_th_perp'], vary = True, min = v_th_perp_b_anti_par_range[0], max = v_th_perp_b_anti_par_range[1])
        p_vdf.add('kappa_b_anti_par', r_beam_anti_par.params['kappa'], vary = True, min = kappa_b_anti_par_range[0], max = kappa_b_anti_par_range[1])
    else :
        p_vdf.add('n_b_anti_par', 0.0, vary=False, min=0, max=1)
        p_vdf.add('u_par_b_anti_par', 0.0, vary=False, min=0, max=1)
        p_vdf.add('v_th_par_b_anti_par', 1.0, vary=False, min=0, max=1)
        p_vdf.add('v_th_perp_b_anti_par', 1.0, vary=False, min=0, max=1)
        p_vdf.add('kappa_b_anti_par', 10.0, vary=False, min=0, max=1)
    
    # Halo
    p_vdf.add('n_h', r_halo.params['n'], vary=True, min=n_h_range[0], max=n_h_range[1])
    p_vdf.add('u_par_h', r_halo.params['u_par'], vary=True, min=u_par_h_range[0], max=u_par_h_range[1], expr='(u_par_c*n_c)/n_h')
    p_vdf.add('v_th_par_h', r_halo.params['v_th_par'], vary=True, min=v_th_par_h_range[0], max=v_th_par_h_range[1])
    p_vdf.add('v_th_perp_h', r_halo.params['v_th_perp'], vary=True, min=v_th_perp_h_range[0], max=v_th_perp_h_range[1])
    p_vdf.add('kappa_h', r_halo.params['kappa'], vary=True, min=kappa_h_range[0], max=kappa_h_range[1])
    # p_vdf.add('delta', r_halo.params['delta'], vary=True, min=5, max=12)
    p_vdf.add('n_sc', n, vary=False, min=None, max=None)

    try :
        r_vdf = vdf_2beam.fit(psd_data, p_vdf, method = 'ampgo', v_par = v_par, v_perp = v_perp, fit_kws=kw_ampgo, max_nfev=50000)#, max_nfev=300)
    except (ValueError, TypeError) :
        r_vdf = None
        print('VDF_twoBeam final fitting failed.')
    
    if r_vdf is not None :
        fit_vdf = vdf_2beam.eval(r_vdf.params, v_par = v_par_mesh, v_perp = v_perp_mesh)
    else:
        fit_vdf = vdf_2beam.eval(p_vdf, v_par = v_par_mesh, v_perp = v_perp_mesh)
        # r_vdf.params = p_vdf

    return r_vdf, fit_vdf


def perform_final_vdf_fit(pa_vec, unc_pa_vec, low_c, n_in, r_core, r_halo, r_beam_par, r_beam_anti_par,
                          anti_par_strahl_cond, par_strahl_cond, v_par_mesh, v_perp_mesh):
    """
    Higher-level function that decides which final VDF fitting function to call based on conditions.
    """
    # All data points 
    countMask_all = (pa_vec[:low_c, :, 4] >=PixelCountThreshold_all) #np.where((pa_vec[:low_c, :, 1] >= 1e-27)&(pa_vec[:low_c, :, 4] >=2))
    psd_masked = pa_vec[:low_c,:,1][countMask_all]
    v_par_masked = pa_vec[:low_c,:,2][countMask_all]
    v_perp_masked = pa_vec[:low_c,:,3][countMask_all]
    psd_unc_masked = unc_pa_vec[:low_c,:][countMask_all]

    # Decide which fitting function to call
    if not anti_par_strahl_cond and not par_strahl_cond:
        r_vdf, fit_vdf = fit_non_beam_vdf(psd_masked, psd_unc_masked, v_par_masked, v_perp_masked, n_in, r_core, r_halo, v_par_mesh, v_perp_mesh)
    elif not anti_par_strahl_cond and par_strahl_cond:
        r_vdf, fit_vdf = fit_one_beam_vdf(psd_masked, psd_unc_masked, v_par_masked, v_perp_masked, n_in, r_core, r_halo, r_beam_par, v_par_mesh, v_perp_mesh)
    elif anti_par_strahl_cond and not par_strahl_cond:
        r_vdf, fit_vdf = fit_one_beam_vdf(psd_masked, psd_unc_masked, v_par_masked, v_perp_masked, n_in, r_core, r_halo, r_beam_anti_par, v_par_mesh, v_perp_mesh)
    else:  # anti_par_strahl_cond and par_strahl_cond both True
        r_vdf, fit_vdf = fit_two_beam_vdf(psd_masked, psd_unc_masked, v_par_masked, v_perp_masked, n_in, r_core, r_halo, r_beam_par, r_beam_anti_par, v_par_mesh, v_perp_mesh)

    return r_vdf, fit_vdf


def iterative_final_vdf_fit(pa_vec, unc_pa_vec, high_h, low_c, n_in, r_core, r_halo, r_beam_par, r_beam_anti_par, beam_data,
                            anti_par_strahl_cond, par_strahl_cond, v_par_mesh, v_perp_mesh,
                            chi_target_range=(0.6, 1.1), max_iterations=20, consecutive_inRange_num=5):
    """
    Perform iterative final VDF fitting until the reduced chi-square over the final fit
    """
    def combined_score(redChi_overall, redChi_beam=None):
        # Euclidean distance from (1,1)
        if redChi_beam is not None:
            dx = redChi_beam - 0.7
            dy = redChi_overall - 0.7
            return np.sqrt(dx*dx + dy*dy)
        else:
            return np.abs(redChi_overall - 0.7)
    
    iteration_results = []
    redChiSqr_beam = None
    redChiSqr = None
    consecutive_inRange = 0
    prev_redChiSqr = float('inf')

    # Create copies of the initial parameters
    current_r_core = copy.deepcopy(r_core)
    current_r_halo = copy.deepcopy(r_halo)
    current_r_beam_par = copy.deepcopy(r_beam_par) if r_beam_par is not None else None
    current_r_beam_anti_par = copy.deepcopy(r_beam_anti_par) if r_beam_anti_par is not None else None
    # Define parameter mappings
    core_param_maps = {'n': 'n_c', 'v_th_par': 'v_th_par_c', 'v_th_perp': 'v_th_perp_c'}
    halo_param_maps = {'n': 'n_h', 'v_th_par': 'v_th_par_h', 'v_th_perp': 'v_th_perp_h', 'kappa': 'kappa_h'}    
    beam_par_param_maps = {'n': 'n_b_par', 'u_par': 'u_par_b_par', 
                           'v_th_par': 'v_th_par_b_par', 'v_th_perp': 'v_th_perp_b_par', 'kappa': 'kappa_b_par'}
    beam_anti_param_maps = {'n': 'n_b_anti_par', 'u_par': 'u_par_b_anti_par', 
                            'v_th_par': 'v_th_par_b_anti_par', 'v_th_perp': 'v_th_perp_b_anti_par', 'kappa': 'kappa_b_anti_par'}

    for iteration in range(1, max_iterations + 1):
        print(f"\nFinal VDF fitting iteration {iteration}.")
        r_vdf, fit_vdf = perform_final_vdf_fit(pa_vec, unc_pa_vec, low_c, n_in, current_r_core, current_r_halo, current_r_beam_par, current_r_beam_anti_par,
                                               anti_par_strahl_cond, par_strahl_cond, v_par_mesh, v_perp_mesh)
        if r_vdf is None: # If can't fit, do next iteration 
            if iteration == 1:
                print("Final VDF fitting failed: No valid fits.")
                return None, None, None
            else:
                print('Final VDF fitting failed. Try again.')
                continue
        
        # Compute reduced chi-square for the overall fit
        redChiSqr = compute_reduced_chi_square_overallFit(r_vdf, pa_vec, high_h, low_c, anti_par_strahl_cond, par_strahl_cond)
        redChiSqr_beam = compute_reduced_chi_square_BeamOnly(r_vdf, beam_data, anti_par_strahl_cond, par_strahl_cond)
        
        if redChiSqr is not None and np.iscomplexobj(redChiSqr):
            redChiSqr = np.real(redChiSqr)
        if redChiSqr_beam is not None and np.iscomplexobj(redChiSqr_beam):
            redChiSqr_beam = np.real(redChiSqr_beam)
        # print(f"Reduced chi-square_Beam (final VDF) at iteration {iteration}: {redChiSqr_beam}")
        print(f"\nIteration {iteration}\nred_chi_Beam: {redChiSqr_beam}, red_chi_Overall: {redChiSqr}")
        
        iteration_results.append((r_vdf, fit_vdf, redChiSqr_beam, redChiSqr))

        if iteration == 1:
            if redChiSqr is None:
                print("Invalid Red-Chisqr_overall, continue...")
                continue
            prev_redChiSqr = redChiSqr
            prev_redChiSqr_beam = redChiSqr_beam

            if 0.6 <= redChiSqr <= 1.2:
                print(f'A good 1st time attempt ==> Updating initial guesses.')
                # Update initial guesses based on last fit
                update_model_params(r_vdf, current_r_core, core_param_maps)
                update_model_params(r_vdf, current_r_halo, halo_param_maps)
                # if par_strahl_cond and current_r_beam_par is not None:
                #     update_model_params(r_vdf, current_r_beam_par, beam_par_param_maps)
                # if anti_par_strahl_cond and current_r_beam_anti_par is not None:
                #     update_model_params(r_vdf, current_r_beam_anti_par, beam_anti_param_maps)

        # Iterations >= 2
        elif redChiSqr is not None:
            # Check improvement in overall chi-square
            overall_improved = abs(redChiSqr - 0.8) < abs(prev_redChiSqr - 0.8)

            beam_improved = False
            if redChiSqr_beam is not None and prev_redChiSqr_beam is not None:
                beam_improved = abs(redChiSqr_beam - 0.8) < abs(prev_redChiSqr_beam - 0.8)
            
            significant_improve = 0.05 * prev_redChiSqr <= abs(prev_redChiSqr - redChiSqr)
            improved = (overall_improved or beam_improved) and significant_improve
            
            if improved:
                print(f'Red-Chisqr_overall: {prev_redChiSqr} -> {redChiSqr} ==> Updating initial guesses.')
                if redChiSqr_beam is not None:
                    print(f'Red-Chisqr_beam: {prev_redChiSqr_beam} -> {redChiSqr_beam} ==> Updating initial guesses.')
                else:
                    print('No beam component in the fit.')
                prev_redChiSqr = redChiSqr
                prev_redChiSqr_beam = redChiSqr_beam

                # Update parameters with tighter constraints for later iterations
                update_model_params(r_vdf, current_r_core, core_param_maps)
                update_model_params(r_vdf, current_r_halo, halo_param_maps)
                if par_strahl_cond and current_r_beam_par is not None:
                    update_model_params(r_vdf, current_r_beam_par, beam_par_param_maps)
                if anti_par_strahl_cond and current_r_beam_anti_par is not None:
                    update_model_params(r_vdf, current_r_beam_anti_par, beam_anti_param_maps)

        
        if redChiSqr_beam is not None:
            in_range_beam = (chi_target_range[0] <= redChiSqr_beam <= chi_target_range[1])
            in_range_overall = (chi_target_range[0] <= redChiSqr <= chi_target_range[1])
            in_range = in_range_beam and in_range_overall
        else:
            in_range = (chi_target_range[0] <= redChiSqr <= chi_target_range[1])
        
        if in_range:
            consecutive_inRange += 1
            print(f"Consecutive in-range count = {consecutive_inRange}")
            if consecutive_inRange >= consecutive_inRange_num:
                if redChiSqr_beam is not None:
                    print(f"\nFinal VDF fitting converged with red_chi_Beam: {redChiSqr_beam:.2f}, red_chi_overall: {redChiSqr:.2f}, after {iteration} iterations.")
                else:
                    print(f"\nFinal VDF fitting converged with red_chi_overall: {redChiSqr:.2f}, after {iteration} iterations.")
                break
                # return r_vdf, fit_vdf, redChiSqr_beam
            else:
                continue
        else:
            consecutive_inRange = 0
            print(f"Out of target Red-Chisqr_beam range. Reset in-range counter.")

    # If no iteration met the target range, choose the closest
    best_score = float('inf')
    best_result = iteration_results[0]  # fallback
    for r_vdf_try, fit_vdf_try, chi_beam_try, chi_overall_try in iteration_results:
        score = combined_score(chi_overall_try,chi_beam_try)
        print(f"chi_beam={chi_beam_try}, chi_overall={chi_overall_try}, score={score}")
        
        if score < best_score:
            print(f"  -> New best! Previous best_score={best_score}, new best_score={score}")
            best_score = score
            best_result = (r_vdf_try, fit_vdf_try, chi_beam_try, chi_overall_try)

    print(f"\nFinal best_score: {best_score}")
    print(f"Final best_result chi values: beam={best_result[2]}, overall={best_result[3]}")
    chosen_r_vdf, chosen_fit_vdf, chosen_red_chi_beam, chosen_red_chi_overall = best_result
    if chosen_red_chi_beam is not None:
        beam_str = f"{chosen_red_chi_beam:.2f}, "
    else:
        beam_str = "None"
    if chosen_red_chi_overall is not None: 
        overall_str = f"{np.real(chosen_red_chi_overall):.2f}"
    else:
        overall_str = "None"
    print(f"Did not achieve target range after {max_iterations} iterations. \n Using closest result with red_chi_Beam={beam_str}, red_chi_overall={overall_str}.")
    return chosen_r_vdf, chosen_fit_vdf, chosen_red_chi_beam


def compute_reduced_chi_square_BeamOnly(r_vdf, beam_data, anti_par_strahl_cond, par_strahl_cond):
    """
    Compute the reduced chi-square for the beam component alone, over the beam energy range.
    """
    
    # Determine which model and how many parameters were used
    beam_count = int(anti_par_strahl_cond) + int(par_strahl_cond)
    # temp_b_pa, temp_b_psd, temp_b_v_par, temp_b_v_perp, temp_b_c = slice_beamDataPoints(pa_vec, high_b, low_b)
    temp_b_pa = beam_data['temp_b_pa']
    temp_b_psd = beam_data['temp_b_psd']
    temp_b_v_par = beam_data['temp_b_v_par']
    temp_b_v_perp = beam_data['temp_b_v_perp']
    temp_b_c = beam_data['temp_b_c']

    if beam_count == 0 or r_vdf is None:
        return None
    elif anti_par_strahl_cond and not par_strahl_cond:
        param_count = 5
        mask = beam_data['mask_AntiPar']
    elif not anti_par_strahl_cond and par_strahl_cond:
        param_count = 5
        mask = beam_data['mask_par']
    else:
        param_count = 10

    if beam_count == 1:
        valid_mask = mask & ~np.isnan(temp_b_psd)

        # Separate zero and nonzero PSD
        nonzero_mask = valid_mask & (temp_b_psd != 0.0)

        beam_params = lmfit.Parameters()
        beam_params.add('n', value=r_vdf.params['n_b'].value, vary=False)
        beam_params.add('u_par', value=r_vdf.params['u_par_b'].value, vary=False)
        beam_params.add('v_th_par', value=r_vdf.params['v_th_par_b'].value, vary=False)
        beam_params.add('v_th_perp', value=r_vdf.params['v_th_perp_b'].value, vary=False)
        beam_params.add('kappa', value=r_vdf.params['kappa_b'].value, vary=False)

        fit_val = beam.eval(beam_params, v_par=temp_b_v_par[nonzero_mask], v_perp=temp_b_v_perp[nonzero_mask])
        deviation_sqr = (temp_b_psd[nonzero_mask] - fit_val)**2
        uncertainty_sqr = (temp_b_psd[nonzero_mask] / np.sqrt(temp_b_c[nonzero_mask]))**2
        red_chi_beam = np.sum(deviation_sqr / uncertainty_sqr) / (np.count_nonzero(valid_mask) - param_count)

    elif beam_count == 2:
        # Remove NaNs and separate zero from nonzero for par beam
        valid_mask_par = beam_data['mask_par'] & ~np.isnan(temp_b_psd)
        nonzero_mask_par = valid_mask_par & (temp_b_psd != 0)

        # Remove NaNs and separate zero from nonzero for anti-par beam
        valid_mask_anti_par = beam_data['mask_AntiPar'] & ~np.isnan(temp_b_psd)
        nonzero_mask_anti_par = valid_mask_anti_par & (temp_b_psd != 0)

        beam_params_par = lmfit.Parameters()
        beam_params_par.add('n', value=r_vdf.params['n_b_par'].value, vary=False)
        beam_params_par.add('u_par', value=r_vdf.params['u_par_b_par'].value, vary=False)
        beam_params_par.add('v_th_par', value=r_vdf.params['v_th_par_b_par'].value, vary=False)
        beam_params_par.add('v_th_perp', value=r_vdf.params['v_th_perp_b_par'].value, vary=False)
        beam_params_par.add('kappa', value=r_vdf.params['kappa_b_par'].value, vary=False)

        beam_params_Antipar = lmfit.Parameters()
        beam_params_Antipar.add('n', value=r_vdf.params['n_b_anti_par'].value, vary=False)
        beam_params_Antipar.add('u_par', value=r_vdf.params['u_par_b_anti_par'].value, vary=False)
        beam_params_Antipar.add('v_th_par', value=r_vdf.params['v_th_par_b_anti_par'].value, vary=False)
        beam_params_Antipar.add('v_th_perp', value=r_vdf.params['v_th_perp_b_anti_par'].value, vary=False)
        beam_params_Antipar.add('kappa', value=r_vdf.params['kappa_b_anti_par'].value, vary=False)

        fit_val_par = beam.eval(beam_params_par, v_par=temp_b_v_par[nonzero_mask_par], v_perp=temp_b_v_perp[nonzero_mask_par])
        fit_val_Antipar = beam.eval(beam_params_Antipar, v_par=temp_b_v_par[nonzero_mask_anti_par], v_perp=temp_b_v_perp[nonzero_mask_anti_par])
        chiSqr_par = np.sum( ((temp_b_psd[nonzero_mask_par] - fit_val_par)**2) / ((temp_b_psd[nonzero_mask_par] / np.sqrt(temp_b_c[nonzero_mask_par]))**2) )
        chiSqr_Antipar = np.sum( ((temp_b_psd[nonzero_mask_anti_par] - fit_val_Antipar)**2) / ((temp_b_psd[nonzero_mask_anti_par] / np.sqrt(temp_b_c[nonzero_mask_anti_par]))**2) )
        
        dof = len(temp_b_psd[valid_mask_par]) + len(temp_b_psd[valid_mask_anti_par]) - param_count
        red_chi_beam = (chiSqr_par + chiSqr_Antipar) / dof

    return red_chi_beam

def compute_reduced_chi_square_overallFit(r_vdf, pa_vec, high_h, low_c, anti_par_strahl_cond, par_strahl_cond):
    """
    Compute the reduced chi-square for the final VDF fit.

    Parameters
    ----------
    r_vdf : lmfit.ModelResult or None
        The result of the final VDF fit.
    pa_vec : ndarray
        Array containing [PA, PSD, v_par, v_perp, counts].
    high_h, high_sc : int
        Energy indices defining the range for chi-square calculation.
    anti_par_strahl_cond, par_strahl_cond : bool
        Flags indicating the presence of anti-parallel or parallel strahl.
    vdf_1beam, vdf_2beam : lmfit.Model
        Fitting models for one-beam and two-beam scenarios.

    Returns
    -------
    red_chi : float or None
        The reduced chi-square. None if r_vdf is None or no valid calculation.
    """
    if r_vdf is None:
        return None

    # Determine which model and how many parameters were used
    beam_count = int(anti_par_strahl_cond) + int(par_strahl_cond)
    if beam_count == 0:
        fit_model = vdf_1beam
        param_count = 9
    elif beam_count == 1:
        fit_model = vdf_1beam
        param_count = 14
    else:
        fit_model = vdf_2beam
        param_count = 19

    red_chi = 0.0
    counter = 0

    # # Compute chi-square. Old one, not correct because of it does not consider the mask.
    # for j in range(high_h, low_c): # (high_h, low_c)
    #     for k in range(pa_vec.shape[1]):
    #         psd_val = pa_vec[j, k, 1]
    #         if not np.isnan(psd_val):
    #             counter += 1
    #             if psd_val != 0.0:
    #                 fit_val = fit_model.eval(r_vdf.params, v_par=pa_vec[j,k,2], v_perp=pa_vec[j,k,3])
    #                 deviation_sqr = (psd_val - fit_val) ** 2
    #                 uncertainty_sqr = (psd_val / np.sqrt(pa_vec[j,k,4])) ** 2
    #                 red_chi += deviation_sqr / uncertainty_sqr
    #             # If psd_val == 0.0, no addition needed as it contributes 0.

    # Check1
    # Mask data
    countMask_all = pa_vec[:low_c, :, 4] >=2 
    psd_masked = np.where(countMask_all, pa_vec[:low_c,:,1],np.nan)
    v_par_masked = np.where(countMask_all, pa_vec[:low_c,:,2],np.nan)
    v_perp_masked = np.where(countMask_all, pa_vec[:low_c,:,3],np.nan)
    count_masked = np.where(countMask_all, pa_vec[:low_c,:,4],np.nan)
    
    # Compute chi-square
    for j in range(high_h, low_c): # (high_h, low_c)
        for k in range(pa_vec.shape[1]):
            psd_val = psd_masked[j, k]
            if not np.isnan(psd_val):
                counter += 1
                if psd_val != 0.0:
                    fit_val = fit_model.eval(r_vdf.params, v_par=v_par_masked[j,k], v_perp=v_perp_masked[j,k])
                    deviation_sqr = (psd_val - fit_val) ** 2
                    uncertainty_sqr = (psd_val / np.sqrt(count_masked[j,k])) ** 2
                    red_chi += deviation_sqr / uncertainty_sqr
                # If psd_val == 0.0, no addition needed as it contributes 0.

    # Compute reduced chi-square: chi-square / (N - P)
    # Ensure denominator is positive
    dof = counter - param_count
    if dof > 0:
        red_chi /= dof
    else:
        red_chi = None

    return red_chi


# Model definitions
def core_halo(v_par, v_perp, n_c, u_par_c, v_th_par_c, v_th_perp_c, n_h, u_par_h, v_th_par_h, v_th_perp_h, kappa_h):
    ''' core : see bi-Max for parameters'''
    c_par = v_par - u_par_c
    c_perp = v_perp
    exponent_c = (c_par/v_th_par_c)**2 + (c_perp/v_th_perp_c)**2

    denominator = (np.pi ** (3/2)) * v_th_par_c * (v_th_perp_c**2)
    coeff_c = n_c / denominator
    term_core = coeff_c * np.exp(- exponent_c)

    ''' halo '''
    denominator = (np.pi ** (3/2)) * v_th_par_h * (v_th_perp_h**2)
    term1_h = n_h / denominator
    gamma_ratio = gamma(kappa_h+1)/(gamma(kappa_h - 0.5)*((kappa_h - 3/2)**(3/2)))
    
    # p, q = 10, 1
    # delta = np.sqrt(1.08*((1/3)*v_th_par_h + (2/3)*v_th_perp_h))
    kappa_part = term1_h*gamma_ratio*(( 1 + ((v_par-u_par_h)**2)/((v_th_par_h**2)*(kappa_h - 3/2)) + (v_perp**2)/((v_th_perp_h**2)*(kappa_h - 3/2)))**(-kappa_h -1))
    term_halo = kappa_part # bi-kappa
    # flattop_part = (1 + ((1/delta)*(((v_par-u_par_h)**2)/(v_th_par_h**2) + (v_perp**2)/((v_th_perp_h**2)))**p))**(-q)
    # term_halo = kappa_part*(1 - flattop_part) # flattened bi-kappa
    
    
    f_core_halo = term_core + term_halo

    return f_core_halo

def combined_model_1beam(v_par, v_perp, n_c, u_par_c, v_th_par_c, v_th_perp_c, 
                   n_h, u_par_h, v_th_par_h, v_th_perp_h, kappa_h, 
                   n_b, u_par_b, v_th_par_b, v_th_perp_b, kappa_b):
    # Conditional component
    beam_1 = beam_nonTruncated(v_par, v_perp, n_b, u_par_b, v_th_par_b, v_th_perp_b, kappa_b)
    beam_2 = beam_truncated(v_par, v_perp, n_b, u_par_b, v_th_par_b, v_th_perp_b, kappa_b)
    if u_par_b < 0 :
        vdf_beam = np.where(v_par < u_par_b, beam_1, beam_2)
    else :
        vdf_beam = np.where(v_par >= u_par_b, beam_1, beam_2)
    
    # ## old beam model
    # vdf_beam = beam_truncated(v_par, v_perp, n_b, u_par_b, v_th_par_b, v_th_perp_b, kappa_b)
    ## core + halo
    vdf_corehalo = core_halo(v_par, v_perp, n_c, u_par_c, v_th_par_c, v_th_perp_c, n_h, u_par_h, v_th_par_h, v_th_perp_h, kappa_h)

    # # stepan's model
    # vdf_core = bi_Max(v_par, v_perp, n_c, u_par_c, v_th_par_c, v_th_perp_c)
    # vdf_halo = bi_kappa_flattop_stepan(v_par, v_perp, n_h, u_par_h, v_th_par_h, v_th_perp_h, kappa_h, delta, u_par_c, v_th_par_c, v_th_perp_c)
    # vdf_corehalo = vdf_core + vdf_halo

    # Sum all components
    total_result = vdf_beam + vdf_corehalo

    return total_result

def combined_model_2beam(v_par, v_perp, n_c, u_par_c, v_th_par_c, v_th_perp_c, 
                   n_h, u_par_h, v_th_par_h, v_th_perp_h, kappa_h, 
                   n_b_par, u_par_b_par, v_th_par_b_par, v_th_perp_b_par, kappa_b_par, 
                   n_b_anti_par, u_par_b_anti_par, v_th_par_b_anti_par, v_th_perp_b_anti_par, kappa_b_anti_par):
    # Conditional component. Parallel beam
    beam_1 = beam_nonTruncated(v_par, v_perp, n_b_par, u_par_b_par, v_th_par_b_par, v_th_perp_b_par, kappa_b_par)
    beam_2 = beam_truncated(v_par, v_perp, n_b_par, u_par_b_par, v_th_par_b_par, v_th_perp_b_par, kappa_b_par)
    if u_par_b_par < 0 :
        vdf_beam_par = np.where(v_par < u_par_b_par, beam_1, beam_2)
    else :
        vdf_beam_par = np.where(v_par >= u_par_b_par, beam_1, beam_2)

    # Conditional component. AntiParallel beam
    beam_1 = beam_nonTruncated(v_par, v_perp, n_b_anti_par, u_par_b_anti_par, v_th_par_b_anti_par, v_th_perp_b_anti_par, kappa_b_anti_par)
    beam_2 = beam_truncated(v_par, v_perp, n_b_anti_par, u_par_b_anti_par, v_th_par_b_anti_par, v_th_perp_b_anti_par, kappa_b_anti_par)
    if u_par_b_anti_par < 0 :
        vdf_beam_anti_par = np.where(v_par < u_par_b_anti_par, beam_1, beam_2)
    else :
        vdf_beam_anti_par = np.where(v_par >= u_par_b_anti_par, beam_1, beam_2)
    
    # ## old beam model
    # vdf_beam = beam_truncated(v_par, v_perp, n_b, u_par_b, v_th_par_b, v_th_perp_b, kappa_b)
    # core + halo
    vdf_corehalo = core_halo(v_par, v_perp, n_c, u_par_c, v_th_par_c, v_th_perp_c, n_h, u_par_h, v_th_par_h, v_th_perp_h, kappa_h)
    
    # # stepan's model
    # vdf_core = bi_Max(v_par, v_perp, n_c, u_par_c, v_th_par_c, v_th_perp_c)
    # vdf_halo = bi_kappa_flattop_stepan(v_par, v_perp, n_h, u_par_h, v_th_par_h, v_th_perp_h, kappa_h, delta, u_par_c, v_th_par_c, v_th_perp_c)
    # vdf_corehalo = vdf_core + vdf_halo

    # Sum all components
    total_result = vdf_beam_par + vdf_beam_anti_par + vdf_corehalo

    return total_result

## Old models
# def core_beam_halo(v_par, v_perp, n_c, u_par_c, v_th_par_c, v_th_perp_c, n_b, u_par_b, v_th_par_b, v_th_perp_b, kappa_b, n_h, u_par_h, v_th_par_h, v_th_perp_h, kappa_h) :
#     ''' core : see bi-Max for parameters'''
#     c_par = v_par - u_par_c
#     c_perp = v_perp
#     exponent_c = (c_par/v_th_par_c)**2 + (c_perp/v_th_perp_c)**2

#     denominator = (np.pi ** (3/2)) * v_th_par_c * (v_th_perp_c**2)
#     coeff_c = n_c / denominator
#     term_core = coeff_c * np.exp(- exponent_c)


#     ''' beam : see bi-kappa for parameters'''
#     D = D_CONST
#     denominator = (np.pi ** (3/2)) * v_th_par_b * (v_th_perp_b**2)
#     D_coeff = (2*np.sqrt(D))/np.sqrt(D+1)
#     coeff_b = (n_b*D_coeff) / denominator
#     gamma_ratio = gamma(kappa_b+1)/(gamma(kappa_b - 0.5)*((kappa_b - 3/2)**(3/2)))

#     term_beam = coeff_b*gamma_ratio*(( 1 + D*((v_par-u_par_b)**2)/((v_th_par_b**2)*(kappa_b - 3/2)) + (v_perp**2)/((v_th_perp_b**2)*(kappa_b - 3/2)))**(-kappa_b -1))

#     ''' halo '''
#     denominator = (np.pi ** (3/2)) * v_th_par_h * (v_th_perp_h**2)
#     term1_h = n_h / denominator

#     gamma_ratio = gamma(kappa_h+1)/(gamma(kappa_h - 0.5)*((kappa_h - 3/2)**(3/2)))

#     term_halo = term1_h*gamma_ratio*(( 1 + ((v_par-u_par_h)**2)/((v_th_par_h**2)*(kappa_h - 3/2)) + (v_perp**2)/((v_th_perp_h**2)*(kappa_h - 3/2)))**(-kappa_h -1))

#     f = term_core + term_beam + term_halo

#     return f

# def core_2beam_halo(v_par, v_perp, n_c, u_par_c, v_th_par_c, v_th_perp_c, n_b_par, u_par_b_par, v_th_par_b_par, v_th_perp_b_par, kappa_b_par, n_b_anti_par, u_par_b_anti_par, v_th_par_b_anti_par, v_th_perp_b_anti_par, kappa_b_anti_par, n_h, u_par_h, v_th_par_h, v_th_perp_h, kappa_h) :

#     ''' core : see bi-Max for parameters'''
#     c_par = v_par - u_par_c
#     c_perp = v_perp
#     exponent_c = (c_par/v_th_par_c)**2 + (c_perp/v_th_perp_c)**2

#     denominator = (np.pi ** (3/2)) * v_th_par_c * (v_th_perp_c**2)
#     coeff_c = n_c / denominator

#     ''' par beam : see bi-kappa for parameters'''
#     D = D_CONST
#     denominator = (np.pi ** (3/2)) * v_th_par_b_par * (v_th_perp_b_par**2)
#     D_coeff = (2*np.sqrt(D))/np.sqrt(D+1)
#     coeff_b = (n_b_par*D_coeff) / denominator
#     gamma_ratio = gamma(kappa_b_par+1)/(gamma(kappa_b_par - 0.5)*((kappa_b_par - 3/2)**(3/2)))

#     term_beam_par = coeff_b*gamma_ratio*(( 1 + D*((v_par-u_par_b_par)**2)/((v_th_par_b_par**2)*(kappa_b_par - 3/2)) + (v_perp**2)/((v_th_perp_b_par**2)*(kappa_b_par - 3/2)))**(-kappa_b_par -1))

#     ''' anti_par beam : see bi-kappa for parameters'''
#     D = D_CONST
#     denominator = (np.pi ** (3/2)) * v_th_par_b_anti_par * (v_th_perp_b_anti_par**2)
#     D_coeff = (2*np.sqrt(D))/np.sqrt(D+1)
#     coeff_b = (n_b_anti_par*D_coeff) / denominator
#     gamma_ratio = gamma(kappa_b_anti_par+1)/(gamma(kappa_b_anti_par - 0.5)*((kappa_b_anti_par - 3/2)**(3/2)))

#     term_beam_anti_par = coeff_b*gamma_ratio*(( 1 + D*((v_par-u_par_b_anti_par)**2)/((v_th_par_b_anti_par**2)*(kappa_b_anti_par - 3/2)) + (v_perp**2)/((v_th_perp_b_anti_par**2)*(kappa_b_anti_par - 3/2)))**(-kappa_b_anti_par -1))

#     ''' halo '''
#     denominator = (np.pi ** (3/2)) * v_th_par_h * (v_th_perp_h**2)
#     term1_h = n_h / denominator

#     gamma_ratio = gamma(kappa_h+1)/(gamma(kappa_h - 0.5)*((kappa_h - 3/2)**(3/2)))

#     term_halo = term1_h*gamma_ratio*(( 1 + ((v_par-u_par_h)**2)/((v_th_par_h**2)*(kappa_h - 3/2)) + (v_perp**2)/((v_th_perp_h**2)*(kappa_h - 3/2)))**(-kappa_h -1))

#     f = coeff_c * np.exp(- exponent_c) + term_beam_par + term_beam_anti_par + term_halo

#     return f

vdf_1beam = lmfit.Model(combined_model_1beam, independent_vars = ['v_par', 'v_perp'], \
    param_names = ['n_c', 'u_par_c', 'v_th_par_c', 'v_th_perp_c', 'n_h', 'u_par_h', 'v_th_par_h', 'v_th_perp_h', 'kappa_h', 'n_b', 'u_par_b', 'v_th_par_b', 'v_th_perp_b', 'kappa_b']) # core_beam_halo

vdf_2beam = lmfit.Model(combined_model_2beam, independent_vars = ['v_par', 'v_perp'], \
    param_names = ['n_c', 'u_par_c', 'v_th_par_c', 'v_th_perp_c', 'n_h', 'u_par_h', 'v_th_par_h', 'v_th_perp_h', 'kappa_h', 'n_b_par', 'u_par_b_par', 'v_th_par_b_par', 'v_th_perp_b_par', 'kappa_b_par', 'n_b_anti_par', 'u_par_b_anti_par', 'v_th_par_b_anti_par', 'v_th_perp_b_anti_par', 'kappa_b_anti_par']) #core_2beam_halo
