import numpy as np
from scipy.special import gamma
from utils import eV_to_vel, int_vdf, moment1st_VDF_para, moment2nd_VDF
import lmfit
from constant import PAW_coeff, n_halo_factor, PixelCountThreshold_halo, n_const_range, u_const_range

stderr_fact = 0.05 # Estimate the error of fitted parameters if their std is 0
# Initial value
def set_halo_initial_params(par_strahl_cond, anti_par_strahl_cond, n_in, r_core):
    """
    Set initial values for halo parameters based on conditions.
    Returns updated halo_init, halo_const.
    """
    
    n_init_h = n_in/n_halo_factor
    # if n_init_h > 0.15*r_core.params['n'].value:
    #     n_init_h = 0.15*r_core.params['n'].value
    halo_init = [n_init_h, 5000, (1*10**9)/2, (1*10**9)/2, 5] # vth-(3*10**9)/2 (2022.03.02) . [n_init_h, 5000, 4e8, 4e8, 4]
    halo_const = [[0, n_in/5], [-2.2e8, 2.2e8], [2.5e8, 2e9], [2.5e8, 2e9], [1.6, 15]] # n, u_par, v_th_par, v_th_perp, kappa

    # Set halo velocity based on beam location
    if not anti_par_strahl_cond and par_strahl_cond:
        halo_init[1] = -1 * eV_to_vel(2)
        halo_const[1] = [-2.2e8, 0]
    elif anti_par_strahl_cond and not par_strahl_cond:
        halo_init[1] = eV_to_vel(2)
        halo_const[1] = [0, 2.2e8]
    return halo_init, halo_const

# Halo Fitting operation
def halo_fit_function(psd_data, v_par, v_perp, psd_unc, halo_const, halo_init, core_param, v_par_mesh, v_perp_mesh) :

    kw_ampgo = {'local_opts':{'maxiter':2000, 'ftol':1e-31},
                'totaliter':50,
                'tabulistsize': 8,
                'eps1': 1e-29, 'eps2': 0.1, 'glbtol': 1e-31, 
                'disp':False}
    # psd_uncertainty = psd_data/np.sqrt(count)
    # psd_unc_safe = np.copy(psd_unc)
    # psd_unc_safe[psd_unc_safe == 0] = 0.1*np.min(psd_unc[psd_unc > 0]) if np.any(psd_unc > 0) else 1e-10

    weight_set = 1.0 #/psd_unc_safe # np.mean(psd_data)#np.median(psd_uncertainty)

    ''' set initial halo guess '''
    p_halo = lmfit.Parameters()
    p_halo.add('n', halo_init[0], vary = True, min = halo_const[0][0], max = halo_const[0][1])
    p_halo.add('u_par', halo_init[1], vary = False, min = halo_const[1][0], max = halo_const[1][1])
    p_halo.add('v_th_par', halo_init[2], vary = False, min = halo_const[2][0], max = halo_const[2][1])
    p_halo.add('v_th_perp', halo_init[3], vary = False, min = halo_const[3][0], max = halo_const[3][1])
    p_halo.add('kappa', halo_init[4], vary = False, min = halo_const[4][0], max = halo_const[4][1])
    
    # # Stepan's model
    # p_halo.add('delta', 6, vary = True, min = 5, max = 12)
    # p_halo.add('u_par_c', core_param.params['u_par'], vary = False)
    # p_halo.add('v_th_par_c', core_param.params['v_th_par'], vary = False)
    # p_halo.add('v_th_perp_c', core_param.params['v_th_perp'], vary = False)

    p_list = list(p_halo.valuesdict().keys())

    try :
        # r_halo = halo.fit(psd_data, p_halo, method = 'ampgo', v_par = v_par, v_perp = v_perp, max_nfev=10000,fit_kws=kw_ampgo, weights=weight_set) #, max_nfev=500
        r_halo = halo.fit(psd_data, p_halo, method = 'leastsq', v_par = v_par, v_perp = v_perp, max_nfev=10000) #, max_nfev=500
    except ValueError :
        print('Halo fitting failed.')
        r_halo = None

    def fit_update(jj, p_halo, r_halo, vary_list, psd_data, v_par, v_perp, halo_const) :

        p_halo.add('n', r_halo.params['n'].value, vary = vary_list[0], min = halo_const[0][0], max = halo_const[0][1])
        p_halo.add('u_par', r_halo.params['u_par'].value, vary = vary_list[1], min = halo_const[1][0], max = halo_const[1][1])
        p_halo.add('v_th_par', r_halo.params['v_th_par'].value, vary = vary_list[2], min = halo_const[2][0], max = halo_const[2][1])
        p_halo.add('v_th_perp', r_halo.params['v_th_perp'].value, vary = vary_list[3], min = halo_const[3][0], max = halo_const[3][1])
        p_halo.add('kappa', r_halo.params['kappa'].value, vary = vary_list[4], min = halo_const[4][0], max = halo_const[4][1])

        # # Stepan's model
        # p_halo.add('delta', r_halo.params['delta'].value, vary = vary_list[5], min = 5, max = 12)
        # p_halo.add('u_par_c', core_param.params['u_par'], vary = False)
        # p_halo.add('v_th_par_c', core_param.params['v_th_par'], vary = False)
        # p_halo.add('v_th_perp_c', core_param.params['v_th_perp'], vary = False)

        try :
            # r_halo = halo.fit(psd_data, p_halo, method = 'ampgo', v_par = v_par, v_perp = v_perp, max_nfev=10000,fit_kws=kw_ampgo, weights=weight_set) #, max_nfev=500, weights=weight_set
            r_halo = halo.fit(psd_data, p_halo, method = 'leastsq', v_par = v_par, v_perp = v_perp, max_nfev=10000)
        except ValueError :
            print(jj, 'Halo updating failed.')
            r_halo = None

        return r_halo

    vary_list = [[True, True, False, False, False, True], 
                 [True, False, True, True, False, True], 
                 [False, False, True, True, True, True], 
                 [False, False, False, True, False, True]]

    r_halo_save = []
    for ii in range(len(vary_list)) :
        if r_halo != None :
            r_halo = fit_update(ii, p_halo, r_halo, vary_list[ii], psd_data, v_par, v_perp, halo_const)
            r_halo_save.append(r_halo)
        else :
            print('Final halo fit doesnt work')
            if ii>2:
                r_halo = r_halo_save[ii-2]
                break
            else:
                break

    if r_halo != None :
        fit_halo = halo.eval(r_halo.params, v_par = v_par_mesh, v_perp = v_perp_mesh)
        if r_halo.errorbars == False :
            for i in range(len(p_list)) :
                r_halo.params[p_list[i]].stderr = stderr_fact*np.abs(r_halo.params[p_list[i]].value)

        for i in range(len(p_list)) :
            if r_halo.params[p_list[i]].stderr == 0 :
                r_halo.params[p_list[i]].stderr = stderr_fact*np.abs(r_halo.params[p_list[i]].value)
    else:
        fit_halo = halo.eval(p_halo, v_par = v_par_mesh, v_perp = v_perp_mesh)

    return r_halo, fit_halo

def calculate_halo_density(r_halo, r_core, v_par_mesh, v_perp_mesh, v_par_arr, v_perp_arr):
    """
    Calculate the actual superthermal density for the beam velocity.
    """
    if r_halo is not None:
        fit_halo_hallow = halo_flattop.eval(r_halo.params, v_par=v_par_mesh, v_perp=v_perp_mesh, r_core=r_core)
        halo_real_n = int_vdf(fit_halo_hallow, v_par_arr, v_perp_arr) / r_halo.params['n'].value # not sure if this is correct. Why divide by n?
    else:
        halo_real_n = 0.15
    return halo_real_n


def slice_haloDataPoints(pa_vec, unc_pa_vec, high_h, low_h):
    """
    Slice the halo data points based on energy range.
    """
    count_mask_h = np.where(pa_vec[high_h:low_h, :, 4] >= PixelCountThreshold_halo)
    temp_h_pa, temp_h_psd, temp_h_v_par, temp_h_v_perp, temp_h_count = (
            np.ravel(pa_vec[high_h:low_h, :, 0][count_mask_h]),
            np.ravel(pa_vec[high_h:low_h, :, 1][count_mask_h]),
            np.ravel(pa_vec[high_h:low_h, :, 2][count_mask_h]),
            np.ravel(pa_vec[high_h:low_h, :, 3][count_mask_h]),
            np.ravel(pa_vec[high_h:low_h, :, 4][count_mask_h])
        )
    temp_h_unc = np.ravel(unc_pa_vec[high_h:low_h, :][count_mask_h])
    return temp_h_pa, temp_h_psd, temp_h_v_par, temp_h_v_perp, temp_h_count, temp_h_unc

# def perform_halo_fit(halo_data, v_par_mesh, v_perp_mesh, anti_par_strahl_cond, par_strahl_cond, unc_pa_vec):
#     """
#     Handle the halo fitting process based on conditions.
#     """
#     r_halo, fit_halo = None, None
#     mask_pitchAngle = halo_data['mask_pitchAngle']

#     if anti_par_strahl_cond and not par_strahl_cond:
#         pa_anti_par_bound = 180 - mean_pad_params['W_180'] * PAW_coeff

#         if pa_anti_par_bound < 90:
#             r_halo, fit_halo = halo_fit_function(temp_h_psd[temp_h_pa < pa_anti_par_bound],
#                                         temp_h_v_par[temp_h_pa < pa_anti_par_bound],
#                                         temp_h_v_perp[temp_h_pa < pa_anti_par_bound], 
#                                         temp_h_unc[temp_h_pa < pa_anti_par_bound], 
#                                         halo_const, halo_init, r_core, v_par_mesh, v_perp_mesh)
#         else:
#             pa_anti_par_bound = 90
#             r_halo, fit_halo = halo_fit_function(temp_h_psd[temp_h_pa < pa_anti_par_bound],
#                                         temp_h_v_par[temp_h_pa < pa_anti_par_bound],
#                                         temp_h_v_perp[temp_h_pa < pa_anti_par_bound], 
#                                         temp_h_unc[temp_h_pa < pa_anti_par_bound], 
#                                         halo_const, halo_init, r_core, v_par_mesh, v_perp_mesh)
#         halo_data_dict.update({'mask_pitchAngle': temp_h_pa < pa_anti_par_bound})
        
#     elif not anti_par_strahl_cond and par_strahl_cond:
#         pa_par_bound = mean_pad_params['W_0'] * PAW_coeff

#         r_halo, fit_halo = halo_fit_function(temp_h_psd[temp_h_pa > pa_par_bound],
#                                     temp_h_v_par[temp_h_pa > pa_par_bound],
#                                     temp_h_v_perp[temp_h_pa > pa_par_bound], 
#                                     temp_h_unc[temp_h_pa > pa_par_bound],
#                                     halo_const, halo_init, r_core, v_par_mesh, v_perp_mesh)
#         halo_data_dict.update({'mask_pitchAngle': temp_h_pa > pa_par_bound})

#     elif not anti_par_strahl_cond and not par_strahl_cond:

#         r_halo, fit_halo = halo_fit_function(temp_h_psd, temp_h_v_par, temp_h_v_perp, temp_h_unc, halo_const, halo_init, r_core, v_par_mesh, v_perp_mesh)
#         halo_data_dict.update({'mask_pitchAngle': np.ones_like(temp_h_pa, dtype=bool)})

#     elif anti_par_strahl_cond and par_strahl_cond:
#         pa_anti_par_bound = 180 - mean_pad_params['W_180'] * PAW_coeff
#         pa_par_bound = mean_pad_params['W_0'] * PAW_coeff

#         r_halo, fit_halo = halo_fit_function(temp_h_psd[np.logical_and(temp_h_pa > pa_par_bound, temp_h_pa < pa_anti_par_bound)],
#                                     temp_h_v_par[np.logical_and(temp_h_pa > pa_par_bound, temp_h_pa < pa_anti_par_bound)],
#                                     temp_h_v_perp[np.logical_and(temp_h_pa > pa_par_bound, temp_h_pa < pa_anti_par_bound)],
#                                     temp_h_unc[np.logical_and(temp_h_pa > pa_par_bound, temp_h_pa < pa_anti_par_bound)],
#                                     halo_const, halo_init, r_core, v_par_mesh, v_perp_mesh)
#         halo_data_dict.update({'mask_pitchAngle': np.logical_and(temp_h_pa > pa_par_bound, temp_h_pa < pa_anti_par_bound)})

#     halo_data_dict.update({'temp_h_pa': temp_h_pa, 'temp_h_psd': temp_h_psd, 'temp_h_v_par': temp_h_v_par, 
#                            'temp_h_v_perp': temp_h_v_perp, 'temp_h_count': temp_h_count, 'temp_h_unc': temp_h_unc,
#                            'halo_init': halo_init, 'halo_const': halo_const, 'r_halo': r_halo, 'fit_halo': fit_halo})

#     halo_real_n = calculate_halo_density(r_halo, r_core, v_par_mesh, v_perp_mesh, v_par_arr, v_perp_arr)
#     return r_halo, halo_real_n, halo_data_dict

def extract_halo_data(pa_vec, high_h, low_h, mean_pad_params, r_core, n_in, anti_par_strahl_cond, par_strahl_cond, unc_pa_vec):
    """
    Extract and initialize halo data points and parameters based on conditions.
    Similar to extract_beam_data for consistency.
    """
    # Initialize halo data dictionary
    halo_data_dict = {
        'temp_h_pa': None, 'temp_h_psd': None, 'temp_h_v_par': None, 
        'temp_h_v_perp': None, 'temp_h_count': None, 'temp_h_unc': None,
        'mask_pitchAngle': None, 'halo_init': None, 'halo_const': None
    }

    # Get initial halo parameters
    halo_init, halo_const = set_halo_initial_params(par_strahl_cond, anti_par_strahl_cond, n_in, r_core)
    
    # Slice data points
    temp_h_pa, temp_h_psd, temp_h_v_par, temp_h_v_perp, temp_h_count, temp_h_unc = slice_haloDataPoints(pa_vec, unc_pa_vec, high_h, low_h)

    # Set pitch angle mask based on conditions
    if anti_par_strahl_cond and not par_strahl_cond:
        pa_anti_par_bound = 180 - mean_pad_params['W_180'] * PAW_coeff
        pa_anti_par_bound = min(pa_anti_par_bound, 130) #90
        mask_pitchAngle = temp_h_pa < pa_anti_par_bound

    elif not anti_par_strahl_cond and par_strahl_cond:
        pa_par_bound = mean_pad_params['W_0'] * PAW_coeff
        mask_pitchAngle = temp_h_pa > pa_par_bound

    elif not anti_par_strahl_cond and not par_strahl_cond:
        mask_pitchAngle = np.ones_like(temp_h_pa, dtype=bool)

    elif anti_par_strahl_cond and par_strahl_cond:
        pa_anti_par_bound = 180 - mean_pad_params['W_180'] * PAW_coeff
        pa_par_bound = mean_pad_params['W_0'] * PAW_coeff
        mask_pitchAngle = np.logical_and(temp_h_pa > pa_par_bound, temp_h_pa < pa_anti_par_bound)

    # Update dictionary with all data
    halo_data_dict.update({
        'temp_h_pa': temp_h_pa, 'temp_h_psd': temp_h_psd,
        'temp_h_v_par': temp_h_v_par, 'temp_h_v_perp': temp_h_v_perp,
        'temp_h_count': temp_h_count, 'temp_h_unc': temp_h_unc,
        'mask_pitchAngle': mask_pitchAngle,
        'halo_init': halo_init, 'halo_const': halo_const
    })

    return halo_data_dict

def Adaptive_halo_fitting(halo_data, v_par_mesh, v_perp_mesh, 
                         r_core, max_iterations=20, chi_target_range=(0.9, 1.1), consecutive_inRange_num=3):
    """
    """
    r_halo, fit_halo = None, None
    red_chi_h = None
    prev_red_chi_h = float('inf')
    consecutive_inRange = 0

    # If no valid data, return None
    if halo_data['temp_h_psd'] is None or len(halo_data['temp_h_psd']) == 0:
        return None, None, None

    iteration_results = []
    
    # Iterative fitting
    for iteration in range(1, max_iterations + 1):
        print(f"\nHalo fitting iteration {iteration}.")

        # Perform halo fit
        r_halo, fit_halo = halo_fit_function(
            halo_data['temp_h_psd'][halo_data['mask_pitchAngle']],
            halo_data['temp_h_v_par'][halo_data['mask_pitchAngle']],
            halo_data['temp_h_v_perp'][halo_data['mask_pitchAngle']],
            halo_data['temp_h_unc'][halo_data['mask_pitchAngle']],
            halo_data['halo_const'],
            halo_data['halo_init'],
            r_core,
            v_par_mesh,
            v_perp_mesh
        )

        if r_halo is None:  # If can't fit, handle first iteration specially
            if iteration == 1:
                print("Halo fitting failed: No valid fits.")
                return None, None, None
            else:
                print('Fitting failed. Try again.')
                continue

        red_chi_h = get_reduced_chi_square_Halo(r_halo, halo_data)
        print(f"Red. chi-square for halo: {red_chi_h}")
        iteration_results.append((r_halo, fit_halo, red_chi_h))

        # First iteration, special handling
        if iteration == 1:
            if red_chi_h is None:
                print("Invalid Red-Chisqr_halo on 1st iteration, continue...")
                continue
            prev_red_chi_h = red_chi_h
            
            if 0.7 <= red_chi_h <= 1.3:
                print(f'A good 1st time attempt ==> Updating initial guesses.')
                # Update initial guesses based on last fit
                halo_init = [r_halo.params['n'].value, r_halo.params['u_par'].value,
                           r_halo.params['v_th_par'].value, r_halo.params['v_th_perp'].value,
                           r_halo.params['kappa'].value]
                if np.abs(halo_init[1]) < 1e3:
                    if halo_init[1] > 0:
                        u_par_h_const = [0, 2e8]
                    else:
                        u_par_h_const = [-2e8, 0]
                else:
                    u_par_h_const = [halo_init[1] * u_const_range[0], halo_init[1] * u_const_range[1]]
                halo_const = [[halo_init[0] * n_const_range[0], halo_init[0] * n_const_range[1]],
                            u_par_h_const,
                            [halo_init[2] * u_const_range[0], halo_init[2] * u_const_range[1]],
                            [halo_init[3] * u_const_range[0], halo_init[3] * u_const_range[1]],
                            [1.6, 10]]
                halo_data.update({'halo_init': halo_init, 'halo_const': halo_const})
            continue

        # Iterations >= 2
        if red_chi_h is None:
            print("Invalid Red-Chisqr_halo, continue...")
            continue

        improved = ((abs(red_chi_h - 1.0) < abs(prev_red_chi_h - 1.0)) & 
                   (0.05*prev_red_chi_h <= abs(prev_red_chi_h-red_chi_h)))
        
        if improved:
            print(f'Red-Chisqr_halo: {prev_red_chi_h} -> {red_chi_h} ==> Updating initial guesses.')
            prev_red_chi_h = red_chi_h

            # Update initial guesses based on last fit
            halo_init = [r_halo.params['n'].value, r_halo.params['u_par'].value,
                        r_halo.params['v_th_par'].value, r_halo.params['v_th_perp'].value,
                        r_halo.params['kappa'].value]
            halo_const = [[halo_init[0] * n_const_range[0], halo_init[0] * n_const_range[1]],
                         [halo_init[1] * u_const_range[0], halo_init[1] * u_const_range[1]],
                         [halo_init[2] * u_const_range[0], halo_init[2] * u_const_range[1]],
                         [halo_init[3] * u_const_range[0], halo_init[3] * u_const_range[1]],
                         [1.6, 10]]
            halo_data.update({'halo_init': halo_init, 'halo_const': halo_const})

        if chi_target_range[0] <= red_chi_h <= chi_target_range[1]:
            consecutive_inRange += 1
            print(f"Consecutive in-range count = {consecutive_inRange}")
            if consecutive_inRange >= consecutive_inRange_num:
                print(f"Consecutive in-range count = {consecutive_inRange}. Fitting converged after {iteration} iterations.")
                break
        else:
            consecutive_inRange = 0
            print(f"Out of target Red-Chisqr_halo range. Reset in-range counter.")

    # Sort results by how close red_chi is to 1.0
    iteration_results.sort(key=lambda x: abs((x[2] or float('inf')) - 1.0))
    best_result = iteration_results[0]
    
    chosen_r_halo, chosen_fit_halo, chosen_red_chi_h = best_result
    print(f"Best result with red_chi_halo={chosen_red_chi_h}.")

    return chosen_r_halo, chosen_fit_halo, chosen_red_chi_h

def get_reduced_chi_square_Halo(r_halo, halo_data):
    # Determine which model and how many parameters were used
    
    # temp_b_pa, temp_b_psd, temp_b_v_par, temp_b_v_perp, temp_b_c = slice_beamDataPoints(pa_vec, high_b, low_b)
    if r_halo is None:
        print('Invalid halo fitting.')
        return None
    
    temp_h_psd = halo_data['temp_h_psd']
    temp_h_v_par = halo_data['temp_h_v_par']
    temp_h_v_perp = halo_data['temp_h_v_perp']
    temp_h_unc = halo_data['temp_h_unc']
    mask = halo_data['mask_pitchAngle']

    param_count = 5
    red_chi_halo = None
    
    valid_mask = mask & ~np.isnan(temp_h_psd)
    # Separate zero and nonzero PSD
    nonzero_mask = valid_mask & (temp_h_psd != 0.0)

    fit_val = halo.eval(r_halo.params, v_par=temp_h_v_par[nonzero_mask], v_perp=temp_h_v_perp[nonzero_mask])
    deviation_sqr = (temp_h_psd[nonzero_mask] - fit_val)**2
    uncertainty_sqr = (temp_h_unc[nonzero_mask] )**2
    red_chi_halo = np.sum(deviation_sqr / uncertainty_sqr) / (np.count_nonzero(valid_mask) - param_count) # valid mask or nonzero mask?
    
    return red_chi_halo



# Model definition
def kappa_vdf(v_par, v_perp, n, u_par, v_th_par, v_th_perp, kappa) :

    """
    Produces a VDF following a bi-Kappa distribution. All in CGS units.

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
    v_th_par : float
        Thermal velocity in parallel direction.
    v_th_perp : float
        Thermal velocity in perpendicular direction.
    kappa : float
        Kappa index controls superthermal density, small kappa is large density

    Returns
    -------
    f : array
        The Velocity distribution function VDF.

    """

    denominator = (np.pi ** (3/2)) * v_th_par * (v_th_perp**2)
    term1 = n / denominator

    gamma_ratio = gamma(kappa+1)/(gamma(kappa - 0.5)*((kappa - 3/2)**(3/2)))

    return term1*gamma_ratio*(( 1 + ((v_par-u_par)**2)/((v_th_par**2)*(kappa - 3/2)) + (v_perp**2)/((v_th_perp**2)*(kappa - 3/2)))**(-kappa -1))

def kappa_flattop_vdf(v_par, v_perp, n, u_par, v_th_par, v_th_perp, kappa) :
    """
    Flat top bi-kappa for Halo fitting. All in CGS units.
    """

    denominator = (np.pi ** (3/2)) * v_th_par * (v_th_perp**2)
    term1 = n / denominator

    gamma_ratio = gamma(kappa+1)/(gamma(kappa - 0.5)*((kappa - 3/2)**(3/2)))

    p, q = 10, 1
    kappa_part = term1*gamma_ratio*(( 1 + ((v_par-u_par)**2)/((v_th_par**2)*(kappa - 3/2)) + (v_perp**2)/((v_th_perp**2)*(kappa - 3/2)))**(-kappa -1))
    delta = 4 #np.sqrt(1.08*((1/3)*v_th_par + (2/3)*v_th_perp)) # Jesse's version # 4, for fit example

    c_par = v_par - u_par
    exponent = (c_par**2)/(v_th_par**2) + (v_perp**2)/(v_th_perp**2)
    flattop_part = (1 + (1/delta)*(exponent)**p)**(-q) # 
    
    # u_par_c = r_core.params['u_par'].value
    # v_th_par_c = r_core.params['v_th_par'].value
    # v_th_perp_c = r_core.params['v_th_perp'].value
    # flattop_part = (1 + ((1/delta)*(((v_par-u_par_c)**2)/(v_th_par_c**2) + (v_perp**2)/((v_th_perp_c**2)))**p))**(-q)    

    return kappa_part*(1 - flattop_part)

def bi_kappa_flattop_stepan(v_par, v_perp, n, u_par, v_th_par, v_th_perp, kappa, delta, u_par_c=None, v_th_par_c=None, v_th_perp_c=None):
    """
    Stepan's flat-top kappa distribution, used for halo electrons with core cutoff.
    delta: Width of flat-top is a free parameter.

    Args:
        Same as kappa(), plus:
        r_core (object, optional): Core fit result for flattop parameters
    """
    # Calculate base bi-kappa distribution
    kappa_part = kappa_vdf(v_par, v_perp, n, u_par, v_th_par, v_th_perp, kappa)
    
    p, q = 10, 1
    if u_par_c is None or v_th_par_c is None or v_th_perp_c is None:
        raise ValueError("u_par_c, v_th_par_c, and v_th_perp_c must be provided as keyword arguments")

    c_par = v_par - u_par_c
    exponent = (c_par**2)/(v_th_par_c**2) + (v_perp**2)/((v_th_perp_c**2))
    flattop_part = (1 + ((1/delta)*exponent)**p)**(-q)    

    return kappa_part * (1 - flattop_part)

halo = lmfit.Model(kappa_vdf, independent_vars = ['v_par', 'v_perp'], \
    param_names = ['n', 'u_par', 'v_th_par', 'v_th_perp', 'kappa'])

# halo = lmfit.Model(bi_kappa_flattop_stepan, independent_vars = ['v_par', 'v_perp'], \
#     param_names = ['n', 'u_par', 'v_th_par', 'v_th_perp', 'kappa', 'delta', 'u_par_c', 'v_th_par_c', 'v_th_perp_c'])

halo_flattop = lmfit.Model(kappa_flattop_vdf, independent_vars = ['v_par', 'v_perp'], \
    param_names = ['n', 'u_par', 'v_th_par', 'v_th_perp', 'kappa'], nan_policy = 'omit')
