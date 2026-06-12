import numpy as np
from utils import eV_to_vel, vel_to_eV, calc_beam_moments_from_pixels_with_deltas
from scipy.special import gamma
import lmfit
from constant import PAW_coeff, stderr_fact, n_kappa_factor, n_const_range, u_const_range, n_beam_factor, kappa_b_init, D_CONST, beam_Vth_bound, beam_kappa_bound, PixelCountThreshold_beam


def beam_nonTruncated(v_par, v_perp, n, u_par, v_th_par, v_th_perp, kappa) :
    """
    Bi-kappa distribution for a beam. All in CGS units.

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
    D = 1

    denominator = (np.pi ** (3/2)) * v_th_par * (v_th_perp**2)
    D_coeff = (2*np.sqrt(D_CONST))/np.sqrt(D_CONST+1)
    term1 = (D_coeff*n) / denominator

    gamma_ratio = gamma(kappa+1)/(gamma(kappa - 0.5)*((kappa - 3/2)**(3/2)))

    # term1*gamma_ratio*(( 1 + D*((v_par-u_par)**2)/((v_th_par**2)*(kappa - 3/2)) + 10*(v_perp**2)/((v_th_perp**2)*(kappa - 3/2)))**(-kappa -1))
    return term1*gamma_ratio*(( 1 + ((v_par-u_par)**2)/((v_th_par**2)*(kappa - 3/2)) + (v_perp**2)/((v_th_perp**2)*(kappa - 3/2)))**(-kappa -1))

def beam_truncated(v_par, v_perp, n, u_par, v_th_par, v_th_perp, kappa) :
    D = D_CONST
    denominator = (np.pi ** (3/2)) * v_th_par * (v_th_perp**2)
    D_coeff = (2*np.sqrt(D_CONST))/np.sqrt(D_CONST+1)
    term1 = (D_coeff*n) / denominator
    gamma_ratio = gamma(kappa+1)/(gamma(kappa - 0.5)*((kappa - 3/2)**(3/2)))
    return term1*gamma_ratio*(( 1 + D*((v_par-u_par)**2)/((v_th_par**2)*(kappa - 3/2)) + (v_perp**2)/((v_th_perp**2)*(kappa - 3/2)))**(-kappa -1))

def combined_beam(v_par, v_perp, n, u_par, v_th_par, v_th_perp, kappa) :
    beam_1 = beam_nonTruncated(v_par, v_perp, n, u_par, v_th_par, v_th_perp, kappa)
    beam_2 = beam_truncated(v_par, v_perp, n, u_par, v_th_par, v_th_perp, kappa)
    if u_par < 0 :
        vdf_beam = np.where(v_par < u_par, beam_1, beam_2)
    else :
        vdf_beam = np.where(v_par >= u_par, beam_1, beam_2)
    return vdf_beam

beam = lmfit.Model(combined_beam, independent_vars=['v_par', 'v_perp'], param_names = ['n', 'u_par', 'v_th_par', 'v_th_perp', 'kappa'])

def beam_fit_function(psd_data, v_par, v_perp, counts, beam_const, beam_init, v_par_mesh, v_perp_mesh) :

    ''' set initial beam guess '''
    p_beam = lmfit.Parameters()
    p_beam.add('n', beam_init[0], vary = True, min = beam_const[0][0], max = beam_const[0][1])
    p_beam.add('u_par', beam_init[1], vary = False, min = beam_const[1][0], max = beam_const[1][1])
    p_beam.add('v_th_par', beam_init[2], vary = False, min = beam_const[2][0], max = beam_const[2][1])
    p_beam.add('v_th_perp', beam_init[3], vary = False, min = beam_const[3][0], max = beam_const[3][1])
    p_beam.add('kappa', beam_init[4], vary = False, min = beam_const[4][0], max = beam_const[4][1])

    p_list = list(p_beam.valuesdict().keys())

    # kw_ampgo = {'local_opts':{'maxiter':50, 'ftol':1e-3},
    #                 'totaliter':50, 
    #                 'tabulistsize': 8,
    #                 'eps1': 0.02, 'eps2': 0.01, 'glbtol': 1e-3, 
    #                 'disp':False} # For weighted fitting
    # kw_ampgo = {'local_opts':{'maxiter':1000, 'ftol':0.001*np.mean(psd_data)},
    #                     'totaliter':40,
    #                     'tabulistsize': 5,
    #                     'eps1': 0.01*np.mean(psd_data), 'eps2': 0.02, 'glbtol': 0.001*np.mean(psd_data), 
    #                     'disp':False}
    kw_ampgo = {'local_opts':{'maxiter':2000, 'ftol':1e-31},
                        'totaliter':80,
                        'tabulistsize': 12,
                        'eps1': 1e-30, 'eps2': 0.03, 'glbtol': 1e-29, # originally eps2: 0.01; 0.02 good; 
                        'disp':False}
    
    # # weights of data. To normalize the residuals to ~ 1
    # def numpy_sig_figs(num, sig_figs=3):
    #     return np.array([float(f"{x:.{sig_figs}g}") for x in np.atleast_1d(num)])
    # psd_uncertainty = psd_data/np.sqrt(counts)
    weight_set = 1.0 #1e26 #1.0/np.mean(psd_data)#np.median(psd_uncertainty)

    try :
        r_beam = beam.fit(psd_data, p_beam, method = 'ampgo',max_nfev=50000,fit_kws=kw_ampgo, v_par = v_par, v_perp = v_perp, weights=weight_set) #, cmcee:fit_kws={'steps':6000, 'nwalkers':50, 'burn': 600}
        # r_beam = beam.fit(psd_data, p_beam, method = 'ampgo', v_par = v_par, v_perp = v_perp, weights = 1)#, max_nfev=300) #ampgo'leastsq', max_nfev=300,'least_squares'
    except ValueError :
        # print('beam dint fit')
        r_beam = None

    def fit_update(p_beam, r_beam, vary_list, psd_data, v_par, v_perp, beam_const) :

        p_beam.add('n', r_beam.params['n'].value, vary = vary_list[0], min = beam_const[0][0], max = beam_const[0][1])
        p_beam.add('u_par', r_beam.params['u_par'].value, vary = vary_list[1], min = beam_const[1][0], max = beam_const[1][1])
        p_beam.add('v_th_par', r_beam.params['v_th_par'].value, vary = vary_list[2], min = beam_const[2][0], max = beam_const[2][1])
        p_beam.add('v_th_perp', r_beam.params['v_th_perp'].value, vary = vary_list[3], min = beam_const[3][0], max = beam_const[3][1])
        p_beam.add('kappa', r_beam.params['kappa'].value, vary = vary_list[4], min = beam_const[4][0], max = beam_const[4][1])


        try :
            r_beam = beam.fit(psd_data, p_beam, method = 'ampgo',max_nfev=50000,fit_kws=kw_ampgo, v_par = v_par, v_perp = v_perp, weights=weight_set)
            # r_beam = beam.fit(psd_data, p_beam, method = 'ampgo', v_par = v_par, v_perp = v_perp, weights = weight_set)#, max_nfev=300) #
        except ValueError:
            # print('beam dint fit')
            r_beam = None

        return r_beam

    vary_list = [[True, True, False, False, False], \
                [True, False, False, True, False], \
                [True, False, True, False, False], \
                [False, False, True, True, True]] #[True, False, True, True, True]

    # for i in range(len(vary_list)) :
    for i in range(3) :
        if r_beam != None :
            r_beam = fit_update(p_beam, r_beam, vary_list[i], psd_data, v_par, v_perp, beam_const)

    if r_beam != None :
        fit_beam = beam.eval(r_beam.params, v_par = v_par_mesh, v_perp = v_perp_mesh)
        if r_beam.errorbars == False :
            for i in range(len(p_list)) :
                r_beam.params[p_list[i]].stderr = stderr_fact*np.abs(r_beam.params[p_list[i]].value)

        for i in range(len(p_list)) :
            if r_beam.params[p_list[i]].stderr == 0 :
                r_beam.params[p_list[i]].stderr = stderr_fact*np.abs(r_beam.params[p_list[i]].value)
    else :
        fit_beam = beam.eval(p_beam, v_par = v_par_mesh, v_perp = v_perp_mesh)

    return r_beam, fit_beam


# New functions
def extract_beam_data(pa_vec, swa_energy, dE_upper_eV, dE_lower_eV, high_b, low_b, mean_pad_params, n_in, r_core, r_halo,
                      anti_par_strahl_cond, par_strahl_cond, beam_energy_thresh, n_kappa_factor = n_kappa_factor, PAW_coeff = PAW_coeff):
    """
    Extract and initialize beam data points and parameters based on conditions.
    Returns extracted data, masks, and initial parameters for beam fitting.
    """
    beam_data_dict = {
        'temp_b_pa': None, 'temp_b_psd': None, 'temp_b_v_par': None, 'temp_b_v_perp': None, 'temp_b_c': None,
        'mask_par': None, 'mask_AntiPar': None,
        'beam_init_par': None, 'beam_init_AntiPar': None,
        'beam_const_par': None, 'beam_const_AntiPar': None
    }

    n_beam_init = n_in - r_core.params['n'].value - r_halo.params['n'].value # Total density from. But somethimes n_beam_init < 0
    if n_beam_init < 0:
        n_beam_init = n_in/n_beam_factor
    # if n_beam_init / n_kappa_factor > 0.1*n_in:
    #     n_beam_init = 0.05*n_in/n_kappa_factor
    # u_halo = r_halo.params['u_par'].value
    # u_beam_init = (r_core.params['n'].value*r_core.params['u_par'].value - halo_real_n*u_halo)/n_beam_init # zero-current condition
    # beam_const = [[0, 0.9*n_in], [None, None], [0, None], [0, None], [3/2, None]]

    # Slice beam data points
    temp_b_pa = pa_vec[high_b:low_b, :, 0]
    temp_b_psd = pa_vec[high_b:low_b, :, 1]
    temp_b_v_par = pa_vec[high_b:low_b, :, 2]
    temp_b_v_perp = pa_vec[high_b:low_b, :, 3]
    temp_b_c = pa_vec[high_b:low_b, :, 4]
    beamCount_mask = pa_vec[high_b:low_b, :, 4] >= PixelCountThreshold_beam  # Beam count threshold

    # energy_sel = swa_energy[high_b:low_b]
    # dE_upper_eV_sel = dE_upper_eV[high_b:low_b]
    # dE_lower_eV_sel = dE_lower_eV[high_b:low_b]
    

    if anti_par_strahl_cond and not par_strahl_cond:
        # Anti-parallel beam conditions
        pa_anti_par_bound = 180 - mean_pad_params['W_180'] * PAW_coeff
        if pa_anti_par_bound > 170:
            pa_anti_par_bound = 170
        if pa_anti_par_bound <= 10:
            pa_anti_par_bound = 170

        mask_AntiPar = (temp_b_pa > pa_anti_par_bound) & beamCount_mask

        # beam_moments = calc_beam_moments_from_pixels_with_deltas(temp_b_psd, temp_b_v_par, temp_b_v_perp, mask_AntiPar, energy_sel, dE_upper_eV_sel, dE_lower_eV_sel)
        maxPSD_AntiPar = np.max(temp_b_psd[mask_AntiPar])

        n_beam_init_AntiPar = 0.6*n_beam_init/ n_kappa_factor # 0.2*n_beam_init/ n_kappa_factor(20220306 18:00)
        v_th_b_init_AntiPar = n_kappa_factor * (np.cbrt((n_beam_init_AntiPar / ((np.pi**(3/2))*maxPSD_AntiPar)))) # eV_to_vel(beam_energy_thresh) # 
        u_beam_init_AntiPar = -1*eV_to_vel(beam_energy_thresh)
        
        beam_init_AntiPar = [n_beam_init_AntiPar, u_beam_init_AntiPar, 1.2*v_th_b_init_AntiPar, 0.8*v_th_b_init_AntiPar, kappa_b_init] #0.9n,0.95*upar (sc_moved_test1)# 0.9par 0.7perp
        # [n_beam_init / n_kappa_factor, u_beam_init_AntiPar, 0.85*v_th_b_init_AntiPar, 0.75*v_th_b_init_AntiPar, kappa_b_init]
        # [1.7*n_beam_init, u_beam_init_AntiPar, 1.3*v_th_b_init_AntiPar, 0.9*v_th_b_init_AntiPar, 1.2*kappa_b_init]
        #  #0.7*v_th_b_init_AntiPar
        # #[n_beam_init / n_kappa_factor, u_beam_init_AntiPar, v_th_b_init_AntiPar, v_th_b_init_AntiPar, kappa_b_init]
        beam_const_AntiPar = [[beam_init_AntiPar[0] * n_const_range[0], beam_init_AntiPar[0] * n_const_range[1]],
                               [u_beam_init_AntiPar * u_const_range[0], u_beam_init_AntiPar * u_const_range[1]],
                               beam_Vth_bound, beam_Vth_bound, beam_kappa_bound]
        beam_data_dict.update({'mask_AntiPar': mask_AntiPar, 'beam_init_AntiPar': beam_init_AntiPar, 'beam_const_AntiPar': beam_const_AntiPar})

    elif not anti_par_strahl_cond and par_strahl_cond:
        # Parallel beam conditions
        pa_par_bound = mean_pad_params['W_0'] * PAW_coeff
        if pa_par_bound < 10:
            pa_par_bound = 10

        mask_par = (temp_b_pa < pa_par_bound) & beamCount_mask
        maxPSD_par = np.max(temp_b_psd[mask_par])
        v_th_b_init_par = n_kappa_factor * ((n_beam_init / ((np.pi**(3/2)) * maxPSD_par))**(1/3)) #eV_to_vel(beam_energy_thresh) # 
        u_beam_init_par = eV_to_vel(beam_energy_thresh)
        
        beam_init_par = [n_beam_init / n_kappa_factor, u_beam_init_par, v_th_b_init_par, v_th_b_init_par, kappa_b_init]
        beam_const_par = [[beam_init_par[0] * n_const_range[0], beam_init_par[0] * n_const_range[1]],
                          [u_beam_init_par * u_const_range[0], u_beam_init_par * u_const_range[1]],
                          beam_Vth_bound, beam_Vth_bound, beam_kappa_bound]
        beam_data_dict.update({'mask_par': mask_par, 'beam_init_par': beam_init_par, 'beam_const_par': beam_const_par})

    elif anti_par_strahl_cond and par_strahl_cond:
        # Double beam scenario
        pa_anti_par_bound = 180 - mean_pad_params['W_180'] * PAW_coeff
        pa_par_bound = mean_pad_params['W_0'] * PAW_coeff

        if pa_par_bound < 10:
            pa_par_bound = 10
        if pa_anti_par_bound > 170:
            pa_anti_par_bound = 170

        mask_par = (temp_b_pa < pa_par_bound) & beamCount_mask
        mask_AntiPar = (temp_b_pa > pa_anti_par_bound) & beamCount_mask

        n_beam_init_half = n_beam_init/2

        maxPSD_par = np.max(temp_b_psd[mask_par])
        maxPSD_AntiPar = np.max(temp_b_psd[mask_AntiPar])
        v_th_b_init_par = n_kappa_factor * ((n_beam_init_half / ((np.pi**(3/2)) * maxPSD_par))**(1/3))
        v_th_b_init_AntiPar = n_kappa_factor * ((n_beam_init_half / ((np.pi**(3/2)) * maxPSD_AntiPar))**(1/3))

        u_beam_init_par = eV_to_vel(beam_energy_thresh)
        u_beam_init_AntiPar = -1*eV_to_vel(beam_energy_thresh)

        beam_init_par = [n_beam_init_half / n_kappa_factor, u_beam_init_par, v_th_b_init_par, v_th_b_init_par, kappa_b_init]
        beam_const_par = [[beam_init_par[0] * n_const_range[0], beam_init_par[0] * n_const_range[1]],
                          [u_beam_init_par * u_const_range[0], u_beam_init_par * u_const_range[1]],
                          beam_Vth_bound, beam_Vth_bound, beam_kappa_bound]

        beam_init_AntiPar = [n_beam_init_half / n_kappa_factor, u_beam_init_AntiPar, v_th_b_init_AntiPar, v_th_b_init_AntiPar, kappa_b_init]
        beam_const_AntiPar = [[beam_init_AntiPar[0] * n_const_range[0], beam_init_AntiPar[0] * n_const_range[1]],
                               [u_beam_init_AntiPar * u_const_range[0], u_beam_init_AntiPar * u_const_range[1]],
                               beam_Vth_bound, beam_Vth_bound, beam_kappa_bound]

        beam_data_dict.update({'mask_par': mask_par, 'beam_init_par': beam_init_par, 'beam_const_par': beam_const_par,
                          'mask_AntiPar': mask_AntiPar, 'beam_init_AntiPar': beam_init_AntiPar, 'beam_const_AntiPar': beam_const_AntiPar})

    beam_data_dict.update({
        'temp_b_pa': temp_b_pa,
        'temp_b_psd': temp_b_psd,
        'temp_b_v_par': temp_b_v_par,
        'temp_b_v_perp': temp_b_v_perp,
        'temp_b_c': temp_b_c
    })

    return beam_data_dict

def perform_beam_fit(beam_data, v_par_mesh, v_perp_mesh, anti_par_strahl_cond, par_strahl_cond):
    """
    Handle the beam fitting process based on extracted data.
    Returns r_beam_par, r_beam_anti_par, fit_beam_par, fit_beam_anti_par.
    """
    r_beam_par, r_beam_anti_par = None, None
    fit_beam_par, fit_beam_anti_par = None, None

    # Extract beam data and initialize parameters
    # beam_data = extract_beam_data(pa_vec, high_b, low_b, mean_pad_params, n_in, r_core, halo_real_n, r_halo,
    #                               anti_par_strahl_cond, par_strahl_cond, beamEnergy_low_bound, n_kappa_factor, PAW_coeff)

    mask_par = beam_data['mask_par']
    mask_AntiPar = beam_data['mask_AntiPar']

    # Fit Parallel Beam
    if par_strahl_cond:
        r_beam_par, fit_beam_par = beam_fit_function(
            beam_data['temp_b_psd'][mask_par],
            beam_data['temp_b_v_par'][mask_par],
            beam_data['temp_b_v_perp'][mask_par],
            beam_data['temp_b_c'][mask_par],
            beam_data['beam_const_par'], beam_data['beam_init_par'], v_par_mesh, v_perp_mesh
        )
        if r_beam_par is not None:
            print(f"SumOfSquareResidual: {np.sum(r_beam_par.residual**2)}")

    # Fit Anti-Parallel Beam
    if anti_par_strahl_cond:
        r_beam_anti_par, fit_beam_anti_par = beam_fit_function(
            beam_data['temp_b_psd'][mask_AntiPar],
            beam_data['temp_b_v_par'][mask_AntiPar],
            beam_data['temp_b_v_perp'][mask_AntiPar],
            beam_data['temp_b_c'][mask_AntiPar],
            beam_data['beam_const_AntiPar'], beam_data['beam_init_AntiPar'], v_par_mesh, v_perp_mesh
        )
        if r_beam_anti_par is not None:
            print(f"SumOfSquareResidual: {np.sum(r_beam_anti_par.residual**2)}")

    return r_beam_par, fit_beam_par, r_beam_anti_par, fit_beam_anti_par

# Iterative beam fitting
def iterative_beam_fitting(beam_data, v_par_mesh, v_perp_mesh, anti_par_strahl_cond, par_strahl_cond, 
                           beamEnergy_low_bound=70, max_iterations=30, chi_target_range=(0.9, 1.1)):
   
    r_beam_par, fit_beam_par, r_beam_anti_par, fit_beam_anti_par = None, None, None, None
    red_chi_b = None

    # If no beams, just return
    if not par_strahl_cond and not anti_par_strahl_cond:
        return r_beam_par, fit_beam_par, r_beam_anti_par, fit_beam_anti_par, red_chi_b

    # Iterative fitting
    iteration_results = []
    for iteration in range(1, max_iterations + 1):
        print(f"Beam fitting iteration {iteration}.")

        # Fit again
        r_beam_par, fit_beam_par, r_beam_anti_par, fit_beam_anti_par = perform_beam_fit( beam_data, v_par_mesh, v_perp_mesh, anti_par_strahl_cond, par_strahl_cond)
        
        # If can't fit, return None
        if r_beam_par is None and r_beam_anti_par is None:
            return r_beam_par, fit_beam_par, r_beam_anti_par, fit_beam_anti_par, red_chi_b
        
        red_chi_b = get_reduced_chi_square_Beam(r_beam_par, r_beam_anti_par, beam_data, anti_par_strahl_cond, par_strahl_cond)
        print(f"Red. chi-square for beam: {red_chi_b}")

        # record results
        iteration_results.append((r_beam_par, fit_beam_par, r_beam_anti_par, fit_beam_anti_par, red_chi_b))

        if red_chi_b is not None and (chi_target_range[0] <= red_chi_b <= chi_target_range[1]):
            print(f"Beam fitting converged with red_chi_beam={red_chi_b} after {iteration} iterations.")
            return r_beam_par, fit_beam_par, r_beam_anti_par, fit_beam_anti_par, red_chi_b
        else:
            print(f"Beam fitting did not converge to desired chi-square range after {iteration} iterations.")

    # Choose the iteration with red_chi_beam closest to chi_target_mid
    best_diff = float('inf')
    best_result = iteration_results[0]  # fallback in case all are None
    chi_target = 1
    for res in iteration_results:
        _, _, _, _, this_chi = res
        if this_chi is not None:
            diff = abs(this_chi - chi_target)
            if diff < best_diff:
                best_diff = diff
                best_result = res

    chosen_r_beam_par, chosen_fit_beam_par, chosen_r_beam_anti_par, chosen_fit_beam_anti_par, chosen_red_chi_b = best_result
    print(f"Using closest result with red_chi_beam={chosen_red_chi_b}.")

    return chosen_r_beam_par, chosen_fit_beam_par, chosen_r_beam_anti_par, chosen_fit_beam_anti_par, chosen_red_chi_b

def get_reduced_chi_square_Beam(r_beam_par, r_beam_anti_par, beam_data, anti_par_strahl_cond, par_strahl_cond):
    # Determine which model and how many parameters were used
    beam_count = int(anti_par_strahl_cond) + int(par_strahl_cond)
    # temp_b_pa, temp_b_psd, temp_b_v_par, temp_b_v_perp, temp_b_c = slice_beamDataPoints(pa_vec, high_b, low_b)
    temp_b_pa = beam_data['temp_b_pa']
    temp_b_psd = beam_data['temp_b_psd']
    temp_b_v_par = beam_data['temp_b_v_par']
    temp_b_v_perp = beam_data['temp_b_v_perp']
    temp_b_c = beam_data['temp_b_c']
    
    red_chi_beam = None
    
    if beam_count == 0:
        return None
    elif anti_par_strahl_cond and not par_strahl_cond:
        param_count = 5
        mask = beam_data['mask_AntiPar']
        r_beam = r_beam_anti_par
    elif not anti_par_strahl_cond and par_strahl_cond:
        param_count = 5
        mask = beam_data['mask_par']
        r_beam = r_beam_par
    else:
        param_count = 10
    
    if beam_count == 1:
        valid_mask = mask & ~np.isnan(temp_b_psd)
        # Separate zero and nonzero PSD
        nonzero_mask = valid_mask & (temp_b_psd != 0.0)

        fit_val = beam.eval(r_beam.params, v_par=temp_b_v_par[nonzero_mask], v_perp=temp_b_v_perp[nonzero_mask])
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

        
        fit_val_par = beam.eval(r_beam_par.params, v_par=temp_b_v_par[nonzero_mask_par], v_perp=temp_b_v_perp[nonzero_mask_par])
        fit_val_Antipar = beam.eval(r_beam_anti_par.params, v_par=temp_b_v_par[nonzero_mask_anti_par], v_perp=temp_b_v_perp[nonzero_mask_anti_par])
        chiSqr_par = np.sum( ((temp_b_psd[nonzero_mask_par] - fit_val_par)**2) / ((temp_b_psd[nonzero_mask_par] / np.sqrt(temp_b_c[nonzero_mask_par]))**2) )
        chiSqr_Antipar = np.sum( ((temp_b_psd[nonzero_mask_anti_par] - fit_val_Antipar)**2) / ((temp_b_psd[nonzero_mask_anti_par] / np.sqrt(temp_b_c[nonzero_mask_anti_par]))**2) )
        
        dof = len(temp_b_psd[valid_mask_par]) + len(temp_b_psd[valid_mask_anti_par]) - param_count
        red_chi_beam = (chiSqr_par + chiSqr_Antipar) / dof
    return red_chi_beam


def random_search_fitting():

    pass

def Adapative_beam_fitting(beam_data, v_par_mesh, v_perp_mesh, anti_par_strahl_cond, par_strahl_cond, 
                           beamEnergy_low_bound=70, max_iterations=30, chi_target_range=(0.6, 1.1), consecutive_inRange_num=5):
   
    r_beam_par, fit_beam_par, r_beam_anti_par, fit_beam_anti_par = None, None, None, None
    red_chi_b = None
    refinement_mode = False
    prev_red_chi_b = float('inf')
    consecutive_inRange = 0

    # If no beams, just return
    if not par_strahl_cond and not anti_par_strahl_cond:
        return r_beam_par, fit_beam_par, r_beam_anti_par, fit_beam_anti_par, red_chi_b

    iteration_results = []
    
    # Iterative fitting
    for iteration in range(1, max_iterations + 1):
        print(f"\nBeam fitting iteration {iteration}.")

        # Fit
        r_beam_par, fit_beam_par, r_beam_anti_par, fit_beam_anti_par = perform_beam_fit( beam_data, v_par_mesh, v_perp_mesh, anti_par_strahl_cond, par_strahl_cond)
        if r_beam_par is None and r_beam_anti_par is None: # If can't fit, do next iteration
            if iteration == 1:
                print("Beam fitting failed: No valid fits.")
                return None, None, None, None, None
            else:
                print('Fitting failed. Try again.')
                continue
        red_chi_b = get_reduced_chi_square_Beam(r_beam_par, r_beam_anti_par, beam_data, anti_par_strahl_cond, par_strahl_cond)
        print(f"Red. chi-square for beam: {red_chi_b}")
        iteration_results.append((r_beam_par, fit_beam_par, r_beam_anti_par, fit_beam_anti_par, red_chi_b)) # record results

        # First iteration, special handling
        if iteration == 1:
            if red_chi_b is None:
                print("Invalid Red-Chisqr_beam on 1st iteration, continue...")
                continue
            prev_red_chi_b = red_chi_b
            
            if 0.6 <= red_chi_b <= 1.1:
                print(f'A good 1st time attempt ==> Updating initial guesses.')
                # Update initial guesses based on last fit
                if r_beam_par is not None:
                    beam_init_par = [r_beam_par.params['n'].value, r_beam_par.params['u_par'].value, r_beam_par.params['v_th_par'].value, r_beam_par.params['v_th_perp'].value, r_beam_par.params['kappa'].value]
                    beam_const_par = [[beam_init_par[0] * n_const_range[0], beam_init_par[0] * n_const_range[1]],
                                    [beam_init_par[1] * u_const_range[0], beam_init_par[1] * u_const_range[1]],
                                    [beam_init_par[2] * n_const_range[0], beam_init_par[2] * n_const_range[1]],
                                    [beam_init_par[3] * n_const_range[0], beam_init_par[3] * n_const_range[1]],
                            [None, None]]
                    beam_data.update({'beam_init_par': beam_init_par, 'beam_const_par': beam_const_par})
                if r_beam_anti_par is not None:
                    beam_init_AntiPar = [r_beam_anti_par.params['n'].value, r_beam_anti_par.params['u_par'].value, r_beam_anti_par.params['v_th_par'].value, r_beam_anti_par.params['v_th_perp'].value, r_beam_anti_par.params['kappa'].value]
                    beam_const_AntiPar = [[beam_init_AntiPar[0] * n_const_range[0], beam_init_AntiPar[0] * n_const_range[1]],
                                [beam_init_AntiPar[1] * u_const_range[0], beam_init_AntiPar[1] * u_const_range[1]],
                                [None, None], [None, None], [None, None]]
                    beam_data.update({'beam_init_AntiPar': beam_init_AntiPar, 'beam_const_AntiPar': beam_const_AntiPar})
            continue
        
        # Iterations >= 2
        if red_chi_b is None:
            print("Invalid Red-Chisqr_beam, continue...")
            continue

        improved = ((abs(red_chi_b - 0.8) < abs(prev_red_chi_b - 0.8)) & (0.05*prev_red_chi_b <= abs(prev_red_chi_b-red_chi_b))) # check for improvement
        if improved:
            print(f'Red-Chisqr_beam: {prev_red_chi_b} -> {red_chi_b} ==> Updating initial guesses.')
            prev_red_chi_b = red_chi_b

            # Update initial guesses based on last fit
            if r_beam_par is not None:
                beam_init_par = [r_beam_par.params['n'].value, r_beam_par.params['u_par'].value, r_beam_par.params['v_th_par'].value, r_beam_par.params['v_th_perp'].value, r_beam_par.params['kappa'].value]
                beam_const_par = [[beam_init_par[0] * n_const_range[0], beam_init_par[0] * n_const_range[1]],
                                  [beam_init_par[1] * u_const_range[0], beam_init_par[1] * u_const_range[1]],
                                  [beam_init_par[2] * n_const_range[0], beam_init_par[2] * n_const_range[1]],
                                  [beam_init_par[3] * n_const_range[0], beam_init_par[3] * n_const_range[1]],
                          [None, None]]
                beam_data.update({'beam_init_par': beam_init_par, 'beam_const_par': beam_const_par})
            if r_beam_anti_par is not None:
                beam_init_AntiPar = [r_beam_anti_par.params['n'].value, r_beam_anti_par.params['u_par'].value, r_beam_anti_par.params['v_th_par'].value, r_beam_anti_par.params['v_th_perp'].value, r_beam_anti_par.params['kappa'].value]
                beam_const_AntiPar = [[beam_init_AntiPar[0] * n_const_range[0], beam_init_AntiPar[0] * n_const_range[1]],
                               [beam_init_AntiPar[1] * u_const_range[0], beam_init_AntiPar[1] * u_const_range[1]],
                               [None, None], [None, None], [None, None]]
                beam_data.update({'beam_init_AntiPar': beam_init_AntiPar, 'beam_const_AntiPar': beam_const_AntiPar})
        
        if chi_target_range[0] <= red_chi_b <= chi_target_range[1]: # Check whether converged to target
            consecutive_inRange += 1
            print(f"Consecutive in-range count = {consecutive_inRange}")
            if consecutive_inRange >= consecutive_inRange_num:
                print(f"Consecutive in-range count = {consecutive_inRange}. Fitting converged after {iteration} iterations.")
                break
        else:
            consecutive_inRange = 0
            print(f"Out of target Red-Chisqr_beam range. Reset in-range counter.")
            # if not refinement_mode:
            #     print(f"Beam fitting converged. Red_chi_beam={red_chi_b} after {iteration} iterations ==> Refinement mode.")
            #     refinement_mode = True
            #     extra_refinement_left = extra_refinement
            # else:
            #     if improved:
            #         extra_refinement_left = extra_refinement
            #     else:
            #         extra_refinement_left -= 1
    
        # Perform extra iterations for refinement
        # if refinement_mode:
        #     # If no extra iters remain, break
        #     if extra_refinement_left <= 0:
        #         print("No further improvement in refinement mode. Stopping Phase 2.")
        #         break
    target_chi = 0.8
    iteration_results.sort(key=lambda x: abs((x[4] or float('inf')) - target_chi))
    best_result = iteration_results[0]
    # # Choose the iteration with red_chi_beam closest to chi_target_mid
    # best_diff = float('inf')
    # best_result = iteration_results[0]  # fallback in case all are None
    # chi_target = 1
    # for res in iteration_results:
    #     _, _, _, _, this_chi = res
    #     if this_chi is not None:
    #         diff = abs(this_chi - chi_target)
    #         if diff < best_diff:
    #             best_diff = diff
    #             best_result = res

    chosen_r_beam_par, chosen_fit_beam_par, chosen_r_beam_anti_par, chosen_fit_beam_anti_par, chosen_red_chi_b = best_result
    print(f"Best result with red_chi_beam={chosen_red_chi_b}.")
    if chosen_r_beam_par is not None:
        print(f"corresponding beam params (para): {chosen_r_beam_par.params}")
    if chosen_r_beam_anti_par is not None:
        print(f"corresponding beam params (anti-para): {chosen_r_beam_anti_par.params}")

    return chosen_r_beam_par, chosen_fit_beam_par, chosen_r_beam_anti_par, chosen_fit_beam_anti_par, chosen_red_chi_b


# # %%
# from utils import eV_to_vel
# eV_to_vel(600)
# red_chi_b = 1.5
# prev_red_chi_b = 2.2
# ((abs(red_chi_b - 1.0) < abs(prev_red_chi_b - 1.0)) & (0.2*prev_red_chi_b <= abs(prev_red_chi_b-red_chi_b)))
# 1.1*0.2