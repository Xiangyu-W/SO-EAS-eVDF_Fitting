from Overall_fitting import vdf_1beam, vdf_2beam
from Beam_fitting import beam
from Halo_fitting import halo_flattop, halo
from utils import int_vdf, moment1st_VDF_para, moment2nd_VDF, vel_to_eV
import lmfit
import numpy as np


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


def compute_reduced_chi_square_HaloOnly(r_vdf, halo_data):
    """
    Compute the reduced chi-square for the halo component alone, over the halo energy range.
    """
    if r_vdf is None:
        return None
    
    # Extract data from the dictionary
    temp_h_psd = halo_data['temp_h_psd']
    temp_h_v_par = halo_data['temp_h_v_par']
    temp_h_v_perp = halo_data['temp_h_v_perp']
    temp_h_c = halo_data['temp_h_count']  # counts for uncertainty calculation
    mask = halo_data['mask_pitchAngle']
    
    # Halo has 5 parameters: n, u_par, v_th_par, v_th_perp, kappa
    param_count = 5
    
    # Create valid mask for data points
    valid_mask = mask & ~np.isnan(temp_h_psd)
    
    # Separate zero and nonzero PSD values
    nonzero_mask = valid_mask & (temp_h_psd != 0.0)
    
    # Create halo parameters object from the overall fit results
    halo_params = lmfit.Parameters()
    halo_params.add('n', value=r_vdf.params['n_h'].value, vary=False)
    halo_params.add('u_par', value=r_vdf.params['u_par_h'].value, vary=False)
    halo_params.add('v_th_par', value=r_vdf.params['v_th_par_h'].value, vary=False)
    halo_params.add('v_th_perp', value=r_vdf.params['v_th_perp_h'].value, vary=False)
    halo_params.add('kappa', value=r_vdf.params['kappa_h'].value, vary=False)
    
    # Evaluate halo model at data points
    fit_val = halo.eval(halo_params, v_par=temp_h_v_par[nonzero_mask], v_perp=temp_h_v_perp[nonzero_mask])
    
    # Calculate squared deviations and uncertainties
    deviation_sqr = (temp_h_psd[nonzero_mask] - fit_val)**2
    uncertainty_sqr = (temp_h_psd[nonzero_mask] / np.sqrt(temp_h_c[nonzero_mask]))**2
    
    # Calculate chi-square and degrees of freedom
    chi_square = np.sum(deviation_sqr / uncertainty_sqr)
    dof = np.count_nonzero(valid_mask) - param_count
    
    # Calculate reduced chi-square if we have enough data points
    if dof > 0:
        red_chi_halo = chi_square / dof
    else:
        red_chi_halo = None
        
    return red_chi_halo
    
    

# Calc 2nd moment of beam VDF
def calc_moment2nd_beam(beamParams, v_par_mesh, v_perp_mesh, v_par_arr, v_perp_arr):
    # Calc 2nd moment of beam VDF
    fit_vdf_beamDistribution = beam.eval(beamParams, v_par = v_par_mesh, v_perp = v_perp_mesh)

    n_beam = int_vdf(fit_vdf_beamDistribution, v_par_arr, v_perp_arr) # electron density. Verified, result correct.
    vBulk_par = moment1st_VDF_para(fit_vdf_beamDistribution, v_par_arr, v_perp_arr)/n_beam # electron para bulk velocity. Verified, result correct.
    p_para_b = moment2nd_VDF(fit_vdf_beamDistribution, v_par_arr, v_perp_arr, vBulk_par) # electron para pressure. verified, result correct.

    eV_to_Erg = 1.602*(10.**(-12.)) # erg/eV
    T_para_b = p_para_b/n_beam/eV_to_Erg # temperature in eV. verified, result correct.
    return T_para_b

def compute_beam_temperature(r_vdf, v_par_mesh, v_perp_mesh, v_par_arr, v_perp_arr, anti_par_strahl_cond, par_strahl_cond):
    
    if r_vdf is None:
        return np.nan, np.nan
    
    beam_count = int(anti_par_strahl_cond) + int(par_strahl_cond)

    if beam_count == 0:
        T_para_b = np.nan
        T_antiPara_b = np.nan
    if beam_count == 1:
        beamParams = lmfit.Parameters()
        beamParams.add('n', value = r_vdf.params['n_b'].value)
        beamParams.add('u_par', value = r_vdf.params['u_par_b'].value)
        beamParams.add('v_th_par', value = r_vdf.params['v_th_par_b'].value)
        beamParams.add('v_th_perp', value = r_vdf.params['v_th_perp_b'].value)
        beamParams.add('kappa', value = r_vdf.params['kappa_b'].value)
        tempT = calc_moment2nd_beam(beamParams, v_par_mesh, v_perp_mesh, v_par_arr, v_perp_arr) # temperature in eV. verified, result correct.
        
        if anti_par_strahl_cond and not par_strahl_cond:
            T_antiPara_b = tempT
            T_para_b = np.nan
        elif not anti_par_strahl_cond and par_strahl_cond:
            T_antiPara_b = np.nan
            T_para_b = tempT

    if beam_count == 2:
        beamParams = lmfit.Parameters()
        beamParams.add('n', value = r_vdf.params['n_b_par'].value)
        beamParams.add('u_par', value = r_vdf.params['u_par_b_par'].value)
        beamParams.add('v_th_par', value = r_vdf.params['v_th_par_b_par'].value)
        beamParams.add('v_th_perp', value = r_vdf.params['v_th_perp_b_par'].value)
        beamParams.add('kappa', value = r_vdf.params['kappa_b_par'].value)
        T_para_b = calc_moment2nd_beam(beamParams, v_par_mesh, v_perp_mesh, v_par_arr, v_perp_arr) # temperature in eV. verified, result correct.
        # (2) anti-para beam
        antiBeamParams = lmfit.Parameters()
        antiBeamParams.add('n', value = r_vdf.params['n_b_anti_par'].value)
        antiBeamParams.add('u_par', value = r_vdf.params['u_par_b_anti_par'].value)
        antiBeamParams.add('v_th_par', value = r_vdf.params['v_th_par_b_anti_par'].value)
        antiBeamParams.add('v_th_perp', value = r_vdf.params['v_th_perp_b_anti_par'].value)
        antiBeamParams.add('kappa', value = r_vdf.params['kappa_b_anti_par'].value)
        T_antiPara_b = calc_moment2nd_beam(antiBeamParams, v_par_mesh, v_perp_mesh, v_par_arr, v_perp_arr) # temperature in eV. verified, result correct.

    return T_para_b, T_antiPara_b

# def calculate_halo_temperature(r_vdf, v_par_mesh, v_perp_mesh, v_par_arr, v_perp_arr):
    # """
    # Calculate the halo temperature. Incorrect, because flat-top halo model in our study is different from Stepan's.
    # """
    # if r_vdf is None:
    #     return np.nan
    
    # haloParams = lmfit.Parameters()
    # haloParams.add('n', value = r_vdf.params['n_h'].value)
    # haloParams.add('u_par', value = r_vdf.params['u_par_h'].value)
    # haloParams.add('v_th_par', value = r_vdf.params['v_th_par_h'].value)
    # haloParams.add('v_th_perp', value = r_vdf.params['v_th_perp_h'].value)
    # haloParams.add('kappa', value = r_vdf.params['kappa_h'].value)
    

    # # Calc 2nd moment of beam VDF
    # # fit_vdf_Halo = halo_flattop.eval(r_vdf.params, v_par = v_par_mesh, v_perp = v_perp_mesh)
    # fit_vdf_Halo = halo_flattop.eval(haloParams, v_par = v_par_mesh, v_perp = v_perp_mesh)

    # n_halo = int_vdf(fit_vdf_Halo, v_par_arr, v_perp_arr) # electron density. 
    # vBulk_par = moment1st_VDF_para(fit_vdf_Halo, v_par_arr, v_perp_arr)/n_halo # electron para bulk velocity. 
    # p_para_h = moment2nd_VDF(fit_vdf_Halo, v_par_arr, v_perp_arr, vBulk_par) # electron para pressure.

    # eV_to_Erg = 1.602*(10.**(-12.)) # erg/eV
    # T_para_h = p_para_h/n_halo/eV_to_Erg # temperature in eV. verified, result correct.
    # return T_para_h