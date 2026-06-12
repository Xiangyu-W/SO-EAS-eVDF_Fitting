# src/pad_fitting.py

import numpy as np
import lmfit
from constant import PAW_coeff

# Constants


# Import the pad_model from your fit functions (not provided fully here).
# from fit_functions import pad_model, pitch_angle_model # Adjust as needed.
def pitch_angle_model(alpha, P_B, P_0, W_0, P_180, W_180) :
    """
    Produces a VDF following a bi-Maxwellian distribution. All in CGS units.

    Parameters
    ----------
    alpha : array
        pitch-angles
    P_B : float
        Base level phase space density
    P_0 : float
        Gaussian height at alpha = 0
    W_0 : float
        Full-width-half-maximum at alpha = 0
    P_180 : float
        Gaussian height at alpha = 180
    W_180 : float
        Full-width-half-maximum at alpha = 180

    Returns
    -------
    f : array
        The pitch angle DF
    """

    return P_B + P_0 * np.exp( -(1/2)*((alpha/(np.sqrt(2)*W_0))**2) ) + P_180 * np.exp( -(1/2)*(((alpha - 180)/(np.sqrt(2)*W_180))**2) )

pad_model = lmfit.Model(pitch_angle_model, independent_vars = ['alpha'], \
    param_names = ['P_B', 'P_0', 'W_0', 'P_180', 'W_180'], nan_policy = 'omit')


def pad_fit_function(meanEnergy_PSD, pitchAngle_center, pitch_angles) :

    '''param_names = ['P_B', 'P_0', 'W_0', 'P_180', 'W_180']'''

    pitchAngle_center = pitchAngle_center[~np.isnan(meanEnergy_PSD)]
    meanEnergy_PSD = meanEnergy_PSD[~np.isnan(meanEnergy_PSD)]

    pad_const = [[np.nanmean(meanEnergy_PSD)*0.9, None], [None, None], [0, 45], [None, None], [0, 45]]
    p_pad = lmfit.Parameters()
    p_pad.add('P_B', np.nanmean(meanEnergy_PSD), vary = False, min = pad_const[0][0], max = pad_const[0][1])
    p_pad.add('P_0', np.sign(meanEnergy_PSD[0] - np.nanmean(meanEnergy_PSD))*meanEnergy_PSD[0], vary = False, min = pad_const[1][0], max = pad_const[1][1])
    p_pad.add('W_0', 10, vary = False, min = pad_const[2][0], max = pad_const[2][1])
    p_pad.add('P_180', np.sign(meanEnergy_PSD[-1] - np.nanmean(meanEnergy_PSD))*meanEnergy_PSD[-1], vary = False, min = pad_const[3][0], max = pad_const[3][1])
    p_pad.add('W_180', 10, vary = False, min = pad_const[4][0], max = pad_const[4][1])

    ''' method employed here '''
    try :
        r_pad = pad_model.fit(meanEnergy_PSD, p_pad, method = 'leastsq', alpha = pitchAngle_center, max_nfev=300, nan_policy = 'omit')
    except (ValueError, TypeError) :
        r_pad = None

    def fit_update(p_pad, r_pad, vary_list, meanEnergy_PSD, alpha, pad_const) :
        p_pad.add('P_B', r_pad.params['P_B'].value, vary = vary_list[0], min = pad_const[0][0], max = pad_const[0][1])
        p_pad.add('P_0', r_pad.params['P_0'].value, vary = vary_list[1], min = pad_const[1][0], max = pad_const[1][1])
        p_pad.add('W_0', r_pad.params['W_0'].value, vary = vary_list[2], min = pad_const[2][0], max = pad_const[2][1])
        p_pad.add('P_180', r_pad.params['P_180'].value, vary = vary_list[3], min = pad_const[3][0], max = pad_const[3][1])
        p_pad.add('W_180', r_pad.params['W_180'].value, vary = vary_list[4], min = pad_const[4][0], max = pad_const[4][1])

        try :
            r_pad = pad_model.fit(meanEnergy_PSD, p_pad, method = 'leastsq', alpha = pitchAngle_center, max_nfev=300, nan_policy = 'omit')
        except (ValueError, TypeError) :
            # print('pad dint fit')

            r_pad = None

        return r_pad

    vary_list = [[True, False, False, False, False], \
    [False, True, False, True, False], \
    [True, False, True, False, True]]#, \
    # [True, True, True, True, True]]

    for i in range(len(vary_list)) :
        if r_pad != None :
            r_pad = fit_update(p_pad, r_pad, vary_list[i], meanEnergy_PSD, pitchAngle_center, pad_const)

    ''' put results back into fit function '''
    if r_pad != None :
        fit_pad = pad_model.eval(r_pad.params, alpha = pitch_angles)
        return r_pad.params, fit_pad
    else:
        # print('dint fit pad model')
        fit_pad = pad_model.eval(p_pad, alpha = pitch_angles)
        return p_pad, fit_pad

def pad_fit_operation(pad, swa_energy, pitch_angle_centers, pitch_angles):

    pad_params_energy, pad_fit_energy = [], []
    for j in range(len(swa_energy)) :
        data_in = pad[j,:,0]
        mask = (data_in != 0.0)
        if len(data_in[mask]) < 4 : # essentially if there were no points.
            temp = [np.nan, np.nan, np.nan, np.nan, np.nan]
            temp_fit_pad = np.full(len(pitch_angles), np.nan)
        else :
            temp_params, temp_fit_pad = pad_fit_function(data_in[mask], pitch_angle_centers[mask], pitch_angles)
            temp = [temp_params['W_0'].value, temp_params['W_180'].value, temp_params['P_0'].value, temp_params['P_180'].value, temp_params['P_B'].value]
        pad_params_energy.append(np.array(temp))
        pad_fit_energy.append(np.array(temp_fit_pad))

    pad_params_energy = np.array(pad_params_energy)
    pad_fit_energy = np.array(pad_fit_energy)

    return np.array(pad_params_energy), np.array(pad_fit_energy)

def evaluate_pad_conditions(mean_pad_params, beam_ratio_cond, deficit_ratio_cond, pitch_angle_centers):
    # Unpack parameters
    P_B = mean_pad_params['P_B'].value if 'P_B' in mean_pad_params else np.nan
    P_0 = mean_pad_params['P_0'].value if 'P_0' in mean_pad_params else np.nan
    P_180 = mean_pad_params['P_180'].value if 'P_180' in mean_pad_params else np.nan

    # Jesse's criterion
    # par_strahl_cond = ((P_0 / P_B) + 1 > beam_ratio_cond) if (not np.isnan(P_0) and not np.isnan(P_B)) else False
    # par_def_cond = ((P_0 / P_B) < deficit_ratio_cond) if (not np.isnan(P_0) and not np.isnan(P_B)) else False
    # anti_par_strahl_cond = ((P_180 / P_B) + 1 > beam_ratio_cond) if (not np.isnan(P_180) and not np.isnan(P_B)) else False
    # anti_par_def_cond = ((P_180 / P_B) < deficit_ratio_cond) if (not np.isnan(P_180) and not np.isnan(P_B)) else False

    PA_anti_deg = PAW_coeff*mean_pad_params['W_180']
    PA_par_deg = PAW_coeff*mean_pad_params['W_0']
    
    # # Chris's paper
    PA_arr = np.arange(0, 185, 5)
    beta_par = np.mean(pad_model.eval(mean_pad_params, alpha = PA_arr[PA_arr <= 45])) / np.mean(pad_model.eval(mean_pad_params, alpha = PA_arr[(PA_arr >= 65) & (PA_arr <= 115)]))-1
    beta_anti_par = np.mean(pad_model.eval(mean_pad_params, alpha = PA_arr[(PA_arr >= 135) & (PA_arr <= 180)]))/np.mean(pad_model.eval(mean_pad_params, alpha = PA_arr[(PA_arr >= 65) & (PA_arr <= 115)]))-1
    par_strahl_cond = (beta_par > 0.6) & (PA_par_deg <= 45) # earlier version did not constrain the strahl pitch angle # & (PA_par_deg < 45)# chris's paper: 2
    par_def_cond = (beta_par < -0.5)
    anti_par_strahl_cond = (beta_anti_par > 0.6) & (PA_anti_deg <= 45) # earlier version did not constrain the strahl pitch angle # & (PA_anti_deg < 45) # chris's paper: 2
    anti_par_def_cond = (beta_anti_par < -0.5)
    return par_strahl_cond, anti_par_strahl_cond, par_def_cond, anti_par_def_cond

