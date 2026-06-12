# src/utils.py
# %%
import numpy as np
import pandas as pd
import re
import lmfit
import math
from scipy.signal import savgol_filter
from math import floor, log10
from pandas import DataFrame
from constant import m_e, eV_to_Erg, m_p, break_E_range


def sh(x):
    print(np.shape(x))

def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)

def make_df(path, data, name):
    DataFrame(data).to_pickle(path+name)

def find_nearest(array, value):
    idx = int(np.searchsorted(array, value, side="left"))
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return int(idx-1)
    else:
        return int(idx)

def eV_to_vel(input) :
    return np.sqrt((2*input*eV_to_Erg)/m_e) # cm/s

def vel_to_eV(input) :
    return (m_e/2)*(input**2)*(1/eV_to_Erg)

def vel_to_eV_p(input) :
    return (m_p/2)*(input**2)*(1/eV_to_Erg)

def eV_to_Kelvin(eV):
    return eV/(8.617323E-05) # E(eV) = k_b*T (kelvin)

def round_to_n(x, n) :
    if np.sign(x) == 1 :
        return np.round(x, -int(floor(log10(x))) + (n - 1))
    if np.sign(x) == -1 :
        return -1*np.round(np.abs(x), -int(floor(log10(np.abs(x)))) + (n - 1))

def smooth_breakE_values(break_E_values):
    # Check if we have enough data for Savitzky-Golay filtering
    if len(break_E_values) >= 11:
        # Apply Savitzky-Golay filter with window length 11 and 2nd order polynomial
        return savgol_filter(break_E_values.flatten(), window_length=11, polyorder=2)
    if len(break_E_values) >= 3:
        window_length = min(len(break_E_values), (len(break_E_values) // 2) * 2 + 1)
        window_length = max(window_length, 3)
        return savgol_filter(break_E_values.flatten(), window_length=window_length, polyorder=2)
    return break_E_values.flatten()

def process_breakE_dict(break_v_dict):

    # First, process all timestamps to get average break energies
    all_break_E_means = []
    all_timestamps = []
    for i, ts in enumerate(break_v_dict['timestamps']):
        temp_break_v = break_v_dict['break_v'][i]
        break_E_arr = vel_to_eV(temp_break_v.values if isinstance(temp_break_v, pd.Series) else np.array([temp_break_v])) # to handle the case when break_v is np.nan
        break_E, useful_breakE = check_breakE_range(break_E_arr)

        if useful_breakE:
            # If break_E is an array, take the mean
            if isinstance(break_E, np.ndarray) and break_E.size > 1:
                mean_break_E = np.mean(break_E)
            else:
                mean_break_E = break_E

            all_break_E_means.append(mean_break_E)
            all_timestamps.append(ts)

    # Convert to numpy arrays for easier processing
    all_break_E_means = np.array(all_break_E_means)
    all_timestamps = np.array(all_timestamps)

    filtered_break_E = smooth_breakE_values(all_break_E_means)
    return {'timestamps': all_timestamps, 'break_E': filtered_break_E}

def process_breakE_dataframe(break_E_df):
    '''
    Build the filtered break-energy dict from a cone-feature-split result DataFrame
    (compute_break_energy_cone_features output with 'timestamp' and 'break_E_eV' columns).
    Keeps epochs whose break energy is finite and inside break_E_range, then applies the
    same Savitzky-Golay smoothing as process_breakE_dict. The returned dict is meant for
    get_break_E(filtered_data=...).
    '''
    break_E_values = break_E_df['break_E_eV'].to_numpy(dtype=float)
    in_range_mask = (
        np.isfinite(break_E_values)
        & (break_E_values >= break_E_range[0])
        & (break_E_values <= break_E_range[1])
    )
    all_timestamps = pd.to_datetime(break_E_df['timestamp']).to_numpy()[in_range_mask]
    filtered_break_E = smooth_breakE_values(break_E_values[in_range_mask])
    return {'timestamps': all_timestamps, 'break_E': filtered_break_E}

def check_breakE_range(break_E_arr):
    upper_limit = break_E_range[1] # eV
    lower_limit = break_E_range[0] # eV
    break_E = np.nan
    useful_breakE = True

    if break_E_arr.size >1:
        element_inRangeCheck = break_E_arr[(break_E_arr>=lower_limit) & (break_E_arr<=upper_limit)].size
        if element_inRangeCheck == 0:
            useful_breakE = False
        else:
            break_E = break_E_arr[(break_E_arr>=lower_limit) & (break_E_arr<=upper_limit)]
    elif break_E_arr.size == 1:
        if break_E_arr>=lower_limit and break_E_arr<=upper_limit:
            break_E = break_E_arr
        else:
            useful_breakE = False
    else:
        useful_breakE = False
    
    return break_E, useful_breakE

def get_break_E(break_v_dict, timestamp, filtered_data=None):
    '''
    Get the break energy at a given timestamp. Or from a 10-min window.
    '''
    if filtered_data is not None:
        break_E = filtered_data['break_E'][np.argmin(np.abs(filtered_data['timestamps'] - timestamp))]
        return break_E
    else:
        break_v_id = break_v_dict['timestamps'].index(timestamp)
        break_E_arr = vel_to_eV(break_v_dict['break_v'][break_v_id].values)
        break_E, useful_breakE = check_breakE_range(break_E_arr)

        if useful_breakE is False:
            useful_breakE_list = []
            t_5min_ahead = timestamp - np.timedelta64(5, 'm')
            t_5min_after = timestamp + np.timedelta64(5, 'm')
            # Create a DataFrame with break_v and timestamps
            break_v_values = [series.values for series in break_v_dict['break_v']]
            id_ahead = np.argmin(np.abs(break_v_dict['timestamps'] - t_5min_ahead))
            id_after = np.argmin(np.abs(break_v_dict['timestamps'] - t_5min_after))

            for i in range(id_ahead, id_after+1):
                break_E_arr = vel_to_eV(break_v_values[i])
                break_E, useful_breakE = check_breakE_range(break_E_arr)
                if useful_breakE is True:
                    useful_breakE_list.append(break_E)
            break_E = np.concatenate(useful_breakE_list)
        else:
            break_E = break_E_arr
        
        # Return the mean
        if isinstance(break_E, np.ndarray):
            return np.mean(break_E)
        else:
            return break_E


def determineCoreBeam_energyIdx(par_strahl_cond, anti_par_strahl_cond, swa_energy, break_E, low_c_val, high_c_val, low_b_val, high_b_val):
    """
    Determine energy ranges for core and beam fitting based on conditions.
    Returns:
    low_c, high_c, low_b, high_b, and the chosen beam energy threshold.
    """
    # Default initialization
    low_b, high_b = None, None
    # low_b_val = break_E+10
    # high_b_val = 500 # eV

    # anti_par_beam_energy_thresh = beam_energy_thresh
    # par_beam_energy_thresh = beam_energy_thresh

    # # Set core and beam energy limits based on conditions
    # if anti_par_strahl_cond and not par_strahl_cond:
    #     low_b_val, high_b_val = anti_par_beam_energy_thresh, 500
    #     high_c_val = anti_par_beam_energy_thresh
        
    # elif not anti_par_strahl_cond and par_strahl_cond:
    #     low_b_val, high_b_val = par_beam_energy_thresh, 500
    #     high_c_val = par_beam_energy_thresh

    # elif anti_par_strahl_cond and par_strahl_cond:
    #     high_c_val = anti_par_beam_energy_thresh
    #     low_b_val, high_b_val = anti_par_beam_energy_thresh, 500
        
    # else:  # no strahl conditions
    #     high_c_val = anti_par_beam_energy_thresh
        
    if anti_par_strahl_cond or par_strahl_cond:
        low_b = np.nanargmin(np.abs(low_b_val - swa_energy))
        high_b = np.nanargmin(np.abs(high_b_val - swa_energy))
    low_c = np.nanargmin(np.abs(low_c_val - swa_energy))
    high_c = np.nanargmin(np.abs(high_c_val - swa_energy))

    return low_c, high_c, low_b, high_b

def determineSWA_energyIdx(lowHalo_E, highHalo_E, swa_energy):
    low_h, high_h = np.nanargmin(np.abs(lowHalo_E - swa_energy)), np.nanargmin(np.abs(highHalo_E - swa_energy))
    return low_h, high_h

def int_vdf(f, v_par, v_perp) :
    int_temp = []
    for i in range(len(v_par)) :
        int_temp.append(np.trapz(np.flipud(v_perp)*np.flipud(f[:,i]), np.flipud(v_perp)))
    return 2*np.pi*np.trapz(np.array(int_temp), v_par)

def moment1st_VDF_para(f, v_par, v_perp):
    
    f2D_Cartesian = f.copy()
    for i in range(len(v_par)) :
        f2D_Cartesian[:,i] = 2*np.pi*np.flipud(v_perp)*np.flipud(f[:,i]) # f3D_Cartesian * V_perp * 2pi = f2D_Cartesian
    
    int_temp = []
    for i in range(len(v_perp)):
        int_temp.append(np.trapz( v_par*f2D_Cartesian[i,:],  v_par)) # 
        
    return np.trapz(np.array(int_temp), np.flipud(v_perp)) #

def moment2nd_VDF(f, v_par, v_perp, vBulk_par) :
    ''' Calculate the second moment of the VDF. 
    f: 2D VDF, 
    v_par is the parallel velocity, 
    v_perp is the perpendicular velocity '''

    f2D_Cartesian = f.copy()
    for i in range(len(v_par)) :
        f2D_Cartesian[:,i] = 2*np.pi*np.flipud(v_perp)*np.flipud(f[:,i]) # f3D_Cartesian * V_perp * 2pi = f2D_Cartesian
    
    int_temp = []
    for i in range(len(v_perp)):
        int_temp.append(np.trapz( (v_par-vBulk_par)**2*f2D_Cartesian[i,:],  v_par)) # 
        
    P_e = m_e*np.trapz(np.array(int_temp), np.flipud(v_perp)) #
    return P_e 


def calc_beam_moments_from_pixels_with_deltas(
    temp_b_psd,
    temp_b_v_par,
    temp_b_v_perp,
    beamCount_mask,
    energy_center_eV_sel,
    dE_upper_eV_sel,
    dE_lower_eV_sel,
    solid_angle_pix=None,
    min_points=5,
):
    """
    Compute physically dimensioned moments from the raw discrete beam VDF
    (energy x pixel direction).

    Works in (v, Ω) space:
        d^3v = v^2 dv dΩ

    Discrete form:
        n      = Σ f_ij v_j^2 Δv_j ΔΩ_i
        u_par  = (1/n) Σ v_par_ij f_ij v_j^2 Δv_j ΔΩ_i
        P_par  = m_e Σ (v_par_ij - u_par)^2 f_ij v_j^2 Δv_j ΔΩ_i
        P_perp = (m_e/2) Σ v_perp_ij^2 f_ij v_j^2 Δv_j ΔΩ_i

        T_par_eV  = P_par  / (n * eV_to_Erg)
        T_perp_eV = P_perp / (n * eV_to_Erg)

    Parameters
    ----
    temp_b_psd       : (N_E_sel, N_pix) array
        PSD f(v, Ω) in the beam region
    temp_b_v_par     : (N_E_sel, N_pix) array
        v_parallel (cm/s)
    temp_b_v_perp    : (N_E_sel, N_pix) array
        v_perp (cm/s)
    beamCount_mask   : (N_E_sel, N_pix) bool array
        valid beam pixels (counts >= threshold)
    energy_center_eV_sel : (N_E_sel,) array
        center energies of the selected channels (eV)
    dE_upper_eV_sel  : (N_E_sel,) array
        energy difference from center to upper bound (eV, >= 0)
    dE_lower_eV_sel  : (N_E_sel,) array
        energy difference from center to lower bound (eV, >= 0)
    m_e              : float
        electron mass (g, cgs)
    solid_angle_pix  : (N_pix,) or (1, N_pix) array, optional
        solid angle ΔΩ_i of each pixel (sr); if None, assume equal areas: 4π/N_pix
    min_points       : int
        minimum number of valid points required

    Returns
    ----
    dict or None
        {
          'n'        : number density (cm^-3)
          'u_par'    : bulk parallel velocity (cm/s)
          'P_par'    : parallel pressure (dyn/cm^2)
          'P_perp'   : perpendicular pressure (dyn/cm^2)
          'T_par_eV' : parallel temperature (eV)
          'T_perp_eV': perpendicular temperature (eV)
        }
    """

    # ---------- shapes ----------
    N_E_sel, N_pix = temp_b_psd.shape

    energy_center_eV_sel = np.asarray(energy_center_eV_sel)
    dE_upper_eV_sel = np.asarray(dE_upper_eV_sel)
    dE_lower_eV_sel = np.asarray(dE_lower_eV_sel)

    if energy_center_eV_sel.shape[0] != N_E_sel:
        raise ValueError("energy_center_eV_sel length must equal N_E_sel")
    if dE_upper_eV_sel.shape[0] != N_E_sel or dE_lower_eV_sel.shape[0] != N_E_sel:
        raise ValueError("dE_upper_eV_sel / dE_lower_eV_sel length must equal N_E_sel")

    # ---------- PSD & mask ----------
    f = np.array(temp_b_psd, copy=True)
    f[~np.isfinite(f)] = 0.0

    if beamCount_mask is not None:
        f = np.where(beamCount_mask, f, 0.0)

    valid_mask = (f > 0.0) & np.isfinite(temp_b_v_par) & np.isfinite(temp_b_v_perp)
    if np.count_nonzero(valid_mask) < min_points:
        return None

    # ---------- energy -> v_center, dv ----------
    # center energy & upper/lower bounds (erg)
    E_c_erg      = energy_center_eV_sel * eV_to_Erg         # (N_E_sel,)
    dE_up_erg    = dE_upper_eV_sel * eV_to_Erg              # (N_E_sel,)
    dE_low_erg   = dE_lower_eV_sel * eV_to_Erg              # (N_E_sel,)

    E_upper_erg  = E_c_erg + dE_up_erg
    E_lower_erg  = np.maximum(E_c_erg - dE_low_erg, 0.0)    # avoid negative energies

    # velocities at the upper/lower bounds
    v_upper = np.sqrt(2.0 * E_upper_erg / m_e)              # (N_E_sel,)
    v_lower = np.sqrt(2.0 * E_lower_erg / m_e)              # (N_E_sel,)

    # dv corresponding to the energy bin width
    dv = v_upper - v_lower                                  # (N_E_sel,)

    # use the velocity at the center energy as v_center (alternatively (v_up+v_low)/2)
    v_center = np.sqrt(2.0 * E_c_erg / m_e)                 # (N_E_sel,)

    # broadcast to 2D
    v_2d  = v_center[:, None]   # (N_E_sel, 1)
    dv_2d = dv[:, None]         # (N_E_sel, 1)

    # ---------- pixel solid angle ΔΩ_i ----------
    if solid_angle_pix is None:
        # equal-area approximation
        dOmega_pix = 4.0 * np.pi / float(N_pix)
        dOmega_2d = np.full((1, N_pix), dOmega_pix)
    else:
        solid_angle_pix = np.asarray(solid_angle_pix)
        if solid_angle_pix.shape not in [(N_pix,), (1, N_pix)]:
            raise ValueError("solid_angle_pix shape must be (N_pix,) or (1, N_pix)")
        dOmega_2d = solid_angle_pix.reshape(1, -1)  # (1, N_pix)

    # ---------- phase-space volume element d^3v = v^2 dv dΩ ----------
    d3v = (v_2d**2) * dv_2d * dOmega_2d    # (N_E_sel, N_pix)

    # ---------- 0th moment: n ----------
    n = np.sum(f * d3v)

    if n <= 0 or not np.isfinite(n):
        return None

    # ---------- 1st moment: u_par ----------
    v_par = np.array(temp_b_v_par, copy=True)
    v_par[~np.isfinite(v_par)] = 0.0

    u_par = np.sum(f * v_par * d3v) / n

    # ---------- 2nd moments: P_par, P_perp ----------
    v_perp = np.array(temp_b_v_perp, copy=True)
    v_perp[~np.isfinite(v_perp)] = 0.0

    # P_par = m_e Σ (v_par - u_par)^2 f d^3v
    P_par = m_e * np.sum(f * (v_par - u_par)**2 * d3v)

    # P_perp = (m_e/2) Σ v_perp^2 f d^3v
    P_perp = 0.5 * m_e * np.sum(f * (v_perp**2) * d3v)

    # ---------- temperatures (eV) ----------
    T_par_eV  = P_par  / (n * eV_to_Erg)
    T_perp_eV = P_perp / (n * eV_to_Erg)

    return {
        'n':         n,
        'u_par':     u_par,
        'P_par':     P_par,
        'P_perp':    P_perp,
        'T_par_eV':  T_par_eV,
        'T_perp_eV': T_perp_eV,
    }


def extract_time_from_filename(filename):
    """
    Extract start and end times from a filename and convert to yyyymmddhh format.
    
    Args:
        filename (str): Filename containing time information like 'VDF_eas_forFitting_2022-03-02 12_2022-03-02 18_1min.pkl'
    
    Returns:
        tuple: (start_time, end_time, folder_name) in yyyymmddhh format, or (None, None, None) if pattern not found
    """
    # Extract time range using regex
    time_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}\s\d{2})_(\d{4}-\d{2}-\d{2}\s\d{2})')
    match = time_pattern.search(filename)
    if match is None:
        time_pattern = re.compile(r'(\d{14})_(\d{14})')
        match = time_pattern.search(filename)
    
    if match:
        # Extract start and end times
        start_time_str = match.group(1)
        end_time_str = match.group(2)
        
        # Convert to yyyymmddhh format
        start_time = start_time_str.replace('-', '').replace(' ', '')
        end_time = end_time_str.replace('-', '').replace(' ', '')
        
        # Create folder name using time range
        folder_name = f"{start_time}_{end_time}"
        
        return start_time, end_time, folder_name
    else:
        return None, None, f"Data_{filename.replace('.pkl', '')}"
