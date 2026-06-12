# main_fitting.py
import os
from pathlib import Path
import sys
import csv
import json
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from data_io import load_data, convert_units
from constant import low_pad_val, high_pad_val, beam_ratio_cond, deficit_ratio_cond, low_c_val
from pad_fitting import pad_fit_operation, pad_fit_function, evaluate_pad_conditions
from utils import determineCoreBeam_energyIdx, determineSWA_energyIdx, eV_to_vel, vel_to_eV, get_break_E, process_breakE_dataframe, extract_time_from_filename
from Core_fitting import set_core_initial_params, perform_core_fit
from Halo_fitting import set_halo_initial_params, get_reduced_chi_square_Halo, extract_halo_data, Adaptive_halo_fitting
from Beam_fitting import perform_beam_fit, iterative_beam_fitting, extract_beam_data, Adapative_beam_fitting
from Overall_fitting import perform_final_vdf_fit, iterative_final_vdf_fit
from Analysis_functions import compute_reduced_chi_square_overallFit, compute_reduced_chi_square_BeamOnly, compute_beam_temperature, compute_reduced_chi_square_HaloOnly
from Plotting import plot_vdf_1D_final_usePAD, plot_2D_VDF, plot_vdf_1D_final,plot_vdf_1D_final_uniformScale

from joblib import Parallel, delayed

REPO_ROOT = Path(__file__).resolve().parents[1]


def portable_path(path):
    """Render a path relative to the repository root when possible, so
    run_config.json and summary.csv stay free of machine-specific paths."""
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def format_time_for_output(value):
    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value.astype('datetime64[s]'))
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    return str(value)


def to_builtin_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def get_lmfit_param_values(result, prefix):
    if result is None or not hasattr(result, 'params'):
        return {}

    values = {}
    for name, param in result.params.items():
        values[f'{prefix}_{name}'] = to_builtin_scalar(param.value)
        if param.stderr is not None:
            values[f'{prefix}_{name}_stderr'] = to_builtin_scalar(param.stderr)
    return values


def make_summary_row(result, source_file, chunk_file, folder_name):
    if result is None:
        return None

    row = {
        'status': 'success',
        'source_file': source_file,
        'chunk_file': portable_path(chunk_file),
        'folder_name': folder_name,
        'epoch_id': result['epoch_id'],
        'time_stamp': format_time_for_output(result['time_stamp']),
        'redChiSqr': to_builtin_scalar(result['redChiSqr']),
        'redChiSqr_beam': to_builtin_scalar(result['redChiSqr_beam']),
        'redChiSqr_halo': to_builtin_scalar(result['redChiSqr_halo']),
        'T_para_b': to_builtin_scalar(result['T_para_b']),
        'T_antiPara_b': to_builtin_scalar(result['T_antiPara_b']),
        'component_energy_json': json.dumps(result['component_energy'], default=to_builtin_scalar),
    }

    row.update(get_lmfit_param_values(result.get('initial_core'), 'initial_core'))
    row.update(get_lmfit_param_values(result.get('initial_halo'), 'initial_halo'))
    row.update(get_lmfit_param_values(result.get('initial_beamPar'), 'initial_beamPar'))
    row.update(get_lmfit_param_values(result.get('initial_beamAntiPar'), 'initial_beamAntiPar'))
    row.update(get_lmfit_param_values(result.get('r_vdf'), 'final'))
    return row


def make_failed_summary_row(epoch_id, epoch_EAS, source_file, chunk_file, folder_name):
    return {
        'status': 'failed',
        'source_file': source_file,
        'chunk_file': portable_path(chunk_file),
        'folder_name': folder_name,
        'epoch_id': epoch_id,
        'time_stamp': format_time_for_output(epoch_EAS[epoch_id]),
    }


def append_summary_rows(summary_path, rows):
    if not rows:
        return

    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows = []
    existing_fields = []
    if summary_path.exists():
        with summary_path.open('r', newline='') as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames or []
            existing_rows = list(reader)

    fieldnames = list(existing_fields)
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    tmp_path = summary_path.with_suffix(summary_path.suffix + '.tmp')
    with tmp_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerows(rows)
    os.replace(tmp_path, summary_path)


def save_pickle_atomic(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with tmp_path.open('wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, path)


def write_run_config(config_path, config):
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = config_path.with_suffix(config_path.suffix + '.tmp')
    with tmp_path.open('w') as f:
        json.dump(config, f, indent=2, default=to_builtin_scalar)
    os.replace(tmp_path, config_path)


def process_single_time_step(i, pad_in, pa_vec_in, unc_pa_vec_in, n_sc, B_avg, epoch_EAS, pitch_angles, swa_energy_all, dE_upper_eV, dE_lower_eV, low_pad, high_pad,scpot_aligned,
                            plot_path, cond_final_plot, filtered_break_E=None):
    # Reseed deterministically per epoch: lmfit's AMPGO draws its random
    # restart points from numpy's global RNG, and joblib workers start with
    # OS-entropy seeds. Seeding from the epoch timestamp makes every fit
    # reproducible across runs, worker counts, and file subsets.
    np.random.seed(int(epoch_EAS[i].astype('datetime64[s]').astype(np.int64)) % (2**31))

    pad = pad_in[i, :, :, :]  # (time, 64, PA_bin 18, [meanPSD, number of points])
    pa_vec = pa_vec_in[i, :, :, :]  # (time, 64, 1024, [PA, PSD, v_par, v_perp, counts])
    scpot = scpot_aligned[i] if scpot_aligned is not None else 0.0
    scpot_value = getattr(scpot, 'values', scpot)
    swa_energy = swa_energy_all[i] if swa_energy_all.ndim == 2 else swa_energy_all  # 2D when SCPOT-shifted, 1D otherwise
    if n_sc is None:
        n_in = 25.0
    else:
        n_in = n_sc[i]
    
    unc_pa_vec = unc_pa_vec_in[i, :, :]

    if np.isnan(pa_vec[:,:,0]).all():
        return None

    # Velocity grid for this time step
    valid_swa_energy = swa_energy[~np.isnan(swa_energy)]
    v_par_arr = np.linspace(-1*np.max(eV_to_vel(valid_swa_energy)), np.max(eV_to_vel(valid_swa_energy)), 1000)
    v_perp_arr = np.linspace(np.max(eV_to_vel(valid_swa_energy)), 0.0, 1000)
    v_par_mesh, v_perp_mesh = np.meshgrid(v_par_arr, v_perp_arr)
    
    
    pitch_angle_centers = np.diff(pitch_angles)/2 + pitch_angles[:-1]
    
    '''1. PAD Fitting'''
    mean_energy_pad = np.nanmean(pad[high_pad:low_pad,:,0], axis = 0) # mean over energy range
    std_energy_pad = np.nanstd(pad[high_pad:low_pad,:,0], axis = 0)/np.sqrt(np.shape(pad[high_pad:low_pad,:,0])[0])
    mean_pad_params, mean_fit_pad = pad_fit_function(mean_energy_pad, pitch_angle_centers, pitch_angles)

    # Perform PAD fitting across all energies
    pad_params_energy, pad_fit_energy = pad_fit_operation(pad=pad, swa_energy=swa_energy, pitch_angle_centers=pitch_angle_centers, pitch_angles=pitch_angles)

    # Evaluate strahl and deficit condition
    par_strahl_cond, anti_par_strahl_cond, par_def_cond, anti_par_def_cond = evaluate_pad_conditions(mean_pad_params, beam_ratio_cond, deficit_ratio_cond, pitch_angle_centers)
    # Energy index of core & beam
    if filtered_break_E is not None:
        break_E = get_break_E(None, epoch_EAS[i], filtered_break_E)
    else:
        break_E = 100
    
    # Energy Range
    high_c_val = break_E - 5 # eV
    low_c_val_shift_scpot = low_c_val - scpot_value # eV. Cut off, previously it's set to be 14 eV. Now it's shifted by SCPOT.
    low_b_val, high_b_val = break_E+0, 1000#500
    low_c, high_c, low_b, high_b = determineCoreBeam_energyIdx(par_strahl_cond, anti_par_strahl_cond, swa_energy, break_E, low_c_val_shift_scpot, high_c_val, low_b_val, high_b_val)


    '''2. Core Fitting'''
    # (1) Set Initial values. (2) Perform Core Fitting. 
    core_maxPSD = np.nanmax(pa_vec[high_c:low_c,:,1])
    core_init, core_const= set_core_initial_params(par_strahl_cond, anti_par_strahl_cond, n_in, core_maxPSD)
    r_core, fit_core = perform_core_fit(pa_vec, low_c, high_c, core_init, core_const, v_par_mesh, v_perp_mesh)
    if r_core is None:
        print('Core fitting failed.')
        return None
    
    '''3. Halo Fitting'''
    # Extract pitch angle, PSD, and velocity data
    low_h_val, high_h_val = break_E+5, 2000 #
    low_h, high_h = determineSWA_energyIdx(low_h_val, high_h_val, swa_energy)
    # halo_init, halo_const = set_halo_initial_params(par_strahl_cond, anti_par_strahl_cond, n_in)
    # r_halo, _, halo_data_dict = process_halo_fit(pa_vec, high_h, low_h, mean_pad_params, halo_const, halo_init, r_core, v_par_mesh, v_perp_mesh, v_par_arr, v_perp_arr, anti_par_strahl_cond, par_strahl_cond, unc_pa_vec)
    halo_data = extract_halo_data(pa_vec, high_h, low_h, mean_pad_params, r_core, n_in, anti_par_strahl_cond, par_strahl_cond, unc_pa_vec)
    r_halo, _, red_chisqr_h = Adaptive_halo_fitting(halo_data, v_par_mesh, v_perp_mesh, r_core, max_iterations=6)
    if r_halo is None:
        print('Halo fitting failed.')
        return None
    else:
        red_chi_halo = get_reduced_chi_square_Halo(r_halo, halo_data)
        print('Halo fitting done. Red_chi-sqr_halo:', red_chi_halo )

    '''4. Beam Fitting'''
    # low_b, high_b = determineSWA_energyIdx(low_b_val, high_b_val, swa_energy)
    beam_data = extract_beam_data(pa_vec, swa_energy, dE_upper_eV, dE_lower_eV, high_b, low_b, mean_pad_params, n_in, r_core, r_halo, anti_par_strahl_cond, par_strahl_cond, beam_energy_thresh=low_b_val)
    r_beam_par, fit_beam_par, r_beam_anti_par, fit_beam_anti_par,red_chisqr_b = Adapative_beam_fitting(beam_data, v_par_mesh, v_perp_mesh, anti_par_strahl_cond, par_strahl_cond, beamEnergy_low_bound=low_b_val, 
                                                                                                        max_iterations=30) #20
    if r_beam_par is None and r_beam_anti_par is None:
        print('Beam fitting failed.')
        # return None
    print('Beam fitting done.')
    
    '''5. Overall Fitting'''
    r_vdf, fit_vdf, redChiSqr_beam = iterative_final_vdf_fit(pa_vec, unc_pa_vec, high_h, low_c, n_in, r_core, r_halo, r_beam_par, r_beam_anti_par, beam_data,
                                                            anti_par_strahl_cond, par_strahl_cond, v_par_mesh, v_perp_mesh, max_iterations=15) #15
    if r_vdf is None:
        print('Overall fitting failed.')
        return None
    print('Overall fitting done.')

    '''6. Reduced chi-square'''
    redChiSqr = compute_reduced_chi_square_overallFit(r_vdf, pa_vec, high_h, low_c, anti_par_strahl_cond, par_strahl_cond)
    redChiSqr_beam = compute_reduced_chi_square_BeamOnly(r_vdf, beam_data, anti_par_strahl_cond, par_strahl_cond)
    redChiSqr_halo = compute_reduced_chi_square_HaloOnly(r_vdf, halo_data)
    print('\nReduced chi-square:', redChiSqr)
    print('Reduced chi-square for Beam:', redChiSqr_beam)
    print('Reduced chi-square for Halo:', redChiSqr_halo)

    T_para_b, T_antiPara_b = compute_beam_temperature(r_vdf, v_par_mesh, v_perp_mesh, v_par_arr, v_perp_arr, anti_par_strahl_cond, par_strahl_cond) # in eV
    
    '''7. Plotting'''
    if cond_final_plot == True:
        # if int(anti_par_strahl_cond) + int(par_strahl_cond) != 0 :
        variables = {'epoch':epoch_EAS[i], 'pa_vec':pa_vec, 'unc_pa_vec':unc_pa_vec, 'pad': pad, 'swa_energy': swa_energy, 'pitch_angles': pitch_angles,
            'mean_energy_pad': mean_energy_pad, 'std_energy_pad':std_energy_pad, 'mean_pad_params':mean_pad_params, 'mean_fit_pad':mean_fit_pad,
            'pad_params_energy':pad_params_energy, 'anti_par_strahl_cond':anti_par_strahl_cond, 'par_strahl_cond':par_strahl_cond,
            'Bx_direc': B_avg[i,0], 'r_vdf': r_vdf, 'fit_vdf': fit_vdf, 'v_par_mesh': v_par_mesh, 'v_perp_mesh': v_perp_mesh, 'v_par_arr': v_par_arr,
            'redChiSqr':redChiSqr, 'redChiSqr_beam':redChiSqr_beam, 'redChiSqr_halo':redChiSqr_halo,
            'plot_path':plot_path, 'fig_name':i}
        indices = {'high_c': high_c, 'low_c': low_c, 'high_h': high_h, 'low_h': low_h, 'high_b': high_b, 'low_b': low_b}
        plot_vdf_1D_final_uniformScale(variables, indices)

        fit_beam_dict = {'fit_b_par': fit_beam_par, 'fit_b_AntiPar': fit_beam_anti_par, 'red_chisqr_b_initial':red_chisqr_b} # Initial fitting
        plot_2D_VDF(epoch_EAS[i], pa_vec, low_c, v_par_mesh, v_perp_mesh, v_par_arr, v_perp_arr, fit_vdf, r_vdf, redChiSqr, redChiSqr_beam, redChiSqr_halo, beam_data, fit_beam_dict, T_para_b, T_antiPara_b, anti_par_strahl_cond, par_strahl_cond, plot_path, figname=i, savePlot=True)
        print('Plotting done.')

    '''8. store reuslt'''
    component_energy = {"core":[swa_energy[low_c], swa_energy[high_c]], "halo":[low_h_val, high_h_val], "beam":[low_b_val, high_b_val]}
    result = {
        "time_stamp": epoch_EAS[i],
        "epoch_id": i,
        "initial_core": r_core,
        "initial_halo": r_halo,
        "initial_beamPar": r_beam_par,
        "initial_beamAntiPar": r_beam_anti_par,
        "r_vdf": r_vdf,
        "fit_vdf": fit_vdf,
        "redChiSqr": redChiSqr,
        "redChiSqr_beam": redChiSqr_beam,
        "redChiSqr_halo": redChiSqr_halo,
        "T_para_b": T_para_b,
        "T_antiPara_b": T_antiPara_b,
        "component_energy": component_energy
        # "strahl_direction":{'par':par_strahl_cond, 'anti_par':anti_par_strahl_cond}
    }
    return result


class _StreamTee:
    """Forward writes to the original stream and a log file, flushing both."""

    def __init__(self, stream, logFile):
        self.stream = stream
        self.logFile = logFile

    def write(self, text):
        self.stream.write(text)
        self.logFile.write(text)
        self.logFile.flush()

    def flush(self):
        self.stream.flush()
        self.logFile.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def tee_stdout_to_log(log_path):
    """Mirror stdout/stderr to a log file while keeping console output.

    Wraps the Python stream objects so the parent process output (file
    progress, tqdm bars, 'Saved chunk' / 'Updated summary') is always written
    to the log. Appends so resumed runs accumulate in one file. Returns the
    open log file handle.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logFile = open(log_path, 'a', buffering=1)
    sys.stdout = _StreamTee(sys.stdout, logFile)
    sys.stderr = _StreamTee(sys.stderr, logFile)
    print(f"\n{'=' * 70}\nRun started: {datetime.now():%Y-%m-%d %H:%M:%S}  ->  {log_path}\n{'=' * 70}")
    return logFile


def main2(pickle_dir=None, pickle_files=None, break_E_dir=None, run_name='example_run',
          n_jobs=4, fit_beg=None, fit_end=None, cond_final_plot=True,
          overwrite_existing_chunks=False):
    """Fit every epoch of the given preprocessed VDF files.

    Defaults run the bundled example dataset. For your own data, point
    pickle_dir/pickle_files at the preprocessed HDF5 files and break_E_dir at
    the matching cone-feature-split break energy results (see docs/TUTORIAL.md).
    """
    # Directories and files; defaults target the bundled example dataset
    if pickle_dir is None:
        pickle_dir = REPO_ROOT / 'examples' / 'example_data'
    if pickle_files is None:
        pickle_files = ['VDF_eas_forFitting_20220302121509_20220302122509_1min.h5']
    if break_E_dir is None:
        break_E_dir = REPO_ROOT / 'examples' / 'example_data'

    pickle_Dir = str(pickle_dir)
    run_path = REPO_ROOT / 'results' / 'fitting' / run_name
    plot_root = run_path / 'plots'
    data_save_path = run_path / 'data'
    chunk_dir = data_save_path / 'chunks'
    summary_path = data_save_path / 'summary.csv'
    config_path = data_save_path / 'run_config.json'

    # Mirror this run's output to logs/fitting_<run_name>.log for easy checking.
    tee_stdout_to_log(REPO_ROOT / 'logs' / f'fitting_{run_name}.log')

    pickle_file_path = list(pickle_files)

    # Cone-feature-split break energy results from compute_break_energy_cone_features.py
    break_E_dir = str(break_E_dir)
    break_E_files = [f'{Path(pfile).stem}_cone_feature_split.pkl' for pfile in pickle_file_path]

    # Load energy delta values (instrument energy bin widths)
    energy_delta_file = REPO_ROOT / 'examples' / 'example_data' / 'energy_deltas.npz'
    energy_delta = np.load(energy_delta_file)
    dE_upper_eV = energy_delta['delta_upper']
    dE_lower_eV = energy_delta['delta_lower']

    write_run_config(config_path, {
        'run_name': run_name,
        'pickle_dir': portable_path(pickle_Dir),
        'pickle_file_path': pickle_file_path,
        'break_E_dir': portable_path(break_E_dir),
        'break_E_files': break_E_files,
        'energy_delta_file': portable_path(energy_delta_file),
        'fit_beg': fit_beg,
        'fit_end': fit_end,
        'n_jobs': n_jobs,
        'overwrite_existing_chunks': overwrite_existing_chunks,
        'cond_final_plot': cond_final_plot,
        'low_pad_val': low_pad_val,
        'high_pad_val': high_pad_val,
        'beam_ratio_cond': beam_ratio_cond,
        'deficit_ratio_cond': deficit_ratio_cond,
        'low_c_val': low_c_val,
    })

    start_idx = 0
    # end_idx = 1
    for f_id, (pfile, break_E_file) in enumerate(zip(pickle_file_path[start_idx:], break_E_files[start_idx:]), start=start_idx):
        print('Processing file:', pfile, 'and', break_E_file)
        _, _, folder_name = extract_time_from_filename(pfile)
        
        # Create output directories for plots and data
        plot_path = str(plot_root / folder_name) + os.sep

        os.makedirs(plot_path, exist_ok=True)
        os.makedirs(chunk_dir, exist_ok=True)

        # Load and convert units
        data_dict0 = load_data(pickle_Dir, pfile)
        data_dict = convert_units(data_dict0)
        break_E_df = pd.read_pickle(os.path.join(break_E_dir, break_E_file))
        filtered_break_E = process_breakE_dataframe(break_E_df)

        # Extract required variables
        pa_vec_in = data_dict['pa_vec_in']        # shape: (time, energy, bins, [PA, PSD, v_par, v_perp, counts])
        pad_in = data_dict['pad_in']              # shape: (time, energy, pa_bins, [meanPSD, ...])
        B_avg = data_dict['B_avg']
        epoch_EAS = data_dict['time_EAS']
        pitch_angles = data_dict['pitch_angles']
        swa_energy_all = data_dict['swa_energy']
        n_sc = data_dict['n_sc']
        u = data_dict['u']                       # proton velocity in cm/s
        unc_pa_vec_in = data_dict['unc_pa_vec_in'] # shape: (time, energy, bins)
        scpot_aligned = data_dict['scpot_aligned']
        
        # Helper variables: swa_energy is 2D (time, energy) when SCPOT-shifted
        if swa_energy_all.ndim == 2:
            # Use the first time step's energies to determine the PAD range
            swa_energy_for_pad = swa_energy_all[0]
        else:
            swa_energy_for_pad = swa_energy_all
        low_pad, high_pad = determineSWA_energyIdx(low_pad_val, high_pad_val, swa_energy_for_pad)

        # valid_swa_energy = swa_energy[~np.isnan(swa_energy)]
        # v_par_arr, v_perp_arr = np.linspace(-1*np.max(eV_to_vel(valid_swa_energy)), np.max(eV_to_vel(valid_swa_energy)), 1000), np.linspace(eV_to_vel(valid_swa_energy)[0], 0.0, 1000)
        # v_par_mesh, v_perp_mesh = np.meshgrid(v_par_arr, v_perp_arr)

        beg = fit_beg if fit_beg is not None else 0
        end = fit_end if fit_end is not None else len(epoch_EAS)
        if beg < 0 or end > len(epoch_EAS) or beg >= end:
            raise ValueError(f'Invalid fit range: beg={beg}, end={end}, len(epoch_EAS)={len(epoch_EAS)}')

        if beg == 0 and end == len(epoch_EAS):
            chunk_label = folder_name
        else:
            chunk_label = f'{folder_name}_epoch{beg:04d}-{end - 1:04d}'
        chunk_file = chunk_dir / f'FitResult_{chunk_label}.pkl'

        if chunk_file.exists() and not overwrite_existing_chunks:
            print(f'Chunk already exists, skipping: {chunk_file}')
            continue

        parallel_results = Parallel(n_jobs = n_jobs)(
            delayed(process_single_time_step)(i, pad_in, pa_vec_in, unc_pa_vec_in, n_sc, B_avg, epoch_EAS, pitch_angles, swa_energy_all,dE_upper_eV,dE_lower_eV, low_pad, high_pad,scpot_aligned,
                            plot_path, cond_final_plot, filtered_break_E)
            for i in tqdm(range(beg, end), desc=f"segment{f_id}")
        ) # range(beg, end)
 
        valid_results = [result for result in parallel_results if result is not None]

        epoch_list = [ r['time_stamp'] for r in valid_results]
        epoch_id_list = [ r['epoch_id'] for r in valid_results]
        initial_Core = [ r['initial_core'] for r in valid_results]
        initial_Halo = [ r['initial_halo'] for r in valid_results]
        initial_BeamPar = [ r['initial_beamPar'] for r in valid_results]
        initial_BeamAntiPar = [ r['initial_beamAntiPar'] for r in valid_results]
        r_vdf_list = [ r['r_vdf'] for r in valid_results]
        fit_vdf_list = [ r['fit_vdf'] for r in valid_results]
        redChiSqr_list = [ r['redChiSqr'] for r in valid_results]
        redChiSqr_beam_list = [ r['redChiSqr_beam'] for r in valid_results]
        redChiSqr_halo_list = [ r['redChiSqr_halo'] for r in valid_results]
        T_para_b_list = [ r['T_para_b'] for r in valid_results]
        T_antiPara_b_list = [ r['T_antiPara_b'] for r in valid_results]
        component_energy = [ r['component_energy'] for r in valid_results]

        failed_epoch_ids = [i for i, result in zip(range(beg, end), parallel_results) if result is None]
        result_dict = {
            "source_file": pfile,
            "break_E_file": break_E_file,
            "folder_name": folder_name,
            "run_name": run_name,
            "fit_range": {"beg": beg, "end": end},
            "failed_epoch_id": failed_epoch_ids,
            "time_stamps": epoch_list,
            "epoch_id": epoch_id_list,
            "swa_energy": swa_energy_all,
            "component_energy": component_energy,
            "initial_core": initial_Core,
            "initial_halo": initial_Halo,
            "initial_beamPar": initial_BeamPar,
            "initial_beamAntiPar": initial_BeamAntiPar,
            "r_vdf": r_vdf_list,
            "fit_vdf": fit_vdf_list,
            "redChiSqr": redChiSqr_list,
            "redChiSqr_beam": redChiSqr_beam_list,
            "redChiSqr_halo": redChiSqr_halo_list,
            "T_para_b": T_para_b_list,
            "T_antiPara_b": T_antiPara_b_list,
        }

        save_pickle_atomic(result_dict, chunk_file)

        summary_rows = [
            make_summary_row(result, pfile, chunk_file, folder_name)
            for result in valid_results
        ]
        summary_rows.extend(
            make_failed_summary_row(epoch_id, epoch_EAS, pfile, chunk_file, folder_name)
            for epoch_id in failed_epoch_ids
        )
        append_summary_rows(summary_path, summary_rows)
        print(f'Saved chunk: {chunk_file}')
        print(f'Updated summary: {summary_path}')

if __name__ == '__main__':
    # Reproducibility is handled per epoch inside process_single_time_step.
    main2()
