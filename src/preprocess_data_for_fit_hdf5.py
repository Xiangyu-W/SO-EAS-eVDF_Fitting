import pandas as pd
import numpy as np
import xarray as xr
from astropy import constants as const
from astropy import units as u
from math import floor, log10
import re
import pickle
import os
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

from cdasws import CdasWs
from cdasws.datarepresentation import DataRepresentation as dr
from download_EAS_Data import download_eas_data

REPO_ROOT = Path(__file__).resolve().parents[1]
# EAS geometric factors / quantum efficiencies, used to convert L1 counts
# to VDF and its uncertainty; shipped alongside the example data.
CALIBRATION_DIR = REPO_ROOT / 'examples' / 'example_data'

# --- Geometric Factor & Quantum Efficiency ---
def load_geometric_factors(instrument, count_data):
    """Load and process geometric factors for specified instrument."""
    GF_dir = f'{CALIBRATION_DIR}{os.sep}'
    GFs = {
        'EAS1': 'EAS1_AGFs_interpolated_from_23July2021_for_Elv_set3_v2024Nov.txt',
        'EAS2': 'EAS2_AGFs_interpolated_from_23July2021_for_Elv_set3_v2024Nov.txt'
    }

    col_title = ['elev'] + [f'azi{i}' for i in range(32)]
    gf_data = pd.read_csv(GF_dir + GFs[instrument], delimiter=",", header=None, names=col_title)
    
    return xr.DataArray(
        gf_data.iloc[:, 1:],
        coords={
            'SWA_EAS_ELEVATION': count_data.coords['SWA_EAS_ELEVATION'],
            'SWA_EAS_AZIMUTH': count_data.coords['SWA_EAS_AZIMUTH']
        },
        dims=['SWA_EAS_ELEVATION', 'SWA_EAS_AZIMUTH']
    )

def load_quantum_efficiency(instrument, count_data):
    """Load and process quantum efficiency for specified instrument."""
    Qe_dir = f'{CALIBRATION_DIR}{os.sep}'
    QEs = {
        'EAS1': 'EAS1_Flight_QuantumEfficiencies_V2020.line',
        'EAS2': 'EAS2_Flight_QuantumEfficiencies_V2020.line'
    }

    qe_data = pd.read_csv(Qe_dir + QEs[instrument], delimiter=" ", header=None)
    energy_coord = f'SWA_{instrument}_ENERGY'

    # Check energy dim name of different version of count data
    energy_dim_old = f'{energy_coord}_dim'
    if energy_dim_old in count_data.dims:
        energy_dim = energy_dim_old
    else:
        energy_dim = energy_coord
    
    # Get energy value
    if energy_coord in count_data.coords:
        energy_values = count_data.coords[energy_coord]
        # use only the first time step's energies if time-dependent
        if 'EPOCH' in energy_values.dims and len(energy_values.dims) > 1:
            energy_values = energy_values.isel(EPOCH=0)
    else:
        # fall back to the dimension itself if the energy coordinate is missing
        energy_values = count_data[energy_dim] if energy_dim in count_data else None
    if energy_values is None:
        raise ValueError(f"Energy coordinate {energy_coord} not found in count_data")
    
    return xr.DataArray(
        qe_data.values.flatten(),
        coords={energy_coord: energy_values}, #count_data.coords[energy_coord].isel(EPOCH=0)
        dims=[energy_dim]
    )

def calculate_vdf_from_count(count_data, gf_data, qe_data, instrument):
    """Calculate VDF from count data using geometric factors and quantum efficiency."""
    dT = 0.00085 * u.s
    m_e_eV = const.m_e.to(u.eV * u.s**2 / u.km**2) # kg to eV*s^2/km^2
    gf_km = gf_data * 1e-10  # cm^2*sr to km^2*sr
    energy_coord = f'SWA_{instrument}_ENERGY'
    
    return m_e_eV**2 * count_data / (2 * dT * count_data[energy_coord]**2 * gf_km * qe_data)


# --- Read EAS Data ---
def readCDFs(filePaths, skip_errors=False):
    cdas = CdasWs()
    data_arrays = []
    dataSet_list = []
    valid_files = []
    invalid_files = []
    
    for filePath in filePaths:
        try:
            temp_eas = cdas.read_data(filePath, data_representation=dr.XARRAY)
            
            # Extract the 'SWA_EAS_Data' variable
            if 'eas1' in filePath.lower():
                data_key = 'SWA_EAS1_Data'
            elif 'eas2' in filePath.lower():
                data_key = 'SWA_EAS2_Data'
            else:
                print(f"Warning: Instrument name not found in {filePath}")
                if skip_errors:
                    invalid_files.append(filePath)
                    continue
                else:
                    raise ValueError(f"Instrument name not found in {filePath}")
            
            if data_key in temp_eas:
                data_array = temp_eas[data_key]
                data_arrays.append(data_array)
                dataSet_list.append(temp_eas)
                valid_files.append(filePath)
            else:
                print(f"Warning: '{data_key}' not found in {filePath}")
                if skip_errors:
                    invalid_files.append(filePath)
                    continue
                else:
                    raise ValueError(f"'{data_key}' not found in {filePath}")
                    
        except Exception as e:
            print(f"Error processing file {filePath}: {str(e)}")
            if skip_errors:
                invalid_files.append(filePath)
                continue
            else:
                raise
    
    if invalid_files:
        print(f"Skipped {len(invalid_files)} invalid files out of {len(filePaths)} total files")
        
    if not data_arrays:
        if skip_errors:
            print("No valid data arrays found. All files were skipped due to errors.")
            return None, []
        else:
            raise ValueError("No data arrays to concatenate.")
    
    if len(data_arrays) == 1:
        return data_arrays[0], dataSet_list
    else:
        # Concatenate DataArrays along 'EPOCH'
        combined_data_array = xr.concat(data_arrays, dim='EPOCH')
        # Remove duplicate times if any
        combined_data_array = combined_data_array.sel(EPOCH=~combined_data_array.get_index('EPOCH').duplicated())
        # Sort by 'EPOCH' to ensure chronological order
        combined_data_array = combined_data_array.sortby('EPOCH')
        return combined_data_array, dataSet_list

def extract_EAS_id(a_string):
    # get the EAS ID from the file name
    eas_id = None
    if 'eas1' in a_string.lower():
        eas_id = 'EAS1'
    elif 'eas2' in a_string.lower():
        eas_id = 'EAS2'
    else:
        print(f"Instrument name not found in {a_string}")
    return eas_id

def check_data_consistency(dataArray, data_set):

    # 1. Check chronological order of 'EPOCH'
    # Convert 'EPOCH' to pandas datetime index for easier handling
    epoch_times = dataArray['EPOCH'].to_pandas()

    # Calculate the time differences between consecutive timestamps
    time_diffs = epoch_times.diff()

    # Check if all time differences are positive
    is_chronological = (time_diffs[1:] > pd.Timedelta(0)).all()
    print(f"Is 'EPOCH' in chronological order? {is_chronological}")

    # 2. Check 'EPOCH' consistency
    original_epochs = xr.concat([ds['EPOCH'] for ds in data_set], dim='EPOCH')
    # Remove duplicates and sort
    original_epochs = original_epochs.sel(EPOCH=~original_epochs.get_index('EPOCH').duplicated())
    original_epochs = original_epochs.sortby('EPOCH')

    # Compare with 'EPOCH' in psdEAS1
    epochs_equal = dataArray['EPOCH'].equals(original_epochs)
    print(f"Do the 'EPOCH' coordinates match between psdEAS1 and original datasets? {epochs_equal}")


    # 3. Check data value consistency across all datasets
    for idx, ds in enumerate(data_set):
        eas_ID = extract_EAS_id(ds.attrs['Logical_source'][0])
        ds_dataArray = ds[f'SWA_{eas_ID}_Data']
        # Select corresponding data from dataArray
        dataArray_sub = dataArray.sel(EPOCH=ds_dataArray['EPOCH'])
        # Compare the data
        data_equal = dataArray_sub.equals(ds_dataArray)
        if not data_equal:
            print(f"Data mismatch found in dataset index: {idx}")
        else:
            print(f"Data is identical for dataset index: {idx}")

def get_EAS_energy(dataset, instrument = None):
    energy_var = f'SWA_{instrument}_ENERGY'
    
    if energy_var not in dataset.variables:
        raise ValueError(f"Variable {energy_var} not found in dataset")
    
    energy_data = dataset[energy_var]
    # check the structure of the energy data
    if 'EPOCH' in energy_data.dims:
        # older versions: energy data includes an EPOCH dimension
        # take the energy values at the first time step
        energy_data = energy_data.isel(EPOCH=10)
    return energy_data

def convert_eV_2Velocity(E):
    me = 9.109E-31

    energy_Joule = E*1.602E-19
    v = np.sqrt(2*energy_Joule/me)/10**3 # km/s
    return v

def getUnitVector(vector):
    return vector/np.linalg.norm(vector)

def getUnitVelocityDirection_EAS(theta, phi, velocity):
    
    theta_gridEAS1, v_gridEAS1, phi_gridEAS1 = np.meshgrid(theta, velocity, phi, indexing='ij')
    vxUnitEAS = np.cos(theta_gridEAS1) * np.cos(phi_gridEAS1)
    vyUnitEAS = np.cos(theta_gridEAS1) * np.sin(phi_gridEAS1)
    vzUnitEAS = -np.sin(theta_gridEAS1)

    return vxUnitEAS, vyUnitEAS, vzUnitEAS

def getVelocityArray_EAS(theta, phi, velocity):
    
    theta_gridEAS1, v_gridEAS1, phi_gridEAS1 = np.meshgrid(theta, velocity, phi, indexing='ij')
    vxEAS = v_gridEAS1*np.cos(theta_gridEAS1) * np.cos(phi_gridEAS1)
    vyEAS = v_gridEAS1*np.cos(theta_gridEAS1) * np.sin(phi_gridEAS1)
    vzEAS = -v_gridEAS1*np.sin(theta_gridEAS1) # '-' is important

    return vxEAS, vyEAS, vzEAS

def transferToSRF(transferMatrix, Vx, Vy, Vz):
    Vx_srf = transferMatrix[0,0]*Vx + transferMatrix[0,1]*Vy + transferMatrix[0,2]*Vz
    Vy_srf = transferMatrix[1,0]*Vx + transferMatrix[1,1]*Vy + transferMatrix[1,2]*Vz
    Vz_srf = transferMatrix[2,0]*Vx + transferMatrix[2,1]*Vy + transferMatrix[2,2]*Vz

    return {'vxSRF':Vx_srf, 'vySRF':Vy_srf, 'vzSRF':Vz_srf}

def transferToRTN(transferMatrices, Vx, Vy, Vz):
    num_matrices = transferMatrices.shape[0]
    shape = Vx.shape  # Shape of Vx, Vy, Vz
    Vx_RTN = np.zeros((num_matrices, *shape))
    Vy_RTN = np.zeros((num_matrices, *shape))
    Vz_RTN = np.zeros((num_matrices, *shape))

    velocity_vector = np.stack((Vx, Vy, Vz), axis=0)  # Shape: (3, 16, 64, 32)

    for i in range(num_matrices):
        Vx_RTN[i], Vy_RTN[i], Vz_RTN[i] = np.einsum('ij,jxyz->ixyz', transferMatrices[i], velocity_vector)


    return {'vxRTN': Vx_RTN, 'vyRTN': Vy_RTN, 'vzRTN': Vz_RTN}

def slice_xarrayEAS(xarrayEAS, timeRange=None, timeFreq=None, sliceMethod='nearest'):
    # Slice the xarrayEAS data to specified range and time resolution
    ## sliceMethod: 'nearest', 'mean'
    if timeFreq is None and timeRange == [xarrayEAS['EPOCH'].values[0], xarrayEAS['EPOCH'].values[-1]]:
        return xarrayEAS

    if timeRange is None:
        timeRange = [xarrayEAS['EPOCH'].values[0], xarrayEAS['EPOCH'].values[-1]]

    data_in_range = xarrayEAS.sel(EPOCH=slice(timeRange[0], timeRange[1]))
    if len(data_in_range['EPOCH']) == 0:
        print(f"Warning: No data found between {timeRange[0]} and {timeRange[1]}")
        return None
    
    # If no time frequency is specified, just return the range-filtered data
    if timeFreq is None:
        return data_in_range
    
    # If we need to resample to a specific frequency
    if sliceMethod == 'mean':
        # Create regular time grid
        time_grid = pd.date_range(start=data_in_range['EPOCH'].values[0], 
                                 end=data_in_range['EPOCH'].values[-1], 
                                 freq=timeFreq)
        
        # For each time point, find data within ±timeFreq/2 and average
        psdEAS_slice_list = []
        for t in time_grid:
            window_data = data_in_range.sel(
                EPOCH=slice(t-pd.Timedelta(timeFreq)/2, t+pd.Timedelta(timeFreq)/2))
            
            # Only add points where we have data
            if len(window_data['EPOCH']) > 0:
                avg_data = window_data.mean(dim='EPOCH')
                psdEAS_slice_list.append(avg_data)
        
        # Only proceed if we have data
        if psdEAS_slice_list:
            EAS_slice = xr.concat(psdEAS_slice_list, dim='EPOCH')
            # Assign the actual time grid points to the result
            EAS_slice = EAS_slice.assign_coords(EPOCH=time_grid[:len(psdEAS_slice_list)])
            return EAS_slice
        else:
            print("Warning: No data found in the specified time range")
            return xr.DataArray()  # Return empty DataArray
            
    elif sliceMethod == 'nearest':
        # Create regular time grid
        time_grid = pd.date_range(start=data_in_range['EPOCH'].values[0], 
                                 end=data_in_range['EPOCH'].values[-1], 
                                 freq=timeFreq)
        
        # Find the nearest actual data point for each time in the grid
        nearest_points = []
        actual_times = []
        
        for t in time_grid:
            # Find the nearest time index
            nearest_idx = np.abs(data_in_range['EPOCH'].values - np.datetime64(t)).argmin()
            nearest_time = data_in_range['EPOCH'].values[nearest_idx]
            
            # Only add if this time hasn't been used already
            if nearest_time not in actual_times:
                nearest_points.append(nearest_idx)
                actual_times.append(nearest_time)
        
        # Select the unique nearest points
        if nearest_points:
            EAS_slice = data_in_range.isel(EPOCH=nearest_points)
            return EAS_slice
        else:
            print("Warning: No data found in the specified time range")
            return xr.DataArray()  # Return empty DataArray
    else:
        raise ValueError("Invalid sliceMethod. Use 'nearest' or 'mean'.")


def get_valid_range_EASdata(timeRange_EAS, countEAS1, countEAS2):
    """
    Compare timeRange_EAS with the time ranges of countEAS1 and countEAS2
    to determine the intersection of valid time ranges.

    Returns:
    - valid_range: List of [valid_start, valid_end] for the intersection of time ranges.
    """
    # Convert timeRange_EAS to pandas timestamps
    start_EAS = pd.to_datetime(timeRange_EAS[0])
    end_EAS = pd.to_datetime(timeRange_EAS[1])

    
    # Determine the time ranges of countEAS1 and countEAS2
    time_countEAS1 = pd.to_datetime(countEAS1['EPOCH'].values)
    time_countEAS2 = pd.to_datetime(countEAS2['EPOCH'].values)

    full_time_range = pd.date_range(start=start_EAS, end=end_EAS, freq='1min')
    
    # Create masks for times where data exists
    # We use the nearest method to find if there's data close to each point in our grid
    has_eas1_data = np.zeros(len(full_time_range), dtype=bool)
    has_eas2_data = np.zeros(len(full_time_range), dtype=bool)
    
    # Tolerance for considering a timestamp "close enough" to have data (e.g., 30 seconds)
    tolerance = pd.Timedelta(seconds=30)
    
    print(f"Checking data availability across {len(full_time_range)} time points...")
    
    # This can be slow for large datasets, so we'll optimize
    # Instead of checking each point, we'll check if each point in full_time_range
    # is within tolerance of any timestamp in timestamps_eas1/timestamps_eas2
    
    # For each time point, check if it's within tolerance of any timestamp
    for i, time_point in enumerate(full_time_range):
        # Check if any EAS1 timestamp is within tolerance
        has_eas1_data[i] = np.any(np.abs((time_countEAS1 - time_point).total_seconds()) <= tolerance.total_seconds())
        
        # Check if any EAS2 timestamp is within tolerance
        has_eas2_data[i] = np.any(np.abs((time_countEAS2 - time_point).total_seconds()) <= tolerance.total_seconds())
        
        # Print progress every 10% of the way
        if i % (len(full_time_range) // 10) == 0:
            print(f"Checked {i}/{len(full_time_range)} time points ({i/len(full_time_range)*100:.1f}%)")
    
    # Combined mask where both EAS1 and EAS2 have data
    has_both_data = has_eas1_data & has_eas2_data
    
    # Find the boundaries of valid regions
    valid_regions = []
    in_valid_region = False
    region_start = None
    
    for i, has_data in enumerate(has_both_data):
        if has_data and not in_valid_region:
            # Start of a new valid region
            in_valid_region = True
            region_start = full_time_range[i]
        elif not has_data and in_valid_region:
            # End of a valid region
            in_valid_region = False
            region_end = full_time_range[i-1]
            valid_regions.append((region_start, region_end))
    
    # Don't forget to add the last region if we're still in one
    if in_valid_region:
        region_end = full_time_range[-1]
        valid_regions.append((region_start, region_end))
    
    print(f"Found {len(valid_regions)} valid time regions")
    for i, (start, end) in enumerate(valid_regions):
        duration = end - start
        print(f"Region {i+1}: {start} to {end} (duration: {duration})")
    
    return valid_regions

def align_eas_datasets(psdEAS1, psdEAS2, countEAS1, countEAS2, timeRange=None):
    # Convert all timestamps to pandas DatetimeIndex
    time_psdEAS1 = pd.DatetimeIndex(psdEAS1['EPOCH'].values)
    time_psdEAS2 = pd.DatetimeIndex(psdEAS2['EPOCH'].values)
    time_countEAS1 = pd.DatetimeIndex(countEAS1['EPOCH'].values)
    time_countEAS2 = pd.DatetimeIndex(countEAS2['EPOCH'].values)
    
    # Find the intersection of all timestamp arrays
    common_times = (time_psdEAS1.intersection(time_psdEAS2)
                   .intersection(time_countEAS1)
                   .intersection(time_countEAS2))
    
    # If timeRange is provided, further restrict to this range
    if timeRange is not None:
        start_time = pd.to_datetime(timeRange[0])
        end_time = pd.to_datetime(timeRange[1])
        common_times = common_times[(common_times >= start_time) & 
                                    (common_times <= end_time)]
    
    # Check if we have common times
    if len(common_times) == 0:
        print("No common time points found across all datasets")
        return None
    
    # Sort the common times
    common_times = common_times.sort_values()
    
    print(f"Found {len(common_times)} common time points across all datasets")
    print(f"Time range: {common_times[0]} to {common_times[-1]}")
    
    # Filter all datasets to only include the common times
    processed_psdEAS1 = psdEAS1.sel(EPOCH=common_times)
    processed_psdEAS2 = psdEAS2.sel(EPOCH=common_times)
    processed_countEAS1 = countEAS1.sel(EPOCH=common_times)
    processed_countEAS2 = countEAS2.sel(EPOCH=common_times)
    
    # Verify that all datasets now have identical time indices
    assert len(processed_psdEAS1['EPOCH']) == len(common_times)
    assert len(processed_psdEAS2['EPOCH']) == len(common_times)
    assert len(processed_countEAS1['EPOCH']) == len(common_times)
    assert len(processed_countEAS2['EPOCH']) == len(common_times)
    
    return {
        'psdEAS1': processed_psdEAS1,
        'psdEAS2': processed_psdEAS2,
        'countEAS1': processed_countEAS1,
        'countEAS2': processed_countEAS2
    }

# Remove SC Potential
def get_RPW_SCPOT(timeRange):
    # Get the magnetic field data from CDAWeb
    cdas = CdasWs()

    timeRange_str = [
    pd.to_datetime(timeRange[0]).strftime('%Y-%m-%dT%H:%M:%SZ'),
    pd.to_datetime(timeRange[-1]).strftime('%Y-%m-%dT%H:%M:%SZ')
    ]

    # Get the magnetic field data
    rpw_data = cdas.get_data('SOLO_L3_RPW-BIA-SCPOT-10-SECONDS',['SCPOT'],timeRange_str[0],timeRange_str[1],dataRepresentation = dr.XARRAY)[1]
    # Altervative: SOLO_L3_RPW-BIA-SCPOT-10-SECONDS

    return rpw_data

def align_rpw_to_eas(rpw_data, eas_epoch, rpw_time_coord='Epoch'):
    from scipy.interpolate import PchipInterpolator
    
    # Extract EAS epoch values if it's a DataArray
    if isinstance(eas_epoch, xr.DataArray):
        eas_epoch_values = eas_epoch.values
    else:
        eas_epoch_values = eas_epoch
    
    # Extract time and values from RPW data
    rpw_time = rpw_data[rpw_time_coord].values
    rpw_values = rpw_data.values
    
    # Convert timestamps to numeric (seconds since epoch)
    rpw_time_numeric = pd.to_datetime(rpw_time).astype('int64') / 1e9
    eas_time_numeric = pd.to_datetime(eas_epoch_values).astype('int64') / 1e9
    
    # Remove NaN values
    valid_mask = ~np.isnan(rpw_values)
    rpw_time_clean = rpw_time_numeric[valid_mask]
    rpw_values_clean = rpw_values[valid_mask]
    
    # Sort by time
    sort_idx = np.argsort(rpw_time_clean)
    rpw_time_clean = rpw_time_clean[sort_idx]
    rpw_values_clean = rpw_values_clean[sort_idx]
    
    # Remove duplicate timestamps
    unique_mask = np.concatenate([[True], np.diff(rpw_time_clean) > 0])
    if not np.all(unique_mask):
        n_duplicates = np.sum(~unique_mask)
        print(f"Info: Removed {n_duplicates} duplicate timestamp(s)")
        rpw_time_clean = rpw_time_clean[unique_mask]
        rpw_values_clean = rpw_values_clean[unique_mask]
    
    # Check if we have valid data
    if len(rpw_values_clean) == 0:
        print("Warning: No valid data after preprocessing. Returning NaN array.")
        aligned_values = np.full(len(eas_epoch_values), np.nan)
    else:
        # PCHIP interpolation
        interpolator = PchipInterpolator(rpw_time_clean, rpw_values_clean, 
                                        extrapolate=False)
        aligned_values = interpolator(eas_time_numeric)
        
        # Mark extrapolated values as NaN
        out_of_range = (eas_time_numeric < rpw_time_clean.min()) | \
                      (eas_time_numeric > rpw_time_clean.max())
        aligned_values[out_of_range] = np.nan
    
    # Create DataArray with aligned values
    # Preserve original attributes but update relevant ones
    attrs = rpw_data.attrs.copy() if hasattr(rpw_data, 'attrs') else {}
    attrs['alignment_method'] = 'PCHIP interpolation'
    attrs['aligned_to'] = 'EAS timestamps'
    
    aligned_data = xr.DataArray(
        aligned_values,
        coords={'EPOCH': eas_epoch_values},
        dims=['EPOCH'],
        name=rpw_data.name if hasattr(rpw_data, 'name') else 'aligned_data',
        attrs=attrs
    )
    
    # Print summary
    n_valid = np.sum(~np.isnan(aligned_values))
    n_total = len(aligned_values)
    print(f"Alignment complete: {n_valid}/{n_total} valid points ({n_valid/n_total*100:.1f}%)")
    
    return aligned_data



# Download and Process MAG, PAS, RPW_Density data
def get_SO_MAG_SRF_Normal(timeRange):
    # Get the magnetic field data from CDAWeb
    cdas = CdasWs()

    timeRange_str = [
    pd.to_datetime(timeRange[0]).strftime('%Y-%m-%dT%H:%M:%SZ'),
    pd.to_datetime(timeRange[-1]).strftime('%Y-%m-%dT%H:%M:%SZ')
    ]

    # Get the magnetic field data
    MAG_data = cdas.get_data('SOLO_L2_MAG-SRF-NORMAL',['B_SRF'],timeRange_str[0],timeRange_str[1],dataRepresentation = dr.XARRAY)[1]

    return MAG_data

def average_MAG_around_timeStamp(magArray, EAS_epoch, timeFreq = None):
    # EAS_epoch: Time stamps of the EAS data
    # timeFreq: time resolution, '1min', '5min', '1H', etc.

    # Average MAG data around the time stamp (t-0.5s, t+0.5s)
    if timeFreq is None:
        magVector = np.array([magArray.sel(EPOCH=slice(t-pd.Timedelta(seconds=0.5), t+pd.Timedelta(seconds=0.5))).mean(dim='EPOCH').values for t in EAS_epoch])
    else:
        timeSlice = pd.date_range(start=EAS_epoch[0], end=EAS_epoch[-1], freq=timeFreq)
        if len(timeSlice) < len(EAS_epoch):
            timeSlice = pd.date_range(start=EAS_epoch[0], end=EAS_epoch[-1] + pd.Timedelta(timeFreq), freq=timeFreq)
        if len(timeSlice) > len(EAS_epoch):
            timeSlice = timeSlice[:len(EAS_epoch)]
        magVector = np.array([magArray.sel(EPOCH=slice(t-pd.Timedelta(seconds=0.5), t+pd.Timedelta(seconds=0.5))).mean(dim='EPOCH').values for t in timeSlice])
    return magVector

def get_SO_PAS_Grnd_Moment(timeRange):
    # Get the magnetic field data from CDAWeb
    cdas = CdasWs()

    timeRange_str = [
    pd.to_datetime(timeRange[0]).strftime('%Y-%m-%dT%H:%M:%SZ'),
    pd.to_datetime(timeRange[-1]).strftime('%Y-%m-%dT%H:%M:%SZ')
    ]

    # Get the magnetic field data
    PAS_data = cdas.get_data('SOLO_L2_SWA-PAS-GRND-MOM',['V_SRF','N'],timeRange_str[0],timeRange_str[1],dataRepresentation = dr.XARRAY)[1]

    return PAS_data

def average_PAS_around_timeStamp(pasArray, EAS_epoch, timeFreq = None):
    # Average MAG data around the time stamp (t-5s, t+5s)

    # EAS_epoch: Time stamps of the EAS data
    # timeFreq: time resolution, '1min', '5min', '1H', etc.
    if timeFreq is None:
        pasVector = np.array([pasArray.sel(Epoch=slice(t-pd.Timedelta(seconds=5), t+pd.Timedelta(seconds=5))).mean(dim='Epoch').values for t in EAS_epoch])
    else:
        timeSlice = pd.date_range(start=EAS_epoch[0], end=EAS_epoch[-1], freq=timeFreq)
        if len(timeSlice) < len(EAS_epoch):
            timeSlice = pd.date_range(start=EAS_epoch[0], end=EAS_epoch[-1] + pd.Timedelta(timeFreq), freq=timeFreq)
        if len(timeSlice) > len(EAS_epoch):
            timeSlice = timeSlice[:len(EAS_epoch)]
        pasVector = np.array([pasArray.sel(Epoch=slice(t-pd.Timedelta(seconds=5), t+pd.Timedelta(seconds=5))).mean(dim='Epoch').values for t in timeSlice])
    
    # Handle NaNs
    for i in range(len(pasVector)):
        if np.isnan(pasVector[i]).all():  # If the entire row is NaN
            prev_idx = next((j for j in range(i-1, -1, -1) if not np.isnan(pasVector[j]).all()), None)
            next_idx = next((j for j in range(i+1, len(pasVector)) if not np.isnan(pasVector[j]).all()), None)
            
            if prev_idx is not None and next_idx is not None:
                pasVector[i] = (pasVector[prev_idx] + pasVector[next_idx]) / 2
            elif prev_idx is not None:
                pasVector[i] = pasVector[prev_idx]
            elif next_idx is not None:
                pasVector[i] = pasVector[next_idx]
                
    return pasVector

def get_SO_RPW_Density(timeRange):
    # Get the magnetic field data from CDAWeb
    cdas = CdasWs()

    timeRange_str = [
    pd.to_datetime(timeRange[0]).strftime('%Y-%m-%dT%H:%M:%SZ'),
    pd.to_datetime(timeRange[-1]).strftime('%Y-%m-%dT%H:%M:%SZ')
    ]

    # Get the magnetic field data
    rpw_data = cdas.get_data('SOLO_L3_RPW-BIA-DENSITY',['DENSITY'],timeRange_str[0],timeRange_str[1],dataRepresentation = dr.XARRAY)[1]

    return rpw_data

def average_RPWDensity_around_timeStamp(rpwArray, EAS_epoch, timeFreq = None):
    # Average RPW density data around the time stamp (t-0.5s, t+0.5s)

    # EAS_epoch: Time stamps of the EAS data
    # timeFreq: time resolution, '1min', '5min', '1H', etc.
    if timeFreq is None:
        rpwDensity = np.array([rpwArray.sel(Epoch=slice(t-pd.Timedelta(seconds=0.5), t+pd.Timedelta(seconds=0.5))).mean(dim='Epoch').values for t in EAS_epoch])
    else:
        timeSlice = pd.date_range(start=EAS_epoch[0], end=EAS_epoch[-1], freq=timeFreq)
        if len(timeSlice) < len(EAS_epoch):
            timeSlice = pd.date_range(start=EAS_epoch[0], end=EAS_epoch[-1] + pd.Timedelta(timeFreq), freq=timeFreq)
        if len(timeSlice) > len(EAS_epoch):
            timeSlice = timeSlice[:len(EAS_epoch)]
        rpwDensity = np.array([rpwArray.sel(Epoch=slice(t-pd.Timedelta(seconds=0.5), t+pd.Timedelta(seconds=0.5))).mean(dim='Epoch').values for t in timeSlice])
    return rpwDensity

## Download EAS L1 Count data
def download_SO_L1_Count_NM3D(timeRange, downloadDir=None):
    # Get the L1 data from SOAR
    from sunpy.net import Fido, attrs as a
    import sunpy_soar

    if downloadDir is None:
        downloadDir = REPO_ROOT / 'data' / 'L1_Count'
    downloadPath = str(downloadDir)

    # Ensure the directory exists
    if not os.path.exists(downloadPath):
        os.makedirs(downloadPath)

    def adjust_timeRange(timeRange, min_hours=6):
        # Ensure at least 1 file can be located
        # Calculate the time difference
        start_dt = pd.to_datetime(timeRange[0])
        end_dt = pd.to_datetime(timeRange[-1])
        time_diff = end_dt - start_dt
        
        # Check if the time range is less than the minimum required duration
        if time_diff < pd.Timedelta(hours=min_hours):
            # Calculate how much time to add on each side to reach the minimum range
            additional_time = (pd.Timedelta(hours=min_hours) - time_diff) / 2
            start_dt = pd.to_datetime(timeRange[0]) - additional_time
            end_dt = pd.to_datetime(timeRange[-1]) + additional_time
        
        # Convert adjusted datetime back to string format
        adjusted_start = start_dt.strftime('%Y-%m-%d %H:%M')
        adjusted_end = end_dt.strftime('%Y-%m-%d %H:%M')
        return adjusted_start, adjusted_end
    
    def select_latest_version(results):
        from copy import deepcopy
        import re
        
        # Check if 'soar' provider exists in the results
        if 'soar' not in results.keys():
            print("No SOAR results found, returning original results")
            return results
        
        # Access the filenames directly from results['soar']['Filename']
        file_info = []
        
        # KEEP THE ORIGINAL CODE AS IS FOR EXTRACTING file_info
        for idx, filename in enumerate(results['soar']['Filename']):
            filename_str = str(filename)
            
            # Extract time period from filename
            time_match = re.search(r'(\d{8}T\d{6}-\d{8}T\d{6})', filename_str)
            if not time_match:
                continue
                    
            time_period = time_match.group(1)
            
            # Extract version from filename
            version_match = re.search(r'V(\d+)\.cdf$', filename_str)
            if not version_match:
                continue
                
            version = int(version_match.group(1))
            
            file_info.append({
                'index': idx,
                'time_period': time_period,
                'version': version,
                'filename': filename_str
            })
        
        # Extract start and end hour from each time period
        hour_groups = {}
        
        for info in file_info:
            # Parse the time_period to extract hour information
            # Format: YYYYMMDDTHHMMSS-YYYYMMDDTHHMMSS
            period_match = re.search(r'(\d{8}T\d{2})\d{4}-(\d{8}T\d{2})\d{4}', info['time_period'])
            if not period_match:
                continue
                
            # Extract hour-precision time parts
            start_hour = period_match.group(1)  # YYYYMMDDTHH
            end_hour = period_match.group(2)    # YYYYMMDDTHH
            
            # Create hour-precision key
            hour_key = f"{start_hour}-{end_hour}"
            
            if hour_key not in hour_groups:
                hour_groups[hour_key] = []
                
            hour_groups[hour_key].append(info)
        
        # For each hour-precision group, select only the file with the highest version
        indices_to_keep = []
        
        for hour_key, group_files in hour_groups.items():
            # Filter to keep only V01 files
            v01_files = [file for file in group_files if file['version'] == 1]
            
            if v01_files:
                # If V01 files exist for this hour group, use them
                for v01_file in v01_files:
                    indices_to_keep.append(v01_file['index'])
                    print(f"For hour period {hour_key}, selecting V01 file: {v01_file['filename']}")
            else:
                # If no V01 files, print a warning and skip this hour group
                print(f"Warning: No V01 files found for hour period {hour_key}. Available versions: {[f['version'] for f in group_files]}")
                print("Skipping this hour period.")
        
        # Keep only the selected files from the SOAR provider
        if indices_to_keep:
            # Sort indices to maintain original order
            indices_to_keep.sort()
            filtered = results['soar'][indices_to_keep]
        else:
            print("No V01 files found for any hour period. Returning empty results.")
            filtered = results['soar'][0:0]  # Empty subset
        
        original_count = len(results['soar'])
        kept_count = len(filtered)
        print(f"Found {len(hour_groups)} unique hour-precision time periods")
        print(f"Filtered {original_count} files to {kept_count} V01 files")
        
        return filtered

    adjust_timeRange = adjust_timeRange(timeRange)
    result_eas1 = Fido.search(a.Instrument('SWA') , a.Time(adjust_timeRange[0],adjust_timeRange[-1]) , a.Level(1) , a.soar.Product('swa-eas1-nm3d'))
    result_eas2 = Fido.search(a.Instrument('SWA') , a.Time(adjust_timeRange[0],adjust_timeRange[-1]) , a.Level(1) , a.soar.Product('swa-eas2-nm3d'))

    # Select the latest version if multiple versions exist
    result_eas1 = select_latest_version(result_eas1)
    result_eas2 = select_latest_version(result_eas2)

    print(f"Found {len(result_eas1)} EAS1 files and {len(result_eas2)} EAS2 files")
    
    # Download
    countEAS1_files = Fido.fetch(result_eas1, path=os.path.join(downloadPath, '{file}'))
    countEAS2_files = Fido.fetch(result_eas2, path=os.path.join(downloadPath, '{file}')) # downloadPath
    
    return countEAS1_files, countEAS2_files

# Data Processing
## Calculate the pitch angle in SRF
def getPitchAngle(velocityEas, magArray, timeArray):

    Vx_srf = velocityEas['vxSRF']
    Vy_srf = velocityEas['vySRF']
    Vz_srf = velocityEas['vzSRF']

    epoch_PA = np.zeros((len(timeArray), Vx_srf.shape[0], Vx_srf.shape[1], Vx_srf.shape[2])) # 4D array, considering epoch

    for i in range(len(timeArray)):

        magVector = magArray[i]

        cosPA = (Vx_srf*magVector[0]+Vy_srf*magVector[1]+Vz_srf*magVector[2])/(np.sqrt(Vx_srf**2 + Vy_srf**2 + Vz_srf**2) * np.sqrt((magVector**2).sum()))
        anglePA = np.degrees(np.arccos(cosPA))

        epoch_PA[i] = anglePA

    return epoch_PA


## subtract Vp from Vx, Vy, Vz
def move_to_PRF(vEAS, vp_Vector_slice):
    '''Proton-rest frame (PRF)'''
    vEAS_prf = {
        'vxPRF': np.zeros_like(vEAS['vxSRF']),
        'vyPRF': np.zeros_like(vEAS['vySRF']),
        'vzPRF': np.zeros_like(vEAS['vzSRF'])
    }

    vEAS_prf['vxPRF'] = vEAS['vxSRF'] - vp_Vector_slice[0]
    vEAS_prf['vyPRF'] = vEAS['vySRF'] - vp_Vector_slice[1]
    vEAS_prf['vzPRF'] = vEAS['vzSRF'] - vp_Vector_slice[2]

    return vEAS_prf

## Calculate v_para and v_perp in SRF
def calc_VparaVperp(vEAS, mag_Vector, vp_Vector):
    '''Calculate v_para and v_perp in SRF, & subtract Vp from Vx, Vy, Vz'''
    # Check if vEAS is time-dependent
    if vEAS['vxSRF'].ndim == 3:
        # Time-independent case: (16, 64, 32)
        time_dependent = False
        vEAS_stack = np.stack([vEAS['vxSRF'], vEAS['vySRF'], vEAS['vzSRF']], axis=0)  # (3, 16, 64, 32)
        n_time = len(mag_Vector)
        shape_suffix = vEAS['vxSRF'].shape
    else:
        # Time-dependent case: (epoch, 16, 64, 32)
        time_dependent = True
        vEAS_stack = np.stack([vEAS['vxSRF'], vEAS['vySRF'], vEAS['vzSRF']], axis=1)  # (epoch, 3, 16, 64, 32)
        n_time = vEAS_stack.shape[0]
        shape_suffix = vEAS['vxSRF'].shape[1:]
    
    # Initialize output arrays
    v_para = np.zeros((n_time, *shape_suffix))
    v_perp1 = np.zeros((n_time, *shape_suffix))
    v_perp2 = np.zeros((n_time, *shape_suffix))

    # Project Vp onto the new coordinate system as well
    vp_para = np.zeros(n_time)
    vp_perp1 = np.zeros(n_time)
    vp_perp2 = np.zeros(n_time)
    
    # vEAS_stack = np.stack([vEAS['vxSRF'], vEAS['vySRF'], vEAS['vzSRF']], axis=0)

    # v_para = np.zeros( (len(mag_Vector), *np.stack(vEAS['vxSRF'].shape)) )
    # v_perp1 = np.zeros( (len(mag_Vector), *np.stack(vEAS['vxSRF'].shape)) )
    # v_perp2 = np.zeros( (len(mag_Vector), *np.stack(vEAS['vxSRF'].shape)) )

    # # Project Vp onto the new coordinate system as well
    # vp_para = np.zeros(len(mag_Vector))
    # vp_perp1 = np.zeros(len(mag_Vector))
    # vp_perp2 = np.zeros(len(mag_Vector))
    
    # for timeIdx in range(len(mag_Vector)):
    for timeIdx in range(n_time):
        # vEAS_prf = move_to_PRF(vEAS, vp_Vector[timeIdx]) # remove Vp from Vx, Vy, Vz at each time step
        # vEAS_stack = np.stack([vEAS_prf['vxPRF'], vEAS_prf['vyPRF'], vEAS_prf['vzPRF']], axis=0)

        ## (1) Project vEAS to new coord
        unitMag = mag_Vector[timeIdx] / np.linalg.norm(mag_Vector[timeIdx])
        unitVp = vp_Vector[timeIdx] / np.linalg.norm(vp_Vector[timeIdx])

        # Get velocity data for this time step
        if time_dependent:
            vEAS_current = vEAS_stack[timeIdx]  # (3, 16, 64, 32)
        else:
            vEAS_current = vEAS_stack

        v_para[timeIdx] = np.tensordot(unitMag, vEAS_current, axes=1) # value of v_para

        # Compute the 1st perpendicular component
        arbitrary_vector = np.array([1, 0, 0]) if np.allclose(unitMag, unitVp) else unitVp # if np.allclose(magVector, [0, 1, 0]) else np.array([0, 1, 0])
        perp_vector1 = np.cross(unitMag, arbitrary_vector)
        perp_vector1 /= np.linalg.norm(perp_vector1)

        # Compute the 2nd perpendicular component
        perp_vector2 = np.cross(unitMag, perp_vector1)

        # Project onto the perpendicular components
        v_perp1[timeIdx] = np.tensordot(perp_vector1, vEAS_current, axes=1)
        v_perp2[timeIdx] = np.tensordot(perp_vector2, vEAS_current, axes=1)

        ## (2) Project Vp onto the new coordinate system as well
        vp_para[timeIdx] = np.dot(unitMag, vp_Vector[timeIdx])
        vp_perp1[timeIdx] = np.dot(perp_vector1, vp_Vector[timeIdx])
        vp_perp2[timeIdx] = np.dot(perp_vector2, vp_Vector[timeIdx])

    v_perp = (np.sqrt(v_perp1**2 + v_perp2**2)) # EAS
    vp_perp = (np.sqrt(vp_perp1**2 + vp_perp2**2)) # Vp

    return v_para, v_perp, {"Vp_para": vp_para, "Vp_perp": vp_perp}

## get the grid index of pitch angle of EAS data
def allocateGridIdx(data, gridBin, rightClosed = False):
    return np.digitize(data, gridBin, right=rightClosed) - 1 # np.digitize start from 1. -1 to keep consistency with array index

def create_time_intervals(time_range, freq='6H'):
    start_time = pd.to_datetime(time_range[0])
    end_time = pd.to_datetime(time_range[1])
    
    # Create regular interval points
    time_intervals = pd.date_range(start=start_time, end=end_time, freq=freq)
    
    # Ensure the last interval covers the end time
    if time_intervals[-1] < end_time:
        if end_time - time_intervals[-1] < pd.Timedelta('1H'):
            # Replace the last interval boundary with end_time
            temp_intervals = time_intervals.tolist()
            temp_intervals[-1] = end_time
            time_intervals = pd.Index(temp_intervals)
        else:
            # Add end_time as an additional interval boundary
            time_intervals = time_intervals.append(pd.DatetimeIndex([end_time]))
    
    # Create pairs of (start, end) intervals
    intervals = [(time_intervals[i], time_intervals[i+1]) 
                for i in range(len(time_intervals)-1)]
    
    print(f"Created {len(intervals)} time intervals")
    for i, (start, end) in enumerate(intervals):
        duration = end - start
        print(f"Interval {i+1}: {start} to {end} (duration: {duration})")
    
    return intervals

## Function to calculate the average PSD for pitch angles and energy
def getPsdPitchAngleEnergy(psdArray, uncArray, pitchAngIdx, pitchAngle):

    def calculate_average(grid, count_grid):
        """Calculate average, adjust sums, and handle NaNs."""
        # adjusted_sums = np.where(count_grid > 0, grid + 1.0, grid)
        average = np.divide(grid, count_grid, out=np.full_like(grid, 0.0), where=count_grid != 0)
        # average[grid == 0.] = np.nan
    
        return average

    psd_eas1 = psdArray['eas1']
    psd_eas2 = psdArray['eas2']
    unc_eas1 = uncArray['eas1']
    unc_eas2 = uncArray['eas2']

    # Initialize the array
    Psd_Pa_energy = np.zeros((psd_eas1.shape[0], psd_eas1.shape[2], len(pitchAngle), 3))
    # pitchAng_E = pd.DataFrame( np.zeros( (psd_eas1.shape[1], len(pitchAngle)) ), index=energyEas, columns=pitchAngle )
    # tempSum = np.zeros( len(pitchAngle) )
    # tempCount = np.zeros_like(tempSum)

    for time in range(psd_eas1.shape[0]):  # The first dimension of psd represents time
    
        # Iterate over energy levels
        for e in range(psd_eas1.shape[2]):  # The 3rd dimension of psd represents energy
            
            # Extract the PSD values and corresponding theta and phi indices for the current energy level
            psd1_slice = psd_eas1[time, :, e, :]
            unc_sliceEas1 = unc_eas1[time, :, e, :]
            pa1Idx_slice = pitchAngIdx['eas1'][time, :, e, :]

            psd2_slice = psd_eas2[time, :, e, :]
            unc_sliceEas2 = unc_eas2[time, :, e, :]
            pa2Idx_slice = pitchAngIdx['eas2'][time, :, e, :]

            PA_binIdx_dataIdx_eas1 = {i: np.where(pa1Idx_slice == i) for i in range( len(pitchAngle))}
            PA_binIdx_dataIdx_eas2 = {i: np.where(pa2Idx_slice == i) for i in range( len(pitchAngle))}


            for (PA_binIdx1, PA_dataIdx_eas1), (PA_binIdx2, PA_dataIdx_eas2) in zip(PA_binIdx_dataIdx_eas1.items(), PA_binIdx_dataIdx_eas2.items()):
                # PA_binIdx1 = PA_binIdx2
                psdSelect_eas1 = psd1_slice[PA_dataIdx_eas1]
                psdSelect_eas2 = psd2_slice[PA_dataIdx_eas2]
                psd_selection = np.concatenate([psdSelect_eas1, psdSelect_eas2])
                unc_selection = np.concatenate([unc_sliceEas1[PA_dataIdx_eas1], unc_sliceEas2[PA_dataIdx_eas2]])

                # Incorrect. Calculate the uncertainty of the mean PSD: taking the unc of each pixels (psd, count) and number of pixels (N) into account
                # nonZeroMask = count_selection > 0
                # psd_filtered = psd_selection[nonZeroMask]
                # count_filtered = count_selection[nonZeroMask]
               
                if len(unc_selection) > 0:
                    unc_meanPSD = np.sqrt( np.sum( (unc_selection/len(unc_selection))**2 ) )
                else:
                    unc_meanPSD = 0
                # unc_meanPSD = np.mean(psd_selection)/np.sqrt(len(psd_selection)) if len(psd_selection) > 0 else 0 # Temporary, used for unc of meanPSD. unc = meanPSD/np.sqrt(N of elements for mean)
                
                # pitchAng_E.loc[energyEas[e], pitchAngle[PA_binIdx1]] = np.mean(psd_selection) if len(psd_selection) > 0 else 0
                Psd_Pa_energy[time, e, PA_binIdx1,:] = [np.mean(psd_selection) if len(psd_selection) > 0 else np.nan, len(psd_selection), unc_meanPSD]
        
    return Psd_Pa_energy

## Concatenate 2 EAS data arrays
def reshape_concatenate(eas1, eas2):
    """
    Reshapes and concatenates two input arrays.

    Parameters:
    eas1 (numpy.ndarray): First input array of shape (time, elevation, energy, azimuth).
    eas2 (numpy.ndarray): Second input array of shape (time, elevation, energy, azimuth).

    Returns:
    numpy.ndarray: The concatenated array of shape (time, energy, elevation * azimuth * 2).
    """
    if eas1.shape != eas2.shape:
        raise ValueError("The input arrays must have the same shape.")
    
    # Extract the original dimensions
    time_dim, elevation_dim, energy_dim, azimuth_dim = eas1.shape
    # Ensure the second dimension is energy and the third is elevation before reshaping
    assert energy_dim == 64 and elevation_dim == 16 and azimuth_dim == 32, "Unexpected dimensions in the input arrays."

    # Reshape the first array
    eas1_reshape = np.transpose(eas1, axes=(0, 2, 1, 3)).reshape(time_dim, energy_dim, elevation_dim * azimuth_dim)
    
    # Reshape the second array
    eas2_reshape = np.transpose(eas2, axes=(0, 2, 1, 3)).reshape(time_dim, energy_dim, elevation_dim * azimuth_dim)
    
    # Concatenate the reshaped arrays along the third dimension
    eas_combined = np.concatenate((eas1_reshape, eas2_reshape), axis=2)
    
    return eas_combined

def save_data_toPickle(data, filePath):
    with open(filePath, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data saved to {filePath}\n")

def _as_hdf5_array(value):
    """Convert common science containers to HDF5-storable arrays without changing precision."""
    if value is None:
        return None, {}

    if hasattr(value, 'values'):
        value = value.values

    arr = np.asarray(value)
    attrs = {}
    if np.issubdtype(arr.dtype, np.datetime64):
        attrs['datetime64_unit'] = 'ns'
        arr = arr.astype('datetime64[ns]').astype(np.int64)
    return arr, attrs

def _write_hdf5_item(group, key, value, compression='gzip', compression_opts=4):
    if isinstance(value, dict):
        subgroup = group.create_group(key)
        subgroup.attrs['python_type'] = 'dict'
        for subkey, subvalue in value.items():
            _write_hdf5_item(subgroup, subkey, subvalue, compression=compression, compression_opts=compression_opts)
        return

    arr, attrs = _as_hdf5_array(value)
    if arr is None:
        dataset = group.create_dataset(key, data=np.array([], dtype=np.float64))
        dataset.attrs['is_none'] = True
        return

    kwargs = {}
    if arr.shape != ():
        kwargs.update({'compression': compression, 'compression_opts': compression_opts, 'shuffle': True})
        if arr.ndim >= 3:
            default_chunks = (30, 64, 1024, 5)
            kwargs['chunks'] = tuple(min(dim, default_chunks[idx]) for idx, dim in enumerate(arr.shape))

    dataset = group.create_dataset(key, data=arr, **kwargs)
    for attr_key, attr_value in attrs.items():
        dataset.attrs[attr_key] = attr_value

def save_data_toHDF5(data, filePath, compression='gzip', compression_opts=4):
    with h5py.File(filePath, 'w') as h5:
        h5.attrs['format'] = 'VDF_eas_forFitting'
        h5.attrs['version'] = 'hdf5_v1'
        for key, value in data.items():
            _write_hdf5_item(h5, key, value, compression=compression, compression_opts=compression_opts)
    print(f"Data saved to {filePath}\n")


def process_eas_data(eas1Files, eas2Files, countEAS1_files=None, countEAS2_files=None, download_dir=None, output_dir=None, timeRes='1min', custom_timeRange=None, shift_SCPOT=False):
    
    if output_dir is None:
        output_dir = str(REPO_ROOT / 'data' / 'processed')


    # Read and concatenate the DataArrays
    psdEAS1,datasetEAS1 = readCDFs(eas1Files) # 10s resolution
    psdEAS2,datasetEAS2 = readCDFs(eas2Files)
    check_data_consistency(psdEAS1, datasetEAS1)
    check_data_consistency(psdEAS2, datasetEAS2)


    # Eas1
    psdEAS1 = psdEAS1.where(psdEAS1 != psdEAS1.attrs['FILLVAL'], np.nan) # Drop invalid elements
    energyEAS1 = get_EAS_energy(datasetEAS1[0], instrument = 'EAS1')#datasetEAS1[0]['SWA_EAS1_ENERGY'][0].copy()
    psd_unit = psdEAS1.attrs['UNITS']
    thetaEAS1 = np.deg2rad(psdEAS1['SWA_EAS_ELEVATION'].values)  # Convert degrees to radians if necessary
    phiEAS1 = np.deg2rad(psdEAS1['SWA_EAS_AZIMUTH'].values)      # Convert degrees to radians if necessary
    psdEAS1['EPOCH'] = psdEAS1['EPOCH'].astype('datetime64[s]')

    # Eas2
    psdEAS2 = psdEAS2.where(psdEAS2 != psdEAS2.attrs['FILLVAL'], np.nan) # Drop invalid elements
    energyEAS2 = get_EAS_energy(datasetEAS2[0], instrument = 'EAS2')#datasetEAS2[0]['SWA_EAS2_ENERGY'][0].copy()
    thetaEAS2 = np.deg2rad(psdEAS2['SWA_EAS_ELEVATION'].values)  # Convert degrees to radians if necessary
    phiEAS2 = np.deg2rad(psdEAS2['SWA_EAS_AZIMUTH'].values)      # Convert degrees to radians if necessary
    psdEAS2['EPOCH'] = psdEAS2['EPOCH'].astype('datetime64[s]')

    vEAS1 = convert_eV_2Velocity(energyEAS1.values) 
    vEAS2 = convert_eV_2Velocity(energyEAS2.values) 
    # RPW data
    print('Getting RPW SCPOT data...')
    t_rpw_start = pd.Timestamp(psdEAS1['EPOCH'].values[0])-pd.Timedelta(seconds=30)
    t_rpw_end = pd.Timestamp(psdEAS1['EPOCH'].values[-1])+pd.Timedelta(seconds=30)
    rpw_data = get_RPW_SCPOT([t_rpw_start, t_rpw_end])

    # Transfer To Spacecraft Frame
    eas1_toSRF = datasetEAS1[0]['EAS1_TO_SRF'].values.copy()
    eas2_toSRF = datasetEAS2[0]['EAS2_TO_SRF'].values.copy()

    vxEAS1, vyEAS1, vzEAS1 = getVelocityArray_EAS(thetaEAS1, phiEAS1, vEAS1)
    vxEAS2, vyEAS2, vzEAS2 = getVelocityArray_EAS(thetaEAS2, phiEAS2, vEAS2)
    normalVec_xz = [0,1,0]
    vEAS1_srf = transferToSRF(eas1_toSRF, vxEAS1, vyEAS1, vzEAS1) # dict: vxSRF, vySRF, vzSRF. Each Array_Shape: (16, 64, 32)
    vEAS2_srf = transferToSRF(eas2_toSRF, vxEAS2, vyEAS2, vzEAS2)

    # Define time range for the dataset
    if custom_timeRange is None:
        timeRange_EAS = [psdEAS1['EPOCH'].values[0], psdEAS1['EPOCH'].values[-1]] # Time range of the EAS data
        print(f"Using default time range: {timeRange_EAS[0]} to {timeRange_EAS[1]}")
    else:
        timeRange_EAS = custom_timeRange
        print(f"Using provided custom time range: {timeRange_EAS[0]} to {timeRange_EAS[1]}")


    # Process count data - download if not provided
    if countEAS1_files is None or countEAS2_files is None:
        print(f"\nDownloading count data for time range {timeRange_EAS[0]} to {timeRange_EAS[1]}")
        countEAS1_files,countEAS2_files = download_SO_L1_Count_NM3D(timeRange_EAS, downloadDir = os.path.join(download_dir, 'L1_Count'))
        
        
    print('Reading EAS1 count data...')
    countEAS1,count_datasetEAS1 = readCDFs(countEAS1_files, skip_errors=True) #
    print('Reading EAS2 count data...')
    countEAS2,count_datasetEAS2 = readCDFs(countEAS2_files, skip_errors=True) #
    
    # Check if we have valid count data
    if countEAS1 is None or countEAS2 is None:
        print("Error: Could not read any valid count data for at least one sensor.")
        return []

    countEAS1['EPOCH'] = countEAS1['EPOCH'].astype('datetime64[s]')
    countEAS2['EPOCH'] = countEAS2['EPOCH'].astype('datetime64[s]')

    # Identify valid time regions
    print("\nIdentifying valid time regions...")
    # valid_regions = get_valid_range_EASdata(timeRange_EAS, countEAS1, countEAS2)
    overlapped_data_dict = align_eas_datasets(psdEAS1, psdEAS2, countEAS1, countEAS2, timeRange = custom_timeRange)
    psdEAS1 = overlapped_data_dict['psdEAS1']
    psdEAS2 = overlapped_data_dict['psdEAS2']
    countEAS1 = overlapped_data_dict['countEAS1']
    countEAS2 = overlapped_data_dict['countEAS2']

    
    # Create time intervals respecting both 6H frequency and valid regions
    print("Creating time intervals...")
    time_ranges = create_time_intervals([psdEAS1['EPOCH'].values[0], psdEAS1['EPOCH'].values[-1]], freq='6H')
    
    if not time_ranges:
        print("No valid time intervals could be created.")
        return []
    
    
    processed_file_paths = []
    for id, timeRange in enumerate(time_ranges[0:]):
        print(f"\nProcessing interval {id+1}/{len(time_ranges)}: {timeRange[0]} to {timeRange[1]}")
        
        # Extract desired period of EAS PSD 
        psd1 = slice_xarrayEAS(psdEAS1, timeRange, timeFreq=timeRes) # psdEAS1 # If needed: slice_psdEAS(psdEAS1, timeRange)
        psd2 = slice_xarrayEAS(psdEAS2, timeRange, timeFreq=timeRes) # psdEAS2 # If needed: slice_psdEAS(psdEAS2, timeRange)
        if psd1 is None or psd2 is None:
            print(f"\nWarning: No PSD data found between {timeRange[0]} and {timeRange[1]}, skip this interval ...")
            continue
        time_stamps = psd1['EPOCH'].values # psdEAS1['EPOCH'].values # time stamps aligned with magVector and pasVector

        count1 = slice_xarrayEAS(countEAS1, timeRange, timeFreq=timeRes)
        count2 = slice_xarrayEAS(countEAS2, timeRange, timeFreq=timeRes)

        # Calculate uncertainties
        GF_EAS1 = load_geometric_factors('EAS1', count1)
        Qe_EAS1 = load_quantum_efficiency('EAS1', count1)
        GF_EAS2 = load_geometric_factors('EAS2', count2)
        Qe_EAS2 = load_quantum_efficiency('EAS2', count2)
        
        # Calculate VDF uncertainties
        unc_vdf_eas1 = calculate_vdf_from_count(np.sqrt(count1), GF_EAS1, Qe_EAS1, 'EAS1')
        unc_vdf_eas2 = calculate_vdf_from_count(np.sqrt(count2), GF_EAS2, Qe_EAS2, 'EAS2')
        print('unc_vdf_eas1.values.shape', unc_vdf_eas1.values.shape)
        print('unc_vdf_eas2.values.shape', unc_vdf_eas2.values.shape)
        
        # Reshape and combine uncertainties
        unc_dataProduct1 = reshape_concatenate(unc_vdf_eas1.values, unc_vdf_eas2.values)

        # MAG data
        magData = get_SO_MAG_SRF_Normal(timeRange) # 0.125s resolution
        B_srf = magData['B_SRF'].copy()
        magVector = average_MAG_around_timeStamp(B_srf, time_stamps)#, timeFreq = timeRes)
        PA_eas1 = getPitchAngle(vEAS1_srf, magVector, time_stamps) # magVector varies with time.
        PA_eas2 = getPitchAngle(vEAS2_srf, magVector, time_stamps)

        # PAS velocity data
        pasData = get_SO_PAS_Grnd_Moment(timeRange) # 4s resolution
        vpVector = average_PAS_around_timeStamp(pasData['V_SRF'], time_stamps)#, timeFreq = timeRes)
        protonDensity = average_PAS_around_timeStamp(pasData['N'], time_stamps)#, timeFreq = timeRes)

        # Define the grid
        grid_interval = 10
        PitchAngleBound = np.arange(0, 180 + grid_interval, grid_interval)# np.arange(0 - grid_interval / 2, 180 + grid_interval, grid_interval)
        PA_MidValue = (PitchAngleBound[:-1] + PitchAngleBound[1:]) / 2

        # Find the Grid Indices of each data point
        pitchAngIdx_Eas1 = allocateGridIdx(PA_eas1, PitchAngleBound, rightClosed=True)
        pitchAngIdx_Eas2 = allocateGridIdx(PA_eas2, PitchAngleBound, rightClosed=True)

        # Pitch Angle Distribution, pad_in
        print('\nGenerating Pitch Angle Distribution...')
        psd_PA_Energy = getPsdPitchAngleEnergy(
            psdArray = {'eas1': psd1.values, 'eas2': psd2.values},
            uncArray = {'eas1': unc_vdf_eas1.values, 'eas2': unc_vdf_eas2.values},
            pitchAngIdx={'eas1': pitchAngIdx_Eas1, 'eas2': pitchAngIdx_Eas2},
            pitchAngle=PA_MidValue
        )  # shape: (time, energy bin = 64, pa bins = 36, (mean PSD [km^6/s^3], number of points) )
        print('psd_PA_Energy done.')

        # Plot PAD vs time figure
        desiredEnergyIdx = np.where((energyEAS1 > 50)&(energyEAS1<2000))[0] # average over 50 - 2000 eV
        pitchAngleBin_and_idxArray = None#{'pitchAngleBound': PitchAngleBound, 'PA_bin_indices_eas1': pitchAngIdx_Eas1, 'PA_bin_indices_eas2': pitchAngIdx_Eas2}
        def get_PAD(psd1, psd2, desiredEnergyIdx, time_stamps, pitchAngleVariables = None):
            if pitchAngleVariables is not None:
                pitchAngleBound = pitchAngleVariables['pitchAngleBound']
                PA_bin_indices_eas1 = pitchAngleVariables['PA_bin_indices_eas1']
                PA_bin_indices_eas2 = pitchAngleVariables['PA_bin_indices_eas2']
            else:
                pitchAngleBound = np.arange(0, 185, 5)
                PA_bin_indices_eas1 = allocateGridIdx(PA_eas1, pitchAngleBound, rightClosed=True)
                PA_bin_indices_eas2 = allocateGridIdx(PA_eas2, pitchAngleBound, rightClosed=True)
            
            midpoints_PA_bins = (pitchAngleBound[:-1] + pitchAngleBound[1:]) / 2
            psd_averaged = pd.DataFrame(np.zeros((len(psd1['EPOCH']), len(pitchAngleBound)-1)),index=psd1['EPOCH'],columns=midpoints_PA_bins)

            for timeId in range(len(time_stamps)):
                # Find indices_Array for each PA_bin
                PA_bin_locations_eas1 = {i: np.where(PA_bin_indices_eas1[timeId] == i) for i in range(0, len(pitchAngleBound)-1)}
                PA_bin_locations_eas2 = {i: np.where(PA_bin_indices_eas2[timeId] == i) for i in range(0, len(pitchAngleBound)-1)}
                
                # sum the PSD over the desired energy range at each PA bins
                for (eas1PA_bin_number, eas1PA_indices_Tuple), (eas2PA_bin_number, eas2PA_indices_Tuple) in zip(PA_bin_locations_eas1.items(), PA_bin_locations_eas2.items()):

                    desiredEnergy_loc_eas1 = np.sort(np.concatenate( [np.where(eas1PA_indices_Tuple[1] == Idx)[0] for Idx in desiredEnergyIdx] )) # For a specific PA bin, find location of all the PSD data that falls into the desired Energy range
                    desiredEnergy_loc_eas2 = np.sort(np.concatenate( [np.where(eas2PA_indices_Tuple[1] == Idx)[0] for Idx in desiredEnergyIdx] )) # For a specific PA bin, find location of all the PSD data that falls into the desired Energy range
                    
                    psd_selection_eas1 = psd1.isel(EPOCH=timeId).values[eas1PA_indices_Tuple[0][desiredEnergy_loc_eas1], eas1PA_indices_Tuple[1][desiredEnergy_loc_eas1], eas1PA_indices_Tuple[2][desiredEnergy_loc_eas1]]
                    psd_selection_eas2 = psd2.isel(EPOCH=timeId).values[eas2PA_indices_Tuple[0][desiredEnergy_loc_eas2], eas2PA_indices_Tuple[1][desiredEnergy_loc_eas2], eas2PA_indices_Tuple[2][desiredEnergy_loc_eas2]]

                    psd_selection = np.concatenate([psd_selection_eas1, psd_selection_eas2])
                    psd_averaged.loc[psd1['EPOCH'].values[timeId], midpoints_PA_bins[eas1PA_bin_number]] = np.mean(psd_selection) if len(psd_selection) > 0 else np.nan
            return psd_averaged
        
        def plot_PAD(psd_averaged, output_dir, vmin=0, vmax=None):
            fig, ax = plt.subplots(1, figsize=(11, 3.5), sharex=True)
            fig.subplots_adjust(left = 0.1,right=0.99,top=0.95,bottom=0.2)

            if vmin is None:
                vmin = np.nanmin(psd_averaged.values)
            if vmax is None:
                vmax = np.nanmax(psd_averaged.values)

            X, Y = np.meshgrid(psd_averaged.index.values, psd_averaged.columns.values)
            pcm = ax.pcolormesh(X.T, Y.T, psd_averaged.values, cmap='jet',shading='nearest', vmin=vmin, vmax=vmax) # cmap='RdBu_r'

            # set x-axis
            date_format = DateFormatter("%H:%M" +"\n" + "%Y-%m-%d")
            ax.xaxis.set_major_formatter(date_format)
            ax.tick_params(axis='x', labelsize=11)

            ax.set_xlabel('SO/EAS', fontsize=14, labelpad=3)
            ax.set_ylabel('Pitch Angle',fontsize=14)

            # set colorbar
            bar = fig.colorbar(pcm, pad=0.02)
            bar.set_label('Phase Space Density [$s^3km^{-6}$]',fontsize=12)
            # set the colorbar ticks to be scientific notation
            bar.formatter.set_powerlimits((0, 0))

            figname = os.path.join(output_dir,'PAD_plots', 'PAD_' + psd_averaged.index[0].strftime('%Y%m%d%H%M%S') + '_' + psd_averaged.index[-1].strftime('%Y%m%d%H%M%S') + '.png')
            os.makedirs(os.path.dirname(figname), exist_ok=True)
            fig.savefig(figname, dpi=200)
            plt.close()
            print(f'\nPAD plot saved to {figname}')

        psd_averaged = get_PAD(psd1, psd2, desiredEnergyIdx, time_stamps, pitchAngleVariables = pitchAngleBin_and_idxArray)
        plot_PAD(psd_averaged, output_dir)
        
        # Download RPW_density
        print('Downloading RPW_density...')
        rpwData = get_SO_RPW_Density(timeRange) # super high resolution
        if rpwData is not None:
            eDensity = average_RPWDensity_around_timeStamp(rpwData['DENSITY'], time_stamps, timeFreq = None)
        else:
            eDensity = protonDensity # If it's not available, use proton density from PAS

        if shift_SCPOT:
            scpot_aligned = align_rpw_to_eas(rpw_data['SCPOT'], time_stamps) # sc potential aligned to EAS time stamps

            def fill_scpot_nan_with_local_mean(scpot_aligned, time_stamps, time_window_hours=3):
                nan_mask = np.isnan(scpot_aligned.values)
                
                if nan_mask.any():
                    nan_indices = np.where(nan_mask)[0]
                    time_window = np.timedelta64(int(time_window_hours * 3600), 's')  # Convert hours to seconds
                    
                    for idx in nan_indices:
                        # Find indices within time window before current time
                        current_time = time_stamps[idx]
                        window_mask = (time_stamps >= current_time - time_window) & (time_stamps <= current_time)
                        
                        # Calculate mean of valid values in window
                        window_values = scpot_aligned.values[window_mask]
                        local_mean = np.nanmean(window_values)
                        
                        # If no valid values in window, use global mean
                        if np.isnan(local_mean):
                            local_mean = np.nanmean(scpot_aligned.values)
                        
                        scpot_aligned.values[idx] = local_mean
                    
                    print(f"Replaced {len(nan_indices)} NaN SC potential values with {time_window_hours}-hour local mean")
                return scpot_aligned
            
            scpot_aligned = fill_scpot_nan_with_local_mean(scpot_aligned, time_stamps, time_window_hours=3)
            n_time = len(time_stamps)
            # Initialize velocity arrays
            vxEAS1_srf_all = np.zeros((n_time, 16, 64, 32))
            vyEAS1_srf_all = np.zeros((n_time, 16, 64, 32))
            vzEAS1_srf_all = np.zeros((n_time, 16, 64, 32))
            
            vxEAS2_srf_all = np.zeros((n_time, 16, 64, 32))
            vyEAS2_srf_all = np.zeros((n_time, 16, 64, 32))
            vzEAS2_srf_all = np.zeros((n_time, 16, 64, 32))

            energy_corrected_eas1_all = np.zeros((n_time, len(energyEAS1)))

            for timeIdx in range(n_time):
                # Convert energy to velocity for this time step
                energy_corrected_eas1 = energyEAS1 - scpot_aligned.values[timeIdx]
                energy_corrected_eas2 = energyEAS2 - scpot_aligned.values[timeIdx]
                # Mark negative energies as invalid 
                energy_corrected_eas1 = np.where(energy_corrected_eas1 > 0, energy_corrected_eas1, np.nan)
                energy_corrected_eas2 = np.where(energy_corrected_eas2 > 0, energy_corrected_eas2, np.nan)

                energy_corrected_eas1_all[timeIdx] = energy_corrected_eas1 # record the corrected energy at each time step

                vEAS1_t = convert_eV_2Velocity(energy_corrected_eas1)  # (64,)
                vEAS2_t = convert_eV_2Velocity(energy_corrected_eas2)  # (64,)
                
                # Calculate velocity components
                vxEAS1_t, vyEAS1_t, vzEAS1_t = getVelocityArray_EAS(thetaEAS1, phiEAS1, vEAS1_t)
                vxEAS2_t, vyEAS2_t, vzEAS2_t = getVelocityArray_EAS(thetaEAS2, phiEAS2, vEAS2_t)
                
                # Transfer to SRF
                vEAS1_srf_t = transferToSRF(eas1_toSRF, vxEAS1_t, vyEAS1_t, vzEAS1_t)
                vEAS2_srf_t = transferToSRF(eas2_toSRF, vxEAS2_t, vyEAS2_t, vzEAS2_t)
                
                vxEAS1_srf_all[timeIdx] = vEAS1_srf_t['vxSRF']
                vyEAS1_srf_all[timeIdx] = vEAS1_srf_t['vySRF']
                vzEAS1_srf_all[timeIdx] = vEAS1_srf_t['vzSRF']
                
                vxEAS2_srf_all[timeIdx] = vEAS2_srf_t['vxSRF']
                vyEAS2_srf_all[timeIdx] = vEAS2_srf_t['vySRF']
                vzEAS2_srf_all[timeIdx] = vEAS2_srf_t['vzSRF']
            
            vEAS1_srf_for_vParaPerp = {'vxSRF': vxEAS1_srf_all, 'vySRF': vyEAS1_srf_all, 'vzSRF': vzEAS1_srf_all}
            vEAS2_srf_for_vParaPerp = {'vxSRF': vxEAS2_srf_all, 'vySRF': vyEAS2_srf_all, 'vzSRF': vzEAS2_srf_all}
            print('vEAS_srf shifted by SCPOT, done.')
        else:
            # Time-independent energy case (original behavior)
            vEAS1_srf_for_vParaPerp = vEAS1_srf
            vEAS2_srf_for_vParaPerp = vEAS2_srf
        
        # Calculate v_para and v_perp. # Not:(1) Subtract Vp from Vx, Vy, Vz; 
        vParaEas1, vPerpEas1, vp_transformed = calc_VparaVperp(vEAS1_srf_for_vParaPerp, magVector, vpVector) 
        vParaEas2, vPerpEas2, _ = calc_VparaVperp(vEAS2_srf_for_vParaPerp, magVector, vpVector)

        psd_combined = reshape_concatenate(psd1.values, psd2.values) # shape: (time, 64, 16*32*2)
        PA_combined = reshape_concatenate(PA_eas1, PA_eas2)
        vPara_combined = reshape_concatenate(vParaEas1, vParaEas2)
        vPerp_combined = reshape_concatenate(vPerpEas1, vPerpEas2)
        counts_combined = reshape_concatenate(count1.values, count2.values)
        print('Generating data product 1 ...')
        dataProduct1 = np.stack((PA_combined, psd_combined, vPara_combined, vPerp_combined, counts_combined), axis=-1) # pa_vec_in

        dataProduct_output = {
            'pa_vec_in': dataProduct1,
            'pad_in': psd_PA_Energy,
            'pitch_angles': PitchAngleBound,
            'B_avg': magVector,
            'time_EAS': time_stamps,
            'swa_energy': energy_corrected_eas1_all if shift_SCPOT else energyEAS1.values, #energyEAS1.values,
            'n_sc': eDensity,
            'u': vpVector,
            'vp_transformed': vp_transformed,
            'n_proton': protonDensity,
            'unc_pa_vec_in': unc_dataProduct1,
            'scpot_aligned': scpot_aligned if shift_SCPOT else None
        }

        # Save output to file
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        fileName = 'VDF_eas_forFitting_' + timeRange[0].strftime('%Y%m%d%H%M%S') + '_' + timeRange[-1].strftime('%Y%m%d%H%M%S') + f'_{timeRes}.h5'
        filePath = os.path.join(output_dir, fileName)
        save_data_toHDF5(dataProduct_output, filePath)
        processed_file_paths.append(filePath)
    
    return processed_file_paths

def compute_break_energy_for_files(processed_file_paths, output_dir, break_energy_output_dir=None,
         cone_half_width_deg=None, break_energy_workers=None):
    """Run the cone-feature-split break energy detection on freshly preprocessed VDF files.

    Defaults the output to results/break_energy/<output_dir name>/ so each
    preprocessing run keeps its own break energy products.
    """
    from compute_break_energy_cone_features import (
        DEFAULT_CONE_HALF_WIDTH_DEG,
        DEFAULT_WORKERS,
        runBreakEnergyPipeline,
    )

    if break_energy_output_dir is None:
        run_name = os.path.basename(os.path.normpath(output_dir))
        break_energy_output_dir = str(REPO_ROOT / 'results' / 'break_energy' / run_name)

    print(f"\nComputing break energy for {len(processed_file_paths)} processed file(s)...")
    print(f"Break energy output directory: {break_energy_output_dir}")
    return runBreakEnergyPipeline(
        inputFilePaths=processed_file_paths,
        outputDir=break_energy_output_dir,
        coneHalfWidthDeg=cone_half_width_deg if cone_half_width_deg is not None else DEFAULT_CONE_HALF_WIDTH_DEG,
        workerCount=break_energy_workers if break_energy_workers is not None else DEFAULT_WORKERS,
    )


def preprocess_main(eas1_files=None, eas2_files=None, count_eas1_files=None, count_eas2_files=None,
         start_time=None, end_time=None, time_resolution='1min', custom_timeRange=None, shift_SCPOT=False,
         download_dir=None, output_dir=None, compute_break_energy=True, break_energy_output_dir=None,
         cone_half_width_deg=None, break_energy_workers=None):

    # Set default paths relative to the repository root if not provided
    if download_dir is None:
        download_dir = str(REPO_ROOT / 'data')

    if output_dir is None:
        output_dir = str(REPO_ROOT / 'data' / 'processed')
    
    # Create download and output directories if they don't exist
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for required parameters
    if not eas1_files and not eas2_files and (not start_time or not end_time):
        raise ValueError("Either provide file paths (eas1_files, eas2_files) or time range (start_time, end_time)")
    
    # Download or use provided EAS data files
    if eas1_files and eas2_files:
        print("Using provided EAS1 and EAS2 data files list.")
    else:
        if not start_time or not end_time:
            raise ValueError("Start time and end time are required for downloading data")
            
        print(f"Downloading EAS data files for period: {start_time} to {end_time}")
        downloaded_files = download_eas_data(start_time, end_time, save_dir=os.path.join(download_dir, 'vdfs'))
        
        # Separate EAS1 and EAS2 files
        eas1_files = [f for f in downloaded_files if 'eas1' in f.lower()]
        eas2_files = [f for f in downloaded_files if 'eas2' in f.lower()]
        
        if not eas1_files or not eas2_files:
            print("Warning: Failed to download either EAS1 or EAS2 files")
            return []
    

    # Process the data files
    print("\nProcessing EAS files...")
    processed_file_paths = process_eas_data(eas1_files, eas2_files,  count_eas1_files, count_eas2_files, download_dir, output_dir,  timeRes=time_resolution, custom_timeRange=custom_timeRange, shift_SCPOT=shift_SCPOT)
    
    print(f"\nProcessing complete. {len(processed_file_paths)} output files generated:")
    for path in processed_file_paths:
        print(f"{path}")

    if compute_break_energy and processed_file_paths:
        compute_break_energy_for_files(
            processed_file_paths,
            output_dir,
            break_energy_output_dir=break_energy_output_dir,
            cone_half_width_deg=cone_half_width_deg,
            break_energy_workers=break_energy_workers,
        )

    return processed_file_paths

if __name__ == "__main__":
    # Template: preprocess one 6-hour segment of SWA-EAS data.
    #
    # Download the L2 PSD and L1 count CDF files from SOAR first (see
    # docs/TUTORIAL.md), place them under data/vdfs/ and data/L1_Count/,
    # and list them below. Leave count files as None to auto-download via
    # sunpy-soar. Each preprocessed output covers the time span of one
    # input CDF pair (typically 6 hours).
    dataDir = REPO_ROOT / 'data' / 'vdfs'
    countDir = REPO_ROOT / 'data' / 'L1_Count'

    eas1File = [str(dataDir / 'solo_L2_swa-eas1-nm3d-psd_20220302T060509-20220302T120459_corrLPP_V01.cdf')]
    eas2File = [str(dataDir / 'solo_L2_swa-eas2-nm3d-psd_20220302T060509-20220302T120459_corrLPP_V01.cdf')]
    countEAS1_files = [str(countDir / 'solo_L1_swa-eas1-NM3D_20220302T060509-20220302T120459_V01.cdf')]
    countEAS2_files = [str(countDir / 'solo_L1_swa-eas2-NM3D_20220302T060509-20220302T120459_V01.cdf')]

    processed_file_paths = preprocess_main(
        eas1_files=eas1File, eas2_files=eas2File,
        count_eas1_files=countEAS1_files, count_eas2_files=countEAS2_files,
        output_dir=str(REPO_ROOT / 'data' / 'processed'),
        shift_SCPOT=True,           # correct electron energies by spacecraft potential (recommended)
        compute_break_energy=True,  # also produce *_cone_feature_split.pkl for the fitting step
    )
    print(processed_file_paths)
