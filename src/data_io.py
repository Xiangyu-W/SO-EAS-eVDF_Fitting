# src/data_io.py

import pickle
import os

import numpy as np
import pandas as pd
import h5py

def _read_hdf5_item(obj):
    if isinstance(obj, h5py.Group):
        return {key: _read_hdf5_item(obj[key]) for key in obj.keys()}

    if obj.attrs.get('is_none', False):
        return None

    data = obj[()]
    if obj.attrs.get('datetime64_unit', None) == 'ns':
        data = data.astype('datetime64[ns]')
    return data

def load_data(pickle_dir, pickle_file):
    from pathlib import Path
    pickle_dir = Path(pickle_dir)
    file_path = pickle_dir / pickle_file

    if file_path.suffix.lower() in {'.h5', '.hdf5'}:
        with h5py.File(file_path, 'r') as h5:
            return {key: _read_hdf5_item(h5[key]) for key in h5.keys()}

    with open(file_path, 'rb') as file:
        return pickle.load(file)

def convert_units(data_dict):
    # Convert units as done previously
    # Example: Convert PSD to cgs if needed
    psd_to_cgs = 1e-30
    km_to_cm = 1e5

    import copy
    data = copy.deepcopy(data_dict)

    data['pad_in'][:,:,:,0] = data['pad_in'][:,:,:,0]*psd_to_cgs
    data['pad_in'][:,:,:,2] = data['pad_in'][:,:,:,2]*psd_to_cgs

    data['pa_vec_in'][:,:,:,1] = data['pa_vec_in'][:,:,:,1]*psd_to_cgs
    data['pa_vec_in'][:,:,:,2] = data['pa_vec_in'][:,:,:,2]*km_to_cm
    data['pa_vec_in'][:,:,:,3] = data['pa_vec_in'][:,:,:,3]*km_to_cm
    data['u'] = data['u']*km_to_cm
    data['vp_transformed']['Vp_para'] = data['vp_transformed']['Vp_para']*km_to_cm
    data['vp_transformed']['Vp_perp'] = data['vp_transformed']['Vp_perp']*km_to_cm
    data['unc_pa_vec_in'] = data['unc_pa_vec_in']*psd_to_cgs

    # Add any other unit conversions here
    return data

def load_OneParticleNoise(pickle_dir):
    one_particle_noise_shape = tuple(np.sum(np.array(pd.read_pickle(pickle_dir+'one_particle_noise_shape')), axis = 1))
    one_particle_noise = np.reshape(np.sum(np.array(pd.read_pickle(pickle_dir+'one_particle_noise')), axis = 1), one_particle_noise_shape)
    
    one_particle_noise = np.mean(one_particle_noise, axis = 0)
    one_particle_noise = np.mean(one_particle_noise, axis = 0)
    one_particle_noise = np.mean(one_particle_noise, axis = 1)
    one_particle_noise = (10**(-12))*one_particle_noise # Convert to cgs. in s^3/cm^6
    return one_particle_noise
