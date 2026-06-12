import numpy as np

# Plot settings
s_set = 2 # scatter size, plotting
lw_set = 1 # linewidth, plotting

# constants #
m_e=9.109*10.**(-28.)
eV_to_J=1.602*(10.**(-19.))
eV_to_Erg = 1.602*(10.**(-12.))
m0=1.2566*10.**(-6.)
kb=1.3806*10.**(-23.)
m_p = 1.6726*10.**(-24)

stderr_fact = 0.05 # Estimate the error of fitted parameters if their std is 0
PAW_coeff = 2*np.sqrt(2*np.log(2))

D_CONST = 10
n_kappa_factor = 0.9*(2*np.sqrt(D_CONST))/np.sqrt(D_CONST+1)
n_const_range = [0.6, 1.4] #[0.5, 1.5] # 
u_const_range = [0.6, 1.4]

n_halo_factor = 10
n_beam_factor = 40

# break energy range
break_E_range = [35, 140] # eV

# SC Potential energy 
low_sc_val, high_sc_val = 0, 12 # eV
low_pad_val, high_pad_val = 61, 1000 # eV

# Halo
# halo_energy_thresh = 100 # eV
PixelCountThreshold_core = 2 # pixels that used in the core fitting
PixelCountThreshold_halo = 2 # pixels that used in the halo fitting
PixelCountThreshold_beam = 2 # pixels that used in the beam fitting
PixelCountThreshold_all = 2

# Beam
kappa_b_init = 4
# beam_energy_thresh = 70 # eV
beam_Vth_bound = [0, 1.5e9] # 5keV: 4.2e9; 600eV: 1.5e9
beam_kappa_bound = [1.6, 20]
# Constants
beam_ratio_cond = 1.2
deficit_ratio_cond = 0.5

# Core
low_c_val = 14 # eV
# high_c_val = beam_energy_thresh # eV

CORE_PARAMS = {
    'n_ratio': {
        'init': 0.8,        # initial density fraction
        'min': 0,    # minimum density fraction
        'max': 1.5    # maximum density fraction
    },
    'u_par': {
        'init': 0.0,         # initial parallel drift velocity
        'min': -2e8,         # minimum parallel drift velocity, cm/s
        'max': 2e8           # maximum parallel drift velocity, cm/s, 11eV
    },
    'v_th_par': {
        'init': 2e8,         # initial parallel thermal speed, cm/s
        'min': 1e8,          # minimum parallel thermal speed, cm/s, 2.8eV
        'max': 3.5e8           # maximum parallel thermal speed, cm/s, 35eV
    },
    'v_th_perp': {
        'init': 2e8,         # initial perpendicular thermal speed
        'min': 1e8,          # minimum perpendicular thermal speed
        'max': 3.5e8           # maximum perpendicular thermal speed
    }
}

# Halo component constants
HALO_PARAMS = {
    'n_ratio': {
        'init': 0.15,       # initial density fraction
        'min': 0.03,   # minimum density fraction
        'max': 0.3     # maximum density fraction
    },
    'u_par': {
        'init': 0.0,         # initial parallel drift velocity
        'min': -2.2e8,         # minimum parallel drift velocity, cm/s
        'max': 2.2e8           # maximum parallel drift velocity, cm/s, 14eV
    },
    'v_th_par': {
        'init': 4.5e8,         # initial parallel thermal speed, 58eV
        'min': 2.5e8,          # minimum parallel thermal speed, 3e8-25eV, 2.5e8-17.8eV
        'max': 7.3e8           # maximum parallel thermal speed, 150eV. 2e9-1137eV
    },
    'v_th_perp': {
        'init': 4.5e8,         # initial perpendicular thermal speed
        'min': 2.5e8,          # minimum perpendicular thermal speed
        'max': 7.3e8        # maximum perpendicular thermal speed 
    },
    'kappa': {
        'init': 4.0,         # initial kappa
        'min': 1.8,          # minimum kappa
        'max': 15.0          # maximum kappa
    },
    'delta': {
        'init': 6.0,         # initial delta (flat-top width)
        'min': 3.0,          # minimum delta
        'max': 10.0           # maximum delta
    }
}