import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.constants as sc
from tqdm import tqdm
from scipy.optimize import minimize
import lmfit
import scipy.special as ss
from scipy.special import gamma

D_CONST = 10

def bi_Max(v_par, v_perp, n, u_par, v_th_par, v_th_perp) :
    """
    Produces a VDF following a bi-Maxwellian distribution. All in CGS units.

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
    u_perp : array
        Bulk velocity in the perpendicular direction.
    v_th_par : float
        Thermal velocity in parallel direction.
    v_th_perp : float
        Thermal velocity in perpendicular direction.

    Returns
    -------
    f : array
        The Velocity distribution function VDF.

    """

    c_par = v_par - u_par
    c_perp = v_perp

    denominator = (np.pi ** (3/2)) * v_th_par * (v_th_perp**2)
    term1 = n / denominator
    exponent = (c_par/v_th_par)**2 + (c_perp/v_th_perp)**2

    f = term1 * np.exp(- exponent)

    return f

def kappa_vdf(v_par, v_perp, n, u_par, v_th_par, v_th_perp, kappa) :

    """
    Produces a VDF following a bi-Maxwellian distribution. All in CGS units.

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

def kappa_beam(v_par, v_perp, n, u_par, v_th_par, v_th_perp, kappa) :

    """
    Jesse's beam distribution, where D===10. All in CGS units.

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
    D = D_CONST

    denominator = (np.pi ** (3/2)) * v_th_par * (v_th_perp**2)
    D_coeff = (2*np.sqrt(D))/np.sqrt(D+1)
    term1 = (D_coeff*n) / denominator

    gamma_ratio = gamma(kappa+1)/(gamma(kappa - 0.5)*((kappa - 3/2)**(3/2)))

    return term1*gamma_ratio*(( 1 + D*((v_par-u_par)**2)/((v_th_par**2)*(kappa - 3/2)) + (v_perp**2)/((v_th_perp**2)*(kappa - 3/2)))**(-kappa -1))

def kappa_flattop_vdf(v_par, v_perp, n, u_par, v_th_par, v_th_perp, kappa) :

    """
    Produces a VDF following a bi-Maxwellian distribution. All in CGS units.

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
    p, q : come from Steverak et al. 2009, found to be convenient.
    delta: is set sto 90% of a gaussian function

    Returns
    -------
    f : array
        The Velocity distribution function VDF.

    """

    denominator = (np.pi ** (3/2)) * v_th_par * (v_th_perp**2)
    term1 = n / denominator

    gamma_ratio = gamma(kappa+1)/(gamma(kappa - 0.5)*((kappa - 3/2)**(3/2)))

    p, q = 10, 1
    delta = np.sqrt(1.08*((1/3)*v_th_par + (2/3)*v_th_perp))
    kappa_part = term1*gamma_ratio*(( 1 + ((v_par-u_par)**2)/((v_th_par**2)*(kappa - 3/2)) + (v_perp**2)/((v_th_perp**2)*(kappa - 3/2)))**(-kappa -1))
    flattop_part = (1 + ((1/delta)*(((v_par-u_par)**2)/(v_th_par**2) + (v_perp**2)/((v_th_perp**2)))**p))**(-q)

    return kappa_part*(1 - flattop_part)

def kappa_log_vdf(v_par, v_perp, n, u_par, v_th_par, v_th_perp, kappa) :

    """
    Produces a VDF following a bi-Maxwellian distribution. All in CGS units.

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

    return np.log(term1*gamma_ratio*(( 1 + ((v_par-u_par)**2)/((v_th_par**2)*(kappa - 3/2)) + (v_perp**2)/((v_th_perp**2)*(kappa - 3/2)))**(- kappa -1)))

def core_beam_halo(v_par, v_perp, n_c, u_par_c, v_th_par_c, v_th_perp_c, n_b, u_par_b, v_th_par_b, v_th_perp_b, n_h, u_par_h, v_th_par_h, v_th_perp_h, kappa_h, u_p, n_sc) :

    ''' core : see bi-Max for parameters'''
    c_par = v_par - u_par_c
    c_perp = v_perp
    exponent_c = (c_par/v_th_par_c)**2 + (c_perp/v_th_perp_c)**2

    denominator = (np.pi ** (3/2)) * v_th_par_c * (v_th_perp_c**2)
    coeff_c = n_c / denominator

    ''' beam : see bi-Max for parameters'''
    c_par = v_par - u_par_b
    c_perp = v_perp
    exponent_b = (c_par/v_th_par_b)**2 + (c_perp/v_th_perp_b)**2

    denominator = (np.pi ** (3/2)) * v_th_par_b * (v_th_perp_b**2)
    coeff_b = n_b / denominator

    ''' halo '''
    denominator = (np.pi ** (3/2)) * v_th_par_h * (v_th_perp_h**2)
    term1_h = n_h / denominator

    gamma_ratio = gamma(kappa+1)/(gamma(kappa - 0.5)*((kappa - 3/2)**(3/2)))

    term_halo = term1_h*gamma_ratio*(( 1 + ((v_par-u_par_h)**2)/((v_th_par_h**2)*(kappa - 3/2)) + (v_perp**2)/((v_th_perp_h**2)*(kappa - 3/2)))**(-kappa -1))

    f = coeff_c * np.exp(- exponent_c) + coeff_b * np.exp(- exponent_b) + term_halo

    return f

def core_kappa_beam_halo(v_par, v_perp, n_c, u_par_c, v_th_par_c, v_th_perp_c, n_b, u_par_b, v_th_par_b, v_th_perp_b, kappa_b, n_h, u_par_h, v_th_par_h, v_th_perp_h, kappa_h, n_sc) :
    ''' core : see bi-Max for parameters'''
    c_par = v_par - u_par_c
    c_perp = v_perp
    exponent_c = (c_par/v_th_par_c)**2 + (c_perp/v_th_perp_c)**2

    denominator = (np.pi ** (3/2)) * v_th_par_c * (v_th_perp_c**2)
    coeff_c = n_c / denominator


    ''' beam : see bi-kappa for parameters'''
    D = D_CONST
    denominator = (np.pi ** (3/2)) * v_th_par_b * (v_th_perp_b**2)
    D_coeff = (2*np.sqrt(D))/np.sqrt(D+1)
    coeff_b = (n_b*D_coeff) / denominator
    gamma_ratio = gamma(kappa_b+1)/(gamma(kappa_b - 0.5)*((kappa_b - 3/2)**(3/2)))

    term_beam = coeff_b*gamma_ratio*(( 1 + D*((v_par-u_par_b)**2)/((v_th_par_b**2)*(kappa_b - 3/2)) + (v_perp**2)/((v_th_perp_b**2)*(kappa_b - 3/2)))**(-kappa_b -1))

    ''' halo '''
    denominator = (np.pi ** (3/2)) * v_th_par_h * (v_th_perp_h**2)
    term1_h = n_h / denominator

    gamma_ratio = gamma(kappa_h+1)/(gamma(kappa_h - 0.5)*((kappa_h - 3/2)**(3/2)))

    term_halo = term1_h*gamma_ratio*(( 1 + ((v_par-u_par_h)**2)/((v_th_par_h**2)*(kappa_h - 3/2)) + (v_perp**2)/((v_th_perp_h**2)*(kappa_h - 3/2)))**(-kappa_h -1))

    f = coeff_c * np.exp(- exponent_c) + term_beam + term_halo

    return f

def core_beam_beam_halo(v_par, v_perp, n_c, u_par_c, v_th_par_c, v_th_perp_c, n_b_par, u_par_b_par, v_th_par_b_par, v_th_perp_b_par, n_b_anti_par, u_par_b_anti_par, v_th_par_b_anti_par, v_th_perp_b_anti_par, n_h, u_par_h, v_th_par_h, v_th_perp_h, kappa, n_sc) :

    ''' core : see bi-Max for parameters'''
    c_par = v_par - u_par_c
    c_perp = v_perp
    exponent_c = (c_par/v_th_par_c)**2 + (c_perp/v_th_perp_c)**2

    denominator = (np.pi ** (3/2)) * v_th_par_c * (v_th_perp_c**2)
    coeff_c = n_c / denominator

    ''' par beam : see bi-Max for parameters'''
    c_par = v_par - u_par_b_par
    c_perp = v_perp
    exponent_b_par = (c_par/v_th_par_b_par)**2 + (c_perp/v_th_perp_b_par)**2

    denominator = (np.pi ** (3/2)) * v_th_par_b_par * (v_th_perp_b_par**2)
    coeff_b_par = n_b_par / denominator

    ''' anti_par beam : see bi-Max for parameters'''
    c_par = v_par - u_par_b_anti_par
    c_perp = v_perp
    exponent_b_anti_par = (c_par/v_th_par_b_anti_par)**2 + (c_perp/v_th_perp_b_anti_par)**2

    denominator = (np.pi ** (3/2)) * v_th_par_b_anti_par * (v_th_perp_b_anti_par**2)
    coeff_b_anti_par = n_b_anti_par / denominator

    ''' halo '''
    denominator = (np.pi ** (3/2)) * v_th_par_h * (v_th_perp_h**2)
    term1_h = n_h / denominator

    gamma_ratio = gamma(kappa+1)/(gamma(kappa - 0.5)*((kappa - 3/2)**(3/2)))

    term_halo = term1_h*gamma_ratio*(( 1 + ((v_par-u_par_h)**2)/((v_th_par_h**2)*(kappa - 3/2)) + (v_perp**2)/((v_th_perp_h**2)*(kappa - 3/2)))**(-kappa -1))

    f = coeff_c * np.exp(- exponent_c) + coeff_b_par * np.exp(- exponent_b_par) + coeff_b_anti_par * np.exp(- exponent_b_anti_par) + term_halo

    return f

def core_kappa_beam_beam_halo(v_par, v_perp, n_c, u_par_c, v_th_par_c, v_th_perp_c, n_b_par, u_par_b_par, v_th_par_b_par, v_th_perp_b_par, kappa_b_par, n_b_anti_par, u_par_b_anti_par, v_th_par_b_anti_par, v_th_perp_b_anti_par, kappa_b_anti_par, n_h, u_par_h, v_th_par_h, v_th_perp_h, kappa_h, n_sc) :

    ''' core : see bi-Max for parameters'''
    c_par = v_par - u_par_c
    c_perp = v_perp
    exponent_c = (c_par/v_th_par_c)**2 + (c_perp/v_th_perp_c)**2

    denominator = (np.pi ** (3/2)) * v_th_par_c * (v_th_perp_c**2)
    coeff_c = n_c / denominator

    ''' par beam : see bi-kappa for parameters'''
    D = D_CONST
    denominator = (np.pi ** (3/2)) * v_th_par_b_par * (v_th_perp_b_par**2)
    D_coeff = (2*np.sqrt(D))/np.sqrt(D+1)
    coeff_b = (n_b_par*D_coeff) / denominator
    gamma_ratio = gamma(kappa_b_par+1)/(gamma(kappa_b_par - 0.5)*((kappa_b_par - 3/2)**(3/2)))

    term_beam_par = coeff_b*gamma_ratio*(( 1 + D*((v_par-u_par_b_par)**2)/((v_th_par_b_par**2)*(kappa_b_par - 3/2)) + (v_perp**2)/((v_th_perp_b_par**2)*(kappa_b_par - 3/2)))**(-kappa_b_par -1))

    ''' anti_par beam : see bi-kappa for parameters'''
    D = D_CONST
    denominator = (np.pi ** (3/2)) * v_th_par_b_anti_par * (v_th_perp_b_anti_par**2)
    D_coeff = (2*np.sqrt(D))/np.sqrt(D+1)
    coeff_b = (n_b_anti_par*D_coeff) / denominator
    gamma_ratio = gamma(kappa_b_anti_par+1)/(gamma(kappa_b_anti_par - 0.5)*((kappa_b_anti_par - 3/2)**(3/2)))

    term_beam_anti_par = coeff_b*gamma_ratio*(( 1 + D*((v_par-u_par_b_anti_par)**2)/((v_th_par_b_anti_par**2)*(kappa_b_anti_par - 3/2)) + (v_perp**2)/((v_th_perp_b_anti_par**2)*(kappa_b_anti_par - 3/2)))**(-kappa_b_anti_par -1))

    ''' halo '''
    denominator = (np.pi ** (3/2)) * v_th_par_h * (v_th_perp_h**2)
    term1_h = n_h / denominator

    gamma_ratio = gamma(kappa_h+1)/(gamma(kappa_h - 0.5)*((kappa_h - 3/2)**(3/2)))

    term_halo = term1_h*gamma_ratio*(( 1 + ((v_par-u_par_h)**2)/((v_th_par_h**2)*(kappa_h - 3/2)) + (v_perp**2)/((v_th_perp_h**2)*(kappa_h - 3/2)))**(-kappa_h -1))

    f = coeff_c * np.exp(- exponent_c) + term_beam_par + term_beam_anti_par + term_halo

    return f

def bi_Max_beam_A(ux, uy, uz, A_beam, v_x_beam, v_y, v_z, v_th_par_beam, v_th_perp_beam):
    """
    Produces a VDF following a bi-Maxwellian distribution for the beam population.

    Parameters
    ----------
    ux : array
        Velocity in the x (parallel) direction.
    uy : array
        Velocity in the y direction.
    uz : array
        Velocity in the z direction.
    A_beam : float
        Amplitude.
    v_x_beam : array
        Bulk velocity in the x (parallel) direction.
    v_y : array
        Bulk velocity in the y direction.
    v_z : array
        Bulk velocity in the z direction.
    v_th_par_beam : float
        Thermal velocity in paraller (x) direction.
    v_th_perp_beam : float
        Thermal velocity in perpendicular direction.

    Returns
    -------
    f : array
        The Velocity distribution function VDF.

    """

    vel_par = ux - v_x_beam

    vy_perp = uy - v_y
    vz_perp = uz - v_z
    vel_perp = np.sqrt((vy_perp * vy_perp) + (vz_perp * vz_perp))

    exponent = ((vel_par * vel_par) / (v_th_par_beam * v_th_par_beam)) + \
        ((vel_perp * vel_perp) / (v_th_perp_beam * v_th_perp_beam))

    f = A_beam * np.exp(- exponent)

    if np.isfinite(f).all() == False:
       return 0

    return f

def bi_Max_combined_A(ux, uy, uz, A, v_x, v_y, v_z, v_th_par, v_th_perp, A_beam, v_x_beam, v_th_par_beam, v_th_perp_beam):
    """
    Produces a VDF following the sum of two bi-Maxwellian distribution.

    Parameters
    ----------
    ux : array
        Velocity in the x (parallel) direction.
    uy : array
        Velocity in the y direction.
    uz : array
        Velocity in the z direction.
    A : float
        Amplitude core.
    v_x : array
        Bulk velocity in the x (parallel) direction.
    v_y : array
        Bulk velocity in the y direction.
    v_z : array
        Bulk velocity in the z direction.
    v_th_par : float
        Thermal velocity in paraller (x) direction.
    v_th_perp : float
        Thermal velocity in perpendicular direction.
    A_beam : float
        Amplitude beam.
    v_x_beam : array
        Bulk velocity in the x (parallel) direction.
    v_th_par_beam : float
        Thermal velocity in paraller (x) direction.
    v_th_perp_beam : float
        Thermal velocity in perpendicular direction.

    Returns
    -------
    f : array
        The Velocity distribution function VDF.

    """
    #CORE
    vel_par = ux - v_x

    vy_perp = uy - v_y
    vz_perp = uz - v_z
    vel_perp = np.sqrt((vy_perp * vy_perp) + (vz_perp * vz_perp))

    exponent_core = ((vel_par * vel_par) / (v_th_par * v_th_par)) + \
        ((vel_perp * vel_perp) / (v_th_perp * v_th_perp))

    f_c = A * np.exp(- exponent_core)

    #BEAM
    vel_par_b = ux - v_x_beam

    vy_perp_b = uy - v_y
    vz_perp_b = uz - v_z
    vel_perp_b = np.sqrt((vy_perp_b * vy_perp_b) + (vz_perp_b * vz_perp_b))

    exponent_beam = ((vel_par_b * vel_par_b) / (v_th_par_beam * v_th_par_beam)) + \
        ((vel_perp_b * vel_perp_b) / (v_th_perp_beam * v_th_perp_beam))

    f_b = A_beam * np.exp(- exponent_beam)

    # print(f_c)

    f = f_c + f_b

    if np.isfinite(f).all() == False:
        return 0

    return f

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

def pitch_angle_width_model(alpha, P, W) :
    """
    Produces a VDF following a bi-Maxwellian distribution. All in CGS units.

    Parameters
    ----------
    alpha : array
        pitch-angles
    P : float
        Gaussian height at alpha = 0
    W : float
        Full-width-half-maximum at alpha = 0

    Returns
    -------
    f : array
        The pitch angle DF
    """

    return  P * np.exp( -(1/2)*((alpha/(np.sqrt(2)*W))**2) )

# My functions
def core_halo(v_par, v_perp, n_c, u_par_c, v_th_par_c, v_th_perp_c, n_h, u_par_h, v_th_par_h, v_th_perp_h, kappa):
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
    gamma_ratio = gamma(kappa+1)/(gamma(kappa - 0.5)*((kappa - 3/2)**(3/2)))
    
    p, q = 10, 1
    delta = np.sqrt(1.08*((1/3)*v_th_par_h + (2/3)*v_th_perp_h))
    kappa_part = term1_h*gamma_ratio*(( 1 + ((v_par-u_par_h)**2)/((v_th_par_h**2)*(kappa - 3/2)) + (v_perp**2)/((v_th_perp_h**2)*(kappa - 3/2)))**(-kappa -1))
    flattop_part = (1 + ((1/delta)*(((v_par-u_par_h)**2)/(v_th_par_h**2) + (v_perp**2)/((v_th_perp_h**2)))**p))**(-q)
    term_halo = kappa_part*(1 - flattop_part)
    # term_halo = term1_h*gamma_ratio*(( 1 + ((v_par-u_par_h)**2)/((v_th_par_h**2)*(kappa - 3/2)) + (v_perp**2)/((v_th_perp_h**2)*(kappa - 3/2)))**(-kappa -1))
    
    f_core_halo = term_core + term_halo

    return f_core_halo

def beam_truncated(v_par, v_perp, n, u_par, v_th_par, v_th_perp, kappa) :

    """
    Truncated bi-kappa distribution. All in CGS units. When energy --> 0 eV.
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

    """
    D = D_CONST

    denominator = (np.pi ** (3/2)) * v_th_par * (v_th_perp**2)
    D_coeff = (2*np.sqrt(D))/np.sqrt(D+1)
    term1 = (D_coeff*n) / denominator

    gamma_ratio = gamma(kappa+1)/(gamma(kappa - 0.5)*((kappa - 3/2)**(3/2)))

    return term1*gamma_ratio*(( 1 + D*((v_par-u_par)**2)/((v_th_par**2)*(kappa - 3/2)) + (v_perp**2)/((v_th_perp**2)*(kappa - 3/2)))**(-kappa -1))


def final_core_1beam_halo(v_par, v_perp, n_c, u_par_c, v_th_par_c, v_th_perp_c, n_b, u_par_b, v_th_par_b, v_th_perp_b, kappa_b, n_h, u_par_h, v_th_par_h, v_th_perp_h, kappa, n_sc) :
    ''' core : see bi-Max for parameters'''
    c_par = v_par - u_par_c
    c_perp = v_perp
    exponent_c = (c_par/v_th_par_c)**2 + (c_perp/v_th_perp_c)**2

    denominator = (np.pi ** (3/2)) * v_th_par_c * (v_th_perp_c**2)
    coeff_c = n_c / denominator

    ''' halo '''
    denominator = (np.pi ** (3/2)) * v_th_par_h * (v_th_perp_h**2)
    term1_h = n_h / denominator

    gamma_ratio = gamma(kappa+1)/(gamma(kappa - 0.5)*((kappa - 3/2)**(3/2)))

    term_halo = term1_h*gamma_ratio*(( 1 + ((v_par-u_par_h)**2)/((v_th_par_h**2)*(kappa - 3/2)) + (v_perp**2)/((v_th_perp_h**2)*(kappa - 3/2)))**(-kappa -1))

    '''Beam'''
    vdf_1 = beam_nonTruncated(v_par, v_perp, n_b, u_par_b, v_th_par_b, v_th_perp_b, kappa_b)
    vdf_2 = beam_truncated(v_par, v_perp, n_b, u_par_b, v_th_par_b, v_th_perp_b, kappa_b)

    if u_par_b >=0:
        conditional_beam_term = np.where(v_par >= u_par_b, vdf_1, vdf_2)
    else:
        conditional_beam_term = np.where(v_par <= u_par_b, vdf_1, vdf_2)

    f = coeff_c * np.exp(- exponent_c) + conditional_beam_term + term_halo

    return f


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
    D_coeff = (2*np.sqrt(D))/np.sqrt(D+1)
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

# Define the combined model
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

    # Sum all components
    total_result = vdf_beam + vdf_corehalo

    return total_result

def combined_model_2beam(v_par, v_perp, n_c, u_par_c, v_th_par_c, v_th_perp_c, 
                   n_h, u_par_h, v_th_par_h, v_th_perp_h, kappa_h,
                   n_b_par, u_par_b_par, v_th_par_b_par, v_th_perp_b_par, kappa_b_par,
                   n_b_anti_par, u_par_b_anti_par, v_th_par_b_anti_par, v_th_perp_b_anti_par, kappa_b_anti_par):
    # Conditional component. Beam_par
    beam_1_par = beam_nonTruncated(v_par, v_perp, n_b_par, u_par_b_par, v_th_par_b_par, v_th_perp_b_par, kappa_b_par)
    beam_2_par = beam_truncated(v_par, v_perp, n_b_par, u_par_b_par, v_th_par_b_par, v_th_perp_b_par, kappa_b_par)
    if u_par_b_par < 0 :
        vdf_beam_par = np.where(v_par < u_par_b_par, beam_1_par, beam_2_par)
    else :
        vdf_beam_par = np.where(v_par >= u_par_b_par, beam_1_par, beam_2_par)
    
    # Beam_anti_par
    beam_1_anti_par = beam_nonTruncated(v_par, v_perp, n_b_anti_par, u_par_b_anti_par, v_th_par_b_anti_par, v_th_perp_b_anti_par, kappa_b_anti_par)
    beam_2_anti_par = beam_truncated(v_par, v_perp, n_b_anti_par, u_par_b_anti_par, v_th_par_b_anti_par, v_th_perp_b_anti_par, kappa_b_anti_par)
    if u_par_b_anti_par < 0 :
        vdf_beam_anti_par = np.where(v_par < u_par_b_anti_par, beam_1_anti_par, beam_2_anti_par)
    else :
        vdf_beam_anti_par = np.where(v_par >= u_par_b_anti_par, beam_1_anti_par, beam_2_anti_par)
    
    # ## old beam model
    # vdf_beam = beam_truncated(v_par, v_perp, n_b, u_par_b, v_th_par_b, v_th_perp_b, kappa_b)
    ## core + halo
    vdf_corehalo = core_halo(v_par, v_perp, n_c, u_par_c, v_th_par_c, v_th_perp_c, n_h, u_par_h, v_th_par_h, v_th_perp_h, kappa_h)

    # Sum all components
    total_result = vdf_beam_par + vdf_beam_anti_par + vdf_corehalo

    return total_result