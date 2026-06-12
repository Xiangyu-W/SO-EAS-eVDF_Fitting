import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend; figures are rendered inside parallel workers
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter, FormatStrFormatter
import matplotlib.lines as mlines
import lmfit
from num2tex import num2tex
from utils import vel_to_eV, eV_to_vel, round_to_n, determineSWA_energyIdx, eV_to_Kelvin
from Beam_fitting import beam
from Core_fitting import core
from Halo_fitting import halo_flattop, halo
from constant import PAW_coeff, s_set, lw_set, low_sc_val, high_sc_val, low_pad_val, high_pad_val, PixelCountThreshold_halo

####################
# Helper functions #
####################
'''Plot fitting results'''
def plot_2D_VDF(epoch, pa_vec, low_c, v_par_mesh, v_perp_mesh, v_par_arr, v_perp_arr, fit_vdf, r_vdf, redChiSqr, redChiSqr_beam, redChiSqr_halo, beam_data, fit_beam, T_para_b, T_antiPara_b, anti_par_strahl_cond, par_strahl_cond,
                plot_path, figname, savePlot=False):
    
    # Mask data
    countMask_all = pa_vec[:low_c, :, 4] >=2 #(pa_vec[:low_c, :, 1] >= 1e-27)&(pa_vec[:low_c, :, 4] >=2) #pa_vec[:low_c,:,4]>=0
    psd_masked = np.where(countMask_all, pa_vec[:low_c,:,1],np.nan)
    # v_par_masked = np.where(countMask_all, pa_vec[:low_c,:,2],np.nan)
    # v_perp_masked = np.where(countMask_all, pa_vec[:low_c,:,3],np.nan)

    psdmin = np.nanmin(psd_masked[psd_masked > 0])
    psdmax = np.nanmax(psd_masked[psd_masked > 0])


    # Extract final fitted parameters for core, halo, and beam
    _, _, final_fit_beam = get_final_fits(r_vdf, v_par_mesh, v_perp_mesh, anti_par_strahl_cond, par_strahl_cond)
    
    # Beam contour levels; a fitted beam with vanishing density can fall
    # entirely below the plotting floor, in which case let matplotlib
    # auto-pick levels (same fallback as the no-beam case)
    BEAM_LEVEL_FLOOR = 5e-29
    beamAboveFloor = final_fit_beam[final_fit_beam > BEAM_LEVEL_FLOOR]
    if beamAboveFloor.size == 0:
        beam_levels = None
    else:
        beam_levels = np.logspace(np.log10(beamAboveFloor.min()),
                                  np.log10(beamAboveFloor.max()*0.9), num=4)


    # 2D plot
    fig, ax = plt.subplots(1,2, figsize=(12.5,5.6))
    plt.subplots_adjust(bottom=0.15, left=0.09,right=0.96,top=0.93, wspace=0.2)

    # ## original data
    plot_2D_beamVDF(pa_vec, beam_data, low_c, fit_beam, v_par_mesh, v_perp_mesh, plot_path, ax=ax[0], figname=None, savePlot=False)

    # Define logarithmically spaced levels
    levels = np.logspace(np.log10(psdmin), np.log10(psdmax), num=30)

    p_fitPsd = ax[1].contourf(v_par_mesh, v_perp_mesh, fit_vdf, levels=levels, cmap='viridis', norm=LogNorm(vmin=psdmin, vmax=psdmax))
    ax[1].contourf(v_par_mesh, -np.flipud(v_perp_mesh), np.flipud(fit_vdf), levels=levels, cmap='viridis', norm=LogNorm(vmin=psdmin, vmax=psdmax))
    
    cbar = fig.colorbar(p_fitPsd, ax=ax[1])
    # Get current colorbar ticks and divide by 1e-27
    ticks = cbar.get_ticks()  # Get tick positions (original values)
    scaled_ticks = ticks / 1e-27  # Normalize ticks to e-27
    def format_tick(t):
        if t < 1:  # For small values, show 3 decimal places
            return f'{t:.1g}'
        else:  # For larger values, show 1 decimal place
            return f'{t:.1f}'
    cbar.set_ticks(ticks)  # Reapply the original tick positions
    cbar.set_ticklabels([format_tick(t) for t in scaled_ticks])  # Format normalized ticks

    cbar.set_label('Phase space density [s$^3$/cm$^{6}$]', fontsize=13)
    # Move the exponent (e-27) to the top of the colorbar
    cbar.ax.annotate('$10^{-27}$', xy=(1.1, 1.01), xycoords='axes fraction',ha='center', va='bottom', fontsize=11)


    beam_contour = ax[1].contour(v_par_mesh, v_perp_mesh, final_fit_beam, levels=beam_levels, colors='red', linestyles='dashed', linewidths=1.4, alpha=0.5)
    ax[1].contour(v_par_mesh, -np.flipud(v_perp_mesh), np.flipud(final_fit_beam), levels=beam_levels, colors='red', linestyles='dashed', linewidths=1.4, alpha=0.5)
    ax[1].clabel(beam_contour, inline=True, fontsize=10, fmt='%.0e')
    # Add a legend for the beam component
    beam_line = plt.Line2D([0], [0], color='red', linestyle='dashed', linewidth=1.2, alpha=0.5, label='Fitted Strahl VDF')
    ax[1].legend(handles=[beam_line], loc='lower right',fontsize=10.8)
    
    ax[1].set_xlabel(r'$v_{par}$ (cm/s)',fontsize=15)
    ax[1].set_ylabel(r'$v_{perp}$ (cm/s)',fontsize=15)
    ax[1].set_title('Overall VDF fitting',fontsize=15)
    ax[1].set_xlim(-2.1e9, 2.1e9)
    ax[1].set_ylim(-2.1e9, 2.1e9)
    
    textstr = temperatureOfComponents_str(r_vdf, T_para_b, T_antiPara_b, anti_par_strahl_cond, par_strahl_cond)

    ax[1].text(0.05,0.98,textstr, fontsize=9.5, transform=ax[1].transAxes, color='#00215E', va='top', ha='left', bbox=dict(facecolor='white', alpha=0.5, edgecolor='grey',linewidth=1)) ##FF5959
    timeStamp = pd.Timestamp(epoch)
    fig.text(0.52, 0.02, f"{timeStamp.strftime('%Y-%m-%d %H:%M:%S')}", ha='center', va='center', fontsize=15)

    # Text, red_chisqr of overall & beam fitting
    chisqrStr_all = f'Reduced $\chi^2={np.real(redChiSqr):.2f}$ (Overall)' #f'Reduced $\chi^2={redChiSqr:.2f}\pm ...$ (Entire VDF)'
    if redChiSqr_beam is not None:
        chisqrStr_beam = f'Reduced $\chi^2={redChiSqr_beam:.2f}$ (Strahl)'
    else:
        chisqrStr_beam = 'Reduced $\chi^2=None$ (Strahl)'
    if redChiSqr_halo is not None:
        chisqrStr_halo = f'Reduced $\chi^2={redChiSqr_halo:.2f}$ (Halo)'
    else:
        chisqrStr_halo = 'Reduced $\chi^2=None$ (Halo)'
    ax[1].text(0.02,0.2, chisqrStr_all+'\n'+chisqrStr_beam+'\n'+chisqrStr_halo, fontsize=9.5, transform=ax[1].transAxes, color='#118B50', va='top', ha='left') 

    if savePlot == True:
        fig.savefig(plot_path+'FittedVDF_2D_ID'+str(figname)+f"_({timeStamp.strftime('%Y-%m-%d %H%M%S')})"+'.png', dpi=100)
    
    plt.close(plt.gcf())
    fig.clf()

def temperatureOfComponents_str(r_vdf, T_para_b, T_antiPara_b, anti_par_strahl_cond, par_strahl_cond):
    '''Return Values of fitted parameters & moments'''
    # Core temperature
    Tc_para_fittedParam = np.round(vel_to_eV(r_vdf.params['v_th_par_c'].value), 2)
    Tc_para_mK = eV_to_Kelvin(Tc_para_fittedParam)/1e6

    # Halo temperature
    Th_para_fittedParam = np.round(vel_to_eV(r_vdf.params['v_th_par_h'].value), 2)
    Th_para_mK = eV_to_Kelvin(Th_para_fittedParam)/1e6

    if anti_par_strahl_cond and par_strahl_cond:  # Two-direction beam
        Ts_antiPara_params = np.round(vel_to_eV(r_vdf.params['v_th_par_b_anti_par'].value), 2)
        Ts_antiPara_mK = eV_to_Kelvin(Ts_antiPara_params)/1e6
        Ts_para_params = np.round(vel_to_eV(r_vdf.params['v_th_par_b_par'].value), 2)
        Ts_para_mK = eV_to_Kelvin(Ts_para_params)/1e6
        textstr = f'Fitted Params\n'+\
            r"$T_{core\parallel}$: "+f'{Tc_para_fittedParam} eV ({Tc_para_mK:.2f} e6 K)\n' + \
            r"$T_{halo\parallel}$: "+f'{Th_para_fittedParam} eV ({Th_para_mK:.2f} e6 K)\n' + \
            r"$T_{s,\parallel}$: "+f'{Ts_para_params} eV ({Ts_para_mK:.2f} e6 K)\n' +\
            r"$T_{s,anti\parallel}$: "+f'{Ts_antiPara_params} eV ({Ts_antiPara_mK:.2f} e6 K)\n' +\
            f'2nd order moment\n' + r"$T_{s,\parallel}$: " + f'{np.round(T_para_b,2)} eV\n' +\
            r"$T_{s,anti\parallel}$: " + f'{np.round(T_antiPara_b,2)} eV'
    elif anti_par_strahl_cond and not par_strahl_cond:
        Ts_antiPara_params = np.round(vel_to_eV(r_vdf.params['v_th_par_b'].value), 2)
        Ts_antiPara_mK = eV_to_Kelvin(Ts_antiPara_params)/1e6
        textstr = f'Fitted Params\n'+\
            r"$T_{core, anti\parallel}$: "+f'{Tc_para_fittedParam} eV ({Tc_para_mK:.2f} e6 K)\n' + \
            r"$T_{halo, anti\parallel}$: "+f'{Th_para_fittedParam} eV ({Th_para_mK:.2f} e6 K)\n' + \
            r"$T_{s, anti\parallel}$: "+f'{Ts_antiPara_params} eV ({Ts_antiPara_mK:.2f} e6 K)\n' +\
            f'2nd order moment\n' + r"$T_{s,anti\parallel}$: " + f'{np.round(T_antiPara_b,2)} eV'
    elif not anti_par_strahl_cond and par_strahl_cond:
        Ts_para_params = np.round(vel_to_eV(r_vdf.params['v_th_par_b'].value), 2)
        Ts_para_mK = eV_to_Kelvin(Ts_para_params)/1e6
        textstr = f'Fitted Params\n'+\
            r"$T_{core\parallel}$: "+f'{Tc_para_fittedParam} eV ({Tc_para_mK:.2f} e6 K)\n' + \
            r"$T_{halo\parallel}$: "+f'{Th_para_fittedParam} eV ({Th_para_mK:.2f} e6 K)\n' + \
            r"$T_{s\parallel}$: "+f'{Ts_para_params} eV ({Ts_para_mK:.2f} e6 K)\n' +\
            f'2nd order moment\n' + r"$T_{s,\parallel}$: " + f'{np.round(T_para_b,2)} eV'
    else:
        textstr=''
    return textstr

def get_final_fits(r_vdf, v_par_mesh, v_perp_mesh, anti_par_strahl_cond, par_strahl_cond):
    """
    Extract final fitted parameters for core, halo, and beam from r_vdf and evaluate them.
    """
    # Core parameters
    core_final = lmfit.Parameters()
    for key in ['n_c','u_par_c','v_th_par_c','v_th_perp_c']:
        core_final.add(key.replace('_c',''), value=r_vdf.params[key].value)
    final_fit_core = core.eval(core_final, v_par=v_par_mesh, v_perp=v_perp_mesh)

    # Beam parameters
    beam_final = lmfit.Parameters()
    if anti_par_strahl_cond and par_strahl_cond:  # Two-direction beam
        for key in ['n_b_par','u_par_b_par','v_th_par_b_par','v_th_perp_b_par','kappa_b_par']:
            beam_final.add(key.replace('_b_par',''), value=r_vdf.params[key].value)
        final_fit_beam_par = beam.eval(beam_final, v_par=v_par_mesh, v_perp=v_perp_mesh)

        beam_final = lmfit.Parameters()
        for key in ['n_b_anti_par','u_par_b_anti_par','v_th_par_b_anti_par','v_th_perp_b_anti_par','kappa_b_anti_par']:
            beam_final.add(key.replace('_b_anti_par',''), value=r_vdf.params[key].value)
        final_fit_beam_anti_par = beam.eval(beam_final, v_par=v_par_mesh, v_perp=v_perp_mesh)

        final_fit_beam = final_fit_beam_par + final_fit_beam_anti_par
    
    elif par_strahl_cond or anti_par_strahl_cond:  # One-direction beam
        for key in ['n_b','u_par_b','v_th_par_b','v_th_perp_b','kappa_b']:
            beam_final.add(key.replace('_b',''), value=r_vdf.params[key].value)
        final_fit_beam = beam.eval(beam_final, v_par=v_par_mesh, v_perp=v_perp_mesh)

    else: # No beam
        final_fit_beam = np.zeros_like(v_par_mesh)  # No beam

    
    # Halo parameters
    core_final.params = core_final
    halo_final = lmfit.Parameters()
    for key in ['n_h','u_par_h','v_th_par_h','v_th_perp_h','kappa_h']: #,'delta' # stepan's model
        halo_final.add(key.replace('_h',''), value=r_vdf.params[key].value)
    # final_fit_halo = halo.eval(halo_final, v_par=v_par_mesh, v_perp=v_perp_mesh)
    final_fit_halo = halo_flattop.eval(halo_final, v_par=v_par_mesh, v_perp=v_perp_mesh) # plotting fit example
    
    # # stepan's model
    # halo_final.add('u_par_c', value=core_final.params['u_par'].value)
    # halo_final.add('v_th_par_c', value=core_final.params['v_th_par'].value)
    # halo_final.add('v_th_perp_c', value=core_final.params['v_th_perp'].value)
    # final_fit_halo = halo.eval(halo_final, v_par=v_par_mesh, v_perp=v_perp_mesh) 

    return final_fit_core, final_fit_halo, final_fit_beam

def plot_2D_beamVDF(pa_vec, beam_data, low_c, fit_beam_dict, v_par_mesh, v_perp_mesh, plot_path, ax= None, figname=None, savePlot=False):
    # Mask data
    mask_forRawData = pa_vec[:low_c, :, 4] >=2 #(pa_vec[:low_c, :, 1] >= 1e-27)&(pa_vec[:low_c, :, 4] >=2) #pa_vec[:low_c,:,4]>=0
    psd_masked = np.where(mask_forRawData, pa_vec[:low_c,:,1],np.nan)
    psdmin = np.nanmin(psd_masked[psd_masked > 0])
    psdmax = np.nanmax(psd_masked[psd_masked > 0])
    levels = np.logspace(np.log10(psdmin), np.log10(psdmax), num=30)

    v_par_masked = np.where(mask_forRawData, pa_vec[:low_c,:,2],np.nan)
    v_perp_masked = np.where(mask_forRawData, pa_vec[:low_c,:,3],np.nan)

    
    # 2D plot
    # 1) If no Axes is passed, create a new figure/axes
    new_fig_created = False
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(10, 7))
        plt.subplots_adjust(bottom=0.15, left=0.1,right=0.95,top=0.95)
        new_fig_created = True

    ## original data
    # All data points
    v_par_atValidPSD = v_par_masked#[psd_masked >= psdmin]
    v_perp_atValidPSD = v_perp_masked#[psd_masked >= psdmin]
    psd_valid = psd_masked#[psd_masked >= psdmin]
    ax.scatter(v_par_atValidPSD, v_perp_atValidPSD, c=psd_valid, cmap='viridis', s=15, alpha=1, norm=LogNorm(vmin=levels[0], vmax=levels[-1]))
    # ax.scatter(v_par_atValidPSD, -v_perp_atValidPSD, c=psd_valid, cmap='viridis', s=15, alpha=1, norm=LogNorm(vmin=levels[0], vmax=levels[-1]))
    
    # ax.scatter(v_par_atValidPSD, v_perp_atValidPSD, edgecolors='C1', facecolors='none', linewidths=1.5, s=15, alpha=0.5, label='Raw Data Points')
    ax.scatter(v_par_atValidPSD, -v_perp_atValidPSD, edgecolors='C1', facecolors='none', linewidths=1.5, s=15, alpha=0.5, label='Raw Data Points')
    # beam_data points
    mask_beam_par = beam_data['mask_par'] # mask for parallel strahl
    mask_beam_AntiPar = beam_data['mask_AntiPar'] # mask for anti-parallel strahl
    if beam_data['mask_par'] is not None:
        beamPSD_par = beam_data['temp_b_psd'][mask_beam_par]
        ax.scatter(beam_data['temp_b_v_par'][mask_beam_par], beam_data['temp_b_v_perp'][mask_beam_par], c=beamPSD_par, cmap='viridis', s=6, alpha=1, norm=LogNorm(vmin=levels[0], vmax=levels[-1]))
        ax.scatter(beam_data['temp_b_v_par'][mask_beam_par], -beam_data['temp_b_v_perp'][mask_beam_par], color='C3', s=6, alpha=0.6, label='Para Beam Data')
    if beam_data['mask_AntiPar'] is not None:
        beamPSD_AntiPar = beam_data['temp_b_psd'][mask_beam_AntiPar]
        ax.scatter(beam_data['temp_b_v_par'][mask_beam_AntiPar], beam_data['temp_b_v_perp'][mask_beam_AntiPar], c=beamPSD_AntiPar, cmap='viridis', s=6, alpha=1, norm=LogNorm(vmin=levels[0], vmax=levels[-1]))
        ax.scatter(beam_data['temp_b_v_par'][mask_beam_AntiPar], -beam_data['temp_b_v_perp'][mask_beam_AntiPar], color='C2', s=6, alpha=0.6, label='Anti-Para Beam Data')

    # fig.colorbar(p_psd, format='%.1e')
    ax.set_xlabel(r'$v_{par}$ (cm/s)',fontsize=15)
    ax.set_ylabel(r'$v_{perp}$ (cm/s)',fontsize=15)
    ax.set_title('Initial Strahl VDF fitting',fontsize=15)
    ax.set_xlim(-1.5e9, 1.5e9)
    ax.set_ylim(-1.5e9, 1.5e9)

    # Grab existing legend entries
    handles, labels = ax.get_legend_handles_labels()

    # fitted data
    if beam_data['mask_par'] is not None:
        fit_beam_par = fit_beam_dict['fit_b_par']
        beam_levels_par = np.logspace(np.log10(np.nanmin(fit_beam_par[fit_beam_par > 5e-29])), 
                              np.log10(np.nanmax(fit_beam_par[fit_beam_par > 5e-29])*0.9), num=4)
        
        beam_contour = ax.contour(v_par_mesh, v_perp_mesh, fit_beam_par, levels=beam_levels_par, colors='#7E1891', linestyles='dashed', linewidths=2, alpha=0.5)
        ax.contour(v_par_mesh, -np.flipud(v_perp_mesh), np.flipud(fit_beam_par), levels=beam_levels_par, colors='#7E1891', linestyles='dashed', linewidths=2, alpha=0.5)
        ax.clabel(beam_contour, inline=True, fontsize=10, fmt='%.0e')
        
        beam_proxy1 = mlines.Line2D([], [], color='#7E1891', linestyle='dashed', linewidth=2, alpha=0.5)
        handles.append(beam_proxy1)
        labels.append("Parallel Strahl") # 4) Append that proxy handle and its label
        
    if beam_data['mask_AntiPar'] is not None:
        fit_beam_AntiPar = fit_beam_dict['fit_b_AntiPar']
        beam_levels_AntiPar = np.logspace(np.log10(np.nanmin(fit_beam_AntiPar[fit_beam_AntiPar > 5e-29])), 
                              np.log10(np.nanmax(fit_beam_AntiPar[fit_beam_AntiPar > 5e-29])*0.9), num=4)
        
        beam_contour = ax.contour(v_par_mesh, v_perp_mesh, fit_beam_AntiPar, levels=beam_levels_AntiPar, colors='k', linestyles='dashed', linewidths=2, alpha=0.5)
        ax.contour(v_par_mesh, -np.flipud(v_perp_mesh), np.flipud(fit_beam_AntiPar), levels=beam_levels_AntiPar, colors='k', linestyles='dashed', linewidths=2, alpha=0.5)
        ax.clabel(beam_contour, inline=True, fontsize=10, fmt='%.0e')
        
        beam_proxy2 = mlines.Line2D([], [], color='k', linestyle='dashed', linewidth=2, alpha=0.5)
        handles.append(beam_proxy2)
        labels.append("Anti-parallel Strahl") # 4) Append that proxy handle and its label
    ax.legend(handles, labels, fontsize=10)

    # Text, red_chisqr of initial beam fitting
    if fit_beam_dict['red_chisqr_b_initial'] is not None:
        chisqrStr = f'Reduced $\chi^2={fit_beam_dict["red_chisqr_b_initial"]:.2f}$ (Strahl)' #f'Reduced $\chi^2={fit_beam_dict["red_chisqr_b_initial"]:.2f}\pm ...$'
    else:
        chisqrStr = 'Reduced $\chi^2=None$ (Strahl)'
    ax.text(0.02,0.08, chisqrStr, fontsize=12, transform=ax.transAxes, color='#118B50', va='top', ha='left') ##FF5959

    if new_fig_created:
        if savePlot == True:
            fig.savefig(plot_path+'FittedResult_2D_Beam_'+str(figname)+'.png', dpi=100)
        plt.close(fig)

## get the grid index of pitch angle of EAS data
def allocateGridIdx(data, gridBin, rightClosed = False):
    return np.digitize(data, gridBin, right=rightClosed) - 1 # np.digitize start from 1. -1 to keep consistency with array index

def flatten_pa_vec(pa_vec, pitch_angles, unc_pa_vec):
    # Store pa_vec in dataframe
    pitch_angle_centers = np.diff(pitch_angles)/2 + pitch_angles[:-1]
    pa_vec_pitchAngIdx = allocateGridIdx(pa_vec[:,:,0], pitch_angles, rightClosed=True)
    
    energy_list = []; pitch_bin_list = []; data_list = []
    for e in range(pa_vec.shape[0]):
        data_pitchAngIdx_atE = pa_vec_pitchAngIdx[e, :]
        
        for PA_binIdx in range(len(pitch_angle_centers)):
            data_slice = np.where(data_pitchAngIdx_atE == PA_binIdx)[0]
            if data_slice.size > 0:
                for dataIdx in data_slice:
                    data_list.append( np.append(pa_vec[e, dataIdx, :], unc_pa_vec[e, dataIdx]) )
                    energy_list.append(e)
                    pitch_bin_list.append(PA_binIdx)
            
    pa_vec_df = pd.DataFrame(data_list, columns=['pitch_angle', 'PSD', 'v_par', 'v_perp', 'count','unc_psd'])
    pa_vec_df['energy_channel'] = energy_list
    pa_vec_df['PA_bin'] = pitch_bin_list
    return pa_vec_df

def plot_vdf_1D_final(variables, indices): 
    
    # Extract variables
    timeStamp = pd.Timestamp(variables['epoch'])
    pa_vec, unc_pa_vec = variables['pa_vec'], variables['unc_pa_vec']
    pad, swa_energy, pitch_angles = variables['pad'], variables['swa_energy'], variables['pitch_angles']
    mean_energy_pad, std_energy_pad, mean_pad_params, mean_fit_pad = variables['mean_energy_pad'], variables['std_energy_pad'], variables['mean_pad_params'], variables['mean_fit_pad']
    pad_params_energy, anti_par_strahl_cond, par_strahl_cond = variables['pad_params_energy'], variables['anti_par_strahl_cond'], variables['par_strahl_cond']
    Bx_direc, r_vdf, fit_vdf, v_par_mesh, v_perp_mesh, v_par_arr = variables['Bx_direc'], variables['r_vdf'], variables['fit_vdf'], variables['v_par_mesh'], variables['v_perp_mesh'], variables['v_par_arr']
    redChiSqr, redChiSqr_beam, redChiSqr_halo = variables['redChiSqr'], variables['redChiSqr_beam'], variables['redChiSqr_halo']
    plot_path, fig_name = variables['plot_path'], variables['fig_name']
    high_c, low_c, high_h, low_h, high_b, low_b = indices['high_c'], indices['low_c'], indices['high_h'], indices['low_h'], indices['high_b'], indices['low_b']

    pitch_angle_centers = np.diff(pitch_angles)/2 + pitch_angles[:-1]
    low_sc, high_sc = determineSWA_energyIdx(low_sc_val, high_sc_val, swa_energy)
    final_fit_core, final_fit_halo, final_fit_beam = get_final_fits(r_vdf, v_par_mesh, v_perp_mesh, anti_par_strahl_cond, par_strahl_cond)
    

    '''-- Plotting Functions --'''
    # color_lines = cmocean.tools.crop_by_percent(cmocean.cm.phase, 15, which='min', N=None)
    # color_lines = color_lines(np.linspace(0, 1, 4 + 1))
    color_lines = ['#E76254','#FFB200','#0D92F4','#376795','#6CAA89','#CD218D']

    fig, ax = plt.subplots(2, 2, figsize = (9.2,9.1))
    plt.subplots_adjust(bottom=0.09,right=0.97,top=0.93, left=0.1, wspace=0.26, hspace=0.23)


    num_pitch_angles = pad.shape[1]
    required_valid_fraction = 0.9
    parallel_idx = None
    anti_parallel_idx = None
    perpendicular_idx = np.shape(pad)[1]//2
    # Start at the first index and move forwards.
    for idx in range(num_pitch_angles):
        valid_fraction = np.mean(~np.isnan(pad[:, idx, 0]))
        if valid_fraction >= required_valid_fraction:
            parallel_idx = idx # e.g., 0
            break
    # Start at the last index and move backwards.
    for idx in range(num_pitch_angles - 1, -1, -1):
        valid_fraction = np.mean(~np.isnan(pad[:, idx, 0]))
        if valid_fraction >= required_valid_fraction:
            anti_parallel_idx = idx # e.g., 17
            break
    # Actual data points => dataframe
    pa_vec_df = flatten_pa_vec(pa_vec, pitch_angles, unc_pa_vec)

    ''' ax[0,0], Parallel direction'''
    # # (1) raw PSD data
    # ax[0,0].scatter(eV_to_vel(swa_energy), pad[:,parallel_idx,0], s = s_set, color = '0.8')
    # ax[0,0].scatter(-1*eV_to_vel(swa_energy), pad[:,anti_parallel_idx,0], s = s_set, color = '0.8')

    # (2) SC electron
    # ax[0,0].scatter(eV_to_vel(swa_energy[high_sc:low_sc]), pad[ high_sc:low_sc, parallel_idx,0], s = s_set, color = 'k')
    # ax[0,0].scatter(-1*eV_to_vel(swa_energy[high_sc:low_sc]), pad[ high_sc:low_sc, anti_parallel_idx,0], s = s_set, color = 'k')

    # (3) core v_par > 0 
    core_df = pa_vec_df[(pa_vec_df['energy_channel'] >= high_c) & (pa_vec_df['energy_channel'] < low_c) & (pa_vec_df['count']>=2) & ((pa_vec_df['PA_bin'] == parallel_idx) | (pa_vec_df['PA_bin'] == anti_parallel_idx))]
    ax[0,0].plot(v_par_mesh[-1,v_par_arr>0], final_fit_core[-1,v_par_arr>0], lw = lw_set, color = color_lines[0], label = 'Core fit', zorder=2) # v_par_mesh[-1,v_par_arr>0]: vperp=0, vpar>0
    ax[0,0].plot(v_par_mesh[-1,v_par_arr<0], final_fit_core[-1,v_par_arr<0], lw = lw_set, color = color_lines[0], zorder=2)
    ax[0,0].errorbar(core_df['v_par'], core_df['PSD'], yerr=core_df['unc_psd'], 
                     alpha=0.65,color = color_lines[0], fmt='o', elinewidth=1.2, markersize=2, zorder=1)

    # Halo
    halo_df = pa_vec_df[(pa_vec_df['energy_channel'] >= high_h) & (pa_vec_df['energy_channel'] < low_h) & (pa_vec_df['count']>=PixelCountThreshold_halo) & ((pa_vec_df['PA_bin'] <= parallel_idx) | (pa_vec_df['PA_bin'] >= anti_parallel_idx))] # & (pa_vec_df['PA_bin'] == parallel_idx)]
    ax[0,0].plot(v_par_mesh[-1,:], final_fit_halo[-1,:], lw = lw_set, color = color_lines[1], label = 'Halo fit', zorder=2)
    ax[0,0].errorbar(halo_df['v_par'], halo_df['PSD'], yerr=halo_df['unc_psd'], 
                     color = color_lines[1], fmt='o', elinewidth=1.2, markersize=2,alpha=0.65, zorder=1)
    
    # (5) If anti par beam
    if anti_par_strahl_cond:
        # halo_df = pa_vec_df[(pa_vec_df['energy_channel'] >= high_h) & (pa_vec_df['energy_channel'] < low_h) & (pa_vec_df['count']>=PixelCountThreshold_halo) & (pa_vec_df['PA_bin'] == parallel_idx)] # & (pa_vec_df['PA_bin'] == parallel_idx)]
        beam_df = pa_vec_df[(pa_vec_df['energy_channel'] >= high_b) & (pa_vec_df['energy_channel'] < low_b) & (pa_vec_df['count']>=2)]
        
        ## Halo. Plot halo v_par > 0, since Halo is not obvious in Anti-Par direction because of the beam
        # ax[0,0].plot(v_par_mesh[-1,v_par_arr>0], final_fit_halo[-1,v_par_arr>0], lw = lw_set, color = color_lines[1])
        # ax[0,0].scatter(eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h,parallel_idx,0], s = s_set, color = color_lines[1])
        # ax[0,0].errorbar(halo_df['v_par'], halo_df['PSD'], yerr=halo_df['PSD']/np.sqrt(halo_df['count']), 
        #              color = color_lines[1], fmt='o', elinewidth=1.2, markersize=2,alpha=0.8)
        # ax[0,0].errorbar(eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h,parallel_idx,0], yerr=pad[high_h:low_h,parallel_idx,2],
        #                  color = color_lines[1], fmt='o', elinewidth=1.2, markersize=2,alpha=0.8)
        
        # for para_id in range(np.abs(parallel_idx),np.abs(parallel_idx)+2) : # plot halo from PA ~0 to 1 more channel, usually 0-20 deg (if parallel_idx = 0)
        #     # ax[0,0].scatter(eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h, para_id,0], s = s_set, color = color_lines[1])
        #     ax[0,0].errorbar(eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h, para_id,0], yerr=pad[high_h:low_h, para_id,0]/np.sqrt(pad[high_h:low_h, para_id,1]),
        #                      color = color_lines[1], fmt='o', elinewidth=1.2, markersize=2,alpha=0.8)

        ## Beam
        ax[0,0].plot(v_par_mesh[-1,v_par_arr<0], final_fit_beam[-1,v_par_arr<0], lw = lw_set, color = color_lines[2], label = 'Strahl fit', zorder=2)
        # Plot anti-par beam points, PA > pa_anti_par_bound
        for anti_id in range(anti_parallel_idx, num_pitch_angles, 1): # plot beam from valid anti-parallel pitch angle to 180 deg
            df_b = beam_df[beam_df['PA_bin'] == anti_id]
            ax[0,0].errorbar(df_b['v_par'].values, df_b['PSD'], yerr=df_b['unc_psd'], 
                             color = color_lines[2], fmt='o', elinewidth=1.2, markersize=2,alpha=0.65, zorder=1) #label = 'Anti-par beam' if anti_id == -1 else '',
            # ax[0,0].errorbar(-1*eV_to_vel(swa_energy[high_b:low_b]), pad[high_b:low_b,anti_id,0], yerr=pad[high_b:low_b,anti_id,2], 
            #                  color = color_lines[2], label = 'Anti-par beam' if anti_id == -1 else '', fmt='o', elinewidth=1.1, markersize=2,alpha=0.8)
    # If par beam
    if par_strahl_cond == True :
        # halo_df = pa_vec_df[(pa_vec_df['energy_channel'] >= high_h) & (pa_vec_df['energy_channel'] < low_h) & (pa_vec_df['count']>=PixelCountThreshold_halo) & (pa_vec_df['PA_bin'] == anti_parallel_idx)] # & (pa_vec_df['PA_bin'] == parallel_idx)]
        beam_df = pa_vec_df[(pa_vec_df['energy_channel'] >= high_b) & (pa_vec_df['energy_channel'] < low_b) & (pa_vec_df['count']>=2)]

        ## Halo. Plot halo v_par < 0, since Halo is not obvious in Par direction because of the beamhalo v_par < 0
        # ax[0,0].plot(v_par_mesh[-1,v_par_arr<0], final_fit_halo[-1,v_par_arr<0], lw = lw_set, color = color_lines[1])
        # ax[0,0].errorbar(halo_df['v_par'], halo_df['PSD'], yerr=halo_df['PSD']/np.sqrt(halo_df['count']), 
        #              color = color_lines[1], fmt='o', elinewidth=1.2, markersize=2,alpha=0.8)
        # ax[0,0].errorbar(-1*eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h, anti_parallel_idx,0], yerr=pad[high_h:low_h, anti_parallel_idx,2], 
        #                  color = color_lines[1],fmt='o', elinewidth=1.1, markersize=2,alpha=0.8)
        # ax[0,0].scatter(-1*eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h, anti_parallel_idx,0], s = s_set, color = color_lines[1])
        # for anti_id in range(anti_parallel_idx-1, anti_parallel_idx+1): # plot halo from PA ~180 to 1 more channel, usually 160-180 deg (if anti_parallel_idx = 17) 
        #     ax[0,0].errorbar(-1*eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h, anti_id,0], yerr=pad[high_h:low_h, anti_id,0]/np.sqrt(pad[high_h:low_h, anti_id,1]),
        #                      color = color_lines[1],fmt='o', elinewidth=1.1, markersize=2,alpha=0.8)
        #     # ax[0,0].scatter(-1*eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h, anti_id,0], s = s_set, color = color_lines[1])

        ## Beam
        ax[0,0].plot(v_par_mesh[-1,v_par_arr>0], final_fit_beam[-1,v_par_arr>0], lw = lw_set, color = color_lines[3], label = 'Strahl fit', zorder=2)
        # Plot par beam points, PA < pa_par_bound
        for para_id in range(0, parallel_idx+1): # plot beam from 0 deg to valid para PA:
            df_b = beam_df[beam_df['PA_bin'] == para_id]
            ax[0,0].errorbar(df_b['v_par'].values, df_b['PSD'], yerr=df_b['unc_psd'], 
                             color = color_lines[2], fmt='o', elinewidth=1.2, markersize=2, alpha=0.65, zorder=1) #label = 'Anti-par beam' if anti_id == -1 else ''
            # ax[0,0].errorbar(eV_to_vel(swa_energy[high_b:low_b]), pad[high_b:low_b, para_id,0], yerr=pad[high_b:low_b, para_id,2], 
            #                  color = color_lines[2], label = 'Par beam' if para_id == 0 else '', fmt='o', elinewidth=1.2, markersize=2,alpha=0.8)
    # (6) Overall fit vdf
    ax[0,0].plot(v_par_mesh[-1,:], fit_vdf[-1,:], lw = lw_set*1.8, ls = '-.', color = 'black', label = 'Final fit', zorder=3) #color_lines[3]
    # ax[0,0].plot(v_par_mesh[-1,v_par_arr>0], fit_vdf[-1,v_par_arr>0], lw = lw_set*1.5, ls = ':', color = color_lines[3])
    # ax[0,0].plot(v_par_mesh[-1,v_par_arr<0], fit_vdf[-1,v_par_arr<0], lw = lw_set*1.5, ls = ':', color = color_lines[3])

    # # noise '''
    # ax[0,0].plot(eV_to_vel(swa_energy), one_particle_noise, c = '0.8',lw = lw_set*1.5)
    # ax[0,0].plot(-1*eV_to_vel(swa_energy), one_particle_noise, c = '0.8',lw = lw_set*1.5)

    # Set axis properties
    ax[0,0].set_yscale('log')
    ax[0,0].set_ylabel('Phase space density [s$^3$/cm$^{6}$]', fontsize=14)
    ax[0,0].set_ylim(np.nanmax(pad[:,2,0])/(2*(10**7)), 2*np.nanmax(pad[:,2,0]))
    # ax[0,0].set_xscale('symlog', linthresh = 8, linscale = 0.5)
    ax[0,0].set_xlabel(r"$V_\parallel$ [$\times 10^9$ cm/s]", fontsize=14)
    ax[0,0].set_xlim(-2e9, 2e9)
    # ax[0,0].legend(loc = 8)
    ax[0,0].legend(loc='best',bbox_to_anchor=(0.62, 0.95))  
    xticks = np.arange(-2e9, 2.1e9, 1e9)  # [-2e9, -1e9, 0, 1e9, 2e9]
    ax[0,0].set_xticks(xticks)
    ax[0,0].set_xticklabels([f'{x/1e9:.1f}' for x in xticks])

    # ticks = ax[0,0].get_xticks()
    # ticks = np.delete(ticks, (3,5))
    # ax[0,0].set_xticks(ticks)

    # ax[0,0].annotate('Parallel direction', xy = (0.64,0.94), xycoords = 'axes fraction')
    if Bx_direc < 0 :
        ax[0,0].annotate('Sunward', xy = (0.045,0.95), xycoords = 'axes fraction', 
                         bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.7, pad=0.2))
        ax[0,0].annotate('Anti-sunward', xy = (0.707,0.95), xycoords = 'axes fraction', 
                         bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.7, pad=0.2))
    if Bx_direc > 0 :
        ax[0,0].annotate('Sunward', xy = (0.707,0.95), xycoords = 'axes fraction', 
                         bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.7, pad=0.2))
        ax[0,0].annotate('Anti-sunward', xy = (0.045,0.95), xycoords = 'axes fraction', 
                         bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.7, pad=0.2))
    ax[0,0].minorticks_on()
    ax[0,0].tick_params(axis='both', which='major', labelsize=11)

    # 2nd x-axis. In eV
    secax = ax[0,0].secondary_xaxis('top')
     
    primary_ticks =  ax[0,0].get_xticks() # Align secondary axis ticks with primary axis ticks
    secondary_ticks = vel_to_eV(primary_ticks)
    secax.set_xticks(primary_ticks, labels=[f'{e/1e2:.1f}' for e in secondary_ticks])
    secax.tick_params(axis='x', which='major', labelsize=11)
    secax.set_xlabel(r"Energy [$\times 10^2$ eV]", fontsize=13) ## secax.set_xlabel('(1e2) eV', loc='left', fontsize=11)

    # Text, red_chisqr of overall & beam fitting
    chisqrStr_all = '$\chi_{red}^2$ = '+f'{np.real(redChiSqr):.2f}' +' (Final fit)' #f'Red-$\chi^2={redChiSqr:.2f}\pm..$(eVDF)'
    # chisqrStr_beam = '$\chi_{red}^2$='+f'{np.real(redChiSqr_beam):.2f}'+' (Strahl)' if redChiSqr_beam is not None else '$\chi_{red}^2=None$ (Strahl)'
    # ax[0,0].text(0.02,0.83, chisqrStr_beam+'\n'+chisqrStr_all, fontsize=8, transform=ax[0,0].transAxes, color='#118B50', va='top', ha='left') 
    ax[0,0].text(0.045,0.925, chisqrStr_all, fontsize=9, transform=ax[0,0].transAxes, color=color_lines[4], va='top', ha='left') #chisqrStr_beam+'\n'+


    ''' ax[0,1], Perp direction'''
    v_par0_idx = np.shape(v_perp_mesh)[1]//2

    # # (1) raw PSD data
    # ax[0,1].scatter(eV_to_vel(swa_energy), pad[:,perpendicular_idx,0], s = s_set, color = '0.8')
    # (2) SC electron
    # ax[0,1].scatter(eV_to_vel(swa_energy[high_sc:low_sc]), pad[high_sc:low_sc,perpendicular_idx,0], s = s_set, color = 'k', label = 'SC electrons')

    # (3) core v_par = 0
    core_df_perp = pa_vec_df[(pa_vec_df['energy_channel'] >= high_c) & (pa_vec_df['energy_channel'] < low_c) & (pa_vec_df['count']>=2) & (pa_vec_df['PA_bin'] == perpendicular_idx)]
    ax[0,1].plot(v_perp_mesh[:, v_par0_idx], final_fit_core[:,v_par0_idx], lw = lw_set, color = color_lines[0], label = 'Core fit', zorder=2)
    
    ax[0,1].errorbar(core_df_perp['v_perp'], core_df_perp['PSD'], yerr=core_df_perp['unc_psd'], 
                     color = color_lines[0], fmt='o', elinewidth=1.2, markersize=2,alpha=0.8, zorder=1)
    # ax[0,1].errorbar(eV_to_vel(swa_energy[high_c:low_c]), pad[high_c:low_c,perpendicular_idx,0], yerr=pad[high_c:low_c,perpendicular_idx,2], 
    #                  color = color_lines[0], label = 'Core', fmt='o', elinewidth=2, markersize=2,alpha=0.8)
    
    # (4) halo v_par = 0 '''
    halo_df_perp = pa_vec_df[(pa_vec_df['energy_channel'] >= high_h) & (pa_vec_df['energy_channel'] < low_h) & (pa_vec_df['count']>=PixelCountThreshold_halo) & ((pa_vec_df['PA_bin'] >= perpendicular_idx-1) & (pa_vec_df['PA_bin'] <= perpendicular_idx+1))] # & (pa_vec_df['PA_bin'] == parallel_idx)]
    ax[0,1].plot(v_perp_mesh[:, v_par0_idx], final_fit_halo[:,v_par0_idx], lw = lw_set, color = color_lines[1], label = 'Halo fit', zorder=2)
    # ax[0,1].scatter(eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h,v_perp_pa_idx,0], s = s_set, color = color_lines[1], label = 'Halo')
    ax[0,1].errorbar(halo_df_perp['v_perp'], halo_df_perp['PSD'], yerr=halo_df_perp['unc_psd'],  #halo_df_perp['PSD']/np.sqrt(halo_df_perp['count']
                     color = color_lines[1], fmt='o', elinewidth=1.2, markersize=2,alpha=0.8, zorder=1)
    # ax[0,1].errorbar(eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h,perpendicular_idx,0], yerr=pad[high_h:low_h,perpendicular_idx,2], 
    #                  color = color_lines[1], label = 'Halo', fmt='o', elinewidth=2, markersize=2,alpha=0.8)
    ## Plot Halo in 6 more PA channels, around PA=90 deg. [-60, 120]
    # for perp_id in range(perpendicular_idx-1,perpendicular_idx+2): # plot halo from PA ~90 to 2 more channels, usually 80-100 deg
    #     # ax[0,1].scatter(eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h,perp_id,0], s = s_set, color = color_lines[1])
    #     ax[0,1].errorbar(eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h,perp_id,0], yerr=pad[high_h:low_h,perp_id,2], 
    #                      color = color_lines[1], fmt='o', elinewidth=2, markersize=2,alpha=0.8)

    # (5) Overall fit vdf
    ax[0,1].plot(v_perp_mesh[:, v_par0_idx], fit_vdf[:,v_par0_idx], lw = lw_set*1.8, ls = '-.', color = 'black' , label = 'Final fit', zorder=3) #color_lines[3]
    # # Noise
    # ax[0,1].plot(eV_to_vel(swa_energy), one_particle_noise, c = '0.8',lw = lw_set*1.5, label = '1-particle noise')


    ax[0,1].set_yscale('log')
    # ax[0,1].set_ylabel('Phase space density [s$^3$/cm$^{6}$]', fontsize=14)
    ax[0,1].legend()
    ax[0,1].set_ylim(np.max(pad[:,2,0])/(2*(10**7)), 2*np.max(pad[:,2,0]))
    # ax[0,1].set_xscale('log')
    ax[0,1].set_xlabel(r"$V_\perp$ [$\times 10^9$ cm/s]", fontsize=14)
    ax[0,1].set_xlim(0, 2e9)
    # ax[0,1].annotate('Perpendicular direction', xy = (0.07,0.04), xycoords = 'axes fraction')
    ax[0,1].minorticks_on()
    ax[0,1].tick_params(axis='both', which='major', labelsize=11)
    ax[0,1].legend(loc='upper left',bbox_to_anchor=(0.65, 0.95), bbox_transform=ax[0,1].transAxes) #.56, .95
    xticks = np.arange(0, 2.1e9, 0.5e9)  # [0, 0.5e9, 1e9, 1.5e9, 2e9]
    ax[0,1].set_xticks(xticks)
    ax[0,1].set_xticklabels([f'{x/1e9:.1f}' for x in xticks]) 

    # 2nd x-axis. In eV
    secax = ax[0,1].secondary_xaxis('top')
    # secax.set_xlabel('(1e2) eV', loc='left', fontsize=11) 
    primary_ticks =  ax[0,1].get_xticks() # Align secondary axis ticks with primary axis ticks
    secondary_ticks = vel_to_eV(primary_ticks)
    secax.set_xticks(primary_ticks, labels=[f'{e/1e2:.1f}' for e in secondary_ticks])
    secax.tick_params(axis='x', which='major', labelsize=11)
    secax.set_xlabel(r"Energy [$\times 10^2$ eV]", fontsize=13)


    ''' ax[1,0], Pitch angle fit '''
    ax[1,0].errorbar(pitch_angle_centers, mean_energy_pad, yerr = std_energy_pad, marker = 'o', linestyle = '', elinewidth = 1.2, markersize=2, color = color_lines[0], label = 'Averaged VDF ['+str(int(low_pad_val))+'-'+str(int(high_pad_val))+'] eV')
    ax[1,0].plot(pitch_angles, mean_fit_pad, color = color_lines[0], label = 'Pitch angle fit')
    ax[1,0].set_ylabel('Phase space density [s$^3$/cm$^{6}$]', fontsize=14)
    ax[1,0].set_xlabel('Pitch angle [$^\circ$]', fontsize=14)

    ax[1,0].set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax[1,0].set_xticklabels(['0', '30', '60', '90', '120', '150', '180'])
    ax[1,0].tick_params(axis='both', which='major', labelsize=11)

    ax[1,0].legend(loc = 2)
    ax[1,0].annotate(text = r'$P_{\mathrm{B}}$ = '+r'${:.2e}$'.format(num2tex(round_to_n(mean_pad_params['P_B'], 3))), xy = (0.05, 0.8), xycoords = 'axes fraction')
    ax[1,0].annotate(text = r'$P_0$ = '+r'${:.2e}$'.format(num2tex(round_to_n(mean_pad_params['P_0'], 3)))+', $\mathrm{PAW}_0$ = '+str(round_to_n(PAW_coeff*mean_pad_params['W_0'], 3)), xy = (0.05, 0.74), xycoords = 'axes fraction')
    ax[1,0].annotate(text = r'$P_{180}$ = '+r'${:.2e}$'.format(num2tex(round_to_n(mean_pad_params['P_180'], 3)))+', $\mathrm{PAW}_{180}$ = '+str(round_to_n(PAW_coeff*mean_pad_params['W_180'], 3)), xy = (0.05, 0.68), xycoords = 'axes fraction')


    ''' PA width fits'''
    paw_0 = PAW_coeff*pad_params_energy[:,0] # pad_params_energy[:,0]: PAW_0. pitch angle width at parallel direction
    paw_0_mask = (pad_params_energy[:,2] > 0) # pad_params_energy[:,2]: P_0. Gaussian height at alpha = 0. This is to find valid paw_0 points.
    paw_180 = PAW_coeff*pad_params_energy[:,1]
    paw_180_mask = (pad_params_energy[:,3] > 0)

    # if r_beam_params != None :
    #     ax[1,1].plot(swa_energy, np.full(len(swa_energy), anti_par_beam_paw_thresh), ls = ':', color = '0.7', label = '$\mathrm{PAW}_{180}$ beam threshold')
    #     ax[1,1].plot([anti_par_beam_energy_thresh, anti_par_beam_energy_thresh], [0, 100], ls = '--', color = '0.7', label = '$\mathrm{PAW}_{180}$ energy threshold')

    ax[1,1].scatter(swa_energy[paw_0_mask], paw_0[paw_0_mask], s = s_set*2, color = color_lines[0], label = r'$\mathrm{PAW}_{0}$ with $P_0 > 0$')
    ax[1,1].scatter(swa_energy[paw_180_mask], paw_180[paw_180_mask], s = s_set*2, color = color_lines[1], label = r'$\mathrm{PAW}_{180}$ with $P_{180} > 0$')

    ax[1,1].plot(swa_energy, PAW_coeff*pad_params_energy[:,0], lw = lw_set, color = color_lines[0], label = r'$\mathrm{PAW}_{0}$')
    ax[1,1].plot(swa_energy, PAW_coeff*pad_params_energy[:,1], lw = lw_set, color = color_lines[1], label = r'$\mathrm{PAW}_{180}$')
    ax[1,1].plot(swa_energy, np.full(len(swa_energy),PAW_coeff*mean_pad_params['W_0']), lw = lw_set, color = color_lines[0], ls = '--', label = r'$\langle \mathrm{PAW}_{0} \rangle$')
    ax[1,1].plot(swa_energy, np.full(len(swa_energy),PAW_coeff*mean_pad_params['W_180']), lw = lw_set, color = color_lines[1], ls = '--', label = r'$\langle \mathrm{PAW}_{180} \rangle$')

    ax[1,1].set_ylabel('Pitch angle width [$^\circ$]', fontsize=14)
    ax[1,1].set_xlabel('Energy [eV]', fontsize=14)
    ax[1,1].tick_params(axis='both', which='major', labelsize=11)

    ax[1,1].set_xlim(20, 1000)
    ax[1,1].set_xscale('log')
    ax[1,1].set_ylim(0, 100)
    ax[1,1].legend()

    fig.text(0.52, 0.025, f"{timeStamp.strftime('%Y-%m-%d %H:%M:%S')}", ha='center', va='center', fontsize=14)

    plt.savefig(plot_path+"Final_fit_ID"+str(fig_name)+f"_({timeStamp.strftime('%Y-%m-%d %H%M%S')})"+".png", dpi=600, bbox_inches = 'tight')
    plt.close(plt.gcf())
    fig.clf()

def plot_vdf_1D_final_uniformScale(variables, indices): 
    
    # Extract variables
    timeStamp = pd.Timestamp(variables['epoch'])
    pa_vec, unc_pa_vec = variables['pa_vec'], variables['unc_pa_vec']
    pad, swa_energy, pitch_angles = variables['pad'], variables['swa_energy'], variables['pitch_angles']
    mean_energy_pad, std_energy_pad, mean_pad_params, mean_fit_pad = variables['mean_energy_pad'], variables['std_energy_pad'], variables['mean_pad_params'], variables['mean_fit_pad']
    pad_params_energy, anti_par_strahl_cond, par_strahl_cond = variables['pad_params_energy'], variables['anti_par_strahl_cond'], variables['par_strahl_cond']
    Bx_direc, r_vdf, fit_vdf, v_par_mesh, v_perp_mesh, v_par_arr = variables['Bx_direc'], variables['r_vdf'], variables['fit_vdf'], variables['v_par_mesh'], variables['v_perp_mesh'], variables['v_par_arr']
    redChiSqr, redChiSqr_beam, redChiSqr_halo = variables['redChiSqr'], variables['redChiSqr_beam'], variables['redChiSqr_halo']
    plot_path, fig_name = variables['plot_path'], variables['fig_name']
    high_c, low_c, high_h, low_h, high_b, low_b = indices['high_c'], indices['low_c'], indices['high_h'], indices['low_h'], indices['high_b'], indices['low_b']

    pitch_angle_centers = np.diff(pitch_angles)/2 + pitch_angles[:-1]
    low_sc, high_sc = determineSWA_energyIdx(low_sc_val, high_sc_val, swa_energy)
    final_fit_core, final_fit_halo, final_fit_beam = get_final_fits(r_vdf, v_par_mesh, v_perp_mesh, anti_par_strahl_cond, par_strahl_cond)
    

    '''-- Plotting Functions --'''
    # color_lines = cmocean.tools.crop_by_percent(cmocean.cm.phase, 15, which='min', N=None)
    # color_lines = color_lines(np.linspace(0, 1, 4 + 1))
    color_lines = ['#E76254','#FFB200','#0D92F4','#376795','#6CAA89','#CD218D']

    # Use GridSpec to manually control axes positions for uniform scale
    # The key is to make ax[0,1] exactly half the width of ax[0,0] in the plotting area
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(9.6, 9.1))
    gs = GridSpec(2, 3, figure=fig, 
                  left=0.1, right=0.97, bottom=0.09, top=0.93,
                  hspace=0.23, wspace=0.45,
                  width_ratios=[2, 1, 0.05])  # Third column is a small spacer
    
    ax = [[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
          [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]]
    ax = np.array(ax)


    num_pitch_angles = pad.shape[1]
    required_valid_fraction = 0.9
    parallel_idx = None
    anti_parallel_idx = None
    perpendicular_idx = np.shape(pad)[1]//2
    # Start at the first index and move forwards.
    for idx in range(num_pitch_angles):
        valid_fraction = np.mean(~np.isnan(pad[:, idx, 0]))
        if valid_fraction >= required_valid_fraction:
            parallel_idx = idx # e.g., 0
            break
    # Start at the last index and move backwards.
    for idx in range(num_pitch_angles - 1, -1, -1):
        valid_fraction = np.mean(~np.isnan(pad[:, idx, 0]))
        if valid_fraction >= required_valid_fraction:
            anti_parallel_idx = idx # e.g., 17
            break
    # Actual data points => dataframe
    pa_vec_df = flatten_pa_vec(pa_vec, pitch_angles, unc_pa_vec)

    ''' ax[0,0], Parallel direction'''
    # # (1) raw PSD data
    # ax[0,0].scatter(eV_to_vel(swa_energy), pad[:,parallel_idx,0], s = s_set, color = '0.8')
    # ax[0,0].scatter(-1*eV_to_vel(swa_energy), pad[:,anti_parallel_idx,0], s = s_set, color = '0.8')

    # (2) SC electron
    # ax[0,0].scatter(eV_to_vel(swa_energy[high_sc:low_sc]), pad[ high_sc:low_sc, parallel_idx,0], s = s_set, color = 'k')
    # ax[0,0].scatter(-1*eV_to_vel(swa_energy[high_sc:low_sc]), pad[ high_sc:low_sc, anti_parallel_idx,0], s = s_set, color = 'k')

    # (3) core v_par > 0 
    core_df = pa_vec_df[(pa_vec_df['energy_channel'] >= high_c) & (pa_vec_df['energy_channel'] < low_c) & (pa_vec_df['count']>=2) & ((pa_vec_df['PA_bin'] == parallel_idx) | (pa_vec_df['PA_bin'] == anti_parallel_idx))]
    ax[0,0].plot(v_par_mesh[-1,v_par_arr>0], final_fit_core[-1,v_par_arr>0], lw = lw_set, color = color_lines[0], label = 'Core fit', zorder=2) # v_par_mesh[-1,v_par_arr>0]: vperp=0, vpar>0
    ax[0,0].plot(v_par_mesh[-1,v_par_arr<0], final_fit_core[-1,v_par_arr<0], lw = lw_set, color = color_lines[0], zorder=2)
    ax[0,0].errorbar(core_df['v_par'], core_df['PSD'], yerr=core_df['unc_psd'], 
                     alpha=0.65,color = color_lines[0], fmt='o', elinewidth=1.2, markersize=2, zorder=1)

    # Halo
    halo_df = pa_vec_df[(pa_vec_df['energy_channel'] >= high_h) & (pa_vec_df['energy_channel'] < low_h) & (pa_vec_df['count']>=PixelCountThreshold_halo) & ((pa_vec_df['PA_bin'] <= parallel_idx) | (pa_vec_df['PA_bin'] >= anti_parallel_idx))] # & (pa_vec_df['PA_bin'] == parallel_idx)]
    ax[0,0].plot(v_par_mesh[-1,:], final_fit_halo[-1,:], lw = lw_set, color = color_lines[1], label = 'Halo fit', zorder=2)
    ax[0,0].errorbar(halo_df['v_par'], halo_df['PSD'], yerr=halo_df['unc_psd'], 
                     color = color_lines[1], fmt='o', elinewidth=1.2, markersize=2,alpha=0.65, zorder=1)
    
    # (5) If anti par beam
    if anti_par_strahl_cond:
        # halo_df = pa_vec_df[(pa_vec_df['energy_channel'] >= high_h) & (pa_vec_df['energy_channel'] < low_h) & (pa_vec_df['count']>=PixelCountThreshold_halo) & (pa_vec_df['PA_bin'] == parallel_idx)] # & (pa_vec_df['PA_bin'] == parallel_idx)]
        beam_df = pa_vec_df[(pa_vec_df['energy_channel'] >= high_b) & (pa_vec_df['energy_channel'] < low_b) & (pa_vec_df['count']>=2)]
        
        ## Halo. Plot halo v_par > 0, since Halo is not obvious in Anti-Par direction because of the beam
        # ax[0,0].plot(v_par_mesh[-1,v_par_arr>0], final_fit_halo[-1,v_par_arr>0], lw = lw_set, color = color_lines[1])
        # ax[0,0].scatter(eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h,parallel_idx,0], s = s_set, color = color_lines[1])
        # ax[0,0].errorbar(halo_df['v_par'], halo_df['PSD'], yerr=halo_df['PSD']/np.sqrt(halo_df['count']), 
        #              color = color_lines[1], fmt='o', elinewidth=1.2, markersize=2,alpha=0.8)
        # ax[0,0].errorbar(eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h,parallel_idx,0], yerr=pad[high_h:low_h,parallel_idx,2],
        #                  color = color_lines[1], fmt='o', elinewidth=1.2, markersize=2,alpha=0.8)
        
        # for para_id in range(np.abs(parallel_idx),np.abs(parallel_idx)+2) : # plot halo from PA ~0 to 1 more channel, usually 0-20 deg (if parallel_idx = 0)
        #     # ax[0,0].scatter(eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h, para_id,0], s = s_set, color = color_lines[1])
        #     ax[0,0].errorbar(eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h, para_id,0], yerr=pad[high_h:low_h, para_id,0]/np.sqrt(pad[high_h:low_h, para_id,1]),
        #                      color = color_lines[1], fmt='o', elinewidth=1.2, markersize=2,alpha=0.8)

        ## Beam
        ax[0,0].plot(v_par_mesh[-1,v_par_arr<0], final_fit_beam[-1,v_par_arr<0], lw = lw_set, color = color_lines[2], label = 'Strahl fit', zorder=2)
        # Plot anti-par beam points, PA > pa_anti_par_bound
        for anti_id in range(anti_parallel_idx, num_pitch_angles, 1): # plot beam from valid anti-parallel pitch angle to 180 deg
            df_b = beam_df[beam_df['PA_bin'] == anti_id]
            ax[0,0].errorbar(df_b['v_par'].values, df_b['PSD'], yerr=df_b['unc_psd'], 
                             color = color_lines[2], fmt='o', elinewidth=1.2, markersize=2,alpha=0.65, zorder=1) #label = 'Anti-par beam' if anti_id == -1 else '',
            # ax[0,0].errorbar(-1*eV_to_vel(swa_energy[high_b:low_b]), pad[high_b:low_b,anti_id,0], yerr=pad[high_b:low_b,anti_id,2], 
            #                  color = color_lines[2], label = 'Anti-par beam' if anti_id == -1 else '', fmt='o', elinewidth=1.1, markersize=2,alpha=0.8)
    # If par beam
    if par_strahl_cond == True :
        # halo_df = pa_vec_df[(pa_vec_df['energy_channel'] >= high_h) & (pa_vec_df['energy_channel'] < low_h) & (pa_vec_df['count']>=PixelCountThreshold_halo) & (pa_vec_df['PA_bin'] == anti_parallel_idx)] # & (pa_vec_df['PA_bin'] == parallel_idx)]
        beam_df = pa_vec_df[(pa_vec_df['energy_channel'] >= high_b) & (pa_vec_df['energy_channel'] < low_b) & (pa_vec_df['count']>=2)]

        ## Halo. Plot halo v_par < 0, since Halo is not obvious in Par direction because of the beamhalo v_par < 0
        # ax[0,0].plot(v_par_mesh[-1,v_par_arr<0], final_fit_halo[-1,v_par_arr<0], lw = lw_set, color = color_lines[1])
        # ax[0,0].errorbar(halo_df['v_par'], halo_df['PSD'], yerr=halo_df['PSD']/np.sqrt(halo_df['count']), 
        #              color = color_lines[1], fmt='o', elinewidth=1.2, markersize=2,alpha=0.8)
        # ax[0,0].errorbar(-1*eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h, anti_parallel_idx,0], yerr=pad[high_h:low_h, anti_parallel_idx,2], 
        #                  color = color_lines[1],fmt='o', elinewidth=1.1, markersize=2,alpha=0.8)
        # ax[0,0].scatter(-1*eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h, anti_parallel_idx,0], s = s_set, color = color_lines[1])
        # for anti_id in range(anti_parallel_idx-1, anti_parallel_idx+1): # plot halo from PA ~180 to 1 more channel, usually 160-180 deg (if anti_parallel_idx = 17) 
        #     ax[0,0].errorbar(-1*eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h, anti_id,0], yerr=pad[high_h:low_h, anti_id,0]/np.sqrt(pad[high_h:low_h, anti_id,1]),
        #                      color = color_lines[1],fmt='o', elinewidth=1.1, markersize=2,alpha=0.8)
        #     # ax[0,0].scatter(-1*eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h, anti_id,0], s = s_set, color = color_lines[1])

        ## Beam
        ax[0,0].plot(v_par_mesh[-1,v_par_arr>0], final_fit_beam[-1,v_par_arr>0], lw = lw_set, color = color_lines[3], label = 'Strahl fit', zorder=2)
        # Plot par beam points, PA < pa_par_bound
        for para_id in range(0, parallel_idx+1): # plot beam from 0 deg to valid para PA:
            df_b = beam_df[beam_df['PA_bin'] == para_id]
            ax[0,0].errorbar(df_b['v_par'].values, df_b['PSD'], yerr=df_b['unc_psd'], 
                             color = color_lines[2], fmt='o', elinewidth=1.2, markersize=2, alpha=0.65, zorder=1) #label = 'Anti-par beam' if anti_id == -1 else ''
            # ax[0,0].errorbar(eV_to_vel(swa_energy[high_b:low_b]), pad[high_b:low_b, para_id,0], yerr=pad[high_b:low_b, para_id,2], 
            #                  color = color_lines[2], label = 'Par beam' if para_id == 0 else '', fmt='o', elinewidth=1.2, markersize=2,alpha=0.8)
    # (6) Overall fit vdf
    ax[0,0].plot(v_par_mesh[-1,:], fit_vdf[-1,:], lw = lw_set*1.8, ls = '-.', color = 'black', label = 'Final fit', zorder=3) #color_lines[3]
    # ax[0,0].plot(v_par_mesh[-1,v_par_arr>0], fit_vdf[-1,v_par_arr>0], lw = lw_set*1.5, ls = ':', color = color_lines[3])
    # ax[0,0].plot(v_par_mesh[-1,v_par_arr<0], fit_vdf[-1,v_par_arr<0], lw = lw_set*1.5, ls = ':', color = color_lines[3])

    # # noise '''
    # ax[0,0].plot(eV_to_vel(swa_energy), one_particle_noise, c = '0.8',lw = lw_set*1.5)
    # ax[0,0].plot(-1*eV_to_vel(swa_energy), one_particle_noise, c = '0.8',lw = lw_set*1.5)

    # Set axis properties
    ax[0,0].set_yscale('log')
    ax[0,0].set_ylabel('Phase space density [s$^3$/cm$^{6}$]', fontsize=14)
    ax[0,0].set_ylim(np.nanmax(pad[:,2,0])/(2*(10**7)), 2*np.nanmax(pad[:,2,0]))
    # ax[0,0].set_xscale('symlog', linthresh = 8, linscale = 0.5)
    ax[0,0].set_xlabel(r"$V_\parallel$ [$\times 10^9$ cm/s]", fontsize=14)
    ax[0,0].set_xlim(-2e9, 2e9)
    # ax[0,0].legend(loc = 8)
    ax[0,0].legend(loc='best',bbox_to_anchor=(0.695, 0.95))  
    xticks = np.arange(-2e9, 2.1e9, 1e9)  # [-2e9, -1e9, 0, 1e9, 2e9]
    ax[0,0].set_xticks(xticks)
    ax[0,0].set_xticklabels([f'{x/1e9:.1f}' for x in xticks])

    # ticks = ax[0,0].get_xticks()
    # ticks = np.delete(ticks, (3,5))
    # ax[0,0].set_xticks(ticks)

    # ax[0,0].annotate('Parallel direction', xy = (0.64,0.94), xycoords = 'axes fraction')
    if Bx_direc < 0 :
        ax[0,0].annotate('Sunward', xy = (0.045,0.95), xycoords = 'axes fraction', 
                         bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.7, pad=0.2))
        ax[0,0].annotate('Anti-sunward', xy = (0.707,0.95), xycoords = 'axes fraction', 
                         bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.7, pad=0.2))
    if Bx_direc > 0 :
        ax[0,0].annotate('Sunward', xy = (0.707,0.95), xycoords = 'axes fraction', 
                         bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.7, pad=0.2))
        ax[0,0].annotate('Anti-sunward', xy = (0.045,0.95), xycoords = 'axes fraction', 
                         bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.7, pad=0.2))
    ax[0,0].minorticks_on()
    ax[0,0].tick_params(axis='both', which='major', labelsize=11)

    # 2nd x-axis. In eV
    secax = ax[0,0].secondary_xaxis('top')
     
    primary_ticks =  ax[0,0].get_xticks() # Align secondary axis ticks with primary axis ticks
    secondary_ticks = vel_to_eV(primary_ticks)
    secax.set_xticks(primary_ticks, labels=[f'{e/1e2:.1f}' for e in secondary_ticks])
    secax.tick_params(axis='x', which='major', labelsize=11)
    secax.set_xlabel(r"Energy [$\times 10^2$ eV]", fontsize=13) ## secax.set_xlabel('(1e2) eV', loc='left', fontsize=11)

    # Text, red_chisqr of overall & beam fitting
    chisqrStr_all = '$\chi_{red}^2$ = '+f'{np.real(redChiSqr):.2f}' +' (Final fit)' #f'Red-$\chi^2={redChiSqr:.2f}\pm..$(eVDF)'
    chisqrStr_beam = '$\chi_{red}^2$='+f'{np.real(redChiSqr_beam):.2f}'+' (Strahl)' if redChiSqr_beam is not None else '$\chi_{red}^2=None$ (Strahl)'
    chisqrStr_halo = '$\chi_{red}^2$='+f'{np.real(redChiSqr_halo):.2f}'+' (Halo)' if redChiSqr_halo is not None else '$\chi_{red}^2=None$ (Halo)'
    # ax[0,0].text(0.02,0.83, chisqrStr_beam+'\n'+chisqrStr_all, fontsize=8, transform=ax[0,0].transAxes, color='#118B50', va='top', ha='left') 
    # ax[0,0].text(0.045,0.925, chisqrStr_all, fontsize=9, transform=ax[0,0].transAxes, color=color_lines[4], va='top', ha='left') #chisqrStr_beam+'\n'+
    ax[0,0].text(0.045,0.925, chisqrStr_all+'\n'+chisqrStr_beam+'\n'+chisqrStr_halo, fontsize=9, transform=ax[0,0].transAxes, color=color_lines[4], va='top', ha='left')


    ''' ax[0,1], Perp direction'''
    v_par0_idx = np.shape(v_perp_mesh)[1]//2

    # # (1) raw PSD data
    # ax[0,1].scatter(eV_to_vel(swa_energy), pad[:,perpendicular_idx,0], s = s_set, color = '0.8')
    # (2) SC electron
    # ax[0,1].scatter(eV_to_vel(swa_energy[high_sc:low_sc]), pad[high_sc:low_sc,perpendicular_idx,0], s = s_set, color = 'k', label = 'SC electrons')

    # (3) core v_par = 0
    core_df_perp = pa_vec_df[(pa_vec_df['energy_channel'] >= high_c) & (pa_vec_df['energy_channel'] < low_c) & (pa_vec_df['count']>=2) & (pa_vec_df['PA_bin'] == perpendicular_idx)]
    ax[0,1].plot(v_perp_mesh[:, v_par0_idx], final_fit_core[:,v_par0_idx], lw = lw_set, color = color_lines[0], label = 'Core fit', zorder=2)
    
    ax[0,1].errorbar(core_df_perp['v_perp'], core_df_perp['PSD'], yerr=core_df_perp['unc_psd'], 
                     color = color_lines[0], fmt='o', elinewidth=1.2, markersize=2,alpha=0.8, zorder=1)
    # ax[0,1].errorbar(eV_to_vel(swa_energy[high_c:low_c]), pad[high_c:low_c,perpendicular_idx,0], yerr=pad[high_c:low_c,perpendicular_idx,2], 
    #                  color = color_lines[0], label = 'Core', fmt='o', elinewidth=2, markersize=2,alpha=0.8)
    
    # (4) halo v_par = 0 '''
    halo_df_perp = pa_vec_df[(pa_vec_df['energy_channel'] >= high_h) & (pa_vec_df['energy_channel'] < low_h) & (pa_vec_df['count']>=PixelCountThreshold_halo) & ((pa_vec_df['PA_bin'] >= perpendicular_idx-1) & (pa_vec_df['PA_bin'] <= perpendicular_idx+1))] # & (pa_vec_df['PA_bin'] == parallel_idx)]
    ax[0,1].plot(v_perp_mesh[:, v_par0_idx], final_fit_halo[:,v_par0_idx], lw = lw_set, color = color_lines[1], label = 'Halo fit', zorder=2)
    # ax[0,1].scatter(eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h,v_perp_pa_idx,0], s = s_set, color = color_lines[1], label = 'Halo')
    ax[0,1].errorbar(halo_df_perp['v_perp'], halo_df_perp['PSD'], yerr=halo_df_perp['unc_psd'],  #halo_df_perp['PSD']/np.sqrt(halo_df_perp['count']
                     color = color_lines[1], fmt='o', elinewidth=1.2, markersize=2,alpha=0.8, zorder=1)
    # ax[0,1].errorbar(eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h,perpendicular_idx,0], yerr=pad[high_h:low_h,perpendicular_idx,2], 
    #                  color = color_lines[1], label = 'Halo', fmt='o', elinewidth=2, markersize=2,alpha=0.8)
    ## Plot Halo in 6 more PA channels, around PA=90 deg. [-60, 120]
    # for perp_id in range(perpendicular_idx-1,perpendicular_idx+2): # plot halo from PA ~90 to 2 more channels, usually 80-100 deg
    #     # ax[0,1].scatter(eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h,perp_id,0], s = s_set, color = color_lines[1])
    #     ax[0,1].errorbar(eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h,perp_id,0], yerr=pad[high_h:low_h,perp_id,2], 
    #                      color = color_lines[1], fmt='o', elinewidth=2, markersize=2,alpha=0.8)

    # (5) Overall fit vdf
    ax[0,1].plot(v_perp_mesh[:, v_par0_idx], fit_vdf[:,v_par0_idx], lw = lw_set*1.8, ls = '-.', color = 'black' , label = 'Final fit', zorder=3) #color_lines[3]
    # # Noise
    # ax[0,1].plot(eV_to_vel(swa_energy), one_particle_noise, c = '0.8',lw = lw_set*1.5, label = '1-particle noise')


    ax[0,1].set_yscale('log')
    # ax[0,1].set_ylabel('Phase space density [s$^3$/cm$^{6}$]', fontsize=14)
    ax[0,1].legend()
    ax[0,1].set_ylim(np.max(pad[:,2,0])/(2*(10**7)), 2*np.max(pad[:,2,0]))
    # ax[0,1].set_xscale('log')
    ax[0,1].set_xlabel(r"$V_\perp$ [$\times 10^9$ cm/s]", fontsize=14)
    ax[0,1].set_xlim(0, 2e9)
    # ax[0,1].annotate('Perpendicular direction', xy = (0.07,0.04), xycoords = 'axes fraction')
    ax[0,1].minorticks_on()
    ax[0,1].tick_params(axis='both', which='major', labelsize=11)
    ax[0,1].legend(loc='upper left',bbox_to_anchor=(0.45, 0.95), bbox_transform=ax[0,1].transAxes) #.56, .95
    xticks = np.arange(0, 2.1e9, 1e9)  # [0, 0.5e9, 1e9, 1.5e9, 2e9]
    ax[0,1].set_xticks(xticks)
    ax[0,1].set_xticklabels([f'{x/1e9:.1f}' for x in xticks]) 

    # 2nd x-axis. In eV
    secax = ax[0,1].secondary_xaxis('top')
    # secax.set_xlabel('(1e2) eV', loc='left', fontsize=11) 
    primary_ticks =  ax[0,1].get_xticks() # Align secondary axis ticks with primary axis ticks
    secondary_ticks = vel_to_eV(primary_ticks)
    secax.set_xticks(primary_ticks, labels=[f'{e/1e2:.1f}' for e in secondary_ticks])
    secax.tick_params(axis='x', which='major', labelsize=11)
    secax.set_xlabel(r"Energy [$\times 10^2$ eV]", fontsize=13)


    ''' ax[1,0], Pitch angle fit '''
    ax[1,0].errorbar(pitch_angle_centers, mean_energy_pad, yerr = std_energy_pad, marker = 'o', linestyle = '', elinewidth = 1.2, markersize=2, color = color_lines[0], label = 'Averaged VDF ['+str(int(low_pad_val))+'-'+str(int(high_pad_val))+'] eV')
    ax[1,0].plot(pitch_angles, mean_fit_pad, color = color_lines[0], label = 'Pitch angle fit')
    ax[1,0].set_ylabel('Phase space density [s$^3$/cm$^{6}$]', fontsize=14)
    ax[1,0].set_xlabel('Pitch angle [$^\circ$]', fontsize=14)

    ax[1,0].set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax[1,0].set_xticklabels(['0', '30', '60', '90', '120', '150', '180'])
    ax[1,0].tick_params(axis='both', which='major', labelsize=11)

    ax[1,0].legend(loc = 2)
    ax[1,0].annotate(text = r'$P_{\mathrm{B}}$ = '+r'${:.2e}$'.format(num2tex(round_to_n(mean_pad_params['P_B'], 3))), xy = (0.05, 0.8), xycoords = 'axes fraction')
    ax[1,0].annotate(text = r'$P_0$ = '+r'${:.2e}$'.format(num2tex(round_to_n(mean_pad_params['P_0'], 3)))+', $\mathrm{PAW}_0$ = '+str(round_to_n(PAW_coeff*mean_pad_params['W_0'], 3)), xy = (0.05, 0.74), xycoords = 'axes fraction')
    ax[1,0].annotate(text = r'$P_{180}$ = '+r'${:.2e}$'.format(num2tex(round_to_n(mean_pad_params['P_180'], 3)))+', $\mathrm{PAW}_{180}$ = '+str(round_to_n(PAW_coeff*mean_pad_params['W_180'], 3)), xy = (0.05, 0.68), xycoords = 'axes fraction')


    ''' PA width fits'''
    paw_0 = PAW_coeff*pad_params_energy[:,0] # pad_params_energy[:,0]: PAW_0. pitch angle width at parallel direction
    paw_0_mask = (pad_params_energy[:,2] > 0) # pad_params_energy[:,2]: P_0. Gaussian height at alpha = 0. This is to find valid paw_0 points.
    paw_180 = PAW_coeff*pad_params_energy[:,1]
    paw_180_mask = (pad_params_energy[:,3] > 0)

    # if r_beam_params != None :
    #     ax[1,1].plot(swa_energy, np.full(len(swa_energy), anti_par_beam_paw_thresh), ls = ':', color = '0.7', label = '$\mathrm{PAW}_{180}$ beam threshold')
    #     ax[1,1].plot([anti_par_beam_energy_thresh, anti_par_beam_energy_thresh], [0, 100], ls = '--', color = '0.7', label = '$\mathrm{PAW}_{180}$ energy threshold')

    ax[1,1].scatter(swa_energy[paw_0_mask], paw_0[paw_0_mask], s = s_set*2, color = color_lines[0]) #, label = r'$\mathrm{PAW}_{0}$ with $P_0 > 0$')
    ax[1,1].scatter(swa_energy[paw_180_mask], paw_180[paw_180_mask], s = s_set*2, color = color_lines[1]) #, label = r'$\mathrm{PAW}_{180}$ with $P_{180} > 0$')

    ax[1,1].plot(swa_energy, PAW_coeff*pad_params_energy[:,0], lw = lw_set, color = color_lines[0]) #, label = r'$\mathrm{PAW}_{0}$')
    ax[1,1].plot(swa_energy, PAW_coeff*pad_params_energy[:,1], lw = lw_set, color = color_lines[1]) #, label = r'$\mathrm{PAW}_{180}$')
    ax[1,1].plot(swa_energy, np.full(len(swa_energy),PAW_coeff*mean_pad_params['W_0']), lw = lw_set, color = color_lines[0], ls = '--', label = r'$\langle \mathrm{PAW}_{0} \rangle$')
    ax[1,1].plot(swa_energy, np.full(len(swa_energy),PAW_coeff*mean_pad_params['W_180']), lw = lw_set, color = color_lines[1], ls = '--', label = r'$\langle \mathrm{PAW}_{180} \rangle$')

    ax[1,1].set_ylabel('Pitch angle width [$^\circ$]', fontsize=14)
    ax[1,1].set_xlabel('Energy [eV]', fontsize=14)
    ax[1,1].tick_params(axis='both', which='major', labelsize=11)

    ax[1,1].set_xlim(20, 1000)
    ax[1,1].set_xscale('log')
    ax[1,1].set_ylim(0, 100)
    ax[1,1].legend()

    fig.text(0.52, 0.025, f"{timeStamp.strftime('%Y-%m-%d %H:%M:%S')}", ha='center', va='center', fontsize=14)

    plt.savefig(plot_path+"Final_fit_ID"+str(fig_name)+f"_({timeStamp.strftime('%Y-%m-%d %H%M%S')})"+".png", dpi=600, bbox_inches = 'tight')
    plt.close(plt.gcf())
    fig.clf()

def plot_vdf_1D_final_usePAD(variables, indices): 
    
    # Extract variables
    timeStamp = pd.Timestamp(variables['epoch'])
    pad, swa_energy, pitch_angles = variables['pad'], variables['swa_energy'], variables['pitch_angles']
    mean_energy_pad, std_energy_pad, mean_pad_params, mean_fit_pad = variables['mean_energy_pad'], variables['std_energy_pad'], variables['mean_pad_params'], variables['mean_fit_pad']
    pad_params_energy, anti_par_strahl_cond, par_strahl_cond = variables['pad_params_energy'], variables['anti_par_strahl_cond'], variables['par_strahl_cond']
    Bx_direc, r_vdf, fit_vdf, v_par_mesh, v_perp_mesh, v_par_arr = variables['Bx_direc'], variables['r_vdf'], variables['fit_vdf'], variables['v_par_mesh'], variables['v_perp_mesh'], variables['v_par_arr']
    redChiSqr, redChiSqr_beam = variables['redChiSqr'], variables['redChiSqr_beam']
    plot_path, fig_name = variables['plot_path'], variables['fig_name']
    high_c, low_c, high_h, low_h, high_b, low_b = indices['high_c'], indices['low_c'], indices['high_h'], indices['low_h'], indices['high_b'], indices['low_b']

    pitch_angle_centers = np.diff(pitch_angles)/2 + pitch_angles[:-1]
    low_sc, high_sc = determineSWA_energyIdx(low_sc_val, high_sc_val, swa_energy)
    final_fit_core, final_fit_halo, final_fit_beam = get_final_fits(r_vdf, v_par_mesh, v_perp_mesh, anti_par_strahl_cond, par_strahl_cond)
    

    '''-- Plotting Functions --'''
    # cmap = cmocean.tools.crop_by_percent(cmocean.cm.balance, 15, which='min', N=None)
    # cmap = plt.cm.Paired #Set1
    # color_lines = cmap(np.linspace(0, 1, 5))
    color_lines = ['#E76254','#FFB200','#0D92F4','#376795','#6CAA89','#CD218D'] #lightyellow'#FFD06F',lightblue'#72BCD5',

    fig, ax = plt.subplots(2, 2, figsize = (9.2,9.1))
    plt.subplots_adjust(bottom=0.09,right=0.97,top=0.93, left=0.1, wspace=0.26, hspace=0.23)

    
    num_pitch_angles = pad.shape[1]
    required_valid_fraction = 0.9
    parallel_idx = None
    anti_parallel_idx = None
    perpendicular_idx = np.shape(pad)[1]//2
    # Start at the first index and move forwards.
    for idx in range(num_pitch_angles):
        valid_fraction = np.mean(~np.isnan(pad[:, idx, 0]))
        if valid_fraction >= required_valid_fraction:
            parallel_idx = idx # e.g., 0
            break
    # Start at the last index and move backwards.
    for idx in range(num_pitch_angles - 1, -1, -1):
        valid_fraction = np.mean(~np.isnan(pad[:, idx, 0]))
        if valid_fraction >= required_valid_fraction:
            anti_parallel_idx = idx # e.g., 17
            break
    
    ''' ax[0,0], Parallel direction'''
    # (1) SC electron
    ax[0,0].scatter(eV_to_vel(swa_energy[high_sc:low_sc]), pad[ high_sc:low_sc, parallel_idx, 0], s = s_set, color = 'k')
    ax[0,0].scatter(-1*eV_to_vel(swa_energy[high_sc:low_sc]), pad[ high_sc:low_sc, anti_parallel_idx,0], s = s_set, color = 'k')

    # (2) core
    ax[0,0].plot(v_par_mesh[-1,v_par_arr>0], final_fit_core[-1,v_par_arr>0], lw = lw_set, color = color_lines[0], label = 'Core fit') # v_par_mesh[-1,v_par_arr>0]: vperp=0, vpar>0
    ax[0,0].plot(v_par_mesh[-1,v_par_arr<0], final_fit_core[-1,v_par_arr<0], lw = lw_set, color = color_lines[0])

    ax[0,0].errorbar(eV_to_vel(swa_energy[high_c:low_c]), pad[high_c:low_c, parallel_idx,0], yerr=pad[high_c:low_c, parallel_idx,2], 
                     alpha=0.65,color = color_lines[0], fmt='o', elinewidth=1.2, markersize=2)
    ax[0,0].errorbar(-1*eV_to_vel(swa_energy[high_c:low_c]), pad[high_c:low_c,anti_parallel_idx,0], yerr=pad[high_c:low_c,anti_parallel_idx,2], 
                     alpha=0.65,color = color_lines[0], fmt='o', elinewidth=1.2, markersize=2)

    # Halo
    ax[0,0].plot(v_par_mesh[-1,:], final_fit_halo[-1,:], lw = lw_set, color = color_lines[1], label = 'Halo fit')
    ax[0,0].errorbar(eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h,parallel_idx,0], yerr=pad[high_h:low_h,parallel_idx,2],
                         color = color_lines[1], fmt='o', elinewidth=1.2, markersize=2,alpha=0.65)
    ax[0,0].errorbar(-1*eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h, anti_parallel_idx,0], yerr=pad[high_h:low_h, anti_parallel_idx,2], 
                         color = color_lines[1],fmt='o', elinewidth=1.2, markersize=2,alpha=0.65)

    # (3) Halo & If anti par beam
    if anti_par_strahl_cond:
        ## Halo. Plot halo v_par > 0, since Halo is not obvious in Anti-Par direction because of the beam
        # ax[0,0].plot(v_par_mesh[-1,v_par_arr>0], final_fit_halo[-1,v_par_arr>0], lw = lw_set, color = color_lines[1], label = 'Halo fit')
        # ax[0,0].errorbar(eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h,parallel_idx,0], yerr=pad[high_h:low_h,parallel_idx,2],
        #                  color = color_lines[1], fmt='o', elinewidth=1.2, markersize=2,alpha=0.65)
        
        # for para_id in range(np.abs(parallel_idx),np.abs(parallel_idx)+2) : ## plot halo from PA ~0 to 1 more channel, usually 0-20 deg (if parallel_idx = 0)
        #     # ax[0,0].scatter(eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h, para_id,0], s = s_set, color = color_lines[1])
        #     ax[0,0].errorbar(eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h, para_id,0], yerr=pad[high_h:low_h, para_id,0]/np.sqrt(pad[high_h:low_h, para_id,1]),
        #                      color = color_lines[1], fmt='o', elinewidth=1.2, markersize=2,alpha=0.8)

        ## anti-par beam
        ax[0,0].plot(v_par_mesh[-1,v_par_arr<0], final_fit_beam[-1,v_par_arr<0], lw = lw_set, color = color_lines[2], label = 'Strahl fit')
        for anti_id in range(anti_parallel_idx - num_pitch_angles, 0, 1): # plot beam from valid anti-parallel pitch angle to 180 deg
            # ax[0,0].scatter(-1*eV_to_vel(swa_energy[high_b:low_b]), pad[high_b:low_b,-j,0], s = s_set, color = color_lines[2], label = 'Anti-par beam' if j == 0 else '')
            ax[0,0].errorbar(-1*eV_to_vel(swa_energy[high_b:low_b]), pad[high_b:low_b,anti_id,0], yerr=pad[high_b:low_b,anti_id,2], 
                             color = color_lines[2], fmt='o', elinewidth=1.2, markersize=2,alpha=0.65)
    # (3) Halo & If par beam
    if par_strahl_cond == True :
        ## Halo. Plot halo v_par < 0, since Halo is not obvious in Par direction because of the beamhalo v_par < 0
        # ax[0,0].plot(v_par_mesh[-1,v_par_arr<0], final_fit_halo[-1,v_par_arr<0], lw = lw_set, color = color_lines[1], label = 'Halo fit')
        # ax[0,0].errorbar(-1*eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h, anti_parallel_idx,0], yerr=pad[high_h:low_h, anti_parallel_idx,2], 
        #                  color = color_lines[1],fmt='o', elinewidth=1.2, markersize=2,alpha=0.65)
        # ax[0,0].scatter(-1*eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h, anti_parallel_idx,0], s = s_set, color = color_lines[1])
        # for anti_id in range(anti_parallel_idx-1, anti_parallel_idx+1): # plot halo from PA ~180 to 1 more channel, usually 160-180 deg (if anti_parallel_idx = 17) 
        #     ax[0,0].errorbar(-1*eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h, anti_id,0], yerr=pad[high_h:low_h, anti_id,0]/np.sqrt(pad[high_h:low_h, anti_id,1]),
        #                      color = color_lines[1],fmt='o', elinewidth=1.1, markersize=2,alpha=0.8)
        #     # ax[0,0].scatter(-1*eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h, anti_id,0], s = s_set, color = color_lines[1])

        ## Par beam
        ax[0,0].plot(v_par_mesh[-1,v_par_arr>0], final_fit_beam[-1,v_par_arr>0], lw = lw_set, color = color_lines[3], label = 'Strahl fit')
        for para_id in range(0, parallel_idx+1): # plot beam from 0 deg to valid para PA:
            ax[0,0].errorbar(eV_to_vel(swa_energy[high_b:low_b]), pad[high_b:low_b, para_id,0], yerr=pad[high_b:low_b, para_id,2], 
                             color = color_lines[2], fmt='o', elinewidth=1.2, markersize=2,alpha=0.65) #, label = 'Strahl' if para_id == 0 else ''
    # (6) Overall fit vdf
    ax[0,0].plot(v_par_mesh[-1,:], fit_vdf[-1,:], lw = lw_set*1.5, ls = '-.', color = color_lines[3], label = 'Overall fit')
    # ax[0,0].plot(v_par_mesh[-1,v_par_arr>0], fit_vdf[-1,v_par_arr>0], lw = lw_set*1.5, ls = ':', color = color_lines[3])
    # ax[0,0].plot(v_par_mesh[-1,v_par_arr<0], fit_vdf[-1,v_par_arr<0], lw = lw_set*1.5, ls = ':', color = color_lines[3])

    # # noise '''
    # ax[0,0].plot(eV_to_vel(swa_energy), one_particle_noise, c = '0.8',lw = lw_set*1.5)
    # ax[0,0].plot(-1*eV_to_vel(swa_energy), one_particle_noise, c = '0.8',lw = lw_set*1.5)

    # Set axis properties
    ax[0,0].set_yscale('log')
    ax[0,0].set_ylabel('Phase space density [s$^3$/cm$^{6}$]', fontsize=14)
    ax[0,0].set_ylim(np.nanmax(pad[:,2,0])/(2*(10**7)), 2*np.nanmax(pad[:,2,0]))
    # ax[0,0].set_xscale('symlog', linthresh = 8, linscale = 0.5)
    ax[0,0].set_xlabel(r'$v_{par}$ (cm/s)', fontsize=14)
    ax[0,0].set_xlim(-2e9, 2e9)
    ax[0,0].legend(loc='best',bbox_to_anchor=(0.62, 0.95))  

    # ticks = ax[0,0].get_xticks()
    # ticks = np.delete(ticks, (3,5))
    # ax[0,0].set_xticks(ticks)

    # ax[0,0].annotate('Parallel direction', xy = (0.64,0.94), xycoords = 'axes fraction')
    if Bx_direc < 0 :
        ax[0,0].annotate('Sunward', xy = (0.045,0.95), xycoords = 'axes fraction', 
                         bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.7, pad=0.2))
        ax[0,0].annotate('Anti-sunward', xy = (0.707,0.95), xycoords = 'axes fraction', 
                         bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.7, pad=0.2))
    if Bx_direc > 0 :
        ax[0,0].annotate('Sunward', xy = (0.707,0.95), xycoords = 'axes fraction', 
                         bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.7, pad=0.2))
        ax[0,0].annotate('Anti-sunward', xy = (0.045,0.95), xycoords = 'axes fraction',
                         bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.7, pad=0.2))
    ax[0,0].minorticks_on()
    ax[0,0].tick_params(axis='both', which='major', labelsize=11)

    # 2nd x-axis. In eV
    secax = ax[0,0].secondary_xaxis('top')
    secax.set_xlabel('(1e2) eV', loc='left', fontsize=11) 
    primary_ticks =  ax[0,0].get_xticks() # Align secondary axis ticks with primary axis ticks
    secondary_ticks = vel_to_eV(primary_ticks)
    secax.set_xticks(primary_ticks, labels=[f'{e/1e2:.1f}' for e in secondary_ticks])
    secax.tick_params(axis='x', which='major', labelsize=11)

    # Text, red_chisqr of overall & beam fitting
    chisqrStr_all = f'Red-$\chi^2={redChiSqr:.2f}$ (Overall)' #f'Red-$\chi^2={redChiSqr:.2f}\pm..$(eVDF)'
    chisqrStr_beam = f'Red-$\chi^2={redChiSqr_beam:.2f}$ (Strahl)' if redChiSqr_beam is not None else 'Red-$\chi^2=None$ (Strahl)'
    ax[0,0].text(0.045,0.925, chisqrStr_beam+'\n'+chisqrStr_all, fontsize=8, transform=ax[0,0].transAxes, color=color_lines[4], va='top', ha='left') 


    ''' ax[0,1], Perp direction'''
    v_par0_idx = np.shape(v_perp_mesh)[1]//2

    # # (1) raw PSD data
    # ax[0,1].scatter(eV_to_vel(swa_energy), pad[:,perpendicular_idx,0], s = s_set, color = '0.8')
    # (2) SC electron
    ax[0,1].scatter(eV_to_vel(swa_energy[high_sc:low_sc]), pad[high_sc:low_sc,perpendicular_idx,0], s = s_set, color = 'k', label = 'SC electrons')

    # (3) core v_par = 0
    ax[0,1].plot(v_perp_mesh[:, v_par0_idx], final_fit_core[:,v_par0_idx], lw = lw_set, color = color_lines[0], label = 'Core fit')
    # ax[0,1].scatter(eV_to_vel(swa_energy[high_c:low_c]), pad[high_c:low_c,v_perp_pa_idx,0], s = s_set, color = color_lines[0], label = 'Core')
    ax[0,1].errorbar(eV_to_vel(swa_energy[high_c:low_c]), pad[high_c:low_c,perpendicular_idx,0], yerr=pad[high_c:low_c,perpendicular_idx,2], 
                     color = color_lines[0], fmt='o', elinewidth=1.2, markersize=2,alpha=0.65)
    
    # (4) halo v_par = 0 '''
    ax[0,1].plot(v_perp_mesh[:, v_par0_idx], final_fit_halo[:,v_par0_idx], lw = lw_set, color = color_lines[1], label = 'Halo fit')
    # ax[0,1].scatter(eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h,v_perp_pa_idx,0], s = s_set, color = color_lines[1], label = 'Halo')
    ax[0,1].errorbar(eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h,perpendicular_idx,0], yerr=pad[high_h:low_h,perpendicular_idx,2], 
                     color = color_lines[1], fmt='o', elinewidth=1.2, markersize=2,alpha=0.65)
    ## Plot Halo in 1 more PA channels, around PA=90 deg. [80, 100]
    for perp_id in range(perpendicular_idx-1,perpendicular_idx+1): # plot halo from PA ~90 to 1 more channels, usually 80-100 deg
        # ax[0,1].scatter(eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h,perp_id,0], s = s_set, color = color_lines[1])
        ax[0,1].errorbar(eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h,perp_id,0], yerr=pad[high_h:low_h,perp_id,2], 
                         color = color_lines[1], fmt='o', elinewidth=1.2, markersize=2,alpha=0.65)

    # (5) Overall fit vdf
    ax[0,1].plot(v_perp_mesh[:, v_par0_idx], fit_vdf[:,v_par0_idx], lw = lw_set*1.5, ls = '-.', color = color_lines[3], label = 'Overall fit')
    # # Noise
    # ax[0,1].plot(eV_to_vel(swa_energy), one_particle_noise, c = '0.8',lw = lw_set*1.5, label = '1-particle noise')


    ax[0,1].set_yscale('log')
    # ax[0,1].set_ylabel('Phase space density [s$^3$/cm$^{6}$]', fontsize=14)
    
    ax[0,1].set_ylim(np.max(pad[:,2,0])/(2*(10**7)), 2*np.max(pad[:,2,0]))
    # ax[0,1].set_xscale('log')
    ax[0,1].set_xlabel(r'$v_{perp}$ (cm/s)', fontsize=14)
    ax[0,1].set_xlim(0, 2e9)
    # ax[0,1].annotate('Perpendicular direction', xy = (0.51,0.95), xycoords = 'axes fraction')
    ax[0,1].minorticks_on()
    ax[0,1].tick_params(axis='both', which='major', labelsize=11)
    ax[0,1].legend(loc='upper left',bbox_to_anchor=(0.56, 0.95), bbox_transform=ax[0,1].transAxes)

    # 2nd x-axis. In eV
    secax = ax[0,1].secondary_xaxis('top')
    secax.set_xlabel('(1e2) eV', loc='left', fontsize=11) 
    primary_ticks =  ax[0,1].get_xticks() # Align secondary axis ticks with primary axis ticks
    secondary_ticks = vel_to_eV(primary_ticks)
    secax.set_xticks(primary_ticks, labels=[f'{e/1e2:.1f}' for e in secondary_ticks])
    secax.tick_params(axis='x', which='major', labelsize=11)


    ''' ax[1,0], Pitch angle fit '''
    ax[1,0].errorbar(pitch_angle_centers, mean_energy_pad, yerr = std_energy_pad, marker = 'o', linestyle = '', elinewidth = 1.2, markersize=2, color = color_lines[0], label = 'Averaged VDF ('+str(int(low_pad_val))+'-'+str(int(high_pad_val))+') eV')
    ax[1,0].plot(pitch_angles, mean_fit_pad, color = color_lines[0], label = 'Pitch angle fit')
    ax[1,0].set_ylabel('Phase space density [s$^3$/cm$^{6}$]', fontsize=14)
    ax[1,0].set_xlabel('Pitch angle [$^\circ$]', fontsize=14)

    ax[1,0].set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax[1,0].set_xticklabels(['0', '30', '60', '90', '120', '150', '180'])
    ax[1,0].tick_params(axis='both', which='major', labelsize=11)

    ax[1,0].legend(loc = 2)
    ax[1,0].annotate(text = r'$P_{\mathrm{B}}$ = '+r'${:.2e}$'.format(num2tex(round_to_n(mean_pad_params['P_B'], 3))), xy = (0.05, 0.8), xycoords = 'axes fraction')
    ax[1,0].annotate(text = r'$P_0$ = '+r'${:.2e}$'.format(num2tex(round_to_n(mean_pad_params['P_0'], 3)))+', $\mathrm{PAW}_0$ = '+str(round_to_n(PAW_coeff*mean_pad_params['W_0'], 3)), xy = (0.05, 0.74), xycoords = 'axes fraction')
    ax[1,0].annotate(text = r'$P_{180}$ = '+r'${:.2e}$'.format(num2tex(round_to_n(mean_pad_params['P_180'], 3)))+', $\mathrm{PAW}_{180}$ = '+str(round_to_n(PAW_coeff*mean_pad_params['W_180'], 3)), xy = (0.05, 0.68), xycoords = 'axes fraction')


    ''' PA width fits'''
    paw_0 = PAW_coeff*pad_params_energy[:,0] # pad_params_energy[:,0]: PAW_0. pitch angle width at parallel direction
    paw_0_mask = (pad_params_energy[:,2] > 0) # pad_params_energy[:,2]: P_0. Gaussian height at alpha = 0. This is to find valid paw_0 points.
    paw_180 = PAW_coeff*pad_params_energy[:,1]
    paw_180_mask = (pad_params_energy[:,3] > 0)

    # if r_beam_params != None :
    #     ax[1,1].plot(swa_energy, np.full(len(swa_energy), anti_par_beam_paw_thresh), ls = ':', color = '0.7', label = '$\mathrm{PAW}_{180}$ beam threshold')
    #     ax[1,1].plot([anti_par_beam_energy_thresh, anti_par_beam_energy_thresh], [0, 100], ls = '--', color = '0.7', label = '$\mathrm{PAW}_{180}$ energy threshold')

    ax[1,1].scatter(swa_energy[paw_0_mask], paw_0[paw_0_mask], s = s_set*2, color = color_lines[1], label = r'$\mathrm{PAW}_{0}$ ($P_0 > 0$)')
    ax[1,1].scatter(swa_energy[paw_180_mask], paw_180[paw_180_mask], s = s_set*2, color = color_lines[2], label = r'$\mathrm{PAW}_{180}$ ($P_{180} > 0$)')

    ax[1,1].plot(swa_energy, PAW_coeff*pad_params_energy[:,0], lw = lw_set, color = color_lines[1], label = r'$\mathrm{PAW}_{0}$')
    ax[1,1].plot(swa_energy, PAW_coeff*pad_params_energy[:,1], lw = lw_set, color = color_lines[2], label = r'$\mathrm{PAW}_{180}$')
    ax[1,1].plot(swa_energy, np.full(len(swa_energy),PAW_coeff*mean_pad_params['W_0']), lw = lw_set, color = color_lines[1], ls = '--') #, label = r'$\langle \mathrm{PAW}_{0} \rangle$'
    ax[1,1].plot(swa_energy, np.full(len(swa_energy),PAW_coeff*mean_pad_params['W_180']), lw = lw_set, color = color_lines[2], ls = '--', label = r'$\langle \mathrm{PAW}_{180} \rangle$')

    ax[1,1].set_ylabel('Pitch angle width [$^\circ$]', fontsize=14)
    ax[1,1].set_xlabel('Energy [eV]', fontsize=14)
    ax[1,1].tick_params(axis='both', which='major', labelsize=11)

    ax[1,1].set_xlim(20, 1000)
    ax[1,1].set_xscale('log')
    ax[1,1].set_ylim(0, 100)
    ax[1,1].legend()

    fig.text(0.52, 0.025, f"{timeStamp.strftime('%Y-%m-%d %H:%M:%S')}", ha='center', va='center', fontsize=14)

    plt.savefig(plot_path+"Final_fit_ID"+str(fig_name)+f"_({timeStamp.strftime('%Y-%m-%d %H%M%S')})"+".png", dpi=200, bbox_inches = 'tight')
    plt.close(plt.gcf())
    fig.clf()

def plot_vdf_1beam_final_eV(variables, indices): 
    
    # Extract variables
    pad, swa_energy, pitch_angles = variables['pad'], variables['swa_energy'], variables['pitch_angles']
    mean_energy_pad, std_energy_pad, mean_pad_params, mean_fit_pad = variables['mean_energy_pad'], variables['std_energy_pad'], variables['mean_pad_params'], variables['mean_fit_pad']
    pad_params_energy, anti_par_strahl_cond, par_strahl_cond = variables['pad_params_energy'], variables['anti_par_strahl_cond'], variables['par_strahl_cond']
    Bx_direc, r_vdf, fit_vdf, v_par_mesh, v_perp_mesh, v_par_arr = variables['Bx_direc'], variables['r_vdf'], variables['fit_vdf'], variables['v_par_mesh'], variables['v_perp_mesh'], variables['v_par_arr']
    plot_path, fig_name = variables['plot_path'], variables['fig_name']
    high_c, low_c, high_h, low_h, high_b, low_b = indices['high_c'], indices['low_c'], indices['high_h'], indices['low_h'], indices['high_b'], indices['low_b']

    pitch_angle_centers = np.diff(pitch_angles)/2 + pitch_angles[:-1]
    low_sc, high_sc = determineSWA_energyIdx(low_sc_val, high_sc_val, swa_energy)
    final_fit_core, final_fit_halo, final_fit_beam = get_final_fits(r_vdf, v_par_mesh, v_perp_mesh, anti_par_strahl_cond, par_strahl_cond)
    

    '''-- Plotting Functions --'''
    color_lines = cmocean.tools.crop_by_percent(cmocean.cm.phase, 15, which='min', N=None)
    color_lines = color_lines(np.linspace(0, 1, 4 + 1))

    fig, ax = plt.subplots(2, 2, figsize = (8,8), constrained_layout=True)

    # Select valid parallel and anti-parallel pitch angle indices
    j, jj = 1, 0
    while (1-np.isnan(pad[:, -j, 0]).mean()) < 0.9:
        j += 1
    
    while (1-np.isnan(pad[:, jj, 0]).mean()) < 0.9:
        jj += 1
    v_par_pa_idx, anti_v_par_pa_idx = jj, -j # parallel, anti-parallel pitch angle indices
    v_perp_pa_idx = np.shape(pad)[1]//2

    ''' ax[0,0], Parallel direction'''
    # (1) raw PSD data
    ax[0,0].scatter(swa_energy, pad[:,v_par_pa_idx,0], s = s_set, color = '0.8')
    ax[0,0].scatter(-1*swa_energy, pad[:,anti_v_par_pa_idx,0], s = s_set, color = '0.8')

    # (2) SC electron
    ax[0,0].scatter(swa_energy[high_sc:low_sc], pad[ high_sc:low_sc, v_par_pa_idx, 0], s = s_set, color = 'k')
    ax[0,0].scatter(-1*swa_energy[high_sc:low_sc], pad[ high_sc:low_sc, anti_v_par_pa_idx,0], s = s_set, color = 'k')

    # (3) core v_par > 0 
    ax[0,0].plot(vel_to_eV(v_par_mesh[-1,v_par_arr>0]), final_fit_core[-1,v_par_arr>0], lw = lw_set, color = color_lines[0]) # v_par_mesh[-1,v_par_arr>0]: vperp=0, vpar>0
    ax[0,0].scatter(swa_energy[high_c:low_c], pad[high_c:low_c,v_par_pa_idx,0], s = s_set, color = color_lines[0])

    # core v_par < 0
    ax[0,0].plot(-1*vel_to_eV(v_par_mesh[-1,v_par_arr<0]), final_fit_core[-1,v_par_arr<0], lw = lw_set, color = color_lines[0])
    ax[0,0].scatter(-1*swa_energy[high_c:low_c], pad[high_c:low_c,anti_v_par_pa_idx,0], s = s_set, color = color_lines[0])

    # (4) halo v_par > 0
    ax[0,0].plot(vel_to_eV(v_par_mesh[-1,v_par_arr>0]), final_fit_halo[-1,v_par_arr>0], lw = lw_set, color = color_lines[1])
    ax[0,0].scatter(eV_to_vel(swa_energy[high_h:low_h]), pad[high_h:low_h,v_par_pa_idx,0], s = s_set, color = color_lines[1])
    
    # halo v_par < 0
    ax[0,0].plot(-1*vel_to_eV(v_par_mesh[-1,v_par_arr<0]), final_fit_halo[-1,v_par_arr<0], lw = lw_set, color = color_lines[1])
    ax[0,0].scatter(-1*swa_energy[high_h:low_h], pad[high_h:low_h,anti_v_par_pa_idx,0], s = s_set, color = color_lines[1])

    ## Plot Halo points in 9 more PA, from PA ~ 0 deg to 10 more channels
    for j in range(1,10) :
        ax[0,0].scatter(swa_energy[high_h:low_h], pad[high_h:low_h, v_par_pa_idx+j,0], s = s_set, color = color_lines[1])
    ## Plot Halo points in 6 more PA, from PA ~ 180 deg to 6 more channels with lower PA
    for j in range(4,10) : 
        ax[0,0].scatter(-1*swa_energy[high_h:low_h], pad[high_h:low_h,anti_v_par_pa_idx-j,0], s = s_set, color = color_lines[1])

    # (5) anti par beam
    if anti_par_strahl_cond:
        ax[0,0].plot(-1*vel_to_eV(v_par_mesh[-1,v_par_arr<0]), final_fit_beam[-1,v_par_arr<0], lw = lw_set, color = color_lines[2], label = 'Anti-par beam fit')
        # Plot anti-par beam points, PA > pa_anti_par_bound
        for j in range(np.abs(anti_v_par_pa_idx)+1) :
            ax[0,0].scatter(-1*swa_energy[high_b:low_b], pad[high_b:low_b,-j,0], s = s_set, color = color_lines[2], label = 'Anti-par beam' if j == 0 else '')
    # par beam
    if par_strahl_cond == True :
        ax[0,0].plot(vel_to_eV(v_par_mesh[-1,v_par_arr>0]), final_fit_beam[-1,v_par_arr>0], lw = lw_set, color = color_lines[3], label = 'Par beam fit')
        # Plot par beam points, PA < pa_par_bound
        for j in range(np.abs(v_par_pa_idx)+1) :
            ax[0,0].scatter(swa_energy[high_b:low_b], pad[high_b:j,0], s = s_set, color = color_lines[3], label = 'Par beam' if j == 0 else '')

    # (6) Overall fit vdf
    ax[0,0].plot(vel_to_eV(v_par_mesh[-1,v_par_arr>0]), fit_vdf[-1,v_par_arr>0], lw = lw_set*1.5, ls = ':', color = color_lines[3])
    ax[0,0].plot(-1*vel_to_eV(v_par_mesh[-1,v_par_arr<0]), fit_vdf[-1,v_par_arr<0], lw = lw_set*1.5, ls = ':', color = color_lines[3])

    # # noise '''
    # ax[0,0].plot(swa_energy, one_particle_noise, c = '0.8',lw = lw_set*1.5)
    # ax[0,0].plot(-1*swa_energy, one_particle_noise, c = '0.8',lw = lw_set*1.5)

    # Set axis properties
    ax[0,0].set_yscale('log')
    ax[0,0].set_xscale('symlog', linthresh = 8, linscale = 0.5)
    ax[0,0].set_xlabel('Energy [eV]')
    ax[0,0].set_ylabel('Phase space density [s$^3$/cm$^{6}$]')
    ax[0,0].set_ylim(np.nanmax(pad[:,2,0])/(2*(10**7)), 2*np.nanmax(pad[:,2,0]))
    ax[0,0].set_xlim(-5000, 5000)
    ax[0,0].legend(loc = 8)

    ticks = ax[0,0].get_xticks()
    ticks = np.delete(ticks, (3,5))
    ax[0,0].set_xticks(ticks)

    ax[0,0].annotate('Parallel direction', xy = (0.64,0.94), xycoords = 'axes fraction')
    if Bx_direc < 0 :
        ax[0,0].annotate('Sunward', xy = (0.045,0.88), xycoords = 'axes fraction')
        ax[0,0].annotate('Anti-sunward', xy = (0.707,0.88), xycoords = 'axes fraction')
    if Bx_direc > 0 :
        ax[0,0].annotate('Sunward', xy = (0.707,0.88), xycoords = 'axes fraction')
        ax[0,0].annotate('Anti-sunward', xy = (0.045,0.88), xycoords = 'axes fraction')
    ax[0,0].minorticks_on()


    ''' ax[0,1], Perp direction'''
    v_par0_idx = np.shape(v_perp_mesh)[1]//2

    # (1) raw PSD data
    ax[0,1].scatter(swa_energy, pad[:,v_perp_pa_idx,0], s = s_set, color = '0.8')
    # (2) SC electron
    ax[0,1].scatter(swa_energy[high_sc:low_sc], pad[high_sc:low_sc,v_perp_pa_idx,0], s = s_set, color = 'k', label = 'SC electrons')

    # (3) core v_par = 0
    ax[0,1].plot(vel_to_eV(v_perp_mesh[:, v_par0_idx]), final_fit_core[:,v_par0_idx], lw = lw_set, color = color_lines[0], label = 'Core fit')
    ax[0,1].scatter(swa_energy[high_c:low_c], pad[high_c:low_c,v_perp_pa_idx,0], s = s_set, color = color_lines[0], label = 'Core')

    # (4) halo v_par = 0 '''
    ax[0,1].plot(vel_to_eV(v_perp_mesh[:, v_par0_idx]), final_fit_halo[:,v_par0_idx], lw = lw_set, color = color_lines[1], label = 'Halo fit')
    ax[0,1].scatter(swa_energy[high_h:low_h], pad[high_h:low_h,v_perp_pa_idx,0], s = s_set, color = color_lines[1], label = 'Halo')
    ## Plot Halo in 6 more PA channels, around PA=90 deg. [-60, 120]
    for j in range(-3,4):
        ax[0,1].scatter(swa_energy[high_h:low_h], pad[high_h:low_h,v_perp_pa_idx+j,0], s = s_set, color = color_lines[1])

    # (5) Overall fit vdf
    ax[0,1].plot(vel_to_eV(v_perp_mesh[:, v_par0_idx]), fit_vdf[:,v_par0_idx], lw = lw_set*1.5, ls = ':', color = color_lines[3], label = 'Model')
    # Noise
    # ax[0,1].plot(swa_energy, one_particle_noise, c = '0.8',lw = lw_set*1.5, label = '1-particle noise')


    ax[0,1].set_yscale('log')
    ax[0,1].set_xscale('log')
    ax[0,1].set_xlabel('Energy [eV]')
    ax[0,1].set_ylabel('Phase space density [s$^3$/cm$^{6}$]')
    ax[0,1].legend()
    ax[0,1].set_ylim(np.max(pad[:,2,0])/(2*(10**7)), 2*np.max(pad[:,2,0]))
    ax[0,1].set_xlim(0.8, 5000)
    ax[0,1].annotate('Perpendicular direction', xy = (0.07,0.04), xycoords = 'axes fraction')


    ''' ax[1,0], Pitch angle fit '''
    ax[1,0].errorbar(pitch_angle_centers, mean_energy_pad, yerr = std_energy_pad, marker = 'o', linestyle = '', ms = s_set, elinewidth = 1.2, markersize=2, color = color_lines[0], label = 'VDF averaged ['+str(int(low_pad_val))+'-'+str(int(high_pad_val))+'] eV')
    ax[1,0].plot(pitch_angles, mean_fit_pad, color = color_lines[0], label = 'Pitch angle fit')
    ax[1,0].set_ylabel('Phase space density [s$^3$/cm$^{6}$]')
    ax[1,0].set_xlabel('Pitch angle [deg.]')

    ax[1,0].set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax[1,0].set_xticklabels(['0', '30', '60', '90', '120', '150', '180'])

    ax[1,0].legend(loc = 2)
    ax[1,0].annotate(text = r'$P_{\mathrm{B}}$ = '+r'${:.2e}$'.format(num2tex(round_to_n(mean_pad_params['P_B'], 3))), xy = (0.05, 0.8), xycoords = 'axes fraction')
    ax[1,0].annotate(text = r'$P_0$ = '+r'${:.2e}$'.format(num2tex(round_to_n(mean_pad_params['P_0'], 3)))+', $\mathrm{PAW}_0$ = '+str(round_to_n(PAW_coeff*mean_pad_params['W_0'], 3)), xy = (0.05, 0.74), xycoords = 'axes fraction')
    ax[1,0].annotate(text = r'$P_{180}$ = '+r'${:.2e}$'.format(num2tex(round_to_n(mean_pad_params['P_180'], 3)))+', $\mathrm{PAW}_{180}$ = '+str(round_to_n(PAW_coeff*mean_pad_params['W_180'], 3)), xy = (0.05, 0.68), xycoords = 'axes fraction')


    ''' PA width fits'''
    paw_0 = PAW_coeff*pad_params_energy[:,0] # pad_params_energy[:,0]: PAW_0. pitch angle width at parallel direction
    paw_0_mask = (pad_params_energy[:,2] > 0) # pad_params_energy[:,2]: P_0. Gaussian height at alpha = 0. This is to find valid paw_0 points.
    paw_180 = PAW_coeff*pad_params_energy[:,1]
    paw_180_mask = (pad_params_energy[:,3] > 0)

    # if r_beam_params != None :
    #     ax[1,1].plot(swa_energy, np.full(len(swa_energy), anti_par_beam_paw_thresh), ls = ':', color = '0.7', label = '$\mathrm{PAW}_{180}$ beam threshold')
    #     ax[1,1].plot([anti_par_beam_energy_thresh, anti_par_beam_energy_thresh], [0, 100], ls = '--', color = '0.7', label = '$\mathrm{PAW}_{180}$ energy threshold')

    ax[1,1].scatter(swa_energy[paw_0_mask], paw_0[paw_0_mask], s = s_set*2, color = color_lines[0], label = r'$\mathrm{PAW}_{0}$ with $P_0 > 0$')
    ax[1,1].scatter(swa_energy[paw_180_mask], paw_180[paw_180_mask], s = s_set*2, color = color_lines[1], label = r'$\mathrm{PAW}_{180}$ with $P_{180} > 0$')

    ax[1,1].plot(swa_energy, PAW_coeff*pad_params_energy[:,0], lw = lw_set, color = color_lines[0], label = r'$\mathrm{PAW}_{0}$')
    ax[1,1].plot(swa_energy, PAW_coeff*pad_params_energy[:,1], lw = lw_set, color = color_lines[1], label = r'$\mathrm{PAW}_{180}$')
    ax[1,1].plot(swa_energy, np.full(len(swa_energy),PAW_coeff*mean_pad_params['W_0']), lw = lw_set, color = color_lines[0], ls = '--', label = r'$\langle \mathrm{PAW}_{0} \rangle$')
    ax[1,1].plot(swa_energy, np.full(len(swa_energy),PAW_coeff*mean_pad_params['W_180']), lw = lw_set, color = color_lines[1], ls = '--', label = r'$\langle \mathrm{PAW}_{180} \rangle$')

    ax[1,1].set_ylabel('Pitch angle width [deg.]')
    ax[1,1].set_xlabel('Energy [eV]')

    ax[1,1].set_xlim(20, 1000)
    ax[1,1].set_xscale('log')
    ax[1,1].set_ylim(0, 100)
    ax[1,1].legend()

    plt.savefig(plot_path+"Final_fit_demo_"+str(fig_name)+".png", dpi=200, bbox_inches = 'tight')
    plt.close(plt.gcf())
    fig.clf()