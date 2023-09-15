# -*- coding: utf-8 -*-
"""
A script to fit SFC-ICP-MS dissolution data 

Created on Wed Jul  5 09:23:21 2023

@author: Birk Fritsch

This script is to be read as a supplement to

Fiedler, Ma, Fritsch, Risse, Lechner, Dworschak, Merklein, Mayrhofer, Hutzler,
Stability of Bipolar Plate Materials for Proton-Exchange Membrane Water Electrolyzers:
Dissolution of Titanium and Stainless Steel inDI Water and Highly Diluted Acid,
ChemElectroChem, 2023, DOI: 10.1002/celc.202300373

Dependencies:
Python 3.10.6 packaged by conda-forge
openpyxl 3.0.10
numpy 1.23.4
matplotlib 3.6.2
pandas 1.5.1
scipy 1.9.3
"""

import os
import re
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from itertools import zip_longest

#read data
def read_data(pattern, path='.'):
    """
    Read all relevant files based on a regular expression pattern within a given directory

    Parameters
    ----------
    pattern : Str in regex syntax.
        A string in regular expression syntax. Is passed to re.compile.
    path : str or path object, optional
        Directory of interest. The default is '.'.

    Returns
    -------
    pd_dict : dict
        A dictionary containing pandas DataFrames as values.
        The corresponding file names are given as keys.

    """
    #create regex object
    regex_object = re.compile(pattern)
    
    #find files within path
    
    files = [file for file in os.listdir(path) if regex_object.search(file)]
    
    pd_dict = {file:pd.read_csv(
                                file,
                                index_col=['overlay',
                                           'name_isotope_analyte',
                                           'id_data_icpms'
                                           ],
                                low_memory=False,
                                )
               for file in files}    
    
    return pd_dict


#fit functions

def lognormal(xdat, area, ln_std, xc):
    """
    Lognormal function shifted by xc and scaled by area. After https://www.originlab.com/doc/Origin-Help/LogNormal-FitFunc
    
    with log_std = w follows:
    
        Mean: mu = exp(ln(xc)+1/2*w^2)
        Standard Deviation: sigma = exp(ln(xc)+1/2*w^2)*sqrt(exp(w^2)-1) 
        Maximum: mode = np.exp(np.log(xc) - ln_std**2) = xc/np.exp(ln_std**2)
    
    Parameters
    ----------
    xdat : np.array
        x axis data.
    area : float
        Integrated area under curve.
    ln_std : float 
        ln of the population standard deviation.
    xc : TYPE
        Center (median) of the curve.
        
    y0: y-axis offes

    Returns
    -------
    array-like
        
        Lognormal distribution
    """
    amplitude = area / (np.sqrt(2*np.pi) * ln_std * xdat)
    numerator = np.log(xdat/xc)**2
    denominator =  2*ln_std**2
    exp = np.exp(-numerator/denominator) 
    
    return amplitude * exp


def lognormal_mode(xc, ln_std):
    """
    Computes the mode (=maximum) of a lognormal function as given by 'lognormal'
    via xc/np.exp(ln_std**2)

    Parameters
    ----------
    xc : float
        Center (median) of the curve.
    ln_std : float
        ln of the population standard deviation.

    Returns
    -------
    float
        Mode of the given curve.

    """
    return xc/np.exp(ln_std**2)


def lognormal_mode_error(xc, ln_std, xc_err, ln_std_err):
    """
    Calculates the error of the mode of a lognormal function calculated via 'lognormal_mode' using 
    Gaussian error propagation

    Parameters
    ----------
    xc : float
        Center (median) of the curve.
    ln_std : float
        ln of the population standard deviation.
    xc_err : float
        Uncertainty of xc.
    ln_std_err : float
        Uncertainty.

    Returns
    -------
    float
        Error of the mode.

    """
    xc_term = xc_err/np.exp(ln_std**2)
    ln_std_term = -2*xc*ln_std * np.exp(-ln_std**2) * ln_std_err
    
    return np.sqrt(xc_term**2 + ln_std_term**2)


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks. Fillvalue is returned in case of division rest.
    
    grouper( 'ABCDEFG', 3, 'x') --> ABC DEF Gxx
    
    
    From https://docs.python.org/3.0/library/itertools.html
    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)



def multi_lognormal_fit(xdat, *params):
    """
    Computes overlapping lognormal peaks using 'lognormal'.

    Parameters
    ----------
    xdat : np.array or similar
        Data of the independent variable.
    *params : list
        A list of subsequent parameters. The entries will be chunked in groups of
        three and subsequently fed to 'lognormal'. The order is expected to follow
        [area, ln_std, xc]. If the length of params is not divisible by 3 (in 
        natural number space), the array will be appended by zeros until a last
        full pair of three is reached.
        Thus, the len(params) / 3 determines the amount of lognormal functions to
        be computed.

    Returns
    -------
    y : np.array
        An array containing a multi lognormal distribution along xdat.
        Its shape matches xdat.

    """	
    y = np.zeros(xdat.shape)
    
    for  area, ln_std, xc in grouper(params, 3, 0):
        y += lognormal(xdat, area, ln_std, xc)
        
    return y


def get_errors(p_opt, p_cov, x_dat, y_dat, y_fit):
    
    """
    create the R² and errors out of fitted data.
    Arguments: p_opt, p_cov, x_dat, y_dat, y_fit
    """
    
    # Get standard errors:
    errors = np.sqrt(np.diag(p_cov))
    
    r_squared = calc_r2(y_dat, y_fit)
    # Corrected R squared
    r_squared_corr = 1 - (1 - r_squared) * (x_dat.shape[0]-1) / (x_dat.shape[0]-len(p_opt))
        
    return errors, r_squared_corr



def calc_r2(y, f):
    """
    Corrected R squared
    """
    sstot = np.sum( (y - y.mean() )**2 )
    ssres = np.sum( (y - f)**2 )

    return 1 - ssres/sstot    


def closest_index(x, arr):
    """
    Get the index position in arr that is closest to x

    Parameters
    ----------
    x : float
        a value whose nearest neighbour in arr is of interest.
    arr : np.array
        array of interest.

    Returns
    -------
    int
        Index of arr matching the nearest neighbour of x.

    """
    arr = np.asarray(arr)
    residual_array = np.abs(arr - x)
    
    return np.argwhere(residual_array == residual_array.min())[0][0]
    

def peak_detection_routine(time,
                           icpms_data,
                           external_ax=None,
                           interval=75,
                           title='test',
                           maximum_peak_number=np.inf,
                           auto_detect=True):
    """
    Detects peaks in 'icpms_data' as a function of 'time'

    Parameters
    ----------
    time : pd.Series
        A pandas Series object containing temporal information.
    icpms_data : pd.Series
        A pandas Series containing a measurement signal per element in time. This can
        for instance be an in situ ICP-MS signal.
    external_ax : Matplotlib axes object, optional
        If an axes object is passed, it is used for plotting. If 'None',
        a new instance will be created. The default is None.
    interval : int, optional
        Amount of datapoints in time matching to a peak tip in icpms_data.
        The default is 75.
    title : str, optional
        A string for storing. It is also set as title of the plot. The default is 'test'.
    maximum_peak_number : int, optional
        A maximum number of peaks to constrain the algorithm if requried. The default is np.inf.
    auto_detect : bool, optional
        If truthy, the algorithm will automatically try to detect peaks.
        If falsy, additional peak positions can be provided on the fly.
        The default is True.

    Returns
    -------
    initially_fitted_params : list
        A list containing fitted parameters of lognormal functions as given by 'lognormal'.
    initially_fitted_params_errors : list
        A list containing uncertainties matching initially_fitted_params.
    initially_fitted_params_r2adj : list
        A list containing corrected R² values for each fitting event.

    """
    
    icpms_Gaussian_data = gaussian_filter1d(icpms_data, 3)
    icpms_filtered = savgol_filter(icpms_Gaussian_data, 131, 5)
    
    #normalize filtered data for peak finding
    min_val_filtered = icpms_filtered.min()
    max_val_filtered = icpms_filtered.max()
    icpms_filtered_normalized = (icpms_filtered-min_val_filtered) / (max_val_filtered-min_val_filtered)
    #find peaks
    for i in range(1,4):
        if i > 1:
            print(f'Reduce prominence {i}-fold')
            
        peak_idx, properties = find_peaks(icpms_filtered_normalized, prominence=0.04/i, distance= interval) #0.029 was a good overall performance
        if len(peak_idx) > 0:
            break
        
    if not auto_detect:
        asked=False
        while not asked:
            f2, ax2 = plt.subplots(layout='constrained', dpi=300, figsize=(10,5))
            ax2.plot(time, icpms_data, '.',)
            ax2.tick_params(direction='in', which='both')
            ax2.plot(time, icpms_filtered, '-.', label='SavGol filtered', color='silver')
    
            ax2.plot(np.array(time)[peak_idx], np.array(icpms_filtered)[peak_idx], 'x',
                    markersize=8, label=f'{len(peak_idx)} Peaks found', zorder=3)
            ax2.legend(fontsize='small', ncol=1, loc=0)
            ax2.set(xlabel='$t$ / s',
                   ylabel='d$M$ d$t^{-1} s^{-1}_\\mathrm{geo}$ / ng s$^{-1}$ cm$^{-2}$',
                   title=title)
            plt.show()
            
            question = input('Would you like to add extra peaks? If so, separate t values by spaces. If not, hit enter: ')
            try:

            #expect space-separated numbers or nothing/blank space
                question_list = question.split()
                #only continue if not nothing
                if question_list:
                    additional_peaks = [float(x) for x in question_list]
                    #get closest index positions
                    additional_peak_idx = [closest_index(i, time) for i in additional_peaks]
                    #add to peak_idx
                    peak_idx = np.concatenate((peak_idx, np.array(additional_peak_idx)))
                    
                    print('Added to peak indices:', additional_peak_idx)
                else:
                    #stop while loop
                    asked=True
            except ValueError:
                print('I did not understand {}. Please try again.'.format(question))
         
                
        
    if external_ax is None:
        f, ax = plt.subplots(layout='constrained', dpi=300,)
        ax.plot(time, icpms_data, '.',)
        ax.tick_params(direction='in', which='both')
    else:
        ax = external_ax
    
    #sort peak by prominence: 
    if auto_detect:
        prominences_sorted_idx = properties['prominences'].argsort()    
        peak_idx = peak_idx[prominences_sorted_idx[::-1]]
    else:
        #sort peak by index
        peak_idx.sort()

    #cap the amount of peaks found to maximum_peak_number :
    if len(peak_idx) > maximum_peak_number:
        peak_idx = peak_idx[:maximum_peak_number]    
    
    ax.plot(time, icpms_filtered, '-.', label='SavGol filtered', color='silver')

    ax.plot(np.array(time)[peak_idx], np.array(icpms_filtered)[peak_idx], 'x',
            markersize=8, label=f'{len(peak_idx)} Peaks found', zorder=3)
        
    #create output containers
    initially_fitted_params = []
    initially_fitted_params_errors = []
    initially_fitted_params_r2adj = []
    used_n = 2
    for n, peak in enumerate(peak_idx):
        
        print('Starting Peak',n, 'at', peak)
        
        #select fit data interval boundaries
        interval_half = interval // 2
              
        min_interval = int(peak-interval_half)
        if min_interval < 0:
            min_interval = 0
        minx = time.iloc[min_interval]
        
        max_interval = int(peak+interval_half*2)
        try:
            maxx = time.iloc[max_interval]
        except IndexError:
            maxx = time.iloc[-1]

        #select respective data
        time_fit = time[(time >= minx) & (time <= maxx)].array
        icpms_fit = icpms_data[(time >= minx) & (time <= maxx)].array
        icpms_filtered_fit = icpms_filtered[(time >= minx) & (time <= maxx)]
        
        ##now, ensure that no local minumum is passed by either boundary, as this could correspond to another peak:
        
        #left_handside:
        cond_left = time_fit <= time.iloc[peak]
        left_min_idx = icpms_filtered_fit[cond_left].argmin()
        #right handside
        cond_right = ~cond_left
        right_min_idx_offset = time_fit[cond_left].shape[0] - 1
        right_min_idx = icpms_filtered_fit[cond_right].argmin() + right_min_idx_offset
        #reslice 
        time_fit = time_fit[left_min_idx:right_min_idx]
        icpms_fit = icpms_fit[left_min_idx:right_min_idx]
        icpms_filtered_fit = icpms_filtered_fit[left_min_idx:right_min_idx]

        #calculation of initial guesses for fitting
        area0 = np.trapz(icpms_filtered_fit, time_fit)
        area_max = np.trapz(icpms_data, time)
        if area0 < 0:
            area0 = 0

        p0 = [area0, #area
             0.2, #ln_std
             time.iloc[peak],#xc
             ]
        #fit boundaries
        lower_bounds =  [0, 0, 0.9*time_fit.min()]
        upper_bounds = [area_max, np.inf, 1.1* time_fit.max()]
        bounds = (lower_bounds, upper_bounds)
        
        try:
            popt, pcov = curve_fit(lognormal, time_fit , icpms_filtered_fit, p0=p0, bounds=bounds, maxfev=int(1e4))
        except Exception as e:
            print(e)
            continue                        

        yfit = lognormal(time_fit, *popt)
        yfit_full = lognormal(time, *popt)
        #errors:
        errors, r2adj = get_errors(popt, pcov, time_fit , icpms_filtered_fit, yfit)
            
        if external_ax is None:
            ax.plot(time_fit, icpms_filtered_fit, '.', label=f'Peak {n+1}: selected',
                    color=plt.get_cmap('gist_rainbow')(n/len(peak_idx)) )
            ax.plot(time, yfit_full, ':', label=f'Peak {n+1}', zorder=2,
                    color=plt.get_cmap('rainbow')(n/len(peak_idx)) )
        #two extra legend entries due to SavGol and Peaks
        used_n = n + 2
        #store fit data
        initially_fitted_params.extend(popt)
        initially_fitted_params_errors.extend(errors)
        initially_fitted_params_r2adj.append(r2adj)
        
    if external_ax is None:
        legend_columns = round(used_n/8)
        if legend_columns == 0:
            legend_columns = 1
        ax.legend(fontsize='small', ncol=legend_columns, loc=0)
        ax.set(xlabel='$t$ / s',
               ylabel='d$M$ d$t^{-1} s^{-1}_\\mathrm{geo}$ / ng s$^{-1}$ cm$^{-2}$',
               title=title)
        
                                    
        for end in ['png', 'pdf', 'svg']:
            f.savefig(f'{title}.{end}', transparent=True, dpi=300)

        plt.show()
    
    
    return initially_fitted_params, initially_fitted_params_errors, initially_fitted_params_r2adj



def model_background(time, icpms_data, fit_params, title='test'):
    """
    Fits a residual backround after peak_detection_routine with another
    lognormal distribution.

    Parameters
    ---------- 
    time : pd.Series
        A pandas Series object containing temporal information.
    icpms_data : pd.Series
        A pandas Series containing a measurement signal per element in time. This can
        for instance be an in situ ICP-MS signal.
    fit_params : list
        A list comprising fit parameters for a set of lognormal functions, as
        provided by peak_detection_routine via 'initially_fitted_params'.
    title : str, optional
        A string to be used for data storage and figure title. The default is 'test'.

    Returns
    -------
    popt : np.array
        Optimal fit parameters of the lognormal distribution.

    """
    #first: retreive fit parameters:
    yfit_guess = np.zeros(time.shape)
    for n, (area, ln_std, xc) in enumerate(grouper(fit_params, 3, 0), 1):
        yfit_guess += lognormal(time, area, ln_std, xc)
    #calculate residuals
    residuals = icpms_data - yfit_guess
    #clip data to 0:
    clip_arr = residuals > 0
    residuals = residuals[clip_arr]
    time_fit = time.array[clip_arr]
    icpms_fit = icpms_data.array[clip_arr]
    ##curve fitting   
    #start displaying
    f, ax = plt.subplots(dpi=300)
    
    ax.plot(time, icpms_data, '.', label = 'Data')
    ax.plot(time_fit, residuals, 'd', label = 'Residuals', markersize=3)

    ax.set(xlabel='$t$ / s',
           ylabel='d$M$ d$t^{-1} s^{-1}_\\mathrm{geo}$ / ng s$^{-1}$ cm$^{-2}$',
           title=title)
    ax.tick_params(direction='in', which='both')

    
    #guess p0:
    area0 = np.trapz(residuals)
    ln_std0 = 0.5#0.8
    xc0 = time.iloc[icpms_fit.argmax()]
    p0 = [area0,
          ln_std0,
          xc0,]    
    
    try:
        popt, pcov = curve_fit(lognormal, time_fit, residuals,
                           p0=p0,
                           maxfev=int(5e5))
    except RuntimeError:
        print('Background fit failed. Use initial guesses instead.')
        popt = p0
    #display background guess:
    yfit = lognormal(time, *popt)
    ax.plot(time, yfit, ':', label='Background guess')
    ax.legend(loc=0)
    
    for end in ['png', 'pdf', 'svg']:
        f.savefig(f'{title}.{end}', dpi=300)
    
    plt.show()
    plt.close('all')
    
    return popt


def match_potential(filename, file_extension='_EC.csv'):
    """
    Loads potential data and interpolates them linearly.

    Parameters
    ----------
    filename : str
        It is assumed that the name of the investigated file can be split at a dot ('.')
        to match a potential file with similar name but the extension "file_extension instead".
    file_extension : 'str'
        see filename
    Returns
    -------
    interpolated_potential : interp1D object
        

    """
    file_pure = file.split('.')[0]
    #add EC ending
    file_ec = file_pure + file_extension
    #load file_ec
    ec_data = pd.read_csv(file_ec)
    #grab relevant columns
    time = ec_data['Timestamp_synchronized__s']
    potential = ec_data['E_WE_uncompensated__VvsRHE']
    #interpolate linearly
    interpolated_potential = interp1d(time, potential)
    
    return interpolated_potential
    


def fit_icpms_evolution(df,
                        savename,
                        maximum_peak_number=np.inf,
                        background_correction=True,
                        auto_detect=True,
                        analyze_materials='all'):
    """
    Main function

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame with a multi index consisting of 
                    ['overlay',
                   'name_isotope_analyte',
                   'id_data_icpms']
        which essentially describe: Run number of the experiment, Name of the analysis, and a unique identifyer
        as provided by 'read_data'.
        Moreover it requires the following columns:
        't_delaycorrected__timestamp_sfc_pc_synchronized__s' --> temporal information
        'dm_dt_S__ng_s_cm2geo_fc_top_cell_Aideal' --> signal of interest
        ''
    savename : str
        Name for saving and figure display.
    maximum_peak_number : int, optional
        passed to peak_detection_routine. The default is np.inf.
    background_correction : bool, optional
        If truthy, a background correction is applied. The default is True.
    auto_detect : bool, optional
        passed to peak_detection_routine. The default is True.
    analyze_materials : str or list, optional
        A possibility to select a subset of the data.
        'all' or list of strings matching entries in the name_isotope_analyte
        column of df to analyze only those. if 'all', no selection is performed.
        The default is 'all'.

    Returns
    -------
    output_df : pd.DataFrame
        Output fit parameters. The DataFrame is also stored as 'xlsx', and 'txt' file.

    """
    runs = sorted(run for run in set([run for run, mat, num in df.index]) if isinstance(run, str))

    len_runs = len(runs)

    #output containers
    output_dct = {}
    plot_dct = {}
    
    material_list = analyze_materials 
    
    #get materials
    if analyze_materials == 'all':
        material_list = sorted(set([mat for run, mat, num in df.index]))
    
    for material in material_list:
        
        #fit
        for meas in runs:
            
            print()
            print(f'\t\t Starting with {material} {meas} on {file}')
            print()
            
            
            #get fit data
            time = df.loc[(meas, material, ), 't_delaycorrected__timestamp_sfc_pc_synchronized__s']
            icpms_data = df.loc[(meas, material, ), 'dm_dt_S__ng_s_cm2geo_fc_top_cell_Aideal']
            #correlated potential
            interpolated_potential = match_potential(savename)
            
            #store for plotting
            plot_dct[f'{material} {meas} time'] = time
            plot_dct[f'{material} {meas} icpms_data'] = icpms_data

            
            #peak detection and initial parameters for fit
            out = peak_detection_routine(time,
                                            icpms_data,
                                            title=f'{savename} {material} {meas} initial fit',
                                            maximum_peak_number=maximum_peak_number,
                                            interval=60,
                                            auto_detect=auto_detect)
            
            initially_fitted_params, initially_fitted_params_errors, initially_fitted_params_r2adj = out
            
            #model background:
            if background_correction:
                background_popt = model_background(time, icpms_data,
                                                   initially_fitted_params,
                                                   title=f'{savename} {material} {meas} background',
                                                   )
                #purposly underestimate background size
                background_popt[0] /= 2
                
                initially_fitted_params = [*initially_fitted_params, *background_popt]
                
            #idea check if peaks overlap. If so, reduce their height
            
            #fit full function
            try:
                #fit twice: Once with starting parameters, once with determined optimal parameters as start
                result_storage = {}
                p0 = initially_fitted_params

                for i in range(2):
                
                    popt, pcov = curve_fit(multi_lognormal_fit, time, icpms_data,
                                           p0=initially_fitted_params, bounds=(0, np.inf),
                                           maxfev=int(1e4))
                    yfit = multi_lognormal_fit(time, *popt)
                    errors, r2adj = get_errors(popt, pcov, time, icpms_data, yfit)
                    result_storage[r2adj] = popt, pcov, yfit, errors, r2adj
                    #overwrite p0 for next run
                    p0 = popt
                #now, select better fit
                popt, pcov, yfit, errors, r2adj = result_storage[max(result_storage)]
                
                label = r'fit $R^2_\mathrm{adj}=$' + f'{round(r2adj*100, 2)}%'
            except Exception as e:
                print(e)
                label = 'fit failed\nstarting params shown'
                popt = initially_fitted_params
                errors = np.full(len(popt), np.inf)
                r2adj = 'Fit failed'
                yfit = multi_lognormal_fit(time, *popt)
            #store for plotting
            plot_dct[f'{material} {meas} yfit'] = yfit
            plot_dct[f'{material} {meas} fit label'] = label
            plot_dct[f'{material} {meas} popt'] = popt
            
            #store data of full fit:
            output_dct[f'{material} {meas} full fit R2adj'] = r2adj
            for n, (vals, val_errs) in enumerate(
                                                zip(grouper(popt, 3, 0),
                                                    grouper(errors, 3, 0),
                                                    ),
                                                   1):
                #unpack
                area, ln_std, xc = vals
                area_err, ln_std_err, xc_err = val_errs
                output_dct[f'{material} {meas} peak {n} area / ng cm-2'] = area
                output_dct[f'{material} {meas} peak {n} area error / ng cm-2'] = area_err
                output_dct[f'{material} {meas} peak {n} ln(std)'] = ln_std
                output_dct[f'{material} {meas} peak {n} ln(std) error'] = ln_std_err
                output_dct[f'{material} {meas} peak {n} xc / s'] = xc
                output_dct[f'{material} {meas} peak {n} xc error / s'] = xc_err
                
                mode = lognormal_mode(xc, ln_std)
                mode_error = lognormal_mode_error(xc, ln_std, xc_err, ln_std_err)
                
                output_dct[f'{material} {meas} peak {n} mode / s'] = mode
                output_dct[f'{material} {meas} peak {n} mode error / s'] = mode_error
                
                try:
                    pot = float(interpolated_potential(mode))
                    left_pot = float(interpolated_potential(mode-mode_error))
                    right_pot = float(interpolated_potential(mode+mode_error))
                except ValueError:
                    pot = left_pot = right_pot = np.nan
                    print(f'Unable to determine the potential for peak {n+1} at {mode} +/- {mode_error} s.')

                #calculate potential error based on mode error:
                potential_error = abs(left_pot - right_pot) / 2
                output_dct[f'{material} {meas} peak {n} mode potential / VvsRHE'] = pot

                output_dct[f'{material} {meas} peak {n} mode potential error / VvsRHE'] = potential_error
                
            #store data of initial guesses:
            for n, (vals, val_errs, r2adj_single) in enumerate(zip(
                                                    grouper(initially_fitted_params, 3, 0),
                                                    grouper(initially_fitted_params_errors, 3, 0),
                                                    initially_fitted_params_r2adj
                                                    ),
                                                 1):
            
                area, ln_std, xc = vals
                area_err, ln_std_err, xc_err = val_errs
                output_dct[f'{material} {meas} peak {n} area_single / ng cm-2'] = area
                output_dct[f'{material} {meas} peak {n} area_single error / ng cm-2'] = area_err
                output_dct[f'{material} {meas} peak {n} ln(std)_single'] = ln_std
                output_dct[f'{material} {meas} peak {n} ln(std)_single error'] = ln_std_err
                output_dct[f'{material} {meas} peak {n} xc_single / s'] = xc
                output_dct[f'{material} {meas} peak {n} xc_single error / s'] = xc_err
                
                mode = lognormal_mode(xc, ln_std)
                mode_error = lognormal_mode_error(xc, ln_std, xc_err, ln_std_err)
                
                output_dct[f'{material} {meas} peak {n} mode_single / s'] = mode
                output_dct[f'{material} {meas} peak {n} mode_single error / s'] = mode_error
                
                try:
                    pot = float(interpolated_potential(mode))
                    left_pot = float(interpolated_potential(mode-mode_error))
                    right_pot = float(interpolated_potential(mode+mode_error))
                except ValueError:
                    pot = left_pot = right_pot = np.nan
                    print(f'Unable to determine the potential for peak {n+1} at {round(mode,4)} +/- {round(mode_error,4)} s.')
                potential_error = abs(left_pot - right_pot) / 2
    
                output_dct[f'{material} {meas} peak {n} mode_single potential / VvsRHE'] = pot
                #calculate potential error based on mode error:

                output_dct[f'{material} {meas} peak {n} mode_single potential error / VvsRHE'] = potential_error
                output_dct[f'{material} {meas} peak {n} R2adj'] = r2adj_single

                
            

        #plot
        f, axes = plt.subplots(nrows=len_runs, figsize=(10,1+3*len_runs),
                               sharex=True, layout='constrained',
                               dpi=300,
                               num='main')
        #ensure that axes is a list even if nrows = 1
        if len_runs == 1:
            axes = np.array([axes])
        
        f.suptitle(file)
        
        axes[0].set_title(material)
        axes[-1].set_xlabel('$t$ / s')
        #ylabel on middle axis
        axes[(len(axes)-1)//2].set_ylabel('d$M$ d$t^{-1} s^{-1}_\\mathrm{geo}$ / ng s$^{-1}$ cm$^{-2}$')
        
        for meas, ax in zip(runs, axes.ravel()):
            #retreive plot data
            time = plot_dct[f'{material} {meas} time']
            icpms_data = plot_dct[f'{material} {meas} icpms_data']
            yfit = plot_dct[f'{material} {meas} yfit']
            label = plot_dct[f'{material} {meas} fit label']
            popt = plot_dct[f'{material} {meas} popt']
            
            ax.plot(time, icpms_data, '.', label='Data')
            ax.tick_params(direction='in', which='both')

            ax.plot(time, yfit, '-', label=label, linewidth=3)
            #plot single lognormal functions
            for n, (area, ln_std, xc) in enumerate(grouper(popt, 3, 0), 1):
                yfit_single = lognormal(time, area, ln_std, xc)
                mode = output_dct[f'{material} {meas} peak {n} mode potential / VvsRHE']
                mode_error = output_dct[f'{material} {meas} peak {n} mode potential error / VvsRHE']
                ax.plot(time, yfit_single, ':', label=f'Peak {n}: {round(mode,3)}$\\pm${round(mode_error,3)} V' + '$_\\mathrm{RHE}$')
        
            ax.legend(fontsize='small', ncol=2, title=f'{meas} run',loc=0)# loc='center left', bbox_to_anchor=(1, 0.5))
        
        #store plot                
        for end in ['png', 'pdf', 'svg']:
            f.savefig(f'{savename}_{material}.{end}', dpi=300)
        
        plt.show()
        plt.close('all')
        
    #store fit data
    output_df = pd.DataFrame.from_dict(output_dct, orient='index', columns=['Value'])
    output_df['Material'] = [val.split()[0] for val in output_df.index]
    output_df['Run'] = [val.split()[1] for val in output_df.index]
    output_df['Peak'] = [int(val.split()[3]) if 'peak' in val.lower() else np.nan for val in output_df.index ]
    output_df['Unit'] = [val.split(' / ')[1] if '/' in val.lower() else np.nan for val in output_df.index]
    output_df['Description'] = [' '.join(val.split()[4:]) if 'peak' in val.lower() else ' '.join(val.split()[2:]) for val in output_df.index]
    output_df.index.name = 'Summary'
    
    output_df.to_csv(f'{savename} Results.txt')
    output_df.to_excel(f'{savename} Results.xlsx')

    return output_df



if __name__ == '__main__':
    
    pattern = '[^(_EC)]\.csv$'    
    pd_dict = read_data(pattern)
    
    #iterate over files:
    for file in sorted(pd_dict, reverse=False):
                        
        print('Starting', file)
        
        #name for saving data:
        savename = file.split(".")[0]
        df = pd_dict[file]
        out_dct = {}
        #exclude nan values
        df = df[~df['t_delaycorrected__timestamp_sfc_pc_synchronized__s'].isnull()]
        df = df[~df['dm_dt_S__ng_s_cm2geo_fc_top_cell_Aideal'].isnull()]
        #exclude negative times
        df = df[df['t_delaycorrected__timestamp_sfc_pc_synchronized__s'] >= 0 ]
        
        #escape routine to accelerate fitting:
        maximum_peak_number = 15 
        
        results = fit_icpms_evolution(df, savename,
                                      maximum_peak_number=maximum_peak_number,
                                      background_correction=True,
                                      auto_detect=True,
                                      analyze_materials='all')            

        
        