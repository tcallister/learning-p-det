import numpy as np
from scipy.interpolate import UnivariateSpline
import astropy.units as u
import pandas as pd
from scipy.stats import ks_2samp
from draw_new_injections import gen_found_injections
import matplotlib.pyplot as plt

def get_pp_data(predicted_probabilities,detection_labels):
    
    sorting = np.argsort(predicted_probabilities)

    probabilities_sorted = predicted_probabilities[sorting]
    detection_labels_sorted = detection_labels[sorting]
    cumulative_detections = np.cumsum(detection_labels_sorted)
    cumulative_trials = np.arange(cumulative_detections.size)

    cumulative_N_vs_p_spline_scaled = UnivariateSpline(probabilities_sorted,
                                                       cumulative_trials/cumulative_trials[-1],
                                                       k=2, s=0.01)
    cumulative_C_vs_N_spline_scaled = UnivariateSpline(cumulative_trials/cumulative_trials[-1],
                                                       cumulative_detections/cumulative_detections[-1],
                                                       k=3, s=0.01)

    dN_dp_spline_scaled = cumulative_N_vs_p_spline_scaled.derivative()
    dC_dN_spline_scaled = cumulative_C_vs_N_spline_scaled.derivative()
    
    p_measured = dC_dN_spline_scaled(cumulative_trials/cumulative_trials[-1])\
                        *(cumulative_detections[-1]/cumulative_trials[-1])
    error = np.sqrt(probabilities_sorted*(1.-probabilities_sorted)\
                        /dN_dp_spline_scaled(probabilities_sorted)/cumulative_trials[-1])
    
    return probabilities_sorted,p_measured,error

def get_pp_data_discrete(predicted_probabilities,detection_labels,p_grid=np.linspace(0,1,100)):

    """
    Helper function to compute data for a PP plot evaluating neural network prediction quality.

    Parameters
    ----------
    predicted_probabilities : `array`
        List of P_det predictions from our neural network emulator
    detection_labels : `array`
        Actual missed/found labels assigned to injections
    p_grid : `array`
        Array defining edges of bins into which we will sort PP-plot data

    Returns
    -------
    p_grid_centers : `array`
        List of length len(p_grid)-1 marking bins of predicted P_det values
    dC_dN_grid : `array`
        Actual fractions of injections in each bin that are detected
    grid_errors : `array`
        Expected standard deviation on detection fractions, given finite sampling
    """
    
    # Sort predicted probabilities, in preparation for binning
    sorting = np.argsort(predicted_probabilities)

    # Apply sorting and count the cumulative number of detections and trials,
    # as we look from the lowest to the highest predicted probabilities
    probabilities_sorted = predicted_probabilities[sorting]
    detection_labels_sorted = detection_labels[sorting]
    cumulative_detections = np.cumsum(detection_labels_sorted)
    cumulative_trials = np.arange(cumulative_detections.size)
    
    # Interpolate cumulative counts onto a uniform grid
    cumulative_N_grid = np.interp(p_grid,probabilities_sorted,cumulative_trials)
    cumulative_C_grid = np.interp(p_grid,probabilities_sorted,cumulative_detections)

    # The fraction of events seen is the ratio between dN_dp, the number of trials per unit p,
    # and dC_dp, the change in cumulative detections per unit p
    dN_dp_grid = np.diff(cumulative_N_grid)/np.diff(p_grid)
    dN_grid = np.diff(cumulative_N_grid)
    dC_dp_grid = np.diff(cumulative_C_grid)/np.diff(p_grid)
    dC_dN_grid = np.diff(cumulative_C_grid)/np.diff(cumulative_N_grid)

    # Compute bin centers and expected errors
    p_grid_centers = (p_grid[1:] + p_grid[:-1])/2.
    dp = np.diff(p_grid_centers)[0]
    grid_errors = np.sqrt(p_grid_centers*(1.-p_grid_centers)/dN_grid)
    
    return p_grid_centers,dC_dN_grid,grid_errors

def ks_test(ann,addDerived,feature_names,input_sc,ndraws,reference_data,population_type,output_prefix,
            parameters_to_check=None):

    # Draw new events
    found_events,nTrials = gen_found_injections(ann,
                            addDerived,
                            feature_names,
                            input_sc,
                            ndraws,
                            10000,
                            pop=population_type)

    # Load reference training data and extract detections
    train_data_all = pd.read_hdf(reference_data)
    addDerived(train_data_all)
    train_data_found = train_data_all[train_data_all.detected==1]

    # Define list of parameters for which to check KS test
    if parameters_to_check==None:
        parameters_to_check = ['m1_source','m1_detector','chirp_mass_detector','total_mass_detector','q',\
            'Xeff','Xdiff','Xp_gen','redshift','luminosity_distance','log_d','cos_inclination']

    # Loop across parameters
    ks_results = {}
    for param in parameters_to_check:

        # Store KS test pvalue
        ks_results[param] = ks_2samp(train_data_found[param],found_events[param]).pvalue

        # Plot
        fig,ax = plt.subplots(figsize=(4,3))
        ax.hist(train_data_found[param],density=True,bins=30,alpha=0.5,label='Train (Found)')
        ax.hist(found_events[param],density=True,bins=30,histtype='step',color='black',zorder=-1,label='Test (Found)')
        ax.text(0.9,0.9,"{0:.2e}".format(ks_results[param]),transform=ax.transAxes,verticalalignment='center',horizontalalignment='right')
        ax.set_xlabel(param)
        plt.tight_layout()
        plt.savefig('{0}_{1}_{2}.jpeg'.format(output_prefix,population_type,param),dpi=100,bbox_inches='tight')

    # Also store integrated detection efficiency
    p_hat = len(found_events)/nTrials
    std_p_hat = np.sqrt(p_hat*(1.-p_hat)/nTrials)
    ks_results['det_efficiency'] = p_hat
    ks_results['std_det_efficiency'] = std_p_hat

    return ks_results
