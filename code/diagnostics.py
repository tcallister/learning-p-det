import numpy as np
from scipy.interpolate import UnivariateSpline

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
    
    sorting = np.argsort(predicted_probabilities)
    
    probabilities_sorted = predicted_probabilities[sorting]
    detection_labels_sorted = detection_labels[sorting]
    cumulative_detections = np.cumsum(detection_labels_sorted)
    cumulative_trials = np.arange(cumulative_detections.size)
    
    cumulative_N_grid = np.interp(p_grid,probabilities_sorted,cumulative_trials)
    cumulative_C_grid = np.interp(p_grid,probabilities_sorted,cumulative_detections)

    dN_dp_grid = np.diff(cumulative_N_grid)/np.diff(p_grid)
    dN_grid = np.diff(cumulative_N_grid)
    dC_dp_grid = np.diff(cumulative_C_grid)/np.diff(p_grid)
    dC_dN_grid = np.diff(cumulative_C_grid)/np.diff(cumulative_N_grid)

    p_grid_centers = (p_grid[1:] + p_grid[:-1])/2.
    dp = np.diff(p_grid_centers)[0]
    grid_errors = np.sqrt(p_grid_centers*(1.-p_grid_centers)/dN_grid)
    
    return p_grid_centers,dC_dN_grid,grid_errors
