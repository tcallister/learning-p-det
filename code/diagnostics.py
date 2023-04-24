import numpy as np
from scipy.interpolate import UnivariateSpline
from astropy.cosmology import Planck15,z_at_value
import astropy.units as u

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

def check_mass(ann,parameter_transform):

    """
    Function to check predicted detection probabilities as a function of mass

    Parameters
    ----------
    ann : `keras.engine.sequential.Sequential`
        Neural network with which to evaluate detection probabilities
    parameter_transform : `func`
        Function that transforms the tuple (m1_det,m2_det,DL,a1,a2,cost1,cost2,cos_inc,ra,sin_dec,pol)
        into parameter space accepted by `ann`

    Returns
    -------
    masses : `list`
        List of mass values
    p_masses : `list`
        Detection probabilites at `masses`
    """

    masses = np.linspace(2,100,100)
    dist = 1.
    z = z_at_value(Planck15.luminosity_distance,dist*u.Gpc).value
    
    param_vec = np.array([masses*(1.+z),
                          masses*(1.+z),
                          dist*np.ones_like(masses),
                          np.ones_like(masses),
                          np.ones_like(masses),
                          np.ones_like(masses),
                          np.ones_like(masses),
                          0.5*np.ones_like(masses),
                          1.73*np.ones_like(masses),
                          -0.2*np.ones_like(masses),
                          0.9*np.ones_like(masses)])
    
    p_masses = ann.predict(parameter_transform(*param_vec),verbose=0).reshape(-1)
    return masses,p_masses

def check_distance(ann,parameter_transform):

    dists = np.linspace(0.1,20,100)
    zs = z_at_value(Planck15.luminosity_distance,15*u.Gpc).value
    
    param_vec = np.array([5.*(1.+zs)*np.ones_like(dists),
                          5.*(1.+zs)*np.ones_like(dists),
                          dists,
                          np.ones_like(dists),
                          np.ones_like(dists),
                          np.ones_like(dists),
                          np.ones_like(dists),
                          0.5*np.ones_like(dists),
                          1.73*np.ones_like(dists),
                          -0.2*np.ones_like(dists),
                          0.9*np.ones_like(dists)])
    
    p_dists = ann.predict(parameter_transform(*param_vec),verbose=0).reshape(-1)
    return dists,p_dists

def check_Xeff_via_costs(ann,parameter_transform,m1=30,m2=30):

    cos_tilts = np.linspace(-1,1,100)
    
    param_vec = np.array([m1*np.ones_like(cos_tilts),
                          m2*np.ones_like(cos_tilts),
                          1.*np.ones_like(cos_tilts),
                          np.ones_like(cos_tilts),
                          np.ones_like(cos_tilts),
                          cos_tilts,
                          cos_tilts,
                          0.5*np.ones_like(cos_tilts),
                          1.73*np.ones_like(cos_tilts),
                          -0.2*np.ones_like(cos_tilts),
                          0.9*np.ones_like(cos_tilts)])
    
    p_cos_tilts = ann.predict(parameter_transform(*param_vec),verbose=0).reshape(-1)
    return cos_tilts,p_cos_tilts

def check_Xeff_via_szs(ann,parameter_transform,m1=30,m2=30):

    szs = np.linspace(-1,1,100)
    
    param_vec = np.array([m1*np.ones_like(szs),
                          m2*np.ones_like(szs),
                          1.*np.ones_like(szs),
                          np.abs(szs),
                          np.abs(szs),
                          np.sign(szs),
                          np.sign(szs),
                          0.5*np.ones_like(szs),
                          1.73*np.ones_like(szs),
                          -0.2*np.ones_like(szs),
                          0.9*np.ones_like(szs)])
    
    ps = ann.predict(parameter_transform(*param_vec),verbose=0).reshape(-1)
    return szs,ps

def check_in_plane_spin(ann,parameter_transform,m1=30,m2=30):

    spin_mags = np.linspace(0,1,100)
    
    param_vec = np.array([m1*np.ones_like(spin_mags),
                          m2*np.ones_like(spin_mags),
                          1.*np.ones_like(spin_mags),
                          spin_mags,
                          np.zeros_like(spin_mags),
                          np.zeros_like(spin_mags),
                          np.zeros_like(spin_mags),
                          0.5*np.ones_like(spin_mags),
                          1.73*np.ones_like(spin_mags),
                          -0.2*np.ones_like(spin_mags),
                          0.9*np.ones_like(spin_mags)])
    
    ps = ann.predict(parameter_transform(*param_vec),verbose=0).reshape(-1)
    return spin_mags,ps
