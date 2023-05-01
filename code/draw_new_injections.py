import numpy as np
from astropy.cosmology import Planck15,z_at_value
import astropy.units as u
import json
import pandas as pd
from functools import partial
from utilities import ANNaverage,generalized_Xp
from training_routines import build_ann

def draw_new_injections(batch_size=1000):

    """
    Function to draw new proposed injection parameters from the O3b injection distribution.

    Parameters
    ----------
    batch_size : `int`
        The number of event parameters to draw and return (default 1000)

    Returns
    -------
    draws : `dict`
        Dictionary containing randomly drawn component masses, Cartesian spins, sky position,
        source orientation, distance, and time. Note that the provided masses are in the *source frame*,
        which differs from the convention used in e.g. `drawing_injections.draw_params()`
    """

    # Parameters governing O3b BBH injection distribution
    # see https://zenodo.org/record/5546676
    min_m1 = 2.
    max_m1 = 100.
    alpha_m1 = -2.35
    min_m2 = 2.
    alpha_m2 = 1.
    aMax = 0.998
    zMax=1.9
    kappa=1.

    # Draw random cumulants in the m1 and m2 distribution and use the inverse CDF to translate into m1 and m2 values
    cs_m1 = np.random.random(batch_size)
    cs_m2 = np.random.random(batch_size)
    m1 = (min_m1**(1.+alpha_m1) + cs_m1*(max_m1**(1.+alpha_m1)-min_m1**(1.+alpha_m1)))**(1./(1.+alpha_m1))
    m2 = (min_m2**(1.+alpha_m2) + cs_m2*(m1**(1.+alpha_m2)-min_m2**(1.+alpha_m2)))**(1./(1.+alpha_m2))

    # Random isotropic spins
    a1 = np.random.uniform(low=0.,high=aMax,size=batch_size)
    a2 = np.random.uniform(low=0.,high=aMax,size=batch_size)
    cost1 = np.random.uniform(low=-1.,high=1.,size=batch_size)
    cost2 = np.random.uniform(low=-1.,high=1.,size=batch_size)
    phi1 = np.random.uniform(low=0.,high=2.*np.pi,size=batch_size)
    phi2 = np.random.uniform(low=0.,high=2.*np.pi,size=batch_size)

    # Manually construct CDF of redshift distribution
    z_grid = np.linspace(1e-6,zMax,1000)
    dVdz_grid = 4.*np.pi*Planck15.differential_comoving_volume(z_grid).to(u.Gpc**3/u.sr).value
    p_z_grid = dVdz_grid*(1.+z_grid)**(kappa-1.)
    c_z_grid = np.cumsum(p_z_grid)
    c_z_grid /= c_z_grid[-1]

    # Draw random cumulant, inverse to obtain a redshift, and compute luminosity distance
    cs_z = np.random.random(batch_size)
    z = np.interp(cs_z,c_z_grid,z_grid)
    DL = Planck15.luminosity_distance(z).to(u.Gpc).value

    # Isotropic sky position and orientation
    cos_inc = np.random.uniform(low=-1.,high=1.,size=batch_size)
    ra = np.random.uniform(low=0.,high=2.*np.pi,size=batch_size)
    sin_dec = np.random.uniform(low=-1.,high=1.,size=batch_size)
    pol = np.random.uniform(low=0.,high=2.*np.pi,size=batch_size)
    
    # Record and return
    draws = pd.DataFrame({'m1_src':m1,
                        'm2_src':m2,
                        'a1':a1,
                        'a2':a2,
                        'cost1':cost1,
                        'cost2':cost2,
                        'phi1':phi1,
                        'phi2':phi2,
                        'z':z,
                        'DL':DL,
                        'cos_inc':cos_inc,
                        'ra':ra,
                        'sin_dec':sin_dec,
                        'pol':pol})

    return draws

def input_transform(params,scaler_stats):

    """
    Function to transform a dictionary of parameters, as output by `draw_new_injections()`, into the rescaled coordinate
    system expected by our P_det emulator.

    Parameters
    ----------
    params : `dict`
        Dictionary of proposed event parameters, as produced by `draw_new_injections()`
    scaler_stats : `dict`
        Dictionary containing ['mean'] and ['std'] keys, with which transformed parameters will be rescaled and recentered.
        This is produced in the neural network training.

    Returns
    -------
    param_vector : `array`
        Array of transformed and rescaled injection parameters to be passed through our neural network P_det emulator
    """
    
    # Derive mass parameters
    m1_det = params.m1_src*(1.+params.z)
    m2_det = params.m2_src*(1.+params.z)
    q = m2_det/m1_det
    eta = q/(1.+q)**2
    Mtot_det = m1_det+m2_det
    Mc_det = Mtot_det*eta**(3./5.)
    
    # Derive spin parameters
    a1 = params.a1
    a2 = params.a2
    cost1 = params.cost1
    cost2 = params.cost2
    phi1 = params.phi1
    phi2 = params.phi2
    DL = params.DL
    Xeff = (a1*cost1 + q*a2*cost2)/(1.+q)
    Xdiff = (a1*cost1 - q*a2*cost2)/(1.+q)
    Xp_gen = generalized_Xp(a1*np.sqrt(1.-cost1**2)*np.cos(phi1),
                            a1*np.sqrt(1.-cost1**2)*np.sin(phi1),
                            a2*np.sqrt(1.-cost2**2)*np.cos(phi2),
                            a2*np.sqrt(1.-cost2**2)*np.sin(phi2),
                            q)
    
    # Package the set of parameters expected by our network
    param_vector = np.array([Mc_det**(5./6.)/DL,
                              np.log(Mtot_det),
                              eta,
                              Xeff,
                              Xdiff,
                              Xp_gen,
                              params.cos_inc,
                              params.ra,
                              params.sin_dec,
                              params.pol])
    
    # Recenter, scale, and return
    return (param_vector.T-scaler_stats['mean'])/scaler_stats['std']

def gen_found_injections(p_det_emulator,input_transformation,ntotal,batch_size=1000):

    """
    Generates new sets of "found" injections drawn from the O3b BBH injected distribution, labeled
    according to our neural network P_det emulator.

    Parameters
    ----------
    p_det_emulator : `tf.keras.models.Sequential` or `ANNaverage`
        The network (or list of networks) to be used for prediction
    input_transformation : `func`
        A function transforming and rescaling the parameters returned by `draw_new_injections` into the space
        expected by `p_det_emulator`. See e.g. `input_transform`
    ntotal : `int`
        The total number of requested found injections
    batch_size : `int`
        Size of individual batches in which proposed injections will be drawn and evaluated (default 1000)
    """

    # Loop until we have the desired number of found injections
    nfound = 0
    min_pdet = 1
    while nfound<=ntotal:

        # Draw new injections
        new_draws = draw_new_injections(batch_size=batch_size)

        # Transform to the expected parameter space
        rescaled_input_parameters = input_transformation(new_draws)

        # Evaluate detection probabilities
        p_det_predictions = p_det_emulator.predict(rescaled_input_parameters,verbose=0).reshape(-1)
        new_draws['p_det'] = p_det_predictions
        min_new_pdet = min(p_det_predictions)
        if min_new_pdet<min_pdet:
            min_pdet = min_new_pdet

        # Probabilistically label "found" injections according to the above probabilities
        random_draws = np.random.random(batch_size)
        found = new_draws[random_draws<p_det_predictions]

        # Record found injections
        # If these are our first ones, start a dataframe
        if nfound==0:
            all_found = found

        # Otherwise, append to existing data frame
        else:
            all_found = pd.concat([all_found,found],ignore_index=True)

        # Iterate counter
        print(len(all_found),min_new_pdet)
        nfound += len(found)

    return all_found

if __name__=="__main__":

    ann1 = build_ann(input_shape=10,layer_width=64,hidden_layers=3,lr=1e-4,leaky_alpha=0.001)
    ann2 = build_ann(input_shape=10,layer_width=64,hidden_layers=3,lr=1e-4,leaky_alpha=0.001)
    ann3 = build_ann(input_shape=10,layer_width=64,hidden_layers=3,lr=1e-4,leaky_alpha=0.001)
    ann1.load_weights('./../data/trained_models/job_21_weights.hdf5')
    ann2.load_weights('./../data/trained_models/job_72_weights.hdf5')
    ann3.load_weights('./../data/trained_models/job_97_weights.hdf5')
    ann_average = ANNaverage([ann1,ann2,ann3])

    with open('./../data/trained_models/job_00_scaler.json','r') as jf:
        scaler_stats = json.load(jf)

    partial_input_transform = partial(input_transform,scaler_stats=scaler_stats)

    gen_found_injections(ann_average,partial_input_transform,1000)
