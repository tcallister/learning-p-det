import numpy as np
from astropy.cosmology import Planck15,z_at_value
import astropy.units as u
import json
import pandas as pd
from functools import partial
from utilities import ANNaverage,generalized_Xp
from training_routines import build_ann

def draw_new_injections(batch_size=1000,
        min_m1=2.,
        max_m1=100.,
        alpha_m1=-2.35,
        min_m2=2.,
        max_m2=100.,
        alpha_m2=1.,
        max_a1=0.998,
        max_a2=0.998,
        zMax=1.9,
        kappa=1.,
        conditional_mass=False):

    """
    Function to draw new proposed injection parameters from the O3b injection distribution.
    See https://zenodo.org/record/5546676

    Parameters
    ----------
    min_m1 : `float`
        Minimum source-frame primary mass
    max_m1 : `float`
        Maximum source-frame primary mass
    alpha_m1 : `float`
        Power-law index on primary mass distribution
    min_m2 : `float`
        Minimum source-frame secondary mass
    max_m2 : `float`
        Maximum source-frame secondary mass
    alpha_m2 : `float`
        Power-law index on secondary mass distribution
    max_a1 : `float`
        Maximum spin magnitude on primary
    max_m2 : `float`
        Maximum spin magnitude on secondary
    zMax : `float`
        Maximum redshift
    kappa : `float`
        Power-law index governing evolution of the merger rate with redshift
    conditional_mass : `bool`
        If `True`, secondary mass distribution is normalized on the range `(min_m2,m1)`. If `False`, normalized on `(min_m2,max_m2)` and results are rejection sampled to enforce `m2<m1`. Default `False`

    Returns
    -------
    draws : `pandas.DataFrame`
        Dictionary containing randomly drawn component masses, Cartesian spins, sky position,
        source orientation, distance, and time. 
    """

    # Draw random cumulants in the m1 and m2 distribution and use the inverse CDF to translate into m1 and m2 values
    cs_m1 = np.random.random(batch_size)
    cs_m2 = np.random.random(batch_size)
    m1 = (min_m1**(1.+alpha_m1) + cs_m1*(max_m1**(1.+alpha_m1)-min_m1**(1.+alpha_m1)))**(1./(1.+alpha_m1))

    # Draw secondary mass, either independently of or conditioned on primary mass
    if conditional_mass==True:
        m2 = (min_m2**(1.+alpha_m2) + cs_m2*(m1**(1.+alpha_m2)-min_m2**(1.+alpha_m2)))**(1./(1.+alpha_m2))
    else:
        m2 = (min_m2**(1.+alpha_m2) + cs_m2*(max_m2**(1.+alpha_m2)-min_m2**(1.+alpha_m2)))**(1./(1.+alpha_m2))

    # Random isotropic spins
    a1 = np.random.uniform(low=0.,high=max_a1,size=batch_size)
    a2 = np.random.uniform(low=0.,high=max_a2,size=batch_size)
    cost1 = np.random.uniform(low=-1.,high=1.,size=batch_size)
    cost2 = np.random.uniform(low=-1.,high=1.,size=batch_size)
    phi1 = np.random.uniform(low=0.,high=2.*np.pi,size=batch_size)
    phi2 = np.random.uniform(low=0.,high=2.*np.pi,size=batch_size)

    # Manually construct CDF of redshift distribution
    z_grid = np.linspace(1e-3,zMax,10000)
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
    dec = np.arcsin(np.random.uniform(low=-1.,high=1.,size=batch_size))
    pol = np.random.uniform(low=0.,high=2.*np.pi,size=batch_size)

    # Derive mass parameters
    m1_det = m1*(1.+z)
    m2_det = m2*(1.+z)
    q = m2_det/m1_det
    eta = q/(1.+q)**2
    Mtot_det = m1_det+m2_det
    Mc_det = Mtot_det*eta**(3./5.)
    Xeff = (a1*cost1 + q*a2*cost2)/(1.+q)
    Xdiff = (a1*cost1 - q*a2*cost2)/(1.+q)
    sint1 = np.sqrt(1.-cost1**2)
    sint2 = np.sqrt(1.-cost2**2)
    Xp_gen = generalized_Xp(a1*sint1*np.cos(phi1),a1*sint1*np.sin(phi1),
                            a2*sint2*np.cos(phi2),a2*sint2*np.sin(phi2),
                            q)
    
    # Record and return
    draws = pd.DataFrame({'m1_src':m1,
                        'm2_src':m2,
                        'm1_detector':m1_det,
                        'm2_detector':m2_det,
                        'chirp_mass_detector':Mc_det,
                        'total_mass_detector':Mtot_det,
                        'q':q,
                        'eta':eta,
                        'a1':a1,
                        'a2':a2,
                        'cost1':cost1,
                        'cost2':cost2,
                        'Xeff':Xeff,
                        'Xdiff':Xdiff,
                        'Xp_gen':Xp_gen,
                        'phi1':phi1,
                        'phi2':phi2,
                        'redshift':z,
                        'luminosity_distance':DL,
                        'cos_inclination':cos_inc,
                        'right_ascension':ra,
                        'declination':dec,
                        'polarization':pol})

    # If we have not already conditioned on primary mass, reject all draws with m2>m1
    if conditional_mass==False:
        draws = draws[draws.m1_src>draws.m2_src]

    return draws

def gen_found_injections(p_det_emulator,addDerived_func,feature_names,scaler,ntotal,batch_size=1000,pop='BBH',verbose=False):

    """
    Generates new sets of "found" injections drawn from the O3b BBH injected distribution, labeled
    according to our neural network P_det emulator.

    Parameters
    ----------
    p_det_emulator : `tf.keras.models.Sequential` or `ANNaverage`
        The network (or list of networks) to be used for prediction
    addDerived_func : `func`
        Function serving to add any additional required derived parameters to the DataFrame returned by `draw_new_injections`.
        Should modify the DataFrame in-place.
    feature_names : `list`
        List of feature names expected by `p_det_emulator`. Should correspond to columns of DataFrame natively returned by
        `draw_new_injections` or added by `addDerived_func`
    scaler : `sklearn.preprocessing.StandardScaler` or `sklearn.preprocessing.QuantileTransformer`
        Preprocessing scaler applied to features extracted from DataFrame.
    ntotal : `int`
        The total number of requested found injections
    batch_size : `int`
        Size of individual batches in which proposed injections will be drawn and evaluated (default 1000)
    pop : `string`
        One of `BBH`, `BNS`, or `NSBH`. Determines how secondary mass is drawn, in accordance with generation of 
        original O3 injections. Default `BBH`.
    verbose : `bool`
        If `True`, additional information will be printed to screen. Default `False`.

    Returns
    -------
    all_found : `pandas.DataFrame`
        Set of new injections labeled as "found"
    nTrials : `int`
        Total number of injections drawn in order to yield the set `all_found` of found injections
    """

    # Loop until we have the desired number of found injections
    nfound = 0
    nTrials = 0
    min_pdet = 1
    while nfound<=ntotal:

        # Draw new injections
        if pop=='BBH':

            # BBH params
            min_m1=2.
            max_m1=100.
            alpha_m1=-2.35
            min_m2=2.
            max_m2=100.
            alpha_m2=1.
            max_a1=0.998
            max_a2=0.998
            zMax=1.9
            kappa=1.
            conditional_mass=True

        elif pop=='NSBH':

            # NSBH params
            min_m1=2.5
            max_m1=60.
            alpha_m1=-2.35
            min_m2=1.
            max_m2=2.5
            alpha_m2=0.
            max_a1=0.998
            max_a2=0.4
            zMax=0.25
            kappa=0.
            conditional_mass=False

        elif pop=='BNS':

            # BNS params
            min_m1=1.0
            max_m1=2.5
            alpha_m1=1.0
            min_m2=1.0
            max_m2=2.5
            alpha_m2=0.
            max_a1=0.4
            max_a2=0.4
            zMax=0.15
            kappa=0.
            conditional_mass=False

        # Take draws
        new_draws = draw_new_injections(batch_size=batch_size,
            min_m1=min_m1,
            max_m1=max_m1,
            alpha_m1=alpha_m1,
            min_m2=min_m2,
            max_m2=max_m2,
            alpha_m2=alpha_m2,
            max_a1=max_a1,
            max_a2=max_a2,
            zMax=zMax,
            kappa=kappa,
            conditional_mass=conditional_mass)

        addDerived_func(new_draws)
        new_draws_features = new_draws[feature_names]

        # Transform to the expected parameter space
        rescaled_input_parameters = scaler.transform(new_draws_features)

        # Evaluate detection probabilities
        p_det_predictions = p_det_emulator.predict(rescaled_input_parameters,verbose=0).reshape(-1)
        new_draws['p_det'] = p_det_predictions
        min_new_pdet = min(p_det_predictions)
        if min_new_pdet<min_pdet:
            min_pdet = min_new_pdet

        # Probabilistically label "found" injections according to the above probabilities
        random_draws = np.random.random(len(new_draws))
        found = new_draws[random_draws<p_det_predictions].copy()

        # Record found injections
        # If these are our first ones, start a dataframe
        if nfound==0:
            all_found = found

        # Otherwise, append to existing data frame
        else:
            all_found = pd.concat([all_found,found],ignore_index=True)

        # Iterate counter
        if verbose:
            print(len(all_found),min_new_pdet)
        nfound += len(found)
        nTrials += len(new_draws)

    return all_found,nTrials


