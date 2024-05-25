import numpy as np
from astropy.cosmology import Planck15
import astropy.units as u
import pandas as pd
import os
dirname = os.path.dirname(__file__)

def getSyntheticInjections():

    newInjections = pd.read_pickle('/home/tcallister/repositories/learning-p-det/code/demo_injections_for_inference.pickle') 

    injectionDict = {}
    injectionDict['a1'] = newInjections['a1'].values
    injectionDict['a2'] = newInjections['a2'].values
    injectionDict['cost1'] = newInjections['cost1'].values
    injectionDict['cost2'] = newInjections['cost2'].values
    injectionDict['m1'] = newInjections['m1_src'].values
    injectionDict['m2'] = newInjections['m2_src'].values
    injectionDict['z'] = newInjections['redshift'].values
    injectionDict['dVdz'] = 4.*np.pi*Planck15.differential_comoving_volume(injectionDict['z']).to(u.Gpc**3/u.sr).value
    injectionDict['nTrials'] = 70140000 #44070000.

    # Draw probabilities
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

    p_m1 = (1.+alpha_m1)*injectionDict['m1']**alpha_m1/(max_m1**(1.+alpha_m1) - min_m1**(1.+alpha_m1))
    p_m2 = (1.+alpha_m2)*injectionDict['m2']**alpha_m2/(injectionDict['m1']**(1.+alpha_m2) - min_m2**(1.+alpha_m2))

    z_grid = np.linspace(0,1.9,5000)
    dVdz_grid = 4.*np.pi*Planck15.differential_comoving_volume(z_grid).to(u.Gpc**3/u.sr).value
    p_z_unnormed = dVdz_grid*(1.+z_grid)**(kappa-1.)
    p_z_grid = p_z_unnormed/np.trapz(p_z_unnormed,z_grid)
    p_z = np.interp(injectionDict['z'],z_grid,p_z_grid)

    injectionDict['p_draw_m1m2z'] = p_m1*p_m2*p_z
    injectionDict['p_draw_a1a2cost1cost2'] = 1./4.
    return injectionDict

def getInjections():

    """
    Function to load and preprocess found injections for use in numpyro likelihood functions.

    Parameters
    ----------
    sample_limit : int
        If reweighting to a different reference population, number of found injections to use. No effect otherwise. Default 1e4
    reweight : bool
        If `True`, reweight injections to the reference population defined in `weighting_function`. Default `False`
    weighting_function : func
        Function defining new reference population to reweight to. No effect if `reweight=True`. Default `reweighting_function_arlnm1_q`

    Returns
    ------- 
    injectionDict : dict
        Dictionary containing found injections and associated draw probabilities, for downstream use in hierarchical inference
    """

    # Load injections
    injectionFile = os.path.join(dirname,"../input/injectionDict_FAR_1_in_1_BBH.pickle")
    injectionDict = np.load(injectionFile,allow_pickle=True)

    # Convert all lists to numpy arrays
    for key in injectionDict:
        if key!='nTrials':
            injectionDict[key] = np.array(injectionDict[key])

    return injectionDict

def getSamples(sample_limit=2000,bbh_only=True):

    """
    Function to load and preprocess BBH posterior samples for use in numpyro likelihood functions.
    
    Parameters
    ----------
    sample_limit : int
        Number of posterior samples to retain for each event, for use in population inference (default 2000)
    bbh_only : bool
        If True, will exclude samples for BNS, NSBH, and mass-gap events (default True)
    reweight : bool
        If True, reweight posteriors to the reference population defined in `weighting_function`. Default `False`
    weighting_function : func
        Function defining new reference population to reweight to. No effect if `reweight=True`. Default `reweighting_function_arlnm1_q`

    Returns
    -------
    sampleDict : dict
        Dictionary containing posterior samples, for downstream use in hierarchical inference
    """

    # Load dictionary with preprocessed posterior samples
    sampleFile = os.path.join(dirname,"./../../autoregressive-bbh-inference/input/sampleDict_FAR_1_in_1_yr.pickle")
    sampleDict = np.load(sampleFile,allow_pickle=True)

    # Remove non-BBH events, if desired
    non_bbh = ['GW170817','S190425z','S190426c','S190814bv','S190917u','S200105ae','S200115j']
    if bbh_only:
        for event in non_bbh:
            print("Removing ",event)
            sampleDict.pop(event)

    # Loop across events
    for event in sampleDict:

        # Uniform draw weights
        draw_weights = np.ones(sampleDict[event]['m1'].size)/sampleDict[event]['m1'].size
        draw_weights[sampleDict[event]['m1']>100] = 0
        sampleDict[event]['downselection_Neff'] = np.sum(draw_weights)**2/np.sum(draw_weights**2)

        # Randomly downselect to the desired number of samples       
        inds_to_keep = np.random.choice(np.arange(sampleDict[event]['m1'].size),size=sample_limit,replace=True,p=draw_weights/np.sum(draw_weights))
        for key in sampleDict[event].keys():
            if key!='downselection_Neff':
                sampleDict[event][key] = sampleDict[event][key][inds_to_keep]

    return sampleDict

if __name__=="__main__":

    test = getInjections()
    print(test['m1'].size)
