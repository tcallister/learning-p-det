import numpy as np
from astropy.cosmology import Planck15
import astropy.units as u
from population_model import massModel
import os,sys
dirname = os.path.dirname(__file__)

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
