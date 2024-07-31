import numpy as np
import os
dirname = os.path.dirname(__file__)


def getInjections():

    """
    Function to load and preprocess found injections for use in population
    inference with numpyro.

    Parameters
    ----------
    None

    Returns
    -------
    injectionDict : dict
        Dictionary containing found injections and associated draw
        probabilities, for downstream use in hierarchical inference
    """

    # Load injections
    injectionFile = os.path.join(dirname, "./../input/injectionDict_FAR_1_in_1_BBH.pickle")
    injectionDict = np.load(injectionFile, allow_pickle=True)

    # Convert all lists to numpy arrays
    for key in injectionDict:
        if key != 'nTrials':
            injectionDict[key] = np.array(injectionDict[key])

    return injectionDict


def getSamples(sample_limit=2000, bbh_only=True, O3_only=True):

    """
    Function to load and preprocess BBH posterior samples for use in
    hierarchical population inference.

    Parameters
    ----------
    sample_limit : int
        Number of posterior samples to retain for each event, for use in
        population inference (default 2000)
    bbh_only : bool
        If True, will exclude samples for BNS, NSBH, and mass-gap events
        (default True)
    O3_only : bool
        If true, will include only events from the O3 observing run
        (default True)

    Returns
    -------
    sampleDict : dict
        Dictionary containing posterior samples, for downstream use in
        hierarchical inference
    """

    # Load dictionary with preprocessed posterior samples
    sampleFile = os.path.join(dirname, "./../input/sampleDict_FAR_1_in_1_yr.pickle")
    sampleDict = np.load(sampleFile, allow_pickle=True)

    # Remove non-BBH events, if desired
    non_bbh = ['GW170817', 'S190425z', 'S190426c', 'S190814bv',
               'S190917u', 'S200105ae', 'S200115j']

    if bbh_only:
        for event in non_bbh:
            print("Removing ", event)
            sampleDict.pop(event)

    # Remove non-O3 events, if desired
    # Conveniently, O1 and O2 events are named "GW..." while O3 events
    # are named "S..."
    if O3_only:
        events = list(sampleDict.keys())
        for event in events:
            if event[0] == "G":
                print("Removing ", event)
                sampleDict.pop(event)

    # Loop across events
    for event in sampleDict:

        # Uniform draw weights
        nPoints = sampleDict[event]['m1'].size
        draw_weights = np.ones(nPoints)/nPoints
        draw_weights[sampleDict[event]['m1'] > 100] = 0
        sampleDict[event]['downselection_Neff'] = np.sum(draw_weights)**2/np.sum(draw_weights**2)

        # Randomly downselect to the desired number of samples
        inds_to_keep = np.random.choice(
                            np.arange(sampleDict[event]['m1'].size),
                            size=sample_limit,
                            replace=True,
                            p=draw_weights/np.sum(draw_weights))

        for key in sampleDict[event].keys():
            if key != 'downselection_Neff':
                sampleDict[event][key] = sampleDict[event][key][inds_to_keep]

    print(sampleDict.keys())
    print(len(sampleDict))

    return sampleDict


if __name__ == "__main__":

    injs = getInjections()
    print(len(injs['m1']))
    getSamples()
