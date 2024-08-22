import numpy as np
from scipy.interpolate import UnivariateSpline
import astropy.units as u
import pandas as pd
from scipy.stats import ks_2samp
from draw_new_injections import gen_found_injections
import matplotlib.pyplot as plt


def ks_test(ann, addDerived, feature_names, input_sc, ndraws, reference_data, population_type,
            output_prefix, parameters_to_check=None, jitted=False):

    """
    Function used to compute KS test statistic p-values between distributions of found pipeline injections
    and distributions of detections as predicted by a neural network.
    Effectively a wrapper of `draw_new_injections.gen_found_injections`.
    Used in `run_network_training.py` to produce training diagnostics and summary plots.

    Parameters
    ----------
    ann : `tf.keras.models.Sequential`
        Network to be used for prediction
    addDerived : `func`
        Function to add any necessary derived features.
    feature_names : `list`
        List of feature names expected by network.
    input_sc : `sklearn.preprocessing.StandardScaler`
        Preprocessing scaler applied to features before passing to network.
    ndraws : `int`
        Number of found events to produce from target population.
    reference_data : `str`
        Filepath containing pipeline injections against which to compare
        neural network predictions.
    population_type : `str`
        String specifying population model from which to draw proposed events.
        See `draw_new_injections.gen_found_injections`.
    output_prefix : `str`
        String containing filepath and naming prefix, prepended to saved jpeg files
    parameters_to_check : `list`
        List of parameter names, specifies for which parameters KS test will be performed.
        If None, KS test is performed for all parameters.
    jitted : `bool`
        Boolean that tells `draw_new_injections.gen_found_injections` whether or not
        to expected a jitted function in place of a tensorflow network model.

    Returns
    -------
    ks_results : `dict`
        Dictionary containing KS test statistic p-values and estimated detection efficiencies.
    """

    # Draw new events
    found_events,nTrials = gen_found_injections(
        ann,
        addDerived,
        feature_names,
        input_sc,
        ndraws,
        10000,
        pop=population_type,
        jitted=jitted)

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
