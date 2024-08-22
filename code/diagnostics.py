import numpy as np
from scipy.interpolate import UnivariateSpline
import astropy.units as u
import pandas as pd
from scipy.stats import ks_2samp
from draw_new_injections import gen_found_injections
import matplotlib.pyplot as plt


def ks_test(ann,addDerived,feature_names,input_sc,ndraws,reference_data,population_type,output_prefix,
            parameters_to_check=None,jitted=False):

    # Draw new events
    found_events,nTrials = gen_found_injections(ann,
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
