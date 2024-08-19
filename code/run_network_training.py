import pandas as pd
import numpy as np
import sys
sys.path.append('/home/tcallister/repositories/learning-p-det/code/')
from training_routines import *
from diagnostics import *
from draw_new_injections import draw_new_injections
from utilities import load_training_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
import matplotlib.pyplot as plt
import json
import pickle

# Check for proper GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def addDerived(data):

    """
    Function to add derived quantities to training injections, to be used
    as input parameters to neural network. Note that this function will modify
    `data` in place.

    Parameters
    ----------
    data : `pandas.DataFrame`
        Set of injections, as returned by `utilities.load_training_data`.

    Returns
    -------
    None
    """

    data['amp_factor_a'] = np.log((data.chirp_mass_detector**(5./6.)/data.luminosity_distance*((1.+data.cos_inclination**2)/2))**2)
    data['amp_factor_b'] = np.log((data.chirp_mass_detector**(5./6.)/data.luminosity_distance*data.cos_inclination)**2)
    data['log_m1'] = np.log(data.m1_detector)
    data['log_Mc'] = np.log(data.chirp_mass_detector)
    data['log_Mtot'] = np.log(data.total_mass_detector)
    data['log_d'] = np.log(data.luminosity_distance)
    data['scaled_eta'] = np.log(0.25-data.eta)
    data['sin_declination'] = np.sin(data.declination)
    data['abs_cos_inclination'] = np.abs(data.cos_inclination)
    data['one_plus_z'] = 1.+data.redshift
    data['cos_pol'] = np.cos(data.polarization%np.pi)
    data['sin_pol'] = np.sin(data.polarization%np.pi)
    data['m1_source'] = data.m1_detector/(1.+data.redshift)
    data['Xdiff_PN'] = (data['a1']*data['cost1'] - data['a2']*data['cost2'])/2.


# List of feature names that will be extracted and used as direct inputs
# for neural network
feature_names = ['amp_factor_a',
                       'amp_factor_b',
                       'chirp_mass_detector',
                       'total_mass_detector',
                       'eta',
                       'q',
                       'luminosity_distance',
                       'right_ascension',
                       'sin_declination',
                       'abs_cos_inclination',
                       'sin_pol',
                       'cos_pol',
                       'Xeff',
                        'Xdiff_PN',
                        'Xp_gen']
    

def input_output(data):

    """
    Helper function to split dataframe into input and output columns, using
    the `feature_names` list defined above.

    Parameters
    ----------
    data : `pandas.DataFrame`
        Set of injections, as returned by `utilities.load_training_data` and
        possibly acted on via `addDerived` above.

    Returns
    -------
    input_data : `numpy.array`
        Array of input features extracted from injection data.
    output_data : `pandas.DataFrame`
        Column of missed/found labels
    """

    input_data = data[feature_names].values
    output_data = data['detected']
    
    return input_data,output_data


def run_training(output, rng_seed):

    """
    Workhorse function that manages training of neural network.
    This function gathers data, launches network training, and
    creates various post-processing plots and diagnostics.

    Parameters
    ----------
    output : `str`
        Prefix that will be prepended to output files.
        Should include filepath, e.g. "output/path/runName_"
    rng_seed: `int`
        Random seed to be used when selecting subsets of training
        data

    Returns
    -------
    None 
    """

    # Instantiate neural network wrapper class
    nn = NeuralNetworkWrapper(
        input_shape=15,
        layer_width=192,
        hidden_layers=4,
        activation='LeakyReLU',
        leaky_alpha=1e-3,
        lr=2e-4,
        loss=NegativeLogLikelihoodAugmented,
        output_bias=np.log(1e-3),
        addDerived = addDerived,
        feature_names = feature_names)

    # Get data and load training data
    train_data,val_data = load_training_data(
        '/project/kicp/tcallister/learning-p-det-data/input_data/',
        n_bbh = 90000,
        n_bbh_hopeless = 180000,
        n_bbh_certain = 180000,
        n_bns = 90000,
        n_bns_hopeless = 180000,
        n_bns_certain = 180000,
        n_nsbh = 90000,
        n_nsbh_hopeless = 180000,
        n_nsbh_certain = 180000,
        n_combined_hopeless = 300000,
        rng_key=rng_seed)

    nn.prepare_data(1024,
                     train_data,
                     val_data)


    # Draw binaries from reference distributions, used to guide recovered
    # detection efficiencies during training.
    # First, specify a BBH population consistent with astrophysical observations
    pop = {'min_m1':5,
           'max_m1':100,
           'alpha_m1':-3,
           'min_m2':2,
           'max_m2':100,
           'alpha_m2':1,
           'max_a1':0.998,
           'max_a2':0.998,
           'zMax':1.9,
           'kappa':4.,
           'conditional_mass':True}

    # Draw and store 2e5 events from this population, and tell the NN
    # what the target detection efficiency should be
    nn.draw_from_reference_population(
           pop,
           200000,
           0.00038360609879489505)

    # Injected BBH distribution
    pop2 = {'min_m1':2,
            'max_m1':100,
            'alpha_m1':-2.35,
            'min_m2':2,
            'max_m2':100,
            'alpha_m2':1,
            'max_a1':0.998,
            'max_a2':0.998,
            'zMax':1.9,
            'kappa':1.,
            'conditional_mass':True}

    nn.draw_from_reference_population(
            pop2,
            200000,
            0.0011009853783046646)

    # Injected NSBH distribution
    pop3 = {'min_m1':2.5,
            'max_m1':60.,
            'alpha_m1':-2.35,
            'min_m2':1.,
            'max_m2':2.5,
            'alpha_m2':0.,
            'max_a1':0.998,
            'max_a2':0.4,
            'zMax':0.25,
            'kappa':0.,
            'conditional_mass':False}

    nn.draw_from_reference_population(
            pop3,
            10000,
            0.011199316130953239)

    # Injected BNS distribution
    pop4 = {'min_m1':1.0,
            'max_m1':2.5,
            'alpha_m1':0.0,
            'min_m2':1.0,
            'max_m2':2.5,
            'alpha_m2':0.,
            'max_a1':0.4,
            'max_a2':0.4,
            'zMax':0.15,
            'kappa':0.,
            'conditional_mass':False}

    nn.draw_from_reference_population(
            pop4,
            10000,
            0.016052332309165637)

    # Loss function parameters
    beta = 0.45
    nn.train_model(1000,beta)

    # Save preprocessing scaler and trained weights
    pickle.dump(nn.input_scaler, open("{0}_input_scaler.pickle".format(output), "wb"))
    nn.model.save_weights('{0}_weights.hdf5'.format(output))

    # Parameters for which to perform ks test
    parameters_to_check = ['m1_source', 'm1_detector','chirp_mass_detector',
                           'total_mass_detector', 'q', 'Xeff', 'Xdiff_PN',
                           'Xp_gen', 'redshift', 'luminosity_distance',
                           'log_d', 'cos_inclination']

    # Perform KS test comparison for each source class
    bbh_ks_results = ks_test(nn.model,
                             nn.addDerived,
                             nn.feature_names,
                             nn.input_scaler,
                             3000,
                             "/project/kicp/tcallister/learning-p-det-data/input_data/bbh_training_data.hdf",
                             "BBH",
                             output,
                             parameters_to_check=parameters_to_check)
    
    bns_ks_results = ks_test(nn.model,
                             nn.addDerived,
                             nn.feature_names,
                             nn.input_scaler,3000,
                             "/project/kicp/tcallister/learning-p-det-data/input_data/bns_training_data.hdf",
                             "BNS",
                             output,
                             parameters_to_check=parameters_to_check)
    
    nsbh_ks_results = ks_test(nn.model,
                              nn.addDerived,
                              nn.feature_names,
                              nn.input_scaler,
                              3000,
                              "/project/kicp/tcallister/learning-p-det-data/input_data/nsbh_training_data.hdf",
                              "NSBH",
                              output,
                              parameters_to_check=parameters_to_check)

    # Save results
    ks_results = {}
    ks_results['BBH'] = bbh_ks_results
    ks_results['BNS'] = bns_ks_results
    ks_results['NSBH'] = nsbh_ks_results

    # Generate additional found injections from a distribution closer
    # to the observed BBH distribution
    new_found,new_counts = gen_found_injections(
                                nn.model,
                                addDerived,
                                nn.feature_names,
                                nn.input_scaler,
                                1000,
                                jitted=False,
                                pop = {'min_m1':5,
                                    'max_m1':100,
                                    'alpha_m1':-3,
                                    'min_m2':2,
                                    'max_m2':100,
                                    'alpha_m2':1,
                                    'max_a1':0.998,
                                    'max_a2':0.998,
                                    'zMax':1.9,
                                    'kappa':3.,
                                    'conditional_mass':True})

    # Compute estimated detection efficiency and expected Poisson variance
    # Save result
    p_hat = len(new_found)/new_counts
    std_p_hat = np.sqrt(p_hat*(1.-p_hat)/new_counts)
    ks_results['alt_pop_1'] = {'det_efficiency':p_hat, 'std_det_efficiency':std_p_hat}

    # Save file
    with open("{0}_ks.json".format(output), 'w') as outfile:
        json.dump(ks_results, outfile)

if __name__=="__main__":

    output = sys.argv[1]
    key = int(sys.argv[2])
    print("Drawing data with key: ",key)
    run_training(output, key)
    
