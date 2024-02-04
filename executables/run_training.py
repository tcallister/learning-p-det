import pandas as pd
import numpy as np
import sys
sys.path.append('./../code/')
from training_routines import *
from diagnostics import *
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
import matplotlib.pyplot as plt
import json
import pickle

def addDerived(data):

    data['amp_factor_a'] = ((data.chirp_mass_detector**(5./6.)/data.luminosity_distance*((1.+data.cos_inclination**2)/2))**2)
    data['amp_factor_b'] = ((data.chirp_mass_detector**(5./6.)/data.luminosity_distance*data.cos_inclination)**2)
    data['sin_declination'] = np.sin(data.declination)

def input_output(data):

    input_data = data[['amp_factor_a',
                       'amp_factor_b',
                       'm1_detector',
                       'chirp_mass_detector',
                       'total_mass_detector',
                       'eta',
                       'luminosity_distance',
                       'redshift',
                       'right_ascension',
                       'sin_declination',
                       'Xeff',
                       'polarization']]

    output_data = data[['detected']]
    return input_data,output_data

def new_scheduler(epoch, lr):
    if epoch%10==0 and epoch:
        return lr*tf.math.exp(-0.25)
    else:
        return lr

def run_training(output):

    train_data = pd.read_hdf('./../data/bbh_training_data.hdf').sample(20000)
    val_data = pd.read_hdf('./../data/bbh_validation_data.hdf').sample(5000)
    bns_train_data = pd.read_hdf('./../data/bns_training_data.hdf').sample(20000)
    bns_val_data = pd.read_hdf('./../data/bns_validation_data.hdf').sample(5000)
    nsbh_train_data = pd.read_hdf('./../data/nsbh_training_data.hdf').sample(20000)
    nsbh_val_data = pd.read_hdf('./../data/nsbh_validation_data.hdf').sample(5000)
    official_hopeless_data = pd.read_hdf('./../data/rpo3-without-hopeless-cut-formatted.hdf').sample(60000)
    official_hopeless_data,val_official_hopeless_data = train_test_split(official_hopeless_data,train_size=0.8)

    train_data = shuffle(train_data.append(official_hopeless_data).\
                         append(bns_train_data).\
                         append(nsbh_train_data)\
                        )
    val_data = shuffle(val_data.append(val_official_hopeless_data).\
                       append(bns_val_data).\
                       append(nsbh_val_data)\
                       )

    # Add derived parameters
    addDerived(train_data)
    addDerived(val_data)

    # Split off inputs and outputs
    train_input,train_output = input_output(train_data)
    val_input,val_output = input_output(val_data)

    # Define quantile transformer and scale inputs
    input_sc = QuantileTransformer(output_distribution='normal')
    input_sc.fit(train_input)
    train_input_scaled = input_sc.transform(train_input)
    val_input_scaled = input_sc.transform(val_input)

    # Save preprocessing scaler
    pickle.dump(input_sc, open("{0}_input_scaler.pickle","wb"))

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        verbose=1,
        patience=40,
        mode='min',
        restore_best_weights=True)

    initial_bias = np.log(train_data[train_data.detected==1].shape[0]/train_data.shape[0])
    ann = build_ann(input_shape=12,layer_width=64,hidden_layers=3,lr=3e-4,leaky_alpha=0.0001,output_bias=initial_bias)
    callbacks = [tf.keras.callbacks.LearningRateScheduler(new_scheduler, verbose=0),early_stopping]
    history = ann.fit(train_input_scaled,
                      train_output,
                      batch_size = 64,
                      epochs = 600,
                      validation_data = (val_input_scaled,val_output),
                      callbacks=callbacks,
                      verbose=1)

    with open("{0}_history.json".format(output),'w') as jf:
        json.dump(history.history,jf)

    ann.save_weights('{0}_weights.hdf5'.format(output))

if __name__=="__main__":

    output = sys.argv[1]
    run_training(output)
    
