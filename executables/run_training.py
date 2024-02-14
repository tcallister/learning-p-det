import pandas as pd
import numpy as np
import sys
sys.path.append('./../code/')
from training_routines import *
from diagnostics import *
from utilities import load_training_data
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
import matplotlib.pyplot as plt
import json
import pickle

def addDerived(data):

    data['amp_factor_a'] = log((data.chirp_mass_detector**(5./6.)/data.luminosity_distance*((1.+data.cos_inclination**2)/2))**2)
    data['amp_factor_b'] = log((data.chirp_mass_detector**(5./6.)/data.luminosity_distance*data.cos_inclination)**2)
    data['log_m1'] = np.log(data.m1_detector)
    data['log_Mc'] = np.log(data.chirp_mass_detector)
    data['log_Mtot'] = np.log(data.total_mass_detector)
    data['sin_declination'] = np.sin(data.declination)
    data['cos_pol'] = np.cos(data.polarization)
    data['sin_pol'] = np.sin(data.polarization)

def input_output(data):

    input_data = data[['amp_factor_a',
                       'amp_factor_b',
                       'log_m1',
                       'log_Mc',
                       'log_Mtot',
                       'q',
                       'log_d',
                       'right_ascension',
                       'sin_declination',
                       'cos_inclination',
                       'sin_pol',
                       'cos_pol',
                       'Xeff']]

    output_data = data[['detected']]
    return input_data,output_data

def new_scheduler(epoch, lr):
    if epoch%10==0 and epoch:
        return lr*tf.math.exp(-0.25)
    else:
        return lr

def run_training(output):

    train_data,val_data = load_training_data('./../data/',
                                            n_bbh = 50000,
                                            n_bns = 50000,
                                            n_nsbh = 50000,
                                            n_hopeless = 100000,
                                            n_certain = 50000,
                                            rng_key=11)

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
    pickle.dump(input_sc, open("{0}_input_scaler.pickle".format(output),"wb"))

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        verbose=1,
        patience=30,
        mode='min',
        restore_best_weights=True)

    initial_bias = np.log(train_data[train_data.detected==1].shape[0]/train_data.shape[0])
    ann = build_ann(input_shape=13,
                    layer_width=128,
                    hidden_layers=4,
                    activation='ReLU',
                    lr=1e-4,
                    dropout=True,
                    dropout_rate=0.5,
                    output_bias=initial_bias)

    callbacks = [early_stopping]
    history = ann.fit(train_input_scaled,
                      train_output,
                      batch_size = 32,
                      epochs = 400,
                      validation_data = (val_input_scaled,val_output),
                      callbacks=callbacks,
                      verbose=1)

    with open("{0}_history.json".format(output),'w') as jf:
        json.dump(history.history,jf)

    ann.save_weights('{0}_weights.hdf5'.format(output))

if __name__=="__main__":

    output = sys.argv[1]
    run_training(output)
    
