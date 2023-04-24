import pandas as pd
import numpy as np
import sys
sys.path.append('./../code/')
from training_routines import *
from diagnostics import *
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
import matplotlib.pyplot as plt
import json

def addDerived(data):

    data['Mc_over_d'] = data.Mc_detector**(5./6.)/data.distance
    data['log_Mtot'] = np.log(data.Mc_detector/data.eta**(3./5.)) 
    data['sin_declination'] = np.sin(data.declination)

def input_output(data):

    input_data = data[['Mc_over_d','log_Mtot','eta','Xeff','Xdiff','Xp_gen','cos_inclination','right_ascension','sin_declination','polarization']]
    output_data = data[['detected']]
    return input_data,output_data

def new_scheduler(epoch, lr):
    if epoch%10==0 and epoch:
        return lr*tf.math.exp(-1.0)
    else:
        return lr

def run_training(output):

    train_data = pd.read_hdf('./../data/training_data.hdf')
    val_data = pd.read_hdf('./../data/validation_data.hdf')

    addDerived(train_data)
    addDerived(val_data)

    train_input,train_output = input_output(train_data)
    val_input,val_output = input_output(val_data)

    input_sc = StandardScaler()
    input_sc.fit(train_input)
    train_input_scaled = input_sc.transform(train_input)
    val_input_scaled = input_sc.transform(val_input)

    scaler_params = {'mean':list(input_sc.mean_),'std':list(input_sc.scale_)}
    with open('{0}/scaler.json'.format(output),'w') as jf:
        json.dump(scaler_params,jf)    

    ann = build_ann(input_shape=10,layer_width=64,hidden_layers=6,lr=1e-4,leaky_alpha=0.001)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=output+"/model_checkpoints/epoch_{epoch:02d}.hdf5",
            save_weights_only=True,
            save_best_only=False)

    callbacks = [tf.keras.callbacks.LearningRateScheduler(new_scheduler, verbose=0),model_checkpoint_callback]
    history = ann.fit(train_input_scaled,
                      train_output,
                      batch_size = 8,
                      epochs = 50,
                      validation_data = (val_input_scaled,val_output),
                      callbacks=callbacks,
                      verbose=0)

    stats = {'loss':history.history['loss'],
            'val_loss':history.history['val_loss']}

    with open("{0}/output_stats.json".format(output),'w') as jf:
        json.dump(stats,jf)

if __name__=="__main__":

    output = sys.argv[1]
    run_training(output)
    
