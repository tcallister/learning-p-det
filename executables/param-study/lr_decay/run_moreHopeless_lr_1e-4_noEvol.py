import pandas as pd
import numpy as np
import sys
sys.path.append('./../../code/')
from training_routines import *
from diagnostics import *
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
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

def run_training(output):

    train_data = pd.read_hdf('./../../data/training_data.hdf')
    val_data = pd.read_hdf('./../../data/validation_data.hdf')
    hopeless_data1 = pd.read_hdf("./../../data/hopeless.hdf")
    hopeless_data2 = pd.read_hdf("./../../data/hopeless_02.hdf")

    hopeless_data = hopeless_data1.append(hopeless_data2)
    hopeless_data.distance = hopeless_data.distance/1000.
    train_data = shuffle(hopeless_data.append(train_data))

    addDerived(train_data)
    addDerived(val_data)

    train_input,train_output = input_output(train_data)
    val_input,val_output = input_output(val_data)

    input_sc = StandardScaler()
    input_sc.fit(train_input)
    train_input_scaled = input_sc.transform(train_input)
    val_input_scaled = input_sc.transform(val_input)

    ann = build_ann(input_shape=10,layer_width=64,hidden_layers=3,lr=1e-4,leaky_alpha=0.001)
    history = ann.fit(train_input_scaled,
                      train_output,
                      batch_size = 8,
                      epochs = 50,
                      validation_data = (val_input_scaled,val_output),
                      verbose=0)

    stats = {'loss':history.history['loss'],
            'val_loss':history.history['val_loss']}

    with open(output,'w') as jf:
        json.dump(stats,jf)

if __name__=="__main__":

    output = sys.argv[1]
    run_training(output)
    
