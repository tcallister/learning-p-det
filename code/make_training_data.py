import h5py
import numpy as np
import pandas as pd
from utilities import generalized_Xp
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Load injections
injection_data = h5py.File('./../input/endo3_bbhpop-LIGO-T2100113-v12.hdf5','r')

# Read injection parameters
injectionData = pd.DataFrame()
injectionData['m1_detector'] = np.array(injection_data['injections']['mass1'][()],dtype='float64')
injectionData['m2_detector'] = np.array(injection_data['injections']['mass2'][()],dtype='float64')
injectionData['dist'] = injection_data['injections']['distance'][()]/1000.
injectionData['cos_inc'] = np.cos(injection_data['injections']['inclination'])
injectionData['ra'] = injection_data['injections']['right_ascension']
injectionData['dec'] = injection_data['injections']['declination']
injectionData['pol'] = injection_data['injections']['polarization']
injectionData['z'] = injection_data['injections']['redshift'][()]

# Some derived mass parameters
injectionData['q'] = injectionData.m2_detector/injectionData.m1_detector
injectionData['eta'] = injectionData.m1_detector*injectionData.m2_detector/(injectionData.m1_detector+injectionData.m2_detector)**2
injectionData['Mc_detector'] = injectionData.eta**(3./5.)*(injectionData.m1_detector+injectionData.m2_detector)
injectionData['Mtot_detector'] = (injectionData.m1_detector+injectionData.m2_detector)

# And some derived spin parameters
s1x = np.array(injection_data['injections']['spin1x'])
s1y = np.array(injection_data['injections']['spin1y'])
s1z = np.array(injection_data['injections']['spin1z'])
s2x = np.array(injection_data['injections']['spin2x'])
s2y = np.array(injection_data['injections']['spin2y'])
s2z = np.array(injection_data['injections']['spin2z'])
injectionData['a1'] = np.sqrt(s1x**2 + s1y**2 + s1z**2)
injectionData['a2'] = np.sqrt(s2x**2 + s2y**2 + s2z**2)
injectionData['cost1'] = s1z/injectionData.a1
injectionData['cost2'] = s2z/injectionData.a2
injectionData['Xeff'] = (s1z + injectionData.q*s2z)/(1.+injectionData.q)
injectionData['Xdiff'] = (s1z - injectionData.q*s2z)/(1.+injectionData.q)
injectionData['Xp_gen'] = generalized_Xp(s1x,s1y,s2x,s2y,injectionData.q)

# Read out false alarm rates
far_gstlal = injection_data['injections']['far_gstlal'][()] 
far_pycbc = injection_data['injections']['far_pycbc_bbh'][()] 
far_pycbc_hyper = injection_data['injections']['far_pycbc_hyperbank'][()] 
far_mbta = injection_data['injections']['far_mbta'][()] 
far_cwb = injection_data['injections']['far_cwb'][()] 

# Take minimum FAR across all searches and assign detection labels
far_min = np.min(np.stack([far_gstlal,far_pycbc,far_pycbc_hyper,far_mbta,far_cwb]),axis=0)
injectionData['detected'] = np.where(far_min<1,1,0)

injectionData = shuffle(injectionData)
train_data,val_data = train_test_split(injectionData,train_size=0.75)

train_data.to_hdf('./../data/training_data.hdf','train')
val_data.to_hdf('./../data/validation_data.hdf','validate')
