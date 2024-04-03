import h5py
import numpy as np
import pandas as pd
from utilities import generalized_Xp
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Load injections
injection_data = h5py.File('./../data/rpo3-test-distribution.hdf','r')

# Read injection parameters
injectionData = pd.DataFrame()
injectionData['m1_detector'] = np.array(injection_data['events']['mass1_detector'][()],dtype='float64')
injectionData['m2_detector'] = np.array(injection_data['events']['mass2_detector'][()],dtype='float64')
injectionData['luminosity_distance'] = injection_data['events']['luminosity_distance'][()]/1000. # Convert from Mpc to Gpc
injectionData['cos_inclination'] = np.cos(injection_data['events']['inclination'])
injectionData['right_ascension'] = injection_data['events']['right_ascension']
injectionData['declination'] = injection_data['events']['declination']
injectionData['polarization'] = injection_data['events']['polarization']
injectionData['redshift'] = injection_data['events']['z'][()]

# Some derived mass parameters
injectionData['q'] = injectionData.m2_detector/injectionData.m1_detector
injectionData['eta'] = injectionData.m1_detector*injectionData.m2_detector/(injectionData.m1_detector+injectionData.m2_detector)**2
injectionData['chirp_mass_detector'] = injectionData.eta**(3./5.)*(injectionData.m1_detector+injectionData.m2_detector)
injectionData['total_mass_detector'] = (injectionData.m1_detector+injectionData.m2_detector)

# And some derived spin parameters
s1x = np.array(injection_data['events']['spin1x'])
s1y = np.array(injection_data['events']['spin1y'])
s1z = np.array(injection_data['events']['spin1z'])
s2x = np.array(injection_data['events']['spin2x'])
s2y = np.array(injection_data['events']['spin2y'])
s2z = np.array(injection_data['events']['spin2z'])
injectionData['a1'] = np.sqrt(s1x**2 + s1y**2 + s1z**2)
injectionData['a2'] = np.sqrt(s2x**2 + s2y**2 + s2z**2)
injectionData['cost1'] = s1z/injectionData.a1
injectionData['cost2'] = s2z/injectionData.a2
injectionData['Xeff'] = (s1z + injectionData.q*s2z)/(1.+injectionData.q)
injectionData['Xdiff'] = (s1z - injectionData.q*s2z)/(1.+injectionData.q)
injectionData['Xp_gen'] = generalized_Xp(s1x,s1y,s2x,s2y,injectionData.q)

injectionData = shuffle(injectionData)
injectionData.to_hdf('./../data/rpo3-test-distribution-formatted.hdf','train')
