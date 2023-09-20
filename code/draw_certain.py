import numpy as np
import pandas as pd
from utilities import *
from drawing_injections import draw_certain

certain = draw_certain(50000)

certain_params = pd.DataFrame(certain)
certain_params['distance'] = certain_params.distance/1000.  # Convert from Mpc to Gpc
certain_params['m1_detector'] = certain_params.mass1
certain_params['m2_detector'] = certain_params.mass2
certain_params['cos_inclination'] = np.cos(certain_params.inclination)
certain_params['q'] = certain_params.m2_detector/certain_params.m1_detector
certain_params['eta'] = certain_params.m1_detector*certain_params.m2_detector/(certain_params.m1_detector+certain_params.m2_detector)**2
certain_params['Mc_detector'] = certain_params.eta**(3./5.)*(certain_params.m1_detector+certain_params.m2_detector)
certain_params['Mtot_detector'] = (certain_params.m1_detector+certain_params.m2_detector)
certain_params['a1'] = np.sqrt(certain_params.spin1x**2 + certain_params.spin1y**2 + certain_params.spin1z**2)
certain_params['a2'] = np.sqrt(certain_params.spin2x**2 + certain_params.spin2y**2 + certain_params.spin2z**2)
certain_params['cost1'] = certain_params.spin1z/certain_params.a1
certain_params['cost2'] = certain_params.spin2z/certain_params.a2
certain_params['Xeff'] = (certain_params.spin1z + certain_params.q*certain_params.spin2z)/(1.+certain_params.q)
certain_params['Xdiff'] = (certain_params.spin1z - certain_params.q*certain_params.spin2z)/(1.+certain_params.q)
certain_params['Xp_gen'] = generalized_Xp(certain_params.spin1x,certain_params.spin1y,
                                           certain_params.spin2x,certain_params.spin2y,
                                           certain_params.q)

certain_params = certain_params[['m1_detector','m2_detector','distance',
                'cos_inclination','right_ascension','declination',
                'polarization','redshift',
                'q','eta','Mc_detector','Mtot_detector',
                'a1','a2','cost1','cost2',
                'Xeff','Xdiff','Xp_gen']]

both_detectors_off = np.random.random(len(certain_params))
certain_params['detected'] = np.ones(len(certain_params))
certain_params['detected'].iloc[both_detectors_off<0.25**2] = 0

certain_params.to_hdf('./../data/certain.hdf','certain')
