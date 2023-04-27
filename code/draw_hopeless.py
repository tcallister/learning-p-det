import numpy as np
import pandas as pd
from drawing_injections import *
from utilities import *

hopeless,findable = draw_hopeless(1000)

hopeless_params = pd.DataFrame(hopeless)
hopeless_params['distance'] = hopeless_params.distance/1000.  # Convert from Mpc to Gpc
hopeless_params['m1_detector'] = hopeless_params.mass1
hopeless_params['m2_detector'] = hopeless_params.mass2
hopeless_params['cos_inclination'] = np.cos(hopeless_params.inclination)
hopeless_params['q'] = hopeless_params.m2_detector/hopeless_params.m1_detector
hopeless_params['eta'] = hopeless_params.m1_detector*hopeless_params.m2_detector/(hopeless_params.m1_detector+hopeless_params.m2_detector)**2
hopeless_params['Mc_detector'] = hopeless_params.eta**(3./5.)*(hopeless_params.m1_detector+hopeless_params.m2_detector)
hopeless_params['Mtot_detector'] = (hopeless_params.m1_detector+hopeless_params.m2_detector)
hopeless_params['a1'] = np.sqrt(hopeless_params.spin1x**2 + hopeless_params.spin1y**2 + hopeless_params.spin1z**2)
hopeless_params['a2'] = np.sqrt(hopeless_params.spin2x**2 + hopeless_params.spin2y**2 + hopeless_params.spin2z**2)
hopeless_params['cost1'] = hopeless_params.spin1z/hopeless_params.a1
hopeless_params['cost2'] = hopeless_params.spin2z/hopeless_params.a2
hopeless_params['Xeff'] = (hopeless_params.spin1z + hopeless_params.q*hopeless_params.spin2z)/(1.+hopeless_params.q)
hopeless_params['Xdiff'] = (hopeless_params.spin1z - hopeless_params.q*hopeless_params.spin2z)/(1.+hopeless_params.q)
hopeless_params['Xp_gen'] = generalized_Xp(hopeless_params.spin1x,hopeless_params.spin1y,
                                           hopeless_params.spin2x,hopeless_params.spin2y,
                                           hopeless_params.q)

hopeless_params = hopeless_params[['m1_detector','m2_detector','distance',
                'cos_inclination','right_ascension','declination',
                'polarization','redshift',
                'q','eta','Mc_detector','Mtot_detector',
                'a1','a2','cost1','cost2',
                'Xeff','Xdiff','Xp_gen']]
hopeless_params['detected'] = 0

hopeless_params.to_hdf('./../data/hopeless.hdf','hopeless')
