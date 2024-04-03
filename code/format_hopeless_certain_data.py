import h5py
import numpy as np
import pandas as pd
from utilities import generalized_Xp
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

"""
Script to format and prepare training data sets comprising "certain" detections (will be detected if one or more instruments are on) and "hopeless" detections (no chance of detection).
"""

def read_and_annotate(hfile,hkey='events'):

    """
    Helper function to load hdf file prepared by the `gwdistributions` and add necessary derived data.

    Parameters
    ----------
    hfile : `h5py.File`
        Opened `h5py.File` containing injection data

    Returns
    -------
    injectionData : `pandas.DataFrame`
        Data frame containing injection data and additional derived parameters
    """

    # Read injection parameters
    injectionData = pd.DataFrame()
    injectionData['m1_detector'] = np.array(hfile[hkey]['mass1_detector'][()],dtype='float64')
    injectionData['m2_detector'] = np.array(hfile[hkey]['mass2_detector'][()],dtype='float64')
    injectionData['luminosity_distance'] = hfile[hkey]['luminosity_distance'][()]/1000. # Convert from Mpc to Gpc
    injectionData['cos_inclination'] = np.cos(hfile[hkey]['inclination'])
    injectionData['right_ascension'] = hfile[hkey]['right_ascension']
    injectionData['declination'] = hfile[hkey]['declination']
    injectionData['polarization'] = hfile[hkey]['polarization']
    injectionData['redshift'] = hfile[hkey]['z'][()]

    # Some derived mass parameters
    injectionData['q'] = injectionData.m2_detector/injectionData.m1_detector
    injectionData['eta'] = injectionData.m1_detector*injectionData.m2_detector/(injectionData.m1_detector+injectionData.m2_detector)**2
    injectionData['chirp_mass_detector'] = injectionData.eta**(3./5.)*(injectionData.m1_detector+injectionData.m2_detector)
    injectionData['total_mass_detector'] = (injectionData.m1_detector+injectionData.m2_detector)

    # And some derived spin parameters
    s1x = np.array(hfile[hkey]['spin1x'])
    s1y = np.array(hfile[hkey]['spin1y'])
    s1z = np.array(hfile[hkey]['spin1z'])
    s2x = np.array(hfile[hkey]['spin2x'])
    s2y = np.array(hfile[hkey]['spin2y'])
    s2z = np.array(hfile[hkey]['spin2z'])
    injectionData['a1'] = np.sqrt(s1x**2 + s1y**2 + s1z**2)
    injectionData['a2'] = np.sqrt(s2x**2 + s2y**2 + s2z**2)
    injectionData['cost1'] = s1z/injectionData.a1
    injectionData['cost2'] = s2z/injectionData.a2
    injectionData['Xeff'] = (s1z + injectionData.q*s2z)/(1.+injectionData.q)
    injectionData['Xdiff'] = (s1z - injectionData.q*s2z)/(1.+injectionData.q)
    injectionData['Xp_gen'] = generalized_Xp(s1x,s1y,s2x,s2y,injectionData.q)

    # SNR info
    injectionData['snr_net'] = hfile[hkey]['snr_net']
    injectionData['observed_snr_net'] = hfile[hkey]['observed_snr_net']

    return injectionData

def mark_detections_certain(hfile,injectionData):

    """
    Simple helper script to annotate certain detections as found/missed

    Parameters
    ----------
    hfile : `str`
        Filepath to hdf file produced by `gwdistributions` containing certain injections.
    injectionData : `pandas.DataFrame`
        Data frame object returned by `read_and_annotate()`.
        Annotated in-place.

    """

    # Probabalitically determine if H1 and L1 were on/off
    H1_off = np.random.random(len(injectionData))<(1.-0.78)
    L1_off = np.random.random(len(injectionData))<(1.-0.78)

    # If both H1 and L1 are off, mark detection as missed
    injectionData.loc[:,'detected'] = np.where(H1_off*L1_off,0,1)

def mark_detections_hopeless(injectionData):

    """
    Simple helper script to mark all injections in a given dataset as missed.

    Parameters
    ----------
    injectionData : `pandas.DataFrame`
        Data frame object returned by `read_and_annotate()`.
        Annotated in-place with detections marked as missed.
    """

    # Mark as missed
    injectionData.loc[:,'detected'] = 0

def format_and_save_certain(hfile,output):

    """
    Wrapper function to read, format, and save certain training datasets.

    Parameters
    ----------
    hfile : `str`
        Filepath to hdf file produced by `gwdistributions` containing certain injections
    output : `str`
        Filepath at which to save reformatted and annotated data
    """

    # Open
    hfile = h5py.File(hfile,'r')

    # Add derived parameters and mark as found/missed
    certain = read_and_annotate(hfile)
    mark_detections_certain(hfile,certain)

    # Shuffle and save
    certain = shuffle(certain)
    certain.to_hdf(output,'train')

def format_and_save_hopeless(hfile,output,hkey='events'):

    """
    Wrapper function to read, format, and save hopeless training datasets.

    Parameters
    ----------
    hfile : `str`
        Filepath to hdf file produced by `gwdistributions` containing hopeless injections
    output : `str`
        Filepath at which to save reformatted and annotated data
    """

    # Open
    hfile = h5py.File(hfile,'r')

    # Add derived parameters and mark as missed injections
    hopeless = read_and_annotate(hfile,hkey)
    mark_detections_hopeless(hopeless)

    # Shuffle and save
    hopeless = shuffle(hopeless)
    hopeless.to_hdf(output,'train')

if __name__=="__main__":

    # Hopeless
    format_and_save_hopeless('./../data/rpo3-bbh-hopeless.hdf','./../data/rpo3-bbh-hopeless-formatted.hdf')
    format_and_save_hopeless('./../data/rpo3-nsbh-hopeless.hdf','./../data/rpo3-nsbh-hopeless-formatted.hdf','events/table')
    format_and_save_hopeless('./../data/rpo3-bns-hopeless.hdf','./../data/rpo3-bns-hopeless-formatted.hdf','events/table')
    format_and_save_hopeless('./../data/rpo3-combined-hopeless.hdf','./../data/rpo3-combined-hopeless-formatted.hdf')

    # Certain
    format_and_save_certain('./../data/rpo3-bbh-certain.hdf','./../data/rpo3-bbh-certain-formatted.hdf')
    format_and_save_certain('./../data/rpo3-nsbh-certain.hdf','./../data/rpo3-nsbh-certain-formatted.hdf')
    format_and_save_certain('./../data/rpo3-bns-certain.hdf','./../data/rpo3-bns-certain-formatted.hdf')







