import h5py
import numpy as np
import pandas as pd
from utilities import generalized_Xp
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def format_real_injection_data(file, output):

    """
    Script to load and prepare LVK O3b CBC injections for use in neural
    network training. Specifically, this function adds some derived quantities
    and imposes a selection cut to label each injection as "found" or "missed"

    Parameters
    ----------
    file : `str`
        Filepath to hdf file containing injection data
    output : `str`
        Filepath at which to save pandas.DataFrame with processed injection
        information, to be used in neural network training.
    """

    # Load
    raw_injection_data = h5py.File(file, 'r')
    print(raw_injection_data.attrs['total_generated'])
    injections = raw_injection_data['injections']

    # Read injection parameters
    injectionData = pd.DataFrame()
    injectionData['m1_detector'] = np.array(injections['mass1'][()], dtype='float64')
    injectionData['m2_detector'] = np.array(injections['mass2'][()], dtype='float64')
    injectionData['luminosity_distance'] = injections['distance'][()]/1000.  # Convert from Mpc to Gpc
    injectionData['cos_inclination'] = np.cos(injections['inclination'])
    injectionData['right_ascension'] = injections['right_ascension']
    injectionData['declination'] = injections['declination']
    injectionData['polarization'] = injections['polarization']
    injectionData['redshift'] = injections['redshift'][()]

    # Some derived mass parameters
    injectionData['q'] = injectionData.m2_detector/injectionData.m1_detector
    injectionData['eta'] = injectionData.m1_detector*injectionData.m2_detector/(injectionData.m1_detector+injectionData.m2_detector)**2
    injectionData['chirp_mass_detector'] = injectionData.eta**(3./5.)*(injectionData.m1_detector+injectionData.m2_detector)
    injectionData['total_mass_detector'] = (injectionData.m1_detector+injectionData.m2_detector)

    # And some derived spin parameters
    s1x = np.array(injections['spin1x'])
    s1y = np.array(injections['spin1y'])
    s1z = np.array(injections['spin1z'])
    s2x = np.array(injections['spin2x'])
    s2y = np.array(injections['spin2y'])
    s2z = np.array(injections['spin2z'])
    injectionData['a1'] = np.sqrt(s1x**2 + s1y**2 + s1z**2)
    injectionData['a2'] = np.sqrt(s2x**2 + s2y**2 + s2z**2)
    injectionData['cost1'] = s1z/injectionData.a1
    injectionData['cost2'] = s2z/injectionData.a2
    injectionData['Xeff'] = (s1z + injectionData.q*s2z)/(1.+injectionData.q)
    injectionData['Xdiff'] = (s1z - injectionData.q*s2z)/(1.+injectionData.q)
    injectionData['Xp_gen'] = generalized_Xp(s1x, s1y, s2x,  s2y, injectionData.q)

    # Read out false alarm rates
    far_gstlal = injections['far_gstlal'][()]
    far_pycbc = injections['far_pycbc_bbh'][()]
    far_pycbc_hyper = injections['far_pycbc_hyperbank'][()]
    far_mbta = injections['far_mbta'][()]
    far_cwb = injections['far_cwb'][()]

    # Take minimum FAR across all searches and assign detection labels
    far_min = np.min(np.stack([far_gstlal, far_pycbc, far_pycbc_hyper, far_mbta, far_cwb]), axis=0)
    injectionData['detected'] = np.where(far_min < 1, 1, 0)

    injectionData = shuffle(injectionData)
    train_data, val_data = train_test_split(injectionData, train_size=0.8)

    train_data.to_hdf(output+'training_data.hdf', 'train')
    val_data.to_hdf(output+'validation_data.hdf', 'validate')


def read_and_annotate(hfile, hkey='events'):

    """
    Helper function to load hdf file prepared by the `gwdistributions` and add
    necessary derived data.

    Parameters
    ----------
    hfile : `h5py.File`
        Opened `h5py.File` containing injection data
    hkey : `str`
        Name of relevant dataset inside hdf object. Default `events`

    Returns
    -------
    injectionData : `pandas.DataFrame`
        Data frame containing injection data and additional derived parameters
    """

    # Read out dataset to be processed
    raw_data = hfile[hkey]

    # Read injection parameters
    injectionData = pd.DataFrame()
    injectionData['m1_detector'] = np.array(raw_data['mass1_detector'][()], dtype='float64')
    injectionData['m2_detector'] = np.array(raw_data['mass2_detector'][()], dtype='float64')
    injectionData['luminosity_distance'] = raw_data['luminosity_distance'][()]/1000.  # Convert from Mpc to Gpc
    injectionData['cos_inclination'] = np.cos(raw_data['inclination'])
    injectionData['right_ascension'] = raw_data['right_ascension']
    injectionData['declination'] = raw_data['declination']
    injectionData['polarization'] = raw_data['polarization']
    injectionData['redshift'] = raw_data['z'][()]

    # Some derived mass parameters
    injectionData['q'] = injectionData.m2_detector/injectionData.m1_detector
    injectionData['eta'] = injectionData.m1_detector*injectionData.m2_detector/(injectionData.m1_detector+injectionData.m2_detector)**2
    injectionData['chirp_mass_detector'] = injectionData.eta**(3./5.)*(injectionData.m1_detector+injectionData.m2_detector)
    injectionData['total_mass_detector'] = (injectionData.m1_detector+injectionData.m2_detector)

    # And some derived spin parameters
    s1x = np.array(raw_data['spin1x'])
    s1y = np.array(raw_data['spin1y'])
    s1z = np.array(raw_data['spin1z'])
    s2x = np.array(raw_data['spin2x'])
    s2y = np.array(raw_data['spin2y'])
    s2z = np.array(raw_data['spin2z'])
    injectionData['a1'] = np.sqrt(s1x**2 + s1y**2 + s1z**2)
    injectionData['a2'] = np.sqrt(s2x**2 + s2y**2 + s2z**2)
    injectionData['cost1'] = s1z/injectionData.a1
    injectionData['cost2'] = s2z/injectionData.a2
    injectionData['Xeff'] = (s1z + injectionData.q*s2z)/(1.+injectionData.q)
    injectionData['Xdiff'] = (s1z - injectionData.q*s2z)/(1.+injectionData.q)
    injectionData['Xp_gen'] = generalized_Xp(s1x, s1y, s2x, s2y, injectionData.q)

    # SNR info
    injectionData['snr_net'] = raw_data['snr_net']
    injectionData['observed_snr_net'] = raw_data['observed_snr_net']

    return injectionData


def mark_detections_certain(hfile, injectionData):

    """
    Simple helper script to annotate certain detections as found/missed

    Parameters
    ----------
    hfile : `str`
        Filepath to hdf file produced by `gwdistributions` containing certain
        injections.
    injectionData : `pandas.DataFrame`
        Data frame object returned by `read_and_annotate()`. Annotated
        in-place.

    """

    # Probabalitically determine if H1 and L1 were on/off
    both_off = np.random.random(len(injectionData)) < 0.0589

    # If both H1 and L1 are off, mark detection as missed
    injectionData.loc[:, 'detected'] = np.where(both_off, 0, 1)


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
    injectionData.loc[:, 'detected'] = 0


def format_and_save_certain(hfile, output):

    """
    Wrapper function to read, format, and save certain training datasets.

    Parameters
    ----------
    hfile : `str`
        Filepath to hdf file produced by `gwdistributions` containing certain
        injections
    output : `str`
        Filepath at which to save reformatted and annotated data
    """

    # Open
    hfile = h5py.File(hfile, 'r')

    # Add derived parameters and mark as found/missed
    certain = read_and_annotate(hfile)
    mark_detections_certain(hfile, certain)

    # Shuffle and save
    certain = shuffle(certain)
    certain.to_hdf(output, 'train')


def format_and_save_hopeless(hfile, output, hkey='events'):

    """
    Wrapper function to read, format, and save hopeless training datasets.

    Parameters
    ----------
    hfile : `str`
        Filepath to hdf file produced by `gwdistributions` containing hopeless
        injections
    output : `str`
        Filepath at which to save reformatted and annotated data
    """

    # Open
    hfile = h5py.File(hfile, 'r')

    # Add derived parameters and mark as missed injections
    hopeless = read_and_annotate(hfile, hkey)
    mark_detections_hopeless(hopeless)

    # Shuffle and save
    hopeless = shuffle(hopeless)
    hopeless.to_hdf(output,
                    'train')


if __name__== "__main__":

    # Process real injections
    format_real_injection_data('./../input/endo3_bbhpop-LIGO-T2100113-v12.hdf5',
                        './../data/training_data/bbh_')
    format_real_injection_data('./../input/endo3_bnspop-LIGO-T2100113-v12.hdf5',
                        './../data/training_data/bns_')
    format_real_injection_data('./../input/endo3_nsbhpop-LIGO-T2100113-v12.hdf5',
                        './../data/training_data/nsbh_')

    # Hopeless injections
    format_and_save_hopeless('./../data/training_data/rpo3-bbh-hopeless.hdf',
                             './../data/training_data/rpo3-bbh-hopeless-formatted.hdf',
                             'events/table')
    format_and_save_hopeless('./../data/training_data/rpo3-nsbh-hopeless.hdf',
                             './../data/training_data/rpo3-nsbh-hopeless-formatted.hdf',
                             'events/table')
    format_and_save_hopeless('./../data/training_data/rpo3-bns-hopeless.hdf',
                             './../data/training_data/rpo3-bns-hopeless-formatted.hdf',
                             'events/table')
    format_and_save_hopeless('./../data/training_data/rpo3-combined-hopeless.hdf',
                             './../data/training_data/rpo3-combined-hopeless-formatted.hdf')

    # Certain injections
    format_and_save_certain('./../data/training_data/rpo3-bbh-certain.hdf',
                            './../data/training_data/rpo3-bbh-certain-formatted.hdf')
    format_and_save_certain('./../data/training_data/rpo3-nsbh-certain.hdf',
                            './../data/training_data/rpo3-nsbh-certain-formatted.hdf')
    format_and_save_certain('./../data/training_data/rpo3-bns-certain.hdf',
                            './../data/training_data/rpo3-bns-certain-formatted.hdf')
