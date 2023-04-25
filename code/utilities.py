import numpy as np
from pycbc.waveform import td_approximants, fd_approximants
from pycbc import types, fft, waveform
from pycbc.detector import Detector
from pycbc import psd
from astropy.cosmology import Planck15,z_at_value
import astropy.units as u

def generalized_Xp(s1x,s1y,s2x,s2y,q):

    """
    Function to compute generalized effective precessing spin parameter,
    as defined in Gerosa+ 2021 (https://arxiv.org/abs/2011.11948)

    Parameters
    ----------
    s1x : `float` or `np.array`
        x-component of the primary's spin
    s1y : `float` or `np.array`
        y-component of the primary's spin
    s2x : `float` or `np.array`
        x-component of the secondary's spin
    s2y : `float` or `np.array`
        y-component of the secondary's spin
    q : `float` or `np.array`

    Returns
    -------
    Xp : `float` or `np.array`
        Generalized precessing spin parameter
    """
    
    # Get total in-plane component spins
    Xp1 = np.sqrt(s1x**2+s1y**2)
    Xp2 = np.sqrt(s2x**2+s2y**2)

    # Cosine of angle between component spins, after projection on the orbital plane
    Xp1_Xp2_cos_dphi = (s1x*s2x + s1y*s2y)

    # Compute precessing spin parameter
    Xp = np.sqrt(Xp1**2 + ((3.+4.*q)/(4.+3.*q)*q*Xp2)**2 + 2.*q*(3.+4.*q)/(4.+3.*q)*Xp1_Xp2_cos_dphi)

    return Xp

def draw_params(DL_max=15e3):

    """
    Helper function to randomly draw sets of CBC parameters.
    Used by `draw_hopeless()` below to generate additional undetectable CBCs
    to assist in neural network training.

    Parameters
    ----------
    None

    Returns
    -------
    paramDict : `dict`
        Dictionary containing randomly drawn component masses, Cartesian spins, sky position,
        source orientation, distance, and time. Note that the provided masses are in the *detector frame*.
    """

    # Draw random masses between 2,102 Msun
    mA = 100.*np.random.random()+2.
    mB = 100.*np.random.random()+2.

    # Random distance
    # Note that we deliberately choose a random distance, rather than following a uniform-in-volume prior,
    # in order to provide training points across a range of distances. Following a uniform-in-volume distribution
    # is clearly more astrophysically realistic, but will result in very few training puts being placed at small
    # and intermediate distances
    DL = DL_max*np.random.random()
    z = z_at_value(Planck15.luminosity_distance,DL*u.Mpc).value

    # Isotropic spins
    a1 = np.random.random()
    a2 = np.random.random()
    cost1 = 2.*np.random.random()-1.
    cost2 = 2.*np.random.random()-1.
    phi1 = 2.*np.pi*np.random.random()
    phi2 = 2.*np.pi*np.random.random()

    # Isotropic sky position and orientation
    cos_inc = 2.*np.random.random()-1.
    ra = 2.*np.pi*np.random.random()
    sin_dec = 2.*np.random.random()-1.
    pol = 2.*np.pi*np.random.random()

    # Get a random time across O3
    O3_start = 1238166018
    O3_end = 1269363618
    time = np.random.random()*(O3_end-O3_start) + O3_start

    # Package it all in a dictionary and return
    paramDict = {'mass1':max(mA,mB)*(1.+z),
                'mass2':min(mA,mB)*(1.+z),
                'distance':DL,
                'redshift':z,
                'spin1x':a1*np.sqrt(1.-cost1**2)*np.cos(phi1),
                'spin1y':a1*np.sqrt(1.-cost1**2)*np.sin(phi1),
                'spin1z':a1*cost1,
                'spin2x':a2*np.sqrt(1.-cost2**2)*np.cos(phi2),
                'spin2y':a2*np.sqrt(1.-cost2**2)*np.sin(phi2),
                'spin2z':a2*cost2,
                'right_ascension':ra,
                'declination':np.arcsin(sin_dec),
                'polarization':pol,
                'inclination':np.arccos(cos_inc),
                'time':time}

    return paramDict

def get_snrs(paramDict,H_psd,L_psd):

    """
    Function to compute the expected SNR of a BBH in O3.
    Used in conjunction with `draw_hopeless()` defined below to generate additional hopeless injections
    to assist in neural network training.

    Parameters
    ----------
    paramDict : `dict`
        Dictionary of BBH parameters, as produced by `draw_params()`.
    H_psd : `pycbc.types.frequencyseries.FrequencySeries`
        Frequency series representing H1 O3 PSD
    L_psd : `pycbc.types.frequencyseries.FrequencySeries`
        Frequency series representing L1 O3 PSD
        
    Returns
    -------
    snr : `float`
        H1L1 network SNR expected from the given set of BBH parameters
    """

    # Generate waveform
    delta_f=1./16.
    sptilde, sctilde = waveform.get_fd_waveform(approximant="IMRPhenomPv2",
                                                template=paramDict,
                                                delta_f=delta_f,
                                                f_lower=20.)

    # Project onto detectors
    H1 = Detector("H1")
    L1 = Detector("L1")
    Fhp,Fhc = H1.antenna_pattern(paramDict['right_ascension'],paramDict['declination'],paramDict['polarization'],paramDict['time'])
    Flp,Flc = L1.antenna_pattern(paramDict['right_ascension'],paramDict['declination'],paramDict['polarization'],paramDict['time'])
    h_H1 = Fhp*sptilde + Fhc*sctilde
    h_L1 = Flp*sptilde + Flc*sctilde

    # Interpolate PSDs
    # If PSD extends to higher frequencies than the GW signal, trim as necessary
    psd_H1_interpolated = psd.interpolate(H_psd,h_H1.delta_f)
    if len(psd_H1_interpolated)>len(h_H1):
        psd_H1_interpolated = psd.interpolate(H_psd,h_H1.delta_f)[:len(h_H1)]
    else:
        psd_H1_interpolated_tmp = types.FrequencySeries(types.zeros(len(h_H1)),delta_f=h_H1.delta_f)
        psd_H1_interpolated_tmp[0:len(psd_H1_interpolated)] = psd_H1_interpolated
        psd_H1_interpolated = psd_H1_interpolated_tmp

    psd_L1_interpolated = psd.interpolate(L_psd,h_L1.delta_f)
    if len(psd_L1_interpolated)>len(h_L1):
        psd_L1_interpolated = psd.interpolate(L_psd,h_L1.delta_f)[:len(h_L1)]
    else:
        psd_L1_interpolated_tmp = types.FrequencySeries(types.zeros(len(h_L1)),delta_f=h_L1.delta_f)
        psd_L1_interpolated_tmp[0:len(psd_L1_interpolated)] = psd_L1_interpolated
        psd_L1_interpolated = psd_L1_interpolated_tmp

    # Compute detector SNRs and add in quadrature
    # Note that PSDs are reported as identically zero in regions where they are in fact undefined,
    # so cut on frequencies at which PSDs are above zero
    snr_H1_sq = 4.*h_H1.delta_f*np.sum(np.array(np.abs(h_H1)**2/psd_H1_interpolated)[psd_H1_interpolated[()]>0.])
    snr_L1_sq = 4.*h_L1.delta_f*np.sum(np.array(np.abs(h_L1)**2/psd_L1_interpolated)[psd_L1_interpolated[()]>0.])
    return snr_H1_sq,snr_L1_sq

def draw_hopeless(nDraws):

    """
    Function to randomly draw hopeless BBHs, for use in neural network training.

    Parameters
    ----------
    nDraws : `int`
        Desired number of hopeless injections

    Returns
    -------
    hopeless_params : `list`
        List of BBH parameters (each of which is a dictionary as return by `draw_params()`) that correspond
        to "undetectable" events, with expected network SNRs below 4.
    findable_params : `list`
        The complement of `hopeless_params`, containing events with expected network SNRs above 4.
    """

    # Load representative PSDs
    psd_delta_f = 1./256
    psd_length = int(4096./psd_delta_f)
    psd_low_frequency_cutoff = 20.
    H_psd = psd.from_txt("./../input/H1-AVERAGE_PSD-1241560818-28800.txt",
                            psd_length,psd_delta_f,psd_low_frequency_cutoff,is_asd_file=False)
    L_psd = psd.from_txt("./../input/L1-AVERAGE_PSD-1241560818-28800.txt",
                            psd_length,psd_delta_f,psd_low_frequency_cutoff,is_asd_file=False)

    # Instantiate variables to count and store hopeless/findable events
    n_hopeless = 0
    n_trials = 0
    hopeless_params = []
    findable_params = []

    # Repeat until we reach the desired number of hopeless injections
    while n_hopeless<nDraws:

        # Draw an event, compute its expected SNR, and check if this exceeds or is below 4
        n_trials+=1
        params = draw_params()
        H1_snr,L1_snr = get_snrs(params,H_psd,L_psd)
        snr = np.sqrt(H1_snr**2 + L1_snr**2)
        if snr<=4:
            n_hopeless+=1
            hopeless_params.append(params)

            if n_hopeless%10==0:
                print(n_hopeless,n_trials)

        else:
            findable_params.append(params)
            
    return hopeless_params,findable_params

def draw_certain(nDraws):

    # Load representative PSDs
    psd_delta_f = 1./256
    psd_length = int(4096./psd_delta_f)
    psd_low_frequency_cutoff = 20.
    H_psd = psd.from_txt("./../input/H1-AVERAGE_PSD-1241560818-28800.txt",
                            psd_length,psd_delta_f,psd_low_frequency_cutoff,is_asd_file=False)
    L_psd = psd.from_txt("./../input/L1-AVERAGE_PSD-1241560818-28800.txt",
                            psd_length,psd_delta_f,psd_low_frequency_cutoff,is_asd_file=False)

    # Instantiate variables to count and store hopeless/findable events
    n_certain = 0
    n_trials = 0
    certain_params = []

    # Repeat until we reach the desired number of hopeless injections
    while n_certain<nDraws:

        # Draw an event, compute its expected SNR, and check if this exceeds or is below 4
        n_trials+=1
        params = draw_params(DL_max=200)
        H1_snr,L1_snr = get_snrs(params,H_psd,L_psd)

        if min(H1_snr,L1_snr)>=100:
            n_certain+=1
            certain_params.append(params)

            if n_certain%10==0:
                print(n_certain,n_trials)

    return certain_params

class ANNaverage():

    """
    Class used as a wrapper around a list of individually-trained neural networks.
    Used to average the predictions across these individual networks to increase overall accuracy.
    """
    
    def __init__(self,ann_list):

        # List of `tf.keras.models.Sequential` neural networks
        self.ann_list = ann_list
        
    def predict(self,params,*args,**kwargs):

        """
        Function to compute predictions across the elements of `self.ann_list` and return their average.

        Parameters
        ----------
        params : `list`
            Input parameters at which to predict outputs

        Returns
        -------
        mean_predictions : `list`
            Mean predictions, taken across `self.ann_list`
        """

        # Compute list of predictions from individual networks, passing any additional arguments
        individual_predictions = [ann.predict(params,*args,**kwargs) for ann in self.ann_list]

        # Compute and return mean!
        return np.mean(individual_predictions,axis=0)
