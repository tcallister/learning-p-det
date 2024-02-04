import numpy as np
from pycbc.waveform import td_approximants, fd_approximants
from pycbc import types, fft, waveform
from pycbc.detector import Detector
from pycbc import psd
from astropy.cosmology import Planck15,z_at_value
import astropy.units as u

def draw_params(DL_max=15e3,dist='uniform'):

    """
    Helper function to randomly draw sets of CBC parameters.
    Used by `draw_hopeless()` and `draw_certain()` below to generate additional undetectable CBCs
    to assist in neural network training.

    Parameters
    ----------
    DL_max : `float`
        Maximum distance to consider, in Mpc (default 15e3)
    dist : `str`
        If 'uniform' (default), will draw uniform source-frame component masses. Otherwise
        will draw masses according to O3 injection distribution

    Returns
    -------
    paramDict : `dict`
        Dictionary containing randomly drawn component masses, Cartesian spins, sky position,
        source orientation, distance, and time. Note that the provided masses are in the *detector frame*.
    """

    # Draw random masses between 2,100 Msun
    if dist=='uniform':

        mA = 98.*np.random.random()+2.
        mB = 98.*np.random.random()+2.
        m1 = max(mA,mB)
        m2 = min(mA,mB)

    elif dist=='log':

        mA = np.exp(np.random.uniform(low=np.log(2.),high=np.log(102.)))
        mB = np.exp(np.random.uniform(low=np.log(2.),high=np.log(102.)))
        m1 = max(mA,mB)
        m2 = min(mA,mB)

    else:

        min_m1 = 2.
        max_m1 = 100.
        alpha_m1 = -2.35
        min_m2 = 2.
        alpha_m2 = 1.

        c_m1 = np.random.random()
        c_m2 = np.random.random()
        m1 = (min_m1**(1.+alpha_m1) + c_m1*(max_m1**(1.+alpha_m1)-min_m1**(1.+alpha_m1)))**(1./(1.+alpha_m1))
        m2 = (min_m2**(1.+alpha_m2) + c_m2*(m1**(1.+alpha_m2)-min_m2**(1.+alpha_m2)))**(1./(1.+alpha_m2))

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
    # Note that these dictionary keys need to correspond to parameters expected by `get_fd_waveform` in `get_snrs()` below
    paramDict = {'mass1':m1*(1.+z),
                'mass2':m2*(1.+z),
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
        params = draw_params(dist='log')#dist='injected')
        H1_snr,L1_snr = get_snrs(params,H_psd,L_psd)
        snr = np.sqrt(H1_snr**2 + L1_snr**2)
        if snr<=5:
            n_hopeless+=1
            hopeless_params.append(params)

            if n_hopeless%10==0:
                print(n_hopeless,n_trials)

        else:
            findable_params.append(params)
            
    return hopeless_params,findable_params

def draw_certain(nDraws):

    """
    Function to randomly draw BBHs whose detection is certain (up to detector duty cycle), for use in neural network training.
    Note that imperfect duty cycle must be accounted for downstream when assigning detection labels to these draws.

    Parameters
    ----------
    nDraws : `int`
        Desired number of certain injections

    Returns
    -------
    certain_params : `list`
        List of BBH parameters (each of which is a dictionary as return by `draw_params()`) that correspond
        to "undetectable" events, with individual-IFO SNRs above 100.
    """

    # Load representative PSDs
    psd_delta_f = 1./256
    psd_length = int(4096./psd_delta_f)
    psd_low_frequency_cutoff = 20.
    H_psd = psd.from_txt("./../input/H1-AVERAGE_PSD-1241560818-28800.txt",
                            psd_length,psd_delta_f,psd_low_frequency_cutoff,is_asd_file=False)
    L_psd = psd.from_txt("./../input/L1-AVERAGE_PSD-1241560818-28800.txt",
                            psd_length,psd_delta_f,psd_low_frequency_cutoff,is_asd_file=False)

    # Instantiate variables to count and store certain detections
    n_certain = 0
    n_trials = 0
    certain_params = []

    # Repeat until we reach the desired number of injections
    while n_certain<nDraws:

        # Draw an event and compute Hanford & Livingston SNRs
        n_trials+=1
        params = draw_params(DL_max=500)
        H1_snr,L1_snr = get_snrs(params,H_psd,L_psd)

        # If at least one detector has expected SNR>100, assume that this event is a certain detection,
        # provided that at least one instrument is online at the time
        if min(H1_snr,L1_snr)>=100:
            n_certain+=1
            certain_params.append(params)

            if n_certain%10==0:
                print(n_certain,n_trials)

    return certain_params

