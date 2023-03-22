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

def draw_params():

    mA = 100.*np.random.random()+2.
    mB = 100.*np.random.random()+2.
    DL = 15e3*np.random.random()
    a1 = np.random.random()
    a2 = np.random.random()
    cost1 = 2.*np.random.random()-1.
    cost2 = 2.*np.random.random()-1.
    phi1 = 2.*np.pi*np.random.random()
    phi2 = 2.*np.pi*np.random.random()
    cos_inc = 2.*np.random.random()-1.
    ra = 2.*np.pi*np.random.random()
    sin_dec = 2.*np.random.random()-1.
    pol = 2.*np.pi*np.random.random()

    z = z_at_value(Planck15.luminosity_distance,DL*u.Mpc).value

    O3_start = 1238166018
    O3_end = 1269363618
    time = np.random.random()*(O3_end-O3_start) + O3_start

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

def get_snr(paramDict,H_psd,L_psd):

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

    snr_H1_sq = 4.*h_H1.delta_f*np.sum(np.array(np.abs(h_H1)**2/psd_H1_interpolated)[psd_H1_interpolated[()]>0.])
    snr_L1_sq = 4.*h_L1.delta_f*np.sum(np.array(np.abs(h_L1)**2/psd_L1_interpolated)[psd_L1_interpolated[()]>0.])
    return np.sqrt(snr_H1_sq + snr_L1_sq)

def draw_hopeless(nDraws):

    psd_delta_f = 1./256
    psd_length = int(4096./psd_delta_f)
    psd_low_frequency_cutoff = 20.
    H_psd = psd.from_txt("./../input/H1-AVERAGE_PSD-1241560818-28800.txt",psd_length,psd_delta_f,psd_low_frequency_cutoff,is_asd_file=False)
    L_psd = psd.from_txt("./../input/L1-AVERAGE_PSD-1241560818-28800.txt",psd_length,psd_delta_f,psd_low_frequency_cutoff,is_asd_file=False)

    n_hopeless = 0
    n_trials = 0
    hopeless_params = []
    findable_params = []
    while n_hopeless<nDraws:

        n_trials+=1
        params = draw_params()
        snr = get_snr(params,H_psd,L_psd)
        if snr<=4:
            n_hopeless+=1
            hopeless_params.append(params)

            if n_hopeless%10==0:
                print(n_hopeless,n_trials)

        else:
            findable_params.append(params)
            
    return hopeless_params,findable_params

#draw_hopeless()
