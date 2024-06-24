import numpy as np
import h5py
import astropy.units as u
from astropy.cosmology import Planck15
import pickle


def loadInjections(ifar_threshold):

    """
    Function to read and format BBH injections to be used downstream for
    hierarchical inference with numpyro. Returns a subset of data for the set
    of injections meeting the IFAR threshold necessary for detection.

    Parameters
    ----------
    ifar_threshold : float
        IFAR threshold in units of years, above which injections are
        considered to have been found.

    Returns
    -------
    m1_det : numpy.array
        Primary source-frame masses of found injections
    m2_det : numpy.array
        Secondary source-frame masses
    s1x_det : numpy.array
        Cartesian-x component of primary spin among found injections
    s1y_det : numpy.array
        Cartesian-y component of primary spin among found injections
    s1z_det : numpy.array
        Cartesian-z component of primary spin among found injections
    s2x_det : numpy.array
        Cartesian-x component of secondary spin among found injections
    s2y_det : numpy.array
        Cartesian-y component of secondary spin among found injections
    s2z_det : numpy.array
        Cartesian-z component of secondary spin among found injections
    z_det : numpy.array
        Redshifts of found injections
    precomputed_p_m1m2z : numpy.array
        Draw probability density of component masses and redshifts for
        found injections
    nTrials : int
        Total number of injections performed
    """

    # Read injection file
    mockDetections = h5py.File('../input/endo3_bbhpop-LIGO-T2100113-v12.hdf5', 'r')

    # Total number of trial injections (detected or not)
    nTrials = mockDetections.attrs['total_generated']
    injections = mockDetections['injections']

    # Read out IFARs and SNRs from search pipelines
    ifar_1 = injections['ifar_gstlal'][()]
    ifar_2 = injections['ifar_pycbc_bbh'][()]
    ifar_3 = injections['ifar_pycbc_hyperbank'][()]
    ifar_4 = injections['ifar_cwb'][()]
    ifar_5 = injections['ifar_mbta'][()]

    # Determine which events pass IFAR threshold (O3) or SNR threshold (O1/O2)
    detected_full = np.where((ifar_1 > ifar_threshold)
                             | (ifar_2 > ifar_threshold)
                             | (ifar_3 > ifar_threshold)
                             | (ifar_4 > ifar_threshold)
                             | (ifar_5 > ifar_threshold))[0]

    # Get properties of detected sources
    m1_det = np.array(injections['mass1_source'][()])[detected_full]
    m2_det = np.array(injections['mass2_source'][()])[detected_full]
    s1x_det = np.array(injections['spin1x'][()])[detected_full]
    s1y_det = np.array(injections['spin1y'][()])[detected_full]
    s1z_det = np.array(injections['spin1z'][()])[detected_full]
    s2x_det = np.array(injections['spin2x'][()])[detected_full]
    s2y_det = np.array(injections['spin2y'][()])[detected_full]
    s2z_det = np.array(injections['spin2z'][()])[detected_full]
    z_det = np.array(injections['redshift'][()])[detected_full]

    # This is dP_draw/(dm1*dm2*dz*ds1x*ds1y*ds1z*ds2x*ds2y*ds2z)
    precomputed_p_m1m2z_spin = np.array(injections['sampling_pdf'][()])[detected_full]

    # In general, we'll want either dP_draw/(dm1*dm2*dz*da1*da2*dcost1*dcost2)
    # or dP_draw/(dm1*dm2*dz*dchi_eff*dchi_p). In preparation for computing
    # these quantities, divide out by the component draw probabilities
    # dP_draw/(ds1x*ds1y*ds1z*ds2x*ds2y*ds2z). Note that injections are uniform
    # in spin magnitude (up to a_max = 0.998) and isotropic, giving the
    # following:

    dP_ds1x_ds1y_ds1z = (1./(4.*np.pi))*(1./0.998)/(s1x_det**2+s1y_det**2+s1z_det**2)
    dP_ds2x_ds2y_ds2z = (1./(4.*np.pi))*(1./0.998)/(s2x_det**2+s2y_det**2+s2z_det**2)
    precomputed_p_m1m2z = precomputed_p_m1m2z_spin/dP_ds1x_ds1y_ds1z/dP_ds2x_ds2y_ds2z

    return m1_det, m2_det, s1x_det, s1y_det, s1z_det, s2x_det, s2y_det, \
        s2z_det, z_det, precomputed_p_m1m2z, nTrials


def genInjectionFile(ifar_threshold, filename):

    """
    Function that saves file of preprocessed found injections, for downstream
    use in hierarchical population analysis with numpyro.

    Parameters
    ----------
    ifar_threshold : float
        IFAR threshold (in years) above which injections are considered to
        have been detected
    filename : str
        File path and name at which to store preprocessed injections

    Returns
    -------
    None
    """

    # Load
    m1_det, m2_det, s1x_det, s1y_det, s1z_det, s2x_det, s2y_det, s2z_det, \
        z_det, p_draw_m1m2z, nTrials = loadInjections(ifar_threshold)

    # Derived parameters
    q_det = m2_det/m1_det
    Xeff_det = (m1_det*s1z_det + m2_det*s2z_det)/(m1_det+m2_det)
    Xp_det = np.maximum(
                np.sqrt(s1x_det**2+s1y_det**2),
                (3.+4.*q_det)/(4.+3.*q_det)*q_det*np.sqrt(s2x_det**2+s2y_det**2))
    a1_det = np.sqrt(s1x_det**2 + s1y_det**2 + s1z_det**2)
    a2_det = np.sqrt(s2x_det**2 + s2y_det**2 + s2z_det**2)
    cost1_det = s1z_det/a1_det
    cost2_det = s2z_det/a2_det

    # Compute marginal draw probabilities for chi_effective and joint chi_effective vs. chi_p probabilities
    """
    p_draw_xeff = np.zeros(Xeff_det.size)
    p_draw_xeff_xp = np.zeros(Xeff_det.size)
    for i in range(p_draw_xeff.size):
        if i%500==0:
            print(i)
        p_draw_xeff[i] = chi_effective_prior_from_isotropic_spins(q_det[i],1.,Xeff_det[i])
        p_draw_xeff_xp[i] = joint_prior_from_isotropic_spins(q_det[i],1.,Xeff_det[i],Xp_det[i],ndraws=10000)
    """

    # Draw probabilities for component spin magnitudes and tilts
    p_draw_a1a2cost1cost2 = (1./2.)**2*(1./0.998)**2*np.ones(a1_det.size)

    # Combine
    #pop_reweight = 1./(p_draw_m1m2z*p_draw_xeff_xp)
    #pop_reweight_XeffOnly = 1./(p_draw_m1m2z*p_draw_xeff)
    #pop_reweight_noSpin = 1./p_draw_m1m2z

    # Also compute factors of dVdz that we will need to reweight these samples during inference later on
    dVdz = 4.*np.pi*Planck15.differential_comoving_volume(z_det).to(u.Gpc**3*u.sr**(-1)).value

    # Store and save
    injectionDict = {
            'm1': m1_det,
            'm2': m2_det,
            'Xeff': Xeff_det,
            'Xp': Xp_det,
            'z': z_det,
            's1z': s1z_det,
            's2z': s2z_det,
            'a1': a1_det,
            'a2': a2_det,
            'cost1': cost1_det,
            'cost2': cost2_det,
            'dVdz': dVdz,
            'p_draw_m1m2z': p_draw_m1m2z,
            'p_draw_a1a2cost1cost2': p_draw_a1a2cost1cost2,
            'nTrials': nTrials
            }

    with open(filename, 'wb') as f:
        pickle.dump(injectionDict, f, protocol=2)


if __name__ == "__main__":

    genInjectionFile(1, '../input/injectionDict_FAR_1_in_1_BBH.pickle')
