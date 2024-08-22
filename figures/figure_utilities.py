import numpy as np
import jax
from jax import config
config.update("jax_enable_x64", True)
from jax.scipy.special import erf, erfinv
import jax.numpy as jnp
from astropy.cosmology import Planck15 as Planck15
import astropy.units as u
import pandas as pd
import sys
sys.path.append("./../code/")
from draw_new_injections import draw_new_injections
from diagnostics import *
from population_model import truncatedNormal, massModel
from getData import getInjections


def gen_found(ann,
              n,
              min_m1,
              max_m1,
              alpha_m1,
              min_m2,
              max_m2,
              alpha_m2,
              max_a1,
              max_a2,
              zMax,
              kappa,
              conditional_mass=True):

    """
    Function to generate a set of found injections using our neural network
    Pdet emulator.

    Parameters
    ----------
    ann : function
        The neural network Pdet emulator.
    n : int
        The number of found injections to generate.
    min_m1 : float
        Minimum primary mass
    max_m1 : float
        Maximum primary mass
    alpha_m1 : float
        Power-law index on primary mass
    min_m2 : float
        Minimum secondary mass
    max_m2 : float
        Maximum secondary mass
    alpha_m2 : float
        Power-law index on secondary mass
    max_a1 : float
        Maximum primary spin
    max_a2 : float
        Maximum secondary spin
    zMax : float
        Maximum redshift to consider
    kappa : float
        Power-law index on growth of merger rate with redshift
    conditional_mass : bool
        If True, define secondary mass distribution as p(m2|m1). If False,
        define secondary mass distribution as p(m2) and rejection sample
        according to the condition m1>m2. Default True.

    Returns
    -------
    all_found : pandas.DataFrame
        DataFrame containing the found injections.
    nTotal : int
        Total number of trials drawn in order to obtain `n` found injections.
    """

    nFound = 0
    nTotal = 0
    while nFound < n:

        # Take draws
        new_draws = draw_new_injections(
            batch_size=10000,
            min_m1=min_m1,
            max_m1=max_m1,
            alpha_m1=alpha_m1,
            min_m2=min_m2,
            max_m2=max_m2,
            alpha_m2=alpha_m2,
            max_a1=max_a1,
            max_a2=max_a2,
            zMax=zMax,
            kappa=kappa,
            conditional_mass=conditional_mass)

        injection_params = jnp.array([
            new_draws.m1_source.values,
            new_draws.m2_source.values,
            new_draws.a1.values,
            new_draws.a2.values,
            new_draws.cost1.values,
            new_draws.cost2.values,
            new_draws.redshift.values,
            new_draws.cos_inclination.values,
            new_draws.polarization.values,
            new_draws.phi1.values-new_draws.phi2.values,
            new_draws.right_ascension.values,
            np.sin(new_draws.declination.values)
            ])

        p_det_predictions = ann(injection_params).reshape(-1)
        random_draws = np.random.random(len(new_draws))
        found = new_draws[np.array(random_draws < p_det_predictions)].copy()

        if nFound == 0:
            all_found = found

        else:
            all_found = pd.concat([all_found, found], ignore_index=True)

        nFound += len(found)
        nTotal += len(new_draws)

    return all_found, nTotal


def make_hist(ax,
              reference,
              new,
              raw,
              param,
              xlim=None,
              ylim=None,
              ref_bins=None,
              new_bins=None,
              raw_bins=None,
              label=None,
              color=None,
              legend=False,
              n_decimal_pvalue=1):

    """
    Helper function to generate histograms for a given parameter among (i) raw
    injected distribution, (ii) found pipeline injections, and (iii) detections
    as predicted by our neural network emulator.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to plot on.
    reference : pandas.DataFrame
        DataFrame containing found pipeline injections
    new : pandas.DataFrame
        DataFrame containing detections as predicted by the neural network
        emulator.
    raw : pandas.DataFrame
        DataFrame containing raw injected distribution.
    param : str
        The parameter to plot.
    xlim : tuple, optional
        The x-axis limits. Default None.
    ylim : tuple, optional
        The y-axis limits. Default None.
    ref_bins : int, optional
        The number of bins to use when histogramming found pipeline injections.
        Default None.
    new_bins : int, optional
        The number of bins to use when histogramming detections as predicted by
        the neural network emulator. Default None.
    raw_bins : int, optional
        The number of bins to use when histogramming raw injected distribution.
        Default None.
    label : str, optional
        The label to use for the x-axis. Default None.
    color : str, optional
        Color to use for histograms. Default None.
    legend : bool, optional
        If True, add a legend to the plot. Default False.

    Returns
    -------
    None
    """

    if ref_bins is None:
        ref_bins = 30
    if new_bins is None:
        new_bins = 30
    if raw_bins is None:
        raw_bins = 30

    if not label:
        label = param

    ax.hist(reference[param], density=True, bins=ref_bins, color=color, alpha=0.5, label='Detected (Actual)')
    ax.hist(new[param], density=True, bins=new_bins, histtype='step', color='black', zorder=-1, label='Detected (Emulator)')
    ax.hist(raw[param], density=True, bins=raw_bins, histtype='step', color='black', zorder=-1, ls=':', label='Intrinsic')

    if legend:
        ax.legend(loc='upper left', fontsize=9.5, frameon=False)

    ks_result = ks_2samp(reference[param], new[param]).pvalue
    test = ("p={0:."+str(n_decimal_pvalue)+"f}").format(100*ks_result)+"\%"

    ax.text(0.93, 0.93,
            test,
            transform=ax.transAxes,
            verticalalignment='center',
            horizontalalignment='right',
            fontsize=9.5)

    ax.set_xlabel(label)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    ax.set_yticklabels([])

    return


# Generate grids over which to integrate various quantities in functions
# defined below
tmp_min = 2.
ref_m1_grid = jnp.linspace(tmp_min, 100, 1000)
ref_z_grid = jnp.linspace(0, 1.9, 1000)
ref_dVdz_grid = 4.*np.pi*Planck15.differential_comoving_volume(ref_z_grid)
ref_dVdz_grid = ref_dVdz_grid.to(u.Gpc**3/u.sr).value
ref_cost_grid = jnp.linspace(-1,1,1000)

# Preemptively load found pipeline injections
injectionDict = getInjections()
a1_det = injectionDict['a1']
a2_det = injectionDict['a2']
cost1_det = injectionDict['cost1']
cost2_det = injectionDict['cost2']
m1_det = injectionDict['m1']
m2_det = injectionDict['m2']
z_det = injectionDict['z']
dVdz_det = injectionDict['dVdz']
p_draw = injectionDict['p_draw_m1m2z']*injectionDict['p_draw_a1a2cost1cost2']


@jax.jit
def get_inj_efficiency(alpha,
                       mu_m1,
                       sig_m1,
                       log_f_peak,
                       mMax,
                       mMin,
                       log_dmMax,
                       log_dmMin,
                       bq,
                       mu_chi,
                       logsig_chi,
                       f_iso, 
                       mu_cost,
                       sig_cost,
                       kappa):

    """
    Function that evaluates detection efficiency for a CBC population with
    given hyperparameters, via reweighting of found pipeline injections.

    Parameters
    ----------
    alpha : float
        Power-law index on "Power-law" part of primary mass distribution
    mu_m1 : float
        Mean of Gaussian peak in primary mass distribution
    sig_m1 : float
        Standard deviation of Gaussian peak in primary mass distribution
    log_f_peak : float
        Log fraction of events in Gaussian peak
    mMax : float
        Primary mass above which the mass distribution is truncated
    mMin : float
        Primary mass below which the mass distribution is truncated
    log_dmMax : float
        Log-scale over which the mass distribution is truncated above mMax
    log_dmMin : float
        Log-scale over which the mass distribution is truncated below mMin
    bq : float
        Power-law index on secondary mass distribution
    mu_chi : float
        Mean component spin magnitude
    logsig_chi : float
        Log standard deviation of component spin magnitude
    f_iso : float
        Fraction of spins in isotropic component
    mu_cost : float
        Mean spin-orbit misalignment angle in non-isotropic component
    sig_cost : float
        Standard deviation of spin-orbit misalignment angle in non-isotropic component
    kappa : float
        Power-law index on growth of merger rate with redshift

    Returns
    -------
    detection_efficiency : float
        Computed detection efficiency
    Neff : float
        Effective number of detections informing estimated detection efficiency
    """

    # Necessary normalization constnats
    p_m1_norm = jnp.trapz(
                    massModel(ref_m1_grid, alpha, mu_m1, sig_m1,
                              10.**log_f_peak, mMax, mMin, 10.**log_dmMax,
                              10.**log_dmMin),
                    ref_m1_grid)
    p_z_norm = jnp.trapz(ref_dVdz_grid*(1.+ref_z_grid)**(kappa-1.), ref_z_grid)

    # Compute probability densities of found injections on proposed population
    p_m1_det = massModel(m1_det, alpha, mu_m1, sig_m1, 10.**log_f_peak, mMax,
                         mMin, 10.**log_dmMax, 10.**log_dmMin)/p_m1_norm
    p_m2_det = (1.+bq)*m2_det**bq/(m1_det**(1.+bq)-tmp_min**(1.+bq))
    p_a1_det = truncatedNormal(a1_det, mu_chi, 10.**logsig_chi, 0, 1)
    p_a2_det = truncatedNormal(a2_det, mu_chi, 10.**logsig_chi, 0, 1)
    p_cost1_det = f_iso/2. + (1.-f_iso)*truncatedNormal(cost1_det, mu_cost, sig_cost, -1, 1)
    p_cost2_det = f_iso/2. + (1.-f_iso)*truncatedNormal(cost2_det, mu_cost, sig_cost, -1, 1)
    p_z_det = dVdz_det*(1.+z_det)**(kappa-1.)/p_z_norm
    p_pop_det = p_m1_det*p_m2_det*p_z_det \
        * p_a1_det*p_a2_det \
        * p_cost1_det*p_cost2_det

    # Form ratio of proposed weights over draw weights
    inj_weights = p_pop_det/(p_draw/1.)

    # Compute net detection efficiency
    detection_efficiency = jnp.sum(inj_weights)/injectionDict['nTrials']
    Neff = jnp.sum(inj_weights)**2/jnp.sum(inj_weights**2)

    return detection_efficiency, Neff


# Preemptively load semianalytic injection results
semianalytic = np.load('BBH_semianalytic.npy',allow_pickle=True)[()]
semianalytic['obs_snr']['dVdz'] = 4.*np.pi*Planck15.differential_comoving_volume(semianalytic['obs_snr']['z']).to(u.Gpc**3/u.sr).value
semianalytic['expected_snr']['dVdz'] = 4.*np.pi*Planck15.differential_comoving_volume(semianalytic['expected_snr']['z']).to(u.Gpc**3/u.sr).value

@jax.jit
def get_semianalytic_efficiency(alpha,
                                mu_m1,
                                sig_m1,
                                log_f_peak,
                                mMax,
                                mMin,
                                log_dmMax,
                                log_dmMin,
                                bq,
                                mu_chi,
                                logsig_chi,
                                f_iso, 
                                mu_cost,
                                sig_cost,
                                kappa,
                                selection='obs'):

    """
    Function that evaluates detection efficiency for a CBC population with
    given hyperparameters, via reweighting of semianalytic injections

    Parameters
    ----------
    alpha : float
        Power-law index on "Power-law" part of primary mass distribution
    mu_m1 : float
        Mean of Gaussian peak in primary mass distribution
    sig_m1 : float
        Standard deviation of Gaussian peak in primary mass distribution
    log_f_peak : float
        Log fraction of events in Gaussian peak
    mMax : float
        Primary mass above which the mass distribution is truncated
    mMin : float
        Primary mass below which the mass distribution is truncated
    log_dmMax : float
        Log-scale over which the mass distribution is truncated above mMax
    log_dmMin : float
        Log-scale over which the mass distribution is truncated below mMin
    bq : float
        Power-law index on secondary mass distribution
    mu_chi : float
        Mean component spin magnitude
    logsig_chi : float
        Log standard deviation of component spin magnitude
    f_iso : float
        Fraction of spins in isotropic component
    mu_cost : float
        Mean spin-orbit misalignment angle in non-isotropic component
    sig_cost : float
        Standard deviation of spin-orbit misalignment angle in non-isotropic component
    kappa : float
        Power-law index on growth of merger rate with redshift

    Returns
    -------
    detection_efficiency : float
        Computed detection efficiency
    Neff : float
        Effective number of detections informing estimated detection efficiency
    """

    if selection=='obs':
        injs = semianalytic['obs_snr']
    else:
        injs = semianalytic['expected_snr']

    # Necessary normalization constnats
    p_m1_norm = jnp.trapz(
                    massModel(ref_m1_grid, alpha, mu_m1, sig_m1,
                              10.**log_f_peak, mMax, mMin, 10.**log_dmMax,
                              10.**log_dmMin),
                    ref_m1_grid)
    p_z_norm = jnp.trapz(ref_dVdz_grid*(1.+ref_z_grid)**(kappa-1.), ref_z_grid)

    # Compute probability densities of found injections on proposed population
    p_m1_semianalytic = massModel(injs['m1'], alpha, mu_m1, sig_m1, 10.**log_f_peak, mMax,
                         mMin, 10.**log_dmMax, 10.**log_dmMin)/p_m1_norm
    p_m2_semianalytic = (1.+bq)*injs['m2']**bq/(injs['m1']**(1.+bq)-tmp_min**(1.+bq))
    p_a1_semianalytic = truncatedNormal(injs['a1'], mu_chi, 10.**logsig_chi, 0, 1)
    p_a2_semianalytic = truncatedNormal(injs['a2'], mu_chi, 10.**logsig_chi, 0, 1)
    p_cost1_semianalytic = f_iso/2. + (1.-f_iso)*truncatedNormal(injs['costheta1'], mu_cost, sig_cost, -1, 1)
    p_cost2_semianalytic = f_iso/2. + (1.-f_iso)*truncatedNormal(injs['costheta2'], mu_cost, sig_cost, -1, 1)
    p_z_semianalytic = injs['dVdz']*(1.+injs['z'])**(kappa-1.)/p_z_norm

    # Impose boundaries
    p_m1_semianalytic = jnp.where(injs['m1']<100., p_m1_semianalytic, 0)
    p_m2_semianalytic = jnp.where(injs['m2']>tmp_min, p_m2_semianalytic, 0)
    p_z_semianalytic = jnp.where(injs['z']<1.9, p_z_semianalytic, 0)

    p_pop_semianalytic = p_m1_semianalytic*p_m2_semianalytic*p_z_semianalytic \
        * p_a1_semianalytic*p_a2_semianalytic \
        * p_cost1_semianalytic*p_cost2_semianalytic

    # Form ratio of proposed weights over draw weights
    inj_weights = p_pop_semianalytic/(injs['pdraw']/1.)

    # Compute net detection efficiency
    detection_efficiency = jnp.sum(inj_weights)/injs['nTrials']
    Neff = jnp.sum(inj_weights)**2/jnp.sum(inj_weights**2)

    return detection_efficiency, Neff


def draw_vals(nTrials):

    """
    Helper function to draw a random set of CDF values that will be transformed
    into a set of injections drawn from each population of interest.
    Additionally draws a fixed set of extrinsic parameters (e.g. sky position
    and polarization angle) that are not varied.

    Parameters
    ----------
    nTrials : int
        Number of trials to draw

    Returns
    -------
    inj_data : list
        List of arrays containing the drawn CDF values (or extrinsic values)
        for each parameter.
    """

    # Dynamic parameters
    m1_trials_cdfs = np.random.random(size=nTrials)
    m2_trials_cdfs = np.random.random(size=nTrials)
    a1_trials_cdfs = np.random.random(size=nTrials)
    a2_trials_cdfs = np.random.random(size=nTrials)
    cost1_trials_cdfs = np.random.random(size=nTrials)
    cost2_trials_cdfs = np.random.random(size=nTrials)
    z_trials_cdfs = np.random.random(size=nTrials)

    # Fixed
    ra_trials = 2.*np.pi*np.random.random(size=nTrials)
    sin_dec_trials = 2.*np.random.random(size=nTrials)-1.
    cos_inclination_trials = 2.*np.random.random(size=nTrials)-1.
    pol_trials = 2.*np.pi*np.random.random(size=nTrials)
    phi12_trials = 2.*np.pi*(np.random.random(size=nTrials)
                             - np.random.random(size=nTrials))

    inj_data = [m1_trials_cdfs, m2_trials_cdfs, a1_trials_cdfs, a2_trials_cdfs,
                cost1_trials_cdfs, cost2_trials_cdfs, z_trials_cdfs, ra_trials,
                sin_dec_trials, cos_inclination_trials, pol_trials,
                phi12_trials]

    for i in range(len(inj_data)):
        inj_data[i] = jnp.array(inj_data[i])

    return inj_data


def get_nn_efficiency(jitted_ann,
                      inj_data,
                      alpha,
                      mu_m1,
                      sig_m1,
                      log_f_peak,
                      mMax,
                      mMin,
                      log_dmMax,
                      log_dmMin,
                      bq,
                      mu_chi,
                      logsig_chi,
                      f_iso,
                      mu_cost,
                      sig_cost,
                      kappa,
                      hybrid):

    """
    Function to evaluate detection efficiency for a CBC population with given
    hyperparameters, via our neural network Pdet emulator.

    Parameters
    ----------
    jitted_ann : function
        JIT-compiled neural network emulator call function, used to compute
        Pdet over generated injections.
    inj_data : list
        List of arrays containing the drawn CDF values (or extrinsic values)
        for each parameter.
    alpha : float
        Power-law index on "Power-law" part of primary mass distribution
    mu_m1 : float
        Mean of Gaussian peak in primary mass distribution
    sig_m1 : float
        Standard deviation of Gaussian peak in primary mass distribution
    log_f_peak : float
        Log fraction of events in Gaussian peak
    mMax : float
        Primary mass above which the mass distribution is truncated
    mMin : float
        Primary mass below which the mass distribution is truncated
    log_dmMax : float
        Log-scale over which the mass distribution is truncated above mMax
    log_dmMin : float
        Log-scale over which the mass distribution is truncated below mMin
    bq : float
        Power-law index on secondary mass distribution
    mu_chi : float
        Mean component spin magnitude
    logsig_chi : float
        Log standard deviation of component spin magnitude
    f_iso : float
        Fraction of spins in isotropic component
    mu_cost : float
        Mean spin-orbit misalignment angle in non-isotropic component
    sig_cost : float
        Standard deviation of spin-orbit misalignment angle in non-isotropic component
    kappa : float
        Power-law index on growth of merger rate with redshift
    hybrid : bool
        If True, use hybrid scheme in which redshift values are drawn from
        a fixed distribution and reweighted to the desired distribution to
        increase sampling efficiency. If False, redshift values will be drawn
        directly from distribution specified by `kappa`. Default True.

    Returns
    -------
    xi : float
        Predicted detection efficiency
    Neff : float
        Effective number of samples informing Monte Carlo estimate of detection efficiency
    Neff_draws : float
        Effective number of injections when reweighted to target population
    """

    # Unpack precomputed CDF values for masses/spins/redshifts, and precomputed
    # extrinsic values
    m1_trials_cdfs, m2_trials_cdfs, a1_trials_cdfs, a2_trials_cdfs, \
        cost1_trials_cdfs, cost2_trials_cdfs, z_trials_cdfs, \
        ra_trials, sin_dec_trials, cos_inclination_trials, pol_trials, \
        phi12_trials = inj_data

    # Numerically compute primary mass CDF
    ref_pdf_m1 = massModel(ref_m1_grid, alpha, mu_m1, sig_m1, 10.**log_f_peak,
                           mMax, mMin, 10.**log_dmMax, 10.**log_dmMin)
    ref_cdf_m1 = jnp.cumsum(ref_pdf_m1)*jnp.diff(ref_m1_grid)[0]
    ref_m1_integral = ref_cdf_m1[-1]
    ref_cdf_m1 /= ref_m1_integral

    # Interpolate m1 and m2 cdfs onto inverse CDF distributions to obtain
    # random draws from specified population
    m1_trials = jnp.interp(m1_trials_cdfs, ref_cdf_m1, ref_m1_grid)
    m2_trials = jnp.power(tmp_min**(1.+bq) + m2_trials_cdfs*(
                        m1_trials**(1.+bq) - tmp_min**(1.+bq)), 1./(1.+bq))

    # Compute inverse CDFs for spin magnitudes and tilts
    sqrt_2 = jnp.sqrt(2.)
    chi_erf_a = erf(-mu_chi/(10.**logsig_chi*sqrt_2))
    chi_erf_b = erf((1-mu_chi)/(10.**logsig_chi*sqrt_2))
    #cost_erf_a = erf((-1-mu_cost)/(sig_cost*sqrt_2))
    #cost_erf_b = erf((1-mu_cost)/(sig_cost*sqrt_2))

    # Interpolate to obtain random component spin values
    a1_trials = mu_chi + 10.**logsig_chi*sqrt_2*erfinv(a1_trials_cdfs*(chi_erf_b-chi_erf_a) + chi_erf_a)
    a2_trials = mu_chi + 10.**logsig_chi*sqrt_2*erfinv(a2_trials_cdfs*(chi_erf_b-chi_erf_a) + chi_erf_a)
    #cost1_trials = mu_cost + sig_cost*sqrt_2*erfinv(cost1_trials_cdfs*(cost_erf_b-cost_erf_a) + cost_erf_a)
    #cost2_trials = mu_cost + sig_cost*sqrt_2*erfinv(cost2_trials_cdfs*(cost_erf_b-cost_erf_a) + cost_erf_a)

    ref_pdf_cost = f_iso/2. + (1.-f_iso)*truncatedNormal(ref_cost_grid, mu_cost, sig_cost, -1, 1)
    ref_cdf_cost = jnp.cumsum(ref_pdf_cost)*jnp.diff(ref_cost_grid)[0]
    ref_cdf_cost /= ref_cdf_cost[-1]
    cost1_trials = jnp.interp(cost1_trials_cdfs,ref_cdf_cost,ref_cost_grid)
    cost2_trials = jnp.interp(cost2_trials_cdfs,ref_cdf_cost,ref_cost_grid)

    # If we are not using hybrid sampling scheme, draw from specified
    # redshift distribution
    if hybrid is False:

        # Construct CDF and interpolate
        ref_pdf_z = ref_dVdz_grid*(1.+ref_z_grid)**(kappa-1.)
        ref_cdf_z = np.cumsum(ref_pdf_z)*(ref_z_grid[1]-ref_z_grid[0])
        ref_z_integral = ref_cdf_z[-1]
        ref_cdf_z /= ref_z_integral
        z_trials = jnp.interp(z_trials_cdfs, ref_cdf_z, ref_z_grid)

        # Dummy factors of unity (not unity in Hybrid case)
        reweight_factors = jnp.ones_like(z_trials)

    # If we are using hybrid scheme...
    else:

        # Fixed redshift distribution
        inj_kappa = -1.

        # Construct CDF and interpolate
        ref_pdf_z = ref_dVdz_grid*(1.+ref_z_grid)**(inj_kappa-1.)
        ref_cdf_z = np.cumsum(ref_pdf_z)*(ref_z_grid[1]-ref_z_grid[0])
        ref_z_integral = ref_cdf_z[-1]
        ref_cdf_z /= ref_z_integral
        z_trials = jnp.interp(z_trials_cdfs, ref_cdf_z, ref_z_grid)

        # Compute PDF and integration norm for specified kappa value
        target_pdf_z = ref_dVdz_grid*(1.+ref_z_grid)**(kappa-1.)
        target_cdf_z = np.cumsum(target_pdf_z)*(ref_z_grid[1]-ref_z_grid[0])
        target_z_integral = target_cdf_z[-1]

        # Factors to achieve reweighting from fixed kappa to desired kappa
        reweight_factors = (1.+z_trials)**(kappa-inj_kappa) \
            * (target_z_integral/ref_z_integral)**(-1)

    # Assemble parameters for drawn/transformed injection values
    converted_params = jnp.array([
         m1_trials,
         m2_trials,
         a1_trials,
         a2_trials,
         cost1_trials,
         cost2_trials,
         z_trials,
         cos_inclination_trials,
         pol_trials,
         phi12_trials,
         ra_trials,
         sin_dec_trials])

    # Evaluate neural network to obtain Pdet for each injection
    p_det = jitted_ann(converted_params).T*reweight_factors
    inj_weights = p_det

    # Compute total detection efficiency and number of effective injections
    xi = jnp.mean(inj_weights)
    Neff = jnp.sum(inj_weights)**2/jnp.sum(inj_weights**2)

    # Compute effective number of draws from the target population
    Neff_draws = jnp.sum(reweight_factors)**2/jnp.sum(reweight_factors**2)

    return xi, Neff, Neff_draws
