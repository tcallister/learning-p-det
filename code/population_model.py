import numpyro
import numpyro.distributions as dist
import jax
import jax.numpy as jnp
from jax.scipy.special import erf,erfinv
from jax import vmap
import numpy as np
import astropy.units as u
from astropy.cosmology import Planck15,z_at_value

logit_std = 2.5
tmp_max = 100.
tmp_min = 2.

def truncatedNormal(samples,mu,sigma,lowCutoff,highCutoff):

    """
    Jax-enabled truncated normal distribution
    
    Parameters
    ----------
    samples : `jax.numpy.array` or float
        Locations at which to evaluate probability density
    mu : float
        Mean of truncated normal
    sigma : float
        Standard deviation of truncated normal
    lowCutoff : float
        Lower truncation bound
    highCutoff : float
        Upper truncation bound

    Returns
    -------
    ps : jax.numpy.array or float
        Probability density at the locations of `samples`
    """

    a = (lowCutoff-mu)/jnp.sqrt(2*sigma**2)
    b = (highCutoff-mu)/jnp.sqrt(2*sigma**2)
    norm = jnp.sqrt(sigma**2*np.pi/2)*(-erf(a) + erf(b))
    ps = jnp.exp(-(samples-mu)**2/(2.*sigma**2))/norm
    return ps

def massModel(m1,alpha,mu_m1,sig_m1,f_peak,mMax,mMin,dmMax,dmMin):

    """
    Baseline primary mass model, described as a mixture between a power law
    and gaussian, with exponential tapering functions at high and low masses

    Parameters
    ----------
    m1 : array or float
        Primary masses at which to evaluate probability densities
    alpha : float
        Power-law index
    mu_m1 : float
        Location of possible Gaussian peak
    sig_m1 : float
        Stanard deviation of possible Gaussian peak
    f_peak : float
        Approximate fraction of events contained within Gaussian peak (not exact due to tapering)
    mMax : float
        Location at which high-mass tapering begins
    mMin : float
        Location at which low-mass tapering begins
    dmMax : float
        Scale width of high-mass tapering function
    dmMin : float
        Scale width of low-mass tapering function

    Returns
    -------
    p_m1s : jax.numpy.array
        Unnormalized array of probability densities
    """

    # Define power-law and peak
    p_m1_pl = (1.+alpha)*m1**(alpha)/(tmp_max**(1.+alpha) - tmp_min**(1.+alpha))
    p_m1_peak = jnp.exp(-(m1-mu_m1)**2/(2.*sig_m1**2))/jnp.sqrt(2.*np.pi*sig_m1**2)

    # Compute low- and high-mass filters
    low_filter = jnp.exp(-(m1-mMin)**2/(2.*dmMin**2))
    low_filter = jnp.where(m1<mMin,low_filter,1.)
    high_filter = jnp.exp(-(m1-mMax)**2/(2.*dmMax**2))
    high_filter = jnp.where(m1>mMax,high_filter,1.)

    # Apply filters to combined power-law and peak
    return (f_peak*p_m1_peak + (1.-f_peak)*p_m1_pl)*low_filter*high_filter

def get_value_from_logit(logit_x,x_min,x_max):

    """
    Function to map a variable `logit_x`, defined on `(-inf,+inf)`, to a quantity `x`
    defined on the interval `(x_min,x_max)`.

    Parameters
    ----------
    logit_x : float
        Quantity to inverse-logit transform
    x_min : float
        Lower bound of `x`
    x_max : float
        Upper bound of `x`

    Returns
    -------
    x : float
       The inverse logit transform of `logit_x`
    dlogit_dx : float
       The Jacobian between `logit_x` and `x`; divide by this quantity to convert a uniform prior on `logit_x` to a uniform prior on `x`
    """

    exp_logit = jnp.exp(logit_x)
    x = (exp_logit*x_max + x_min)/(1.+exp_logit)
    dlogit_dx = 1./(x-x_min) + 1./(x_max-x)

    return x,dlogit_dx

def baseline(sampleDict,injectionDict):

    """
    Implementation of a Gaussian effective spin distribution for inference within `numpyro`

    Parameters
    ----------
    sampleDict : dict
        Precomputed dictionary containing posterior samples for each event in our catalog
    injectionDict : dict
        Precomputed dictionary containing successfully recovered injections
    """
    
    # Sample our hyperparameters
    # alpha: Power-law index on primary mass distribution
    # mu_m1: Location of gaussian peak in primary mass distribution
    # sig_m1: Width of gaussian peak
    # f_peak: Fraction of events comprising gaussian peak
    # mMax: Location at which BBH mass distribution tapers off
    # mMin: Lower boundary at which BBH mass distribution tapers off
    # dmMax: Taper width above maximum mass
    # dmMin: Taper width below minimum mass
    # bq: Power-law index on the conditional secondary mass distribution p(m2|m1)
    # mu: Mean of the chi-effective distribution
    # logsig_chi: Log10 of the chi-effective distribution's standard deviation

    #logR20 = numpyro.deterministic("logR20",-0.5)
    alpha = numpyro.deterministic("alpha",-3.5)
    mu_m1 = numpyro.deterministic("mu_m1",35.)
    mMin = numpyro.deterministic("mMin",9.)
    bq = numpyro.deterministic("bq",1.)

    sig_m1 = numpyro.deterministic("sig_m1",3.)
    log_f_peak = numpyro.deterministic("log_f_peak",-3.)
    mMax = numpyro.deterministic("mMax",80.)
    log_dmMin = numpyro.deterministic("log_dmMin",0.)
    log_dmMax = numpyro.deterministic("log_dmMax",1.)
    mu_chi = numpyro.deterministic("mu_chi",0.)
    logsig_chi = numpyro.deterministic("logsig_chi",-0.4)
    sig_cost = numpyro.deterministic("sig_cost",0.4)

    kappa = numpyro.sample("kappa",dist.Normal(0,5))
    #alpha = numpyro.sample("alpha",dist.Normal(-2,3))
    logR20 = numpyro.sample("logR20",dist.Uniform(-12,12))
    R20 = numpyro.deterministic("R20",10.**logR20)

    """
    logR20 = numpyro.sample("logR20",dist.Uniform(-12,12))
    alpha = numpyro.sample("alpha",dist.Normal(-2,3))
    mu_m1 = numpyro.sample("mu_m1",dist.Uniform(20,50))
    mMin = numpyro.sample("mMin",dist.Uniform(5,15))
    bq = numpyro.sample("bq",dist.Normal(0,3))
    kappa = numpyro.sample("kappa",dist.Normal(0,5))
    R20 = numpyro.deterministic("R20",10.**logR20)

    logit_sig_m1 = numpyro.sample("logit_sig_m1",dist.Normal(0,logit_std))
    logit_log_f_peak = numpyro.sample("logit_log_f_peak",dist.Normal(0,logit_std))
    logit_mMax = numpyro.sample("logit_mMax",dist.Normal(0,logit_std))
    logit_log_dmMin = numpyro.sample("logit_log_dmMin",dist.Normal(0,logit_std))
    logit_log_dmMax = numpyro.sample("logit_log_dmMax",dist.Normal(0,logit_std))
    logit_mu_chi = numpyro.sample("logit_mu_chi",dist.Normal(0,logit_std))
    logit_logsig_chi = numpyro.sample("logit_logsig_chi",dist.Normal(0,logit_std))
    logit_sig_cost = numpyro.sample("logit_sig_cost",dist.Normal(0,logit_std))

    sig_m1,jac_sig_m1 = get_value_from_logit(logit_sig_m1,2.,15.)
    log_f_peak,jac_log_f_peak = get_value_from_logit(logit_log_f_peak,-3,0.)
    mMax,jac_mMax = get_value_from_logit(logit_mMax,50.,100.)
    log_dmMin,jac_log_dmMin = get_value_from_logit(logit_log_dmMin,-1,1)
    log_dmMax,jac_log_dmMax = get_value_from_logit(logit_log_dmMax,0.5,1.5)
    mu_chi,jac_mu_chi = get_value_from_logit(logit_mu_chi,0.,1.)
    logsig_chi,jac_logsig_chi = get_value_from_logit(logit_logsig_chi,-1.,0.)
    sig_cost,jac_sig_cost = get_value_from_logit(logit_sig_cost,0.5,1.5)

    numpyro.deterministic("sig_m1",sig_m1)
    numpyro.deterministic("log_f_peak",log_f_peak)
    numpyro.deterministic("mMax",mMax)
    numpyro.deterministic("log_dmMin",log_dmMin)
    numpyro.deterministic("log_dmMax",log_dmMax)
    numpyro.deterministic("mu_chi",mu_chi)
    numpyro.deterministic("logsig_chi",logsig_chi)
    numpyro.deterministic("sig_cost",sig_cost)

    numpyro.factor("p_sig_m1",logit_sig_m1**2/(2.*logit_std**2)-jnp.log(jac_sig_m1))
    numpyro.factor("p_log_f_peak",logit_log_f_peak**2/(2.*logit_std**2)-jnp.log(jac_log_f_peak))
    numpyro.factor("p_mMax",logit_mMax**2/(2.*logit_std**2)-jnp.log(jac_mMax))
    numpyro.factor("p_log_dmMin",logit_log_dmMin**2/(2.*logit_std**2)-jnp.log(jac_log_dmMin))
    numpyro.factor("p_log_dmMax",logit_log_dmMax**2/(2.*logit_std**2)-jnp.log(jac_log_dmMax))
    numpyro.factor("p_mu_chi",logit_mu_chi**2/(2.*logit_std**2)-jnp.log(jac_mu_chi))
    numpyro.factor("p_logsig_chi",logit_logsig_chi**2/(2.*logit_std**2)-jnp.log(jac_logsig_chi))
    numpyro.factor("p_sig_cost",logit_sig_cost**2/(2.*logit_std**2)-jnp.log(jac_sig_cost))
    """

    # Fixed params
    mu_cost = 1.

    # Normalization
    p_m1_norm = massModel(20.,alpha,mu_m1,sig_m1,10.**log_f_peak,mMax,mMin,10.**log_dmMax,10.**log_dmMin)
    p_z_norm = (1.+0.2)**kappa

    # Read out found injections
    # Note that `pop_reweight` is the inverse of the draw weights for each event
    a1_det = injectionDict['a1']
    a2_det = injectionDict['a2']
    cost1_det = injectionDict['cost1']
    cost2_det = injectionDict['cost2']
    m1_det = injectionDict['m1']
    m2_det = injectionDict['m2']
    z_det = injectionDict['z']
    dVdz_det = injectionDict['dVdz']
    p_draw = injectionDict['p_draw_m1m2z']*injectionDict['p_draw_a1a2cost1cost2']

    # Compute proposed population weights
    p_m1_det = massModel(m1_det,alpha,mu_m1,sig_m1,10.**log_f_peak,mMax,mMin,10.**log_dmMax,10.**log_dmMin)/p_m1_norm
    p_m2_det = (1.+bq)*m2_det**bq/(m1_det**(1.+bq)-tmp_min**(1.+bq))
    p_a1_det = truncatedNormal(a1_det,mu_chi,10.**logsig_chi,0,1)
    p_a2_det = truncatedNormal(a2_det,mu_chi,10.**logsig_chi,0,1)
    p_cost1_det = truncatedNormal(cost1_det,mu_cost,sig_cost,-1,1)
    p_cost2_det = truncatedNormal(cost2_det,mu_cost,sig_cost,-1,1)
    p_z_det = dVdz_det*(1.+z_det)**(kappa-1.)/p_z_norm 
    R_pop_det = R20*p_m1_det*p_m2_det*p_z_det*p_a1_det*p_a2_det*p_cost1_det*p_cost2_det

    # Form ratio of proposed weights over draw weights
    inj_weights = R_pop_det/(p_draw)
    
    # As a fit diagnostic, compute effective number of injections
    nEff_inj = jnp.sum(inj_weights)**2/jnp.sum(inj_weights**2)
    nObs = 1.0*len(sampleDict)
    numpyro.deterministic("nEff_inj_per_event",nEff_inj/nObs)

    # Compute net detection efficiency and add to log-likelihood
    Nexp = jnp.sum(inj_weights)/injectionDict['nTrials']
    numpyro.factor("rate",-Nexp)
    
    # This function defines the per-event log-likelihood
    # m1_sample: Primary mass posterior samples
    # m2_sample: Secondary mass posterior samples
    # z_sample: Redshift posterior samples
    # dVdz_sample: Differential comoving volume at each sample location
    # Xeff_sample: Effective spin posterior samples
    # priors: PE priors on each sample
    #def logp(m1_sample,m2_sample,z_sample,dVdz_sample,a1_sample,a2_sample,cost1_sample,cost2_sample,priors):
    def logp(m1_sample,m2_sample,z_sample,dVdz_sample,a1_sample,a2_sample,cost1_sample,cost2_sample,priors):

        # Compute proposed population weights
        p_m1 = massModel(m1_sample,alpha,mu_m1,sig_m1,10.**log_f_peak,mMax,mMin,10.**log_dmMax,10.**log_dmMin)/p_m1_norm
        p_m2 = (1.+bq)*m2_sample**bq/(m1_sample**(1.+bq)-tmp_min**(1.+bq))
        p_a1 = truncatedNormal(a1_sample,mu_chi,10.**logsig_chi,0,1)
        p_a2 = truncatedNormal(a2_sample,mu_chi,10.**logsig_chi,0,1)
        p_cost1 = truncatedNormal(cost1_sample,mu_cost,sig_cost,-1,1)
        p_cost2 = truncatedNormal(cost2_sample,mu_cost,sig_cost,-1,1)
        p_z = dVdz_sample*(1.+z_sample)**(kappa-1.)/p_z_norm
        R_pop = R20*p_m1*p_m2*p_z*p_a1*p_a2*p_cost1*p_cost2

        mc_weights = R_pop/priors

        #key,key_u = random.split(key)
        #m_choice = random.choice(key,m1_sample,p=mc_weights/jnp.sum(mc_weights))
        
        # Compute effective number of samples and return log-likelihood
        n_eff = jnp.sum(mc_weights)**2/jnp.sum(mc_weights**2)     
        return jnp.log(jnp.mean(mc_weights)),n_eff #m_choice
    
    # Map the log-likelihood function over each event in our catalog
    log_ps,n_effs = vmap(logp)(
                        jnp.array([sampleDict[k]['m1'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['m2'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['z'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['dVc_dz'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['a1'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['a2'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['cost1'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['cost2'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['z_prior'] for k in sampleDict]))
        
    # As a diagnostic, save minimum number of effective samples across all events
    numpyro.deterministic('min_log_neff',jnp.min(jnp.log10(n_effs)))

    # Tally log-likelihoods across our catalog
    numpyro.factor("logp",jnp.sum(log_ps))

def baseline_dynamicInjections(sampleDict,injectionCDFs,Pdet):

    """
    Implementation of a Gaussian effective spin distribution for inference within `numpyro`

    Parameters
    ----------
    sampleDict : dict
        Precomputed dictionary containing posterior samples for each event in our catalog
    injectionDict : dict
        Precomputed dictionary containing successfully recovered injections
    """
    
    #############################
    # Sample our hyperparameters
    #############################

    # alpha: Power-law index on primary mass distribution
    # mu_m1: Location of gaussian peak in primary mass distribution
    # sig_m1: Width of gaussian peak
    # f_peak: Fraction of events comprising gaussian peak
    # mMax: Location at which BBH mass distribution tapers off
    # mMin: Lower boundary at which BBH mass distribution tapers off
    # dmMax: Taper width above maximum mass
    # dmMin: Taper width below minimum mass
    # bq: Power-law index on the conditional secondary mass distribution p(m2|m1)
    # mu: Mean of the chi-effective distribution
    # logsig_chi: Log10 of the chi-effective distribution's standard deviation

    #logR20 = numpyro.deterministic("logR20",-0.5)
    #R20 = numpyro.deterministic("R20",10.**logR20)
    alpha = numpyro.deterministic("alpha",-3.5)
    mu_m1 = numpyro.deterministic("mu_m1",35.)
    mMin = numpyro.deterministic("mMin",9.)
    bq = numpyro.deterministic("bq",1.)

    sig_m1 = numpyro.deterministic("sig_m1",3.)
    log_f_peak = numpyro.deterministic("log_f_peak",-3.)
    mMax = numpyro.deterministic("mMax",80.)
    log_dmMin = numpyro.deterministic("log_dmMin",0.)
    log_dmMax = numpyro.deterministic("log_dmMax",1.)
    mu_chi = numpyro.deterministic("mu_chi",0.)
    logsig_chi = numpyro.deterministic("logsig_chi",-0.4)
    sig_cost = numpyro.deterministic("sig_cost",0.4)

    kappa = numpyro.sample("kappa",dist.Normal(0,5))
    #alpha = numpyro.sample("alpha",dist.Normal(-2,3))
    logR20 = numpyro.sample("logR20",dist.Uniform(-12,12))
    R20 = numpyro.deterministic("R20",10.**logR20)

    """
    logR20 = numpyro.sample("logR20",dist.Uniform(-12,12))
    alpha = numpyro.sample("alpha",dist.Normal(-2,3))
    mu_m1 = numpyro.sample("mu_m1",dist.Uniform(20,50))
    mMin = numpyro.sample("mMin",dist.Uniform(5,15))
    bq = numpyro.sample("bq",dist.Normal(0,3))
    kappa = numpyro.sample("kappa",dist.Normal(0,5))
    R20 = numpyro.deterministic("R20",10.**logR20)

    logit_sig_m1 = numpyro.sample("logit_sig_m1",dist.Normal(0,logit_std))
    logit_log_f_peak = numpyro.sample("logit_log_f_peak",dist.Normal(0,logit_std))
    logit_mMax = numpyro.sample("logit_mMax",dist.Normal(0,logit_std))
    logit_log_dmMin = numpyro.sample("logit_log_dmMin",dist.Normal(0,logit_std))
    logit_log_dmMax = numpyro.sample("logit_log_dmMax",dist.Normal(0,logit_std))
    logit_mu_chi = numpyro.sample("logit_mu_chi",dist.Normal(0,logit_std))
    logit_logsig_chi = numpyro.sample("logit_logsig_chi",dist.Normal(0,logit_std))
    logit_sig_cost = numpyro.sample("logit_sig_cost",dist.Normal(0,logit_std))

    sig_m1,jac_sig_m1 = get_value_from_logit(logit_sig_m1,2.,15.)
    log_f_peak,jac_log_f_peak = get_value_from_logit(logit_log_f_peak,-3,0.)
    mMax,jac_mMax = get_value_from_logit(logit_mMax,50.,100.)
    log_dmMin,jac_log_dmMin = get_value_from_logit(logit_log_dmMin,-1,1)
    log_dmMax,jac_log_dmMax = get_value_from_logit(logit_log_dmMax,0.5,1.5)
    mu_chi,jac_mu_chi = get_value_from_logit(logit_mu_chi,0.,1.)
    logsig_chi,jac_logsig_chi = get_value_from_logit(logit_logsig_chi,-1.,0.)
    sig_cost,jac_sig_cost = get_value_from_logit(logit_sig_cost,0.5,1.5)

    numpyro.deterministic("sig_m1",sig_m1)
    numpyro.deterministic("log_f_peak",log_f_peak)
    numpyro.deterministic("mMax",mMax)
    numpyro.deterministic("log_dmMin",log_dmMin)
    numpyro.deterministic("log_dmMax",log_dmMax)
    numpyro.deterministic("mu_chi",mu_chi)
    numpyro.deterministic("logsig_chi",logsig_chi)
    numpyro.deterministic("sig_cost",sig_cost)

    numpyro.factor("p_sig_m1",logit_sig_m1**2/(2.*logit_std**2)-jnp.log(jac_sig_m1))
    numpyro.factor("p_log_f_peak",logit_log_f_peak**2/(2.*logit_std**2)-jnp.log(jac_log_f_peak))
    numpyro.factor("p_mMax",logit_mMax**2/(2.*logit_std**2)-jnp.log(jac_mMax))
    numpyro.factor("p_log_dmMin",logit_log_dmMin**2/(2.*logit_std**2)-jnp.log(jac_log_dmMin))
    numpyro.factor("p_log_dmMax",logit_log_dmMax**2/(2.*logit_std**2)-jnp.log(jac_log_dmMax))
    numpyro.factor("p_mu_chi",logit_mu_chi**2/(2.*logit_std**2)-jnp.log(jac_mu_chi))
    numpyro.factor("p_logsig_chi",logit_logsig_chi**2/(2.*logit_std**2)-jnp.log(jac_logsig_chi))
    numpyro.factor("p_sig_cost",logit_sig_cost**2/(2.*logit_std**2)-jnp.log(jac_sig_cost))
    """

    # Fixed params
    mu_cost = 1.

    # Normalization
    p_m1_norm = massModel(20.,alpha,mu_m1,sig_m1,10.**log_f_peak,mMax,mMin,10.**log_dmMax,10.**log_dmMin)
    p_z_norm = (1.+0.2)**kappa

    ########################################
    # Compute expected number of detections
    ########################################

    # Read out injection CDFs
    inj_m1_cdfs = injectionCDFs['inj_m1_cdfs']
    inj_m2_cdfs = injectionCDFs['inj_m2_cdfs']
    inj_a1_cdfs = injectionCDFs['inj_a1_cdfs']
    inj_a2_cdfs = injectionCDFs['inj_a2_cdfs']
    inj_cost1_cdfs = injectionCDFs['inj_cost1_cdfs']
    inj_cost2_cdfs = injectionCDFs['inj_cost2_cdfs']
    inj_z_cdfs = injectionCDFs['inj_z_cdfs']

    # Define f(m1) over reference grid
    # Use this to obtain the integral over f(m1)/f(m_ref), build CDF, and interpolate to obtain m1 values
    reference_f_m1 = massModel(injectionCDFs['reference_m1_grid'],alpha,mu_m1,sig_m1,10.**log_f_peak,mMax,mMin,10.**log_dmMax,10.**log_dmMin)
    reference_cdf_m1 = jnp.cumsum(reference_f_m1)*injectionCDFs['dm1']      # This is the cumulative integral of f(m)
    reference_f_m1_integral = reference_cdf_m1[-1]                          # This is the integral Int(f(m)*dm)
    reference_cdf_m1 /= reference_f_m1_integral                             # This is cdf(m)
    inj_m1 = jnp.interp(inj_m1_cdfs,reference_cdf_m1,injectionCDFs['reference_m1_grid'])

    # Secondary masses
    inj_m2 = jnp.power(tmp_min**(1.+bq) + inj_m2_cdfs*(inj_m1**(1.+bq) - tmp_min**(1.+bq)),1./(1.+bq))

    # Spin magnitudes
    sqrt_2 = jnp.sqrt(2.)
    chi_erf_a = erf(-mu_chi/(10.**logsig_chi*sqrt_2))
    chi_erf_b = erf((1-mu_chi)/(10.**logsig_chi*sqrt_2))
    inj_a1 = mu_chi + 10.**logsig_chi*sqrt_2*erfinv(inj_a1_cdfs*(chi_erf_b-chi_erf_a) + chi_erf_a)
    inj_a2 = mu_chi + 10.**logsig_chi*sqrt_2*erfinv(inj_a2_cdfs*(chi_erf_b-chi_erf_a) + chi_erf_a)

    # Spin tilts
    cost_erf_a = erf((-1-mu_cost)/(sig_cost*sqrt_2))
    cost_erf_b = erf((1-mu_cost)/(sig_cost*sqrt_2))
    inj_cost1 = mu_cost + sig_cost*sqrt_2*erfinv(inj_cost1_cdfs*(cost_erf_b-cost_erf_a) + cost_erf_a)
    inj_cost2 = mu_cost + sig_cost*sqrt_2*erfinv(inj_cost2_cdfs*(cost_erf_b-cost_erf_a) + cost_erf_a)

    # Redshifts
    #inj_kappa = -1.
    #reference_f_z = injectionCDFs['reference_dVdz_grid']*(1.+injectionCDFs['reference_z_grid'])**(inj_kappa-1.)
    reference_f_z = injectionCDFs['reference_dVdz_grid']*(1.+injectionCDFs['reference_z_grid'])**(kappa-1.)
    reference_cdf_z = jnp.cumsum(reference_f_z)*injectionCDFs['dz']
    reference_f_z_integral = reference_cdf_z[-1]
    reference_cdf_z /= reference_f_z_integral
    inj_z = jnp.interp(inj_z_cdfs,reference_cdf_z,injectionCDFs['reference_z_grid'])

    injection_params = jnp.array([
        inj_m1,
        inj_m2,
        inj_a1,
        inj_a2,
        inj_cost1,
        inj_cost2,
        inj_z,
        injectionCDFs['cos_inclination'],
        injectionCDFs['polarization'],
        injectionCDFs['phi12'],
        injectionCDFs['right_ascension'],
        injectionCDFs['sin_declination']
        ])

    p_dets = Pdet(injection_params)

    Nexp = R20*(reference_f_m1_integral/p_m1_norm)*(reference_f_z_integral/p_z_norm)*jnp.mean(p_dets)
    #Nexp = R20*(reference_f_m1_integral/p_m1_norm)*(reference_f_z_integral/p_z_norm)*jnp.mean(Pdet(injection_params).T*(1.+inj_z)**(kappa-inj_kappa))
    numpyro.factor("rate",-Nexp)

    # As a fit diagnostic, compute effective number of injections
    nEff_inj = jnp.sum(p_dets)**2/jnp.sum(p_dets**2)
    nObs = 1.0*len(sampleDict)
    numpyro.deterministic("nEff_inj_per_event",nEff_inj/nObs)
    
    # This function defines the per-event log-likelihood
    # m1_sample: Primary mass posterior samples
    # m2_sample: Secondary mass posterior samples
    # z_sample: Redshift posterior samples
    # dVdz_sample: Differential comoving volume at each sample location
    # Xeff_sample: Effective spin posterior samples
    # priors: PE priors on each sample
    #def logp(m1_sample,m2_sample,z_sample,dVdz_sample,a1_sample,a2_sample,cost1_sample,cost2_sample,priors):
    def logp(m1_sample,m2_sample,z_sample,dVdz_sample,a1_sample,a2_sample,cost1_sample,cost2_sample,priors):

        # Compute proposed population weights
        p_m1 = massModel(m1_sample,alpha,mu_m1,sig_m1,10.**log_f_peak,mMax,mMin,10.**log_dmMax,10.**log_dmMin)/p_m1_norm
        p_m2 = (1.+bq)*m2_sample**bq/(m1_sample**(1.+bq)-tmp_min**(1.+bq))
        p_a1 = truncatedNormal(a1_sample,mu_chi,10.**logsig_chi,0,1)
        p_a2 = truncatedNormal(a2_sample,mu_chi,10.**logsig_chi,0,1)
        p_cost1 = truncatedNormal(cost1_sample,mu_cost,sig_cost,-1,1)
        p_cost2 = truncatedNormal(cost2_sample,mu_cost,sig_cost,-1,1)
        p_z = dVdz_sample*(1.+z_sample)**(kappa-1.)/p_z_norm
        R_pop = R20*p_m1*p_m2*p_z*p_a1*p_a2*p_cost1*p_cost2

        mc_weights = R_pop/priors

        #key,key_u = random.split(key)
        #m_choice = random.choice(key,m1_sample,p=mc_weights/jnp.sum(mc_weights))
        
        # Compute effective number of samples and return log-likelihood
        n_eff = jnp.sum(mc_weights)**2/jnp.sum(mc_weights**2)     
        return jnp.log(jnp.mean(mc_weights)),n_eff #m_choice
    
    # Map the log-likelihood function over each event in our catalog
    log_ps,n_effs = vmap(logp)(
                        jnp.array([sampleDict[k]['m1'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['m2'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['z'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['dVc_dz'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['a1'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['a2'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['cost1'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['cost2'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['z_prior'] for k in sampleDict]))
        
    # As a diagnostic, save minimum number of effective samples across all events
    numpyro.deterministic('min_log_neff',jnp.min(jnp.log10(n_effs)))

    # Tally log-likelihoods across our catalog
    numpyro.factor("logp",jnp.sum(log_ps))

