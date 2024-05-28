import numpyro
nChains = 1
numpyro.set_host_device_count(nChains)
numpyro.set_platform(platform='gpu')
from numpyro.infer import NUTS,MCMC
from jax.config import config
config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
import arviz as az
from population_model import baseline_dynamicInjections
from getData import *
import h5py
from astropy.cosmology import Planck15
import astropy.units as u
import sys
sys.path.append('../../p-det-O3/')
from p_det_O3.emulator import p_det_O3
from scipy.stats.qmc import Sobol

# Get dictionaries holding injections and posterior samples
sampleDict = getSamples(sample_limit=3000)

# Set up dictionary with precomputed quantities for sensitivity estimation
"""
nTrials = int(1e5)
injectionDict = {
    'inj_m1_cdfs':jnp.array(np.random.random(size=nTrials)),
    'inj_m2_cdfs':jnp.array(np.random.random(size=nTrials)),
    'inj_a1_cdfs':jnp.array(np.random.random(size=nTrials)),
    'inj_a2_cdfs':jnp.array(np.random.random(size=nTrials)),
    'inj_cost1_cdfs':jnp.array(np.random.random(size=nTrials)),
    'inj_cost2_cdfs':jnp.array(np.random.random(size=nTrials)),
    'inj_z_cdfs':jnp.array(np.random.random(size=nTrials))
    }
"""

sobol_sampler = Sobol(d=7, scramble=True)
sobol_samples = jnp.array(sobol_sampler.random_base2(m=19))
injectionDict = {
    'inj_m1_cdfs':sobol_samples[:,0],
    'inj_m2_cdfs':sobol_samples[:,1],
    'inj_a1_cdfs':sobol_samples[:,2],
    'inj_a2_cdfs':sobol_samples[:,3],
    'inj_cost1_cdfs':sobol_samples[:,4],
    'inj_cost2_cdfs':sobol_samples[:,5],
    'inj_z_cdfs':sobol_samples[:,6],
    }
nTrials = int(2**19)

# Fixed parameters that we'll not vary
injectionDict['right_ascension'] = jnp.array(2.*np.pi*np.random.random(size=nTrials))
injectionDict['sin_declination'] = jnp.array(2.*np.random.random(size=nTrials)-1.)
injectionDict['cos_inclination'] = jnp.array(2.*np.random.random(size=nTrials)-1.)
injectionDict['polarization'] = jnp.array(2.*np.pi*np.random.random(size=nTrials))
phi1 = jnp.array(2.*np.pi*np.random.random(size=nTrials))
phi2 = jnp.array(2.*np.pi*np.random.random(size=nTrials))
injectionDict['phi12'] = phi1-phi2 

# Finally, population with reference grids
injectionDict['reference_m1_grid'] = jnp.linspace(2.,100.,400)
injectionDict['dm1'] = jnp.diff(injectionDict['reference_m1_grid'])[0]
injectionDict['reference_z_grid'] = jnp.linspace(0.,1.9,400)
injectionDict['dz'] = jnp.diff(injectionDict['reference_z_grid'])[0]
injectionDict['reference_dVdz_grid'] = 4.*np.pi*Planck15.differential_comoving_volume(injectionDict['reference_z_grid']).to(u.Gpc**3/u.sr).value

p_det = p_det_O3(model_weights="/project/kicp/tcallister/trained_models/bbhOnly_3e4_1122_b_0.35_scaledSigmoid_full/job_16_weights.hdf5",
        scaler="/project/kicp/tcallister/trained_models/bbhOnly_3e4_1122_b_0.35_scaledSigmoid_full/job_16_input_scaler.pickle")

# Set up NUTS sampler over our likelihood
kernel = NUTS(baseline_dynamicInjections)
mcmc = MCMC(kernel,num_warmup=500,num_samples=1000,num_chains=nChains)

# Choose a random key and run over our model
rng_key = random.PRNGKey(122)
rng_key,rng_key_ = random.split(rng_key)
mcmc.run(rng_key_,sampleDict,injectionDict,p_det)
mcmc.print_summary()

# Save out data
data = az.from_numpyro(mcmc)
az.to_netcdf(data,"output_dynamicInjections_altNN2.cdf")

