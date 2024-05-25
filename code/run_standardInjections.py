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
from population_model import baseline
from getData import *
import h5py
import sys

# Get dictionaries holding injections and posterior samples
injectionDict = getInjections()
sampleDict = getSamples(sample_limit=2000)

# Set up NUTS sampler over our likelihood
kernel = NUTS(baseline)
mcmc = MCMC(kernel,num_warmup=500,num_samples=3000,num_chains=nChains)

# Choose a random key and run over our model
rng_key = random.PRNGKey(119)
rng_key,rng_key_ = random.split(rng_key)
mcmc.run(rng_key_,sampleDict,injectionDict)
mcmc.print_summary()

# Save out data
data = az.from_numpyro(mcmc)
az.to_netcdf(data,"output_standardInjections.cdf")

