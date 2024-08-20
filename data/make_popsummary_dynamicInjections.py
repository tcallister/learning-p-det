import arviz as az
from popsummary.popresult import PopulationResult
import numpy as np
import sys
sys.path.append('../code/')
from population_model import massModel
from popsummary_generation_utilities import evaluate_over_grids

# Open popsummary object
result = PopulationResult(
        fname='popsummary_dynamicInjections.h5',
        )

# Load numpyro data
cdf_data = az.from_netcdf("output_dynamicInjections.cdf")
samps = cdf_data.posterior.stack(draws=("chain", "draw"))

# Extract hyperparameters to be stored
hyperposterior_samples = np.transpose(np.array([
        samps.R20.values,
        samps.alpha.values,
        samps.bq.values,
        samps.f_iso.values,
        samps.kappa.values,
        samps.log_dmMin.values,
        samps.log_dmMax.values,
        samps.log_f_peak.values,
        samps.logsig_chi.values,
        samps.mMax.values,
        samps.mMin.values,
        samps.mu_chi.values,
        samps.mu_m1.values,
        samps.sig_cost.values,
        samps.sig_m1.values,
        samps.nEff_inj_per_event.values,
        samps.min_log_neff.values
        ]))

# Metadata for each hyperparam (key, latex string, and docstring)
hyperparam_info = [
        ['R_ref', "$\\R_\\mathrm{ref}$", "Differential merger rate [units Gpc**(-3) yr**(-1) Msun**(-1) at m1=20Msun and z=0.2"],
        ['alpha', "$\\alpha$", "Power-law index on the primary mass distribution"],
        ['beta_q', "$\\beta_q$", "Power-law index on conditional mass ratio distribution p(m2|m1)"],
        ['f_iso', "$f_\\mathrm{iso}$", "Fraction of component spins contained in preferentially aligned mixture component"],
        ['kappa', "$\\kappa$", "Power law on the redshift evolution of the merger rate, scaling as (1+z)**kappa"],
        ['log_dmMin', "$\\log \\delta m_\\mathrm{min}$", "Log10 scale length over which primary mass distribution tapers below mMin"],
        ['log_dmMax', "$\\log \\delta m_\\mathrm{max}$", "Log10 scale length over which primary mass distribution tapers above mMax"],
        ['log_f_peak', "$\\log f_\\mathrm{peak}$", "Log10 of primary mass peak mixing fraction parameter"],
        ['logsig_chi', "$\\log \\sigma_{\\chi}$", "Log10 of standard deviation of component spin magnitude distribution"],
        ['mMax', "$m_\\mathrm{max}$", "Mass above which primary mass distribution is tapered to zero"],
        ['mMin', "$m_\\mathrm{min}$", "Mass below which primary mass distribution is tapered to zero"],
        ['mu_chi', "$\\mu_{\\chi}$", "Mean of component spin magnitude distribution"],
        ['mu_m1', "$\\mu_m$", "Mean of a Gaussian peak in the primary mass distribution"],
        ['sig_cost', "$\\sigma_u$", "Standard deviation of the cosine-spin-tilt distribution"],
        ['sig_m1', "$\\sigma_m$", "Standard deviation of a Gaussian peak in the primary mass distribution"],
        ['nEff_inj_per_event', "$N_\\mathrm{inj}/N_\\mathrm{obs}$", "Number of effective injections per observed CBC event"],
        ['min_log_neff', "$\\mathrm{min}\\log N_\\mathrm{eff}$", "Log10 of effective posterior samples, minimized across events"]
        ]

# Set metadata
hyperparams,hyperparams_latex,hyperparams_description = np.transpose(hyperparam_info)
result.set_metadata("hyperparameters",hyperparams.tolist(),overwrite=True)
result.set_metadata("hyperparameter_latex_labels",hyperparams_latex.tolist(),overwrite=True)
result.set_metadata("hyperparameter_descriptions",hyperparams_description.tolist(),overwrite=True)

# Save hyperposterior samples
result.set_hyperparameter_samples(hyperposterior_samples,overwrite=True)

# Get probabilities/rates evaluated over grids of CBC parameters 
nDraws = 3000
m1_grid, q_grid, z_grid, chi_grid, cost_grid,\
    p_m1s, p_qs, R_zs, p_chis, p_costs = evaluate_over_grids(samps, nDraws)

result.set_rates_on_grids('p_mass1',
                          grid_params='m1',
                          positions=m1_grid,
                          rates=p_m1s,
                          attribute_keys=['units', 'description'],
                          attributes=['Msun**(-1)', 'Marginal probability distribution of primary BBH masses.'],
                          overwrite=True)

result.set_rates_on_grids('p_mass_ratio',
                          grid_params='q',
                          positions=q_grid,
                          rates=p_qs,
                          attribute_keys=['units', 'description'],
                          attributes=['', 'Marginal probability distribution of primary BBH mass ratios.'],
                          overwrite=True)

result.set_rates_on_grids('rate_vs_redshift',
                          grid_params='z',
                          positions=z_grid,
                          rates=R_zs,
                          attribute_keys=['units', 'description'],
                          attributes=['Gpc**(-3) yr**(-1)', 'Total merger rate of BBHs per unit comoving volume, as a function of redshift.'],
                          overwrite=True)

result.set_rates_on_grids('p_chi',
                          grid_params='chi',
                          positions=chi_grid,
                          rates=p_chis,
                          attribute_keys=['units', 'description'],
                          attributes=['', 'Marginal probability distribution of primary BBH component spin magnitudes.'],
                          overwrite=True)

result.set_rates_on_grids('p_cos_theta',
                          grid_params='cos_theta',
                          positions=cost_grid,
                          rates=p_costs,
                          attribute_keys=['units', 'description'],
                          attributes=['', 'Marginal probability distribution of cosine BBH spin-orbit tilt angles.'],
                          overwrite=True)
