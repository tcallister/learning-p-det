import numpy as np
import sys
import tqdm
sys.path.append('./../code/')
from population_model import massModel, truncatedNormal


def evaluate_over_grids(samps, nDraws):

    """
    Helper function to evaluate population posteriors over grids of masses, redshifts, and spins.
    Used by `make_popsummary_standardInjections.py` and `make_popsummary_dynamicInjections.py`

    Parameters
    ----------
    samps : `arviz.InferenceData`
        Object holding posterior samples, as returned by `arviz.from_netcdf()`
    nDraws : `int`
        Number of draws to make from the provided posterior

    Returns
    -------
    m_grid : `numpy.array`
        Grid of primary masses
    q_grid : `numpy.array`
        Grid of mass ratios
    z_grid : `numpy.array`
        Grid of redshifts
    chi_grid : `numpy.array`
        Grid of component spin magnitudes
    cost_grid : `numpy.array`
        Grid of component (cosine) spin-orbit tilts
    p_ms : `numpy.array`
        Posterior samples for the normalized primary mass distribution.
        Shape `(nDraws, m_grid.size)`.
    p_qs : `numpy.array`
        Posterior samples for the normalized mass ratio distribution.
        Shape `(nDraws, q_grid.size)`.
    R_zs : `numpy.array`
        Posterior samples for total merger rate as a function of redshift.
        Shape `(nDraws, z_grid.size)`.
    p_chis : `numpy.array`
        Posterior samples for the component spin magnitude probability distribution.
        Shape `(nDraws, chi_grid.size)`.
    p_costs : `numpy.array`
        Posterior samples for the component spin tilt probability distribution.
        Shape `(nDraws, cost_grid.size)`.
    """

    # Define grids
    m_grid = np.linspace(2,100,500)
    q_grid = np.linspace(0,1,499)
    z_grid = np.linspace(0,1.5,500)
    M,Q = np.meshgrid(m_grid,q_grid)
    chi_grid = np.linspace(0,1,500)
    cost_grid = np.linspace(-1,1,500)
    
    # Initialize arrays to hold population results
    R_zs = np.zeros((nDraws,z_grid.size))
    p_ms = np.zeros((nDraws,m_grid.size))
    p_qs = np.zeros((nDraws,q_grid.size))
    p_chis = np.zeros((nDraws,chi_grid.size))
    p_costs = np.zeros((nDraws,cost_grid.size))
    
    # Loop across samples
    for i in tqdm.tqdm(range(nDraws)):

        # 1. Redshift distribution
        # ----------------------------

        # Evaluate (unnormalized) primary mass probability distribution
        p_m = massModel(m_grid,
                        samps.alpha.values[i],
                        samps.mu_m1.values[i],
                        samps.sig_m1.values[i],
                        10.**samps.log_f_peak.values[i],
                        samps.mMax.values[i],
                        samps.mMin.values[i],
                        10.**samps.log_dmMax.values[i],
                        10.**samps.log_dmMin.values[i])

        # Mass distribution at reference value (at which rate R20 is defined)
        p_m_ref = massModel(20.,
                        samps.alpha.values[i],
                        samps.mu_m1.values[i],
                        samps.sig_m1.values[i],
                        10.**samps.log_f_peak.values[i],
                        samps.mMax.values[i],
                        samps.mMin.values[i],
                        10.**samps.log_dmMax.values[i],
                        10.**samps.log_dmMin.values[i])
    
        # Integrate over masses to get net merger rate as a function of redshift
        R_zs[i,:] = samps.R20.values[i]*np.trapz(p_m/p_m_ref,m_grid)*((1.+z_grid)/(1.+0.2))**samps.kappa.values[i]

        # 2. Mass and mass ratio distributions
        # ------------------------------------

        # Evaluate mass and mass ratio distributions over 2D grids
        p_m = massModel(M,
                        samps.alpha.values[i],
                        samps.mu_m1.values[i],
                        samps.sig_m1.values[i],
                        10.**samps.log_f_peak.values[i],
                        samps.mMax.values[i],
                        samps.mMin.values[i],
                        10.**samps.log_dmMax.values[i],
                        10.**samps.log_dmMin.values[i])

        bq = samps.bq.values[i]
        p_q = (1.+bq)*Q**bq/(1.-(2./M)**(1.+bq))
        p_q[Q<=2/M] = 0
        p_2D = p_m*p_q

        # Integrate over each axis to get 1D marginal distributions
        p_m = np.sum(p_2D,axis=0)
        p_ms[i,:] = p_m/np.trapz(p_m,m_grid)
        p_q = np.sum(p_2D,axis=1)
        p_qs[i,:] = p_q/np.trapz(p_q,q_grid)

        # 3. Component spin distributions
        # -------------------------------
            
        # Evaluate and normalize spin magnitude distribution
        p_chi = truncatedNormal(chi_grid,
                    samps.mu_chi.values[i],
                    10.**samps.logsig_chi.values[i],
                    0,1)
        
        p_chis[i,:] = p_chi/np.trapz(p_chi,chi_grid)
        
        # Evaluate and normalize spin tilt distribution
        f_iso = samps.f_iso.values[i]
        p_cost = f_iso*(1./2.) \
            + (1.-f_iso)*truncatedNormal(cost_grid, samps.mu_cost.values[i], samps.sig_cost.values[i], -1, 1)

        p_costs[i,:] = p_cost/np.trapz(p_cost,cost_grid)
        
    return m_grid, q_grid, z_grid, chi_grid, cost_grid, p_ms, p_qs, R_zs, p_chis, p_costs
