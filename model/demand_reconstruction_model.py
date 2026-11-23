#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 12:54:31 2026

@author: cameronmoore
"""

import pytensor.tensor as pt
import pymc as pm
import numpy as np


# %%

## helper function for one term fourier expension
def fourier_seasonality(t, T1, a0, R1,phi):

    return (
        a0
        + R1 * pt.cos(T1 * np.pi * t / 6 - phi))

## returns mean of lognormal distribution
def inv_demand(p, mu, sigma, loc):
    p = pt.clip(p, 1e-12, 1.0 - 1e-12)  # numerical safety
    z = pt.sqrt(2.0) * pm.math.erfinv(2.0 * p - 1.0)  # Φ⁻¹(p)
    return loc + pt.exp(mu + sigma * z)

def std_norm_cdf(z):
    return 0.5 * (1.0 + pt.erf(z / pt.sqrt(2.0)))

## defines the integral of the product of the rate structure and lognormal pdf
def rev_function_lognorm_symbolic_rate_loc(
    mu, sigma, b,
    m_0, m_1, m_2, m_3, m_4, m_5, m_6,
    x_1, x_2, x_3, x_4, x_5, x_6,
    conn,p_null,
    demand_val=0.0,tau=1e-3):
    
    mu = mu + 1e-10
    sigma = sigma + 1e-10

    def softplus(x):
        # basic definition: log(1 + exp(x))
        return pt.log1p(pt.exp(x))

    def smooth_pos_part(z):
        return softplus(z / tau) * tau + 1e-10

    # Standard normal CDF via erf: Phi(z) = 0.5*(1 + erf(z / sqrt(2)))
    def std_norm_cdf(z):
        return 0.5 * (1.0 + pt.erf(z / pt.sqrt(2.0)))

    def lognorm_segment(m, x_low, x_high, b_val, x_thresh):
        ## helper function for calculating rev for a piece of rate structure
        x_low_eff=pt.minimum(pt.maximum(x_low,x_thresh),x_high)+1e-10

        ln_low = pt.log(x_low_eff)
        ln_high = pt.log(x_high)

        # convert to standard normal
        z_mean_low = (ln_low - mu - sigma**2) / sigma
        z_mean_high = (ln_high - mu - sigma**2) / sigma
        z_prob_low = (ln_low - mu) / sigma
        z_prob_high = (ln_high - mu) / sigma

        mean_prefactor = pt.exp(mu + (sigma ** 2) / 2.0)

        return (
            m / 1.0 * mean_prefactor * (std_norm_cdf(z_mean_high) - std_norm_cdf(z_mean_low)) +
            b_val * (std_norm_cdf(z_prob_high) - std_norm_cdf(z_prob_low)))

    # Block 1 (0 -> x1)
    i_1 = lognorm_segment(m_0, 0.0, x_1, b, demand_val)

    # Block 2
    y_1 = b + m_0 * x_1
    b_1 = y_1 - m_1 * x_1
    i_2 = lognorm_segment(m_1, x_1, x_2, b_1, demand_val)

    # Block 3
    y_2 = y_1 + m_1 * (x_2 - x_1)
    b_2 = y_2 - m_2 * x_2
    i_3 = lognorm_segment(m_2, x_2, x_3, b_2, demand_val)

    # Block 4
    y_3 = y_2 + m_2 * (x_3 - x_2)
    b_3 = y_3 - m_3 * x_3
    i_4 = lognorm_segment(m_3, x_3, x_4, b_3, demand_val)

    # Block 5
    y_4 = y_3 + m_3 * (x_4 - x_3)
    b_4 = y_4 - m_4 * x_4
    i_5 = lognorm_segment(m_4, x_4, x_5, b_4, demand_val)

    # Block 6
    y_5 = y_4 + m_4 * (x_5 - x_4)
    b_5 = y_5 - m_5 * x_5
    i_6 = lognorm_segment(m_5, x_5, x_6, b_5, demand_val)

    # Block 7 (tail to ∞)
    y_6 = y_5 + m_5 * (x_6 - x_5)
    b_6 = y_6 - m_6 * x_6

    tail_lower = pt.maximum(x_6, demand_val)+1e-10
    ln_tail = pt.log(tail_lower)
    z_mean_tail = (ln_tail - mu - sigma**2) / sigma
    z_prob_tail = (ln_tail - mu) / sigma

    tail = (
        m_6 * pt.exp(mu + (sigma ** 2) / 2.0) * (1.0 - std_norm_cdf(z_mean_tail)) +
        b_6 * (1.0 - std_norm_cdf(z_prob_tail))
    )

    total = i_1 + i_2 + i_3 + i_4 + i_5 + i_6 + tail
    total = softplus((total - 1e-10) / tau) * tau + 1e-10
   

    return conn*(1-p_null) * total + p_null*b*conn



# %%

## defines hierarchical model 
def demand_reconstruction_model(monthly_train_df,yearly_train_df,
                                household_obs_vals,household_obs_idx,yearID_obs_idx,include_household=True,
                                hyperpriors={},fix_CV=None):
    
    utility_to_utility_month = monthly_train_df["PWSID_idx"].values.astype("int64")
    month_to_utility_month = monthly_train_df["Month"].values.astype("int64")
    utility_year_to_utility_month = monthly_train_df["yearID_idx"].values.astype("int64")

    n_yearID = monthly_train_df['yearID'].nunique()
    n_util=monthly_train_df['PWSID'].nunique()
    
    with pm.Model() as hierarchical_model:
        
        
        obs_dem = pm.Data("obs_dem", np.log(monthly_train_df["dem"]*monthly_train_df["conn"]).values)
        obs_rev_data = pm.Data("obs_rev", np.log(np.array(yearly_train_df['rev']).astype(np.float64).reshape(-1)))

        m_0_vec=pm.Data('m_0',monthly_train_df['m_0'].values)
        m_1_vec=pm.Data('m_1',monthly_train_df['m_1'].values)
        m_2_vec=pm.Data('m_2',monthly_train_df['m_2'].values)
        m_3_vec=pm.Data('m_3',monthly_train_df['m_3'].values)
        m_4_vec=pm.Data('m_4',monthly_train_df['m_4'].values)
        m_5_vec=pm.Data('m_5',monthly_train_df['m_5'].values)
        m_6_vec=pm.Data('m_6',monthly_train_df['m_6'].values)
    
        x_1_vec=pm.Data('x_1',monthly_train_df['x_1'].values)
        x_2_vec=pm.Data('x_2',monthly_train_df['x_2'].values)
        x_3_vec=pm.Data('x_3',monthly_train_df['x_3'].values)
        x_4_vec=pm.Data('x_4',monthly_train_df['x_4'].values)
        x_5_vec=pm.Data('x_5',monthly_train_df['x_5'].values)
        x_6_vec=pm.Data('x_6',monthly_train_df['x_6'].values)
        conn_vec=pm.Data('conn',monthly_train_df['conn'].values)
        b_vec=pm.Data('b',monthly_train_df['b'].values)
        
        t_vec=pm.Data('t_vec',month_to_utility_month)
        
        null_rate=3

        
        ## specify hyperpriors as dictionary
        ## seasonality
        mu_phi=hyperpriors['mu_phi']()
        kappa_phi=hyperpriors['kappa_phi']()
        
        mu_a0_dbar=hyperpriors['mu_a0_dbar']()
        sigma_a0_dbar=hyperpriors['sigma_a0_dbar']()
        
        mu_a0_CV=hyperpriors['mu_a0_CV']()
        sigma_a0_CV=hyperpriors['sigma_a0_CV']()

        alpha_A1_frac_dbar=hyperpriors['alpha_A1_frac_dbar']()
        beta_A1_frac_dbar=hyperpriors['beta_A1_frac_dbar']()
        
        alpha_A1_frac_CV=hyperpriors['alpha_A1_frac_CV']()
        beta_A1_frac_CV=hyperpriors['beta_A1_frac_CV']()
        
        p_null=hyperpriors['p_null']()
        
        dem_error=hyperpriors['dem_error']()
        rev_error=hyperpriors['rev_error']()
        
        ## utility parameters
        phi_vec = pm.VonMises("phi_vec", mu=mu_phi, kappa=kappa_phi, shape=n_util)
        
        a0_dbar_vec=pm.LogNormal('a0_dbar_vec',mu=mu_a0_dbar,sigma=sigma_a0_dbar,shape=n_util)
        a0_CV_vec=pm.LogNormal('a0_CV_vec',mu=mu_a0_CV,sigma=sigma_a0_CV,shape=n_util)
       
        A1_frac_dbar=pm.Beta('A1_frac_dbar',alpha=alpha_A1_frac_dbar,beta=beta_A1_frac_dbar,shape=n_util)
        A1_frac_CV=pm.Beta('A1_frac_CV',alpha=alpha_A1_frac_CV,beta=beta_A1_frac_CV,shape=n_util)
        
        A1_dbar_vec=pm.Deterministic("A1_dbar_vec",A1_frac_dbar*a0_dbar_vec)
        A1_CV_vec=pm.Deterministic("A1_CV_vec",A1_frac_CV*a0_CV_vec)
        
        ## expand to observation index
        phi_ind = phi_vec[utility_to_utility_month]
        
        a0_dbar_ind = a0_dbar_vec[utility_to_utility_month]
        A1_dbar_ind = A1_dbar_vec[utility_to_utility_month]
        a0_CV_ind = a0_CV_vec[utility_to_utility_month]
        A1_CV_ind = A1_CV_vec[utility_to_utility_month]
        
    
        ## model seasonality
        dbar_season=fourier_seasonality(t=t_vec,T1=1,a0=a0_dbar_ind,R1=A1_dbar_ind,phi=phi_ind)
        
        if fix_CV is None:
            cv_season=fourier_seasonality(t=t_vec,T1=1,a0=a0_CV_ind,R1=A1_CV_ind,phi=phi_ind)
        else: 
            cv_season=pt.ones_like(t_vec, dtype="float64")*fix_CV
        
        ## convert to mu and sigma
        sigma_ind=pt.sqrt(pt.log(1+cv_season**2))
        mu_ind=pt.log(dbar_season)-(sigma_ind**2)/2  
        
        ## calculate observed metrics
        
        ### total utility demand (from both lognorm and exponential)
        d_ind = pt.exp(mu_ind + 0.5 * sigma_ind**2)
        d_total= d_ind*conn_vec*(1-p_null)+conn_vec*p_null/null_rate
        log_d_total = pt.log(pt.clip(d_total, 1e-10, np.inf))  
        
        ### annual revenue
        pred_rev_monthly=rev_function_lognorm_symbolic_rate_loc(mu_ind,sigma_ind,b_vec,m_0_vec,m_1_vec,m_2_vec,m_3_vec,m_4_vec,m_5_vec,m_6_vec,x_1_vec,x_2_vec,x_3_vec,x_4_vec,x_5_vec,x_6_vec,conn_vec,p_null)
        pred_rev_yearly = pt.extra_ops.bincount(utility_year_to_utility_month,weights=pt.reshape(pred_rev_monthly, (-1,)),minlength=n_yearID)
        log_pred_rev_yearly = pm.Deterministic("log_pred_rev_yearly", pt.log(pt.clip(pred_rev_yearly, 1e-10, np.inf)))
        
        ## constrain total monthly demand and annual reveue
        pm.Normal("demand_obs", mu=log_d_total, sigma=dem_error, observed=obs_dem)
        pm.Normal("rev_obs", mu=log_pred_rev_yearly, sigma=rev_error, observed=obs_rev_data)
        
        ## add household observations
        if include_household:
            household_obs = pm.Data("household_obs", household_obs_vals)                     # shape (N,)
            household_obs_idx = pm.Data("household_obs_idx", household_obs_idx.astype("int64"))  # shape (N,)
             
            ## model household demand for utility_months that we have obsevations for
            mu_hh=mu_ind[household_obs_idx]
            sigma_hh=sigma_ind[household_obs_idx]
             
            
            ## calculate likelihood
            log_hh  = pt.log(pt.clip(household_obs, 1e-10, np.inf))
            household_logpdf = (-0.5 * (pt.log(2 * np.pi) + 2 * pt.log(sigma_hh)+ ((log_hh - mu_hh) ** 2) / (sigma_hh ** 2))- log_hh)
  
            ## model chance of null values
            null_logpdf=pm.logp(pm.Exponential.dist(lam=null_rate),household_obs)     
             
            ## combine 
            log_p_null = pt.log(p_null)
            log_p_main = pt.log(1 - p_null)
            mixture_loglik = pm.math.logsumexp(pt.stack([log_p_null + null_logpdf,log_p_main + household_logpdf]),axis=0)
            pm.Potential("household_likelihood", pt.sum(mixture_loglik))
            
            
        ## save parameters
        pm.Deterministic("mu_ind",mu_ind)
        pm.Deterministic("sigma_ind", sigma_ind)
        pm.Deterministic("pred_rev_yearly", pred_rev_yearly)
        
            
        
    return hierarchical_model
