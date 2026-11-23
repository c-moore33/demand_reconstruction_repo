#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 14:09:58 2026

@author: cameronmoore
"""


import pymc as pm
import pandas as pd
import arviz as az
from demand_reconstruction_model import demand_reconstruction_model
import random
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--version", type=int, default=6)
parser.add_argument("--cluster", action="store_true")
args = parser.parse_args()

version=args.version
cluster=args.cluster
print(version)
print(cluster)

# %%

## replace with local filepath
if cluster:
    train_df=pd.read_csv('demand_reconstruction/data/model_train.csv')
else:
    train_df=pd.read_csv('../preprocessing/cleaned_data/model_train.csv')

## get training datasets
monthly_train_df=train_df.loc[:,['yearID','yearID_idx','PWSID','PWSID_idx','utility_month','utility_month_idx','Month','dem','conn',
                                 'b','m_0','m_1','m_2','m_3','m_4','m_5','m_6','x_1','x_2','x_3','x_4','x_5','x_6']]
yearly_train_df=train_df.loc[:,['yearID','yearID_idx','PWSID','PWSID_idx','rev']]
yearly_train_df=yearly_train_df.drop_duplicates()


## replace with local filepath
if cluster:
    household_dem_df=pd.read_csv('demand_reconstruction/data/model_household_dem_df.csv')
else:
    household_dem_df=pd.read_csv('../preprocessing/cleaned_data/model_household_dem_df.csv')
    
household_dem_df=household_dem_df.drop(columns='Unnamed: 0')



# %%


### choose withhold scenario

### adding in utilities (randomly selected)

### initial randomization 
# withhold_PWSIDS = ['CA4410017','CA0110005','CA3410001','CA1910102','CA1910017']
# random.seed(42)
# random.shuffle(withhold_PWSIDS)

### only version 1 can be run without access to household billing data
### corresponds to no observation model 

if version==1:
    ### 1: all utilities (withhold 0)
    output_name='demand_reconstruction_model_withold_0'
    withhold_PWSIDS=[]
    fix_CV=None
    
elif version==2:
    ### 2: 4 utilites (withhold 1) 
    output_name='demand_reconstruction_model_withold_1.0'
    withhold_PWSIDS=['CA1910102']
    fix_CV=None

elif version==3:
    ### 3: 3 utilites (withhold 2)
    output_name='demand_reconstruction_model_withold_2'
    withhold_PWSIDS=['CA1910102','CA0110005']
    fix_CV=None

elif version==4:
    ### 4: 2 utilites (withhold 3)
    output_name='demand_reconstruction_model_withold_3'
    withhold_PWSIDS=['CA1910102','CA0110005','CA3410001']
    fix_CV=None


elif version==5:
    ## 5: 1 utility (withhold 4)
    output_name='demand_reconstruction_model_withold_4'
    withhold_PWSIDS=['CA1910102','CA0110005','CA3410001','CA1910017']
    fix_CV=None


elif version==6:
    ## 6: no utility (withhold 5)
    output_name='demand_reconstruction_model_withold_5'
    withhold_PWSIDS=['CA1910102', 'CA0110005', 'CA3410001', 'CA1910017', 'CA4410017']
    fix_CV=None

elif version==7:
    ### remove each utility, leave the rest in (remove CA1910102 redundancy)
    ## 7
    output_name='demand_reconstruction_model_withold_1.1'
    withhold_PWSIDS=['CA0110005']
    fix_CV=None

elif version==8:
    ## 8
    output_name='demand_reconstruction_model_withold_1.2'
    withhold_PWSIDS=['CA3410001']
    fix_CV=None

elif version==9:
    output_name='demand_reconstruction_model_withold_1.3'
    withhold_PWSIDS=['CA1910017']
    fix_CV=None

elif version==10:
    output_name='demand_reconstruction_model_withold_1.4'
    withhold_PWSIDS=['CA4410017']
    fix_CV=None
    

else:
    print('version not specified')

# %%

## parse version

household_dem_df=household_dem_df.loc[~household_dem_df.apply(lambda x: x['yearID'][4:] in withhold_PWSIDS,axis=1),:]
household_dem_df=household_dem_df.reset_index(drop=True)
if household_dem_df.shape[0]==0:
    include_household=False
else:
    include_household=True


## align household observation
obs_yearIDs=household_dem_df['yearID'].unique()
obs_yearIDS_df=monthly_train_df.loc[monthly_train_df['yearID'].apply(lambda x: x in obs_yearIDs),:]
obs_yearIDS_idx=obs_yearIDS_df.index.values
    

## fill in observation values
household_dem_obs=household_dem_df['dem'].values
household_dem_df['obs_idx']=np.nan   
for yearID in obs_yearIDs:
    for month in range(12):
            
        tmp=obs_yearIDS_df.loc[(obs_yearIDS_df['yearID']==yearID)&(obs_yearIDS_df['Month']==month),:]
            
        idx=tmp.index.values[0]
            
        household_dem_df.loc[(household_dem_df['yearID']==yearID)&(household_dem_df['month']==month),'obs_idx']=idx
                 
household_obs_idx=household_dem_df['obs_idx'].values
            
## add indicator to aggregate df  
monthly_train_df['has_hh']=monthly_train_df['yearID'].apply(lambda x: x in household_dem_df['yearID'].unique()).sum()




# %%


### specify hyperpriors
hyperpriors={
    'mu_phi':lambda: pm.Normal("mu_phi", mu=np.pi, sigma=1),
    'kappa_phi':lambda: pm.HalfNormal("kappa_phi", sigma=3),
    'mu_a0_dbar':lambda :pm.Normal('mu_a0_dbar',mu=2,sigma=1),
    'sigma_a0_dbar':lambda :pm.Exponential('sigma_a0_dbar',lam=1),
    'mu_a0_CV':lambda :pm.Normal('mu_a0_CV',mu=0, sigma=0.25),
    'sigma_a0_CV':lambda :pm.Exponential('sigma_a0_CV',lam=4),
    'alpha_A1_frac_dbar':lambda :pm.TruncatedNormal('alpha_A1_frac_dbar',mu=3,sigma=0.5,lower=0.01),
    'beta_A1_frac_dbar':lambda :pm.TruncatedNormal('beta_A1_frac_dbar',mu=3,sigma=0.5,lower=0.01),
    'alpha_A1_frac_CV':lambda :pm.TruncatedNormal('alpha_A1_frac_CV',mu=2,sigma=0.5,lower=0.01),
    'beta_A1_frac_CV':lambda :pm.TruncatedNormal('beta_A1_frac_CV',mu=10,sigma=0.5,lower=0.01),
    'p_null':lambda :pm.Beta('p_null',alpha=0.5,beta=5),
    'dem_error':lambda :pm.Exponential('dem_error',lam=4),
    'rev_error':lambda :pm.Exponential('rev_error',lam=4)
        }

    

## run model
hierarchical_model = demand_reconstruction_model(monthly_train_df, yearly_train_df,
                                                 household_dem_obs,household_obs_idx,obs_yearIDS_idx,include_household=include_household,
                                                 hyperpriors=hyperpriors,
                                                 fix_CV=fix_CV)

with hierarchical_model:
    trace_1=pm.sample(
        draws=20000,
        tune=5000,
        chains=5,
        step=pm.Metropolis(),
        cores=8,
        random_seed=70,
        progressbar=True,
        discard_tuned_samples=True)
    
## output to NetCDF file
az.to_netcdf(trace_1, "%s.nc"%output_name)
