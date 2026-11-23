#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:45:10 2026

@author: cameronmoore
"""
# from pathlib import Path
import sys

## change to your local filepath
sys.path.insert(0, "/Users/cameron/Library/CloudStorage/Box-Box/Cameron/central_computer/demand_reconstruction_public")

import pandas as pd 
from glob import glob
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from utility_class import Utility

#%%

## read model training datasets

## change to your local filepath
train_df=pd.read_csv('../preprocessing/cleaned_data/model_train.csv')

## read full model results
## change to your local filepath
## pick appropiate model output
file_path='../../../../../../../Desktop/traces/%s'
trace=az.from_netcdf(file_path%'demand_reconstruction_model_withold_0.nc')

## read in census data
## change to your local filepath
census_data_df=pd.read_csv('../preprocessing/cleaned_data/census_data.csv')


## intiate metric df and add census data
metric_df=train_df.loc[:,['yearID','Month','rev','dem','conn']]
metric_df=metric_df.rename(columns={'dem':'obs_dem','rev':'obs_rev'})
metric_df=metric_df.merge(census_data_df,left_on='yearID',right_on='Unnamed: 0',how='left')
metric_df=metric_df.drop(columns='Unnamed: 0')


# %%


## generate posterior demand distributions

mu_post=trace.posterior['mu_ind'][:,:,:].values
sigma_post=trace.posterior['sigma_ind'][:,:,:].values
p_null_post=trace.posterior['p_null'][:,:].values


mu_post=mu_post.reshape(-1,mu_post.shape[-1])
sigma_post=sigma_post.reshape(-1,sigma_post.shape[-1])
p_null_post=p_null_post.reshape(-1)
model_post=lognorm(scale=np.exp(mu_post),s=sigma_post)


### sample demand
demand_sample=model_post.rvs()

# %%



### expected model parameters
exp_mu=mu_post.mean(axis=0)
exp_sigma=sigma_post.mean(axis=0)
exp_CV=np.mean(np.sqrt(np.exp(sigma_post**2)-1),axis=0)
exp_dem=demand_sample.mean(axis=0)


metric_df['exp_mu']=exp_mu
metric_df['exp_sigma']=exp_sigma
metric_df['exp_CV']=exp_CV
metric_df['exp_dem']=exp_dem


# %%

## calculate relevant metrics 

#### start with metrics that don't depend on rate structure

#### proportion of users using at or below 6000 gals/month 
prop_6000g=model_post.cdf(6000/748.052)
metric_df['prop_6000g']=prop_6000g.mean(axis=0)


## proportion of users in last block 
last_block_thresholds=train_df.loc[:,['x_1','x_2','x_3','x_4','x_5','x_6']].max(axis=1).values
prop_last_block=1-model_post.cdf(last_block_thresholds)
prop_last_block=prop_last_block.mean(axis=0)
metric_df['prop_last_block']=prop_last_block

## slope of last block, ibr indicator, start of last block
metric_df['slope_last_block']=train_df.loc[:,['m_0','m_1','m_2','m_3','m_4','m_5','m_6']].max(axis=1)
metric_df['is_ibr']=train_df.loc[:,['m_0','m_1','m_2','m_3','m_4','m_5','m_6']].nunique(axis=1)>1
metric_df['start_last_block']=train_df.loc[:,['x_1','x_2','x_3','x_4','x_5','x_6']].max(axis=1)

# %%
## calculate metrics that require utility objects (parsing rate structures)
#### includes revenue gini and final costs

## intialize arrays

### household revenue sample for revenue gini
revenue_sample=np.zeros_like(demand_sample)
total_rev_sample=np.zeros_like(demand_sample)



### loop through yearIDs and define utility object
for i,yearID in enumerate(list(train_df['yearID'].unique())):
    
    mask=train_df.loc[train_df['yearID']==yearID,:].index


    yearID_samples=demand_sample[:,mask]

    utility=Utility(yearID,validated_filepath='../preprocessing/cleaned_data/%s')


    utility_rev=utility.household_rev_function(yearID_samples,vector=True)
    
    revenue_sample[:,mask]=utility_rev
    
    
       
    ## total monthly utily rev
    total_rev_sample[:,mask]=utility.null_model_rev(mu=mu_post[:,mask],sigma=sigma_post[:,mask],p_null=p_null_post[:,None])

    

### calculate revenue gini
revenue_sample=np.sort(revenue_sample,axis=0)
n=revenue_sample.shape[0]
rank=np.arange(1,n+1).reshape(-1, 1)
revenue_gini=(2*np.sum(rank*revenue_sample,axis=0))/(n*np.sum(revenue_sample,axis=0))-(n+1)/n




## save metrics
metric_df['revenue_gini']=revenue_gini
metric_df['total_monthly_rev']=total_rev_sample.mean(axis=0)
metric_df.to_csv('metric_results.csv')
