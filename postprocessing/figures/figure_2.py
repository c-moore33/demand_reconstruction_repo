#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:00:16 2026

@author: cameronmoore
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from scipy.stats import lognorm
from sklearn.metrics import r2_score


## change to proper local filepath
train_df=pd.read_csv('../../preprocessing/cleaned_data/model_train.csv')

## get training datasets
monthly_train_df=train_df.loc[:,['yearID','yearID_idx','PWSID','PWSID_idx','utility_month','utility_month_idx','Month','dem','conn','sf_demand',
                                 'b','m_0','m_1','m_2','m_3','m_4','m_5','m_6','x_1','x_2','x_3','x_4','x_5','x_6']]
yearly_train_df=train_df.loc[:,['yearID','yearID_idx','PWSID','PWSID_idx','rev']]
yearly_train_df=yearly_train_df.drop_duplicates()

## change to proper local filepath
file_path='../../model/%s'


## choose trace
full_trace='demand_reconstruction_model_withold_5.nc'

month_names=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']





# %%

## get observed data
obs_dem=monthly_train_df.loc[:,['yearID','dem','conn']]
monthly_train_df['total_dem']=monthly_train_df['dem']*monthly_train_df['conn']
obs_dem=monthly_train_df.loc[:,['yearID','total_dem']].groupby('yearID').sum()
               

obs_rev=yearly_train_df.loc[:,['yearID','rev']]
obs_rev=obs_rev.set_index('yearID')

# %%


## retrieve posterior samples
trace=az.from_netcdf(file_path%full_trace)

mu_post=trace.posterior.mu_ind[:,:,:].values
sigma_post=trace.posterior.sigma_ind[:,:,:].values
p_post=trace.posterior.p_null[:,:].values
rev_post=trace.posterior.pred_rev_yearly.values


mu_post=mu_post.reshape(-1,mu_post.shape[2])
sigma_post=sigma_post.reshape(-1,sigma_post.shape[2])
p_post=p_post.reshape(-1,1)
rev_post=rev_post.reshape(-1,rev_post.shape[2])

n_conn=monthly_train_df['conn'].values[np.newaxis,:]



## total demand samples
exp_dem=np.exp(mu_post+(sigma_post**2)/2)
total_dem=exp_dem*n_conn*(1-p_post)+n_conn*p_post/3




## aggregate by year
total_dem_df=pd.DataFrame(total_dem.T)
total_dem_df['yearID']=monthly_train_df['yearID']
total_annual_dem_df=total_dem_df.groupby('yearID').sum()

total_annual_exp_dem=total_annual_dem_df.mean(axis=1)




## revenue samples
exp_rev=rev_post.mean(axis=0)
exp_rev=pd.Series(exp_rev,index=yearly_train_df['yearID'])


## compare median RCIW for 4 vs 1 utils
model=lognorm(scale=np.exp(mu_post),s=sigma_post)
median_post=model.median()


q_975=np.percentile(median_post,97.5,axis=0)
q_25=np.percentile(median_post,2.5,axis=0)
mean_post=np.mean(median_post,axis=0)

RCIW_post=(q_975-q_25)/mean_post


util_4_PWSIDS=train_df['PWSID'].value_counts().loc[(train_df['PWSID'].value_counts()==48)].index
util_4_RCIW=RCIW_post[train_df['PWSID'].apply(lambda x: x in util_4_PWSIDS)]
print(util_4_RCIW.mean())


util_1_PWSIDS=train_df['PWSID'].value_counts().loc[(train_df['PWSID'].value_counts()==12)].index
util_1_RCIW=RCIW_post[train_df['PWSID'].apply(lambda x: x in util_1_PWSIDS)]
print(util_1_RCIW.mean())



# %%


## get posterior samples of highlight util 
highlight_yearID='2020CA0110003'
d_max=60
d_space=np.linspace(0,d_max,1000)
d_bins=np.linspace(0,d_max,40)


month_1_idx=monthly_train_df.loc[(monthly_train_df['yearID']==highlight_yearID)&(monthly_train_df['Month']==1),:].index[0]
month_2_idx=monthly_train_df.loc[(monthly_train_df['yearID']==highlight_yearID)&(monthly_train_df['Month']==7),:].index[0]



month_1_mu_post=mu_post[:,month_1_idx]
month_2_mu_post=mu_post[:,month_2_idx]

month_1_sigma_post=sigma_post[:,month_1_idx]
month_2_sigma_post=sigma_post[:,month_2_idx]

month_1_model=lognorm(scale=np.exp(month_1_mu_post[:,None]),s=month_1_sigma_post[:,None])
month_2_model=lognorm(scale=np.exp(month_2_mu_post[:,None]),s=month_2_sigma_post[:,None])




## cdf posterior samples
month_1_cdf_post=month_1_model.cdf(d_space[None,:])
month_1_cdf_low=np.percentile(month_1_cdf_post,2.5,axis=0)
month_1_cdf_high=np.percentile(month_1_cdf_post,97.5,axis=0)
month_1_cdf_exp=np.mean(month_1_cdf_post,axis=0)

month_2_cdf_post=month_2_model.cdf(d_space[None,:])
month_2_cdf_low=np.percentile(month_2_cdf_post,2.5,axis=0)
month_2_cdf_high=np.percentile(month_2_cdf_post,97.5,axis=0)
month_2_cdf_exp=np.mean(month_2_cdf_post,axis=0)


## demand posterior samples
month_1_demand_sample=month_1_model.rvs().reshape(-1)
month_2_demand_sample=month_2_model.rvs().reshape(-1)


## median samples
median_1_sample=month_1_model.median()
median_2_sample=month_2_model.median()

median_1_low=np.percentile(median_1_sample,2.5)
median_1_high=np.percentile(median_1_sample,97.5)

median_2_low=np.percentile(median_2_sample,2.5)
median_2_high=np.percentile(median_2_sample,97.5)

median_1_plot=np.linspace(median_1_low,median_1_high,1000)
median_2_plot=np.linspace(median_2_low,median_2_high,1000)

# %%

## generate figure

## formatting 

## font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'


## label sizes
plt.rcParams['axes.titlesize'] = 14     # Font size for individual subplot titles
plt.rcParams['xtick.labelsize'] = 10    # Font size for X-axis tick labels
plt.rcParams['ytick.labelsize'] = 10    # Font size for Y-axis tick labels
plt.rcParams['figure.titlesize'] = 18   # Font size for the overarching main figure title (suptitle)
plt.rcParams['axes.labelsize'] = 12 
plt.rcParams["axes.grid"] = False




### intialize plots

fig = plt.figure(figsize=(10,8))

gs = fig.add_gridspec(2, 2, hspace=0.4)  # increase/decrease as needed

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

true_line=np.linspace(0,1e10,1000)




## plot demand


ax1.scatter(obs_dem,total_annual_exp_dem)




ax1.plot(true_line,true_line,'--',color='#666666',zorder=5)
ax1.scatter(obs_dem,total_annual_exp_dem,color='#005587',zorder=8)
ax1.scatter(obs_dem.loc[highlight_yearID],total_annual_exp_dem.loc[highlight_yearID],color='red',zorder=9)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(True)




## plot rev

ax2.plot(true_line,true_line,'--',color='#666666',zorder=5)
ax2.scatter(obs_rev,exp_rev,color='#005587',zorder=8)
ax2.scatter(obs_rev.loc[highlight_yearID],exp_rev.loc[highlight_yearID],color='red',zorder=9)

ax2.set_xscale('log')
ax2.set_yscale('log')

print(r2_score(np.log(obs_dem),np.log(total_annual_exp_dem)))
print(r2_score(np.log(obs_rev),np.log(exp_rev)))
ax2.grid(True)



## plot cdfs


ax3.plot(d_space,month_1_cdf_exp,color='#993399')
ax3.fill_between(d_space,month_1_cdf_low,month_1_cdf_high,color='#993399',alpha=0.5)


ax3.plot(d_space,month_2_cdf_exp,color='#339898')
ax3.fill_between(d_space,month_2_cdf_low,month_2_cdf_high,color='#339898',alpha=0.5)


ax3.plot(d_space,np.ones_like(d_space)*0.5,'--',color='#666666')
ax3.plot(median_1_plot,np.ones_like(median_1_plot)*0.5,color='red',linewidth=3)
ax3.plot(median_2_plot,np.ones_like(median_2_plot)*0.5,color='red',linewidth=3)
ax3.grid(True)




ax4.hist(month_1_demand_sample,bins=d_bins,color='#993399',alpha=0.5,density=True)
ax4.hist(month_2_demand_sample,bins=d_bins,color='#339898',alpha=0.5,density=True)
ax4.grid(True, axis="x")




## figure labels

ax1.set_title('Total demand')
ax1.set_xlabel('Observed log demand (ccf)')
ax1.set_ylabel('Expected log demand (ccf)')

ax2.set_title('Total revenue')
ax2.set_xlabel('Observed log revenue ($)')
ax2.set_ylabel('Expected log revenue ($)')

ax3.set_title('Household demand distribution')
ax3.set_xlabel('Household usage (ccf per month)')
ax3.set_ylabel('Cumulative probability')

ax4.set_title('Household demand sample')
ax4.set_xlabel('Household usage (ccf per month)')
ax4.set_ylabel('Density')




## set limits


ax1.set_xlim(obs_dem.values.min()*2.72e-1,obs_dem.values.max()*2.72)
ax1.set_ylim(obs_dem.values.min()*2.72e-1,obs_dem.values.max()*2.72)

ax2.set_xlim(obs_rev.values.min()*2.72e-1,obs_rev.values.max()*2.72)
ax2.set_ylim(obs_rev.values.min()*2.72e-1,obs_rev.values.max()*2.72)


ax3.set_xlim(0,d_max)
ax3.set_ylim(0,1)

ax4.set_xlim(0,d_max)
ax4.set_ylim(0)




for ax, label in zip([ax1, ax2, ax3, ax4], ['a', 'b', 'c','d']):
    pos = ax.get_position()
    fig.text(
        pos.x0 - 0.04, pos.y1 + 0.01, label,
        ha='right', va='bottom',
        fontsize=16, fontweight='black',fontfamily='Arial'
    )
    
    
plt.show()



