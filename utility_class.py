#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 11:17:23 2025

@author: cameronmoore
"""

import pandas as pd
import numpy as np
from scipy.stats import norm




## requires df of yearID, rate structure, and number of connections
## must specify proper local filepath to this df


class Utility:
    def __init__(self,utility_id,validated=True,rates=pd.Series([]),validated_filepath='preprocessing/cleaned_data/%s'):
        ##
        if validated==True:
            
            validated_df=pd.read_csv(validated_filepath%'utility_init.csv')
            validated_df=validated_df.set_index('yearID')
            
            self.id=str(utility_id)
   
            ## parse cost structure
            
            
            if self.id in validated_df.index and 'conn' in validated_df.columns:
                self.N=validated_df.loc[self.id,'conn']
            else:
                self.N=1000
            self.b=validated_df.loc[self.id,'b']
            self.m_0=max(validated_df.loc[self.id,'m_0'],0) 
            self.x_1=max(validated_df.loc[self.id,'x_1'],0)
            self.m_1=max(validated_df.loc[self.id,'m_1'],self.m_0)
            self.x_2=max(validated_df.loc[self.id,'x_2'],self.x_1)
            self.m_2=max(validated_df.loc[self.id,'m_2'],self.m_1)
            self.x_3=max(validated_df.loc[self.id,'x_3'],self.x_2)
            self.m_3=max(validated_df.loc[self.id,'m_3'],self.m_2)
            self.x_4=max(validated_df.loc[self.id,'x_4'],self.x_3)
            self.m_4=max(validated_df.loc[self.id,'m_4'],self.m_3)
            self.x_5=max(validated_df.loc[self.id,'x_5'],self.x_4)
            self.m_5=max(validated_df.loc[self.id,'m_5'],self.m_4)
            self.x_6=max(validated_df.loc[self.id,'x_6'],self.x_5)
            self.m_6=max(validated_df.loc[self.id,'m_6'],self.m_5)
          
            
        else:
            self.b=rates.loc['b']
            self.m_0=max(rates.loc['m_0'],0) 
            self.x_1=max(rates.loc['x_1'],0)
            self.m_1=max(rates.loc['m_1'],self.m_0)
            self.x_2=max(rates.loc['x_2'],self.x_1)
            self.m_2=max(rates.loc['m_2'],self.m_1)
            self.x_3=max(rates.loc['x_3'],self.x_2)
            self.m_3=max(rates.loc['m_3'],self.m_2)
            self.x_4=max(rates.loc['x_4'],self.x_3)
            self.m_4=max(rates.loc['m_4'],self.m_3)
            self.x_5=max(rates.loc['x_5'],self.x_4)
            self.m_5=max(rates.loc['m_5'],self.m_4)
            self.x_6=max(rates.loc['x_6'],self.x_5)
            self.m_6=max(rates.loc['m_6'],self.m_5)
            
        

    ## parses water rate structure. can either take demand as a scalar or vector
    def household_rev_function(self,d,vector=False):

        ## inflection points of piecewise function
        inflections={}
        inflections['y_1']=self.b+self.m_0*max(0,(self.x_1-0))
        inflections['y_2']=inflections['y_1']+self.m_1*max(0,(self.x_2-self.x_1))
        inflections['y_3']=inflections['y_2']+self.m_2*max(0,(self.x_3-self.x_2))
        inflections['y_4']=inflections['y_3']+self.m_3*max(0,(self.x_4-self.x_3))
        inflections['y_5']=inflections['y_4']+self.m_4*max(0,(self.x_5-self.x_4))
        inflections['y_6']=inflections['y_5']+self.m_4*max(0,(self.x_6-self.x_5))

        ## if scalar value
        if vector==False:
            if self.x_1==0.0:
                y=self.b+d*self.m_0
            elif d<self.x_1:
                y=self.b+d*self.m_0
            elif d<self.x_2:
                y=(d-self.x_1)*self.m_1+inflections['y_1']
            elif d<self.x_3:
                y=(d-self.x_2)*self.m_2+inflections['y_2']
            elif d<self.x_4:
                y=(d-self.x_3)*self.m_3+inflections['y_3']
            elif d<self.x_5:
                y=(d-self.x_4)*self.m_4+inflections['y_4']
            elif d<self.x_6:
                y=(d-self.x_5)*self.m_5+inflections['y_5']
            else:
                y=(d-self.x_6)*self.m_6+inflections['y_6']

        ## if vector value
        else:
            y=np.ones_like(d)
            if self.x_1==0.0:
                y=self.b+d*self.m_0
            else:
                y[d<self.x_1]=self.b+d[d<self.x_1]*self.m_0
            y[(self.x_1<=d)&(d <self.x_2)]=((d[(self.x_1<=d)&(d <self.x_2)])-self.x_1)*self.m_1+inflections['y_1']
            y[(self.x_2<=d)&(d <self.x_3)]=((d[(self.x_2<=d)&(d <self.x_3)])-self.x_2)*self.m_2+inflections['y_2']
            y[(self.x_3<=d)&(d <self.x_4)]=((d[(self.x_3<=d)&(d <self.x_4)])-self.x_3)*self.m_3+inflections['y_3']
            y[(self.x_4<=d)&(d <self.x_5)]=((d[(self.x_4<=d)&(d <self.x_5)])-self.x_4)*self.m_4+inflections['y_4']
            y[(self.x_5<=d)&(d <self.x_6)]=((d[(self.x_5<=d)&(d <self.x_6)])-self.x_5)*self.m_4+inflections['y_5']
            y[self.x_6<=d]=((d[self.x_6<=d])-self.x_6)*self.m_6+inflections['y_6']

        return y

  

    
    ### can calculate the expected revenue of mixture model of 3 param lognormal and exponential 
    def null_model_rev(self, mu, sigma, p_null,loc=np.array([0]),d=0):
        
        with np.errstate(divide='ignore'):

            standard_norm = norm(loc=0, scale=1)
            N=self.N
            
            x_1=self.x_1
            x_2=self.x_2
            x_3=self.x_3
            x_4=self.x_4
            x_5=self.x_5
            x_6=self.x_6
          
                
            x_1=np.maximum(d,x_1)
            x_2=np.maximum(d,x_2)
            x_3=np.maximum(d,x_3)
            x_4=np.maximum(d,x_4)
            x_5=np.maximum(d,x_5)
            x_6=np.maximum(d,x_6)
                
            d_delta=d-loc
            x_1_delta=x_1-loc
            x_2_delta=x_2-loc
            x_3_delta=x_3-loc
            x_4_delta=x_4-loc
            x_5_delta=x_5-loc
            x_6_delta=x_6-loc
            
    
            x_1_delta[x_1_delta<0]=0
            x_2_delta[x_2_delta<0]=0
            x_3_delta[x_3_delta<0]=0
            x_4_delta[x_4_delta<0]=0
            x_5_delta[x_5_delta<0]=0
            x_6_delta[x_6_delta<0]=0
    
                
            
            i_1 =(self.m_0*(loc*(standard_norm.cdf((np.log(x_1_delta)-mu)/sigma)-standard_norm.cdf((np.log(d_delta)-mu)/sigma))
                            +np.exp(mu+(sigma**2)/2)*(standard_norm.cdf((np.log(x_1_delta)-mu-sigma**2)/sigma)-standard_norm.cdf((np.log(d_delta)-mu-sigma**2)/sigma)))
                  +self.b*(standard_norm.cdf((np.log(x_1_delta)-mu)/sigma)-standard_norm.cdf((np.log(d_delta)-mu)/sigma)))
            
            
            
        
            ## block 2
            y_1 = self.b + self.m_0 * self.x_1
            b_1 = y_1 - self.m_1 * self.x_1
            i_2 =(self.m_1*(loc*(standard_norm.cdf((np.log(x_2_delta)-mu)/sigma)-standard_norm.cdf((np.log(x_1_delta)-mu)/sigma))
                            +np.exp(mu+(sigma**2)/2)*(standard_norm.cdf((np.log(x_2_delta)-mu-sigma**2)/sigma)-standard_norm.cdf((np.log(x_1_delta)-mu-sigma**2)/sigma)))
                  +b_1*(standard_norm.cdf((np.log(x_2_delta)-mu)/sigma)-standard_norm.cdf((np.log(x_1_delta)-mu)/sigma)))
    
    
            ## block 3
            y_2 = y_1 + self.m_1 * (self.x_2 - self.x_1)
            b_2 = y_2 - self.m_2 * self.x_2
            i_3 =(self.m_2*(loc*(standard_norm.cdf((np.log(x_3_delta)-mu)/sigma)-standard_norm.cdf((np.log(x_2_delta)-mu)/sigma))
                            +np.exp(mu+(sigma**2)/2)*(standard_norm.cdf((np.log(x_3_delta)-mu-sigma**2)/sigma)-standard_norm.cdf((np.log(x_2_delta)-mu-sigma**2)/sigma)))
                  +b_2*(standard_norm.cdf((np.log(x_3_delta)-mu)/sigma)-standard_norm.cdf((np.log(x_2_delta)-mu)/sigma)))
    
        
            ## block 4
            y_3 = y_2 + self.m_2 * (self.x_3 - self.x_2)
            b_3 = y_3 - self.m_3 * self.x_3
            i_4 =(self.m_3*(loc*(standard_norm.cdf((np.log(x_4_delta)-mu)/sigma)-standard_norm.cdf((np.log(x_3_delta)-mu)/sigma))
                            +np.exp(mu+(sigma**2)/2)*(standard_norm.cdf((np.log(x_4_delta)-mu-sigma**2)/sigma)-standard_norm.cdf((np.log(x_3_delta)-mu-sigma**2)/sigma)))
                  +b_3*(standard_norm.cdf((np.log(x_4_delta)-mu)/sigma)-standard_norm.cdf((np.log(x_3_delta)-mu)/sigma)))
    
        
            ## block 5
            y_4 = y_3 + self.m_3 * (self.x_4 - self.x_3)
            b_4 = y_4 - self.m_4 * self.x_4
            i_5 =(self.m_4*(loc*(standard_norm.cdf((np.log(x_5_delta)-mu)/sigma)-standard_norm.cdf((np.log(x_4_delta)-mu)/sigma))
                            +np.exp(mu+(sigma**2)/2)*(standard_norm.cdf((np.log(x_5_delta)-mu-sigma**2)/sigma)-standard_norm.cdf((np.log(x_4_delta)-mu-sigma**2)/sigma)))
                  +b_4*(standard_norm.cdf((np.log(x_5_delta)-mu)/sigma)-standard_norm.cdf((np.log(x_4_delta)-mu)/sigma)))
            
        
            ## block 6
            y_5 = y_4 + self.m_4 * (self.x_5 - self.x_4)
            b_5 = y_5 - self.m_5 * self.x_5
            i_6 =(self.m_5*(loc*(standard_norm.cdf((np.log(x_6_delta)-mu)/sigma)-standard_norm.cdf((np.log(x_5_delta)-mu)/sigma))
                            +np.exp(mu+(sigma**2)/2)*(standard_norm.cdf((np.log(x_6_delta)-mu-sigma**2)/sigma)-standard_norm.cdf((np.log(x_5_delta)-mu-sigma**2)/sigma)))
                  +b_5*(standard_norm.cdf((np.log(x_6_delta)-mu)/sigma)-standard_norm.cdf((np.log(x_5_delta)-mu)/sigma)))
            
        
            ## block 7
            y_6 = y_5 + self.m_5 * (self.x_6 - self.x_5)
            b_6 = y_6 - self.m_6 * self.x_6  
            i_7 =(self.m_6*(loc*(1-standard_norm.cdf((np.log(x_6_delta)-mu)/sigma))
                            +np.exp(mu+(sigma**2)/2)*(1-standard_norm.cdf((np.log(x_6_delta)-mu-sigma**2)/sigma)))
                  +b_6*(1-standard_norm.cdf((np.log(x_6_delta)-mu)/sigma)))
            
            return (1-p_null)*N * (i_1 + i_2 + i_3 + i_4 + i_5 + i_6 + i_7)+p_null*N*1/3


    

