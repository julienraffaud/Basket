
# coding: utf-8

# In[115]:


import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import math
from time import time


def drift(array):
    log = np.log(array.tail(252)).diff()
    mu =  np.mean(log)
    v = log.var()
    d = mu - (.5*v)
    return d.values*252


def correlation_matrix(array):
    return np.array(np.log(array.tail(252)).diff().corr())


def std(array):
    std = np.log(array.tail(252)).diff().std()*np.sqrt(252)
    return std.values


def tensor_mc(array,T,trials,steps):
    d = drift(array) ; vol = std(array) ; s1 = spt.iloc[0].values
    independent_simulations = []
    
    for _ in range(len(s1)):
        
        r = d[_] ; S0 = s1[_] ; sig = vol[_] ; T = 1  
        M = steps ; dt = T/M ; I = trials
        S = S0*np.exp(np.cumsum((r - 0.5*sig**2)*dt+sig*math.sqrt(dt)*                                np.random.standard_normal((M+1,I)),axis=0)) ; S[0] = S0
        independent_simulations.append(S.T)
    tensor = np.array(independent_simulations)
    
    return tensor


def cholesky_simulation(array,T,trials,steps):
    correl = correlation_matrix(array)
    cholesky = np.linalg.cholesky(correl)
    tensor = tensor_mc(array,T,trials,steps)
    m = np.mean(array).values
    std = array.std().values
    
    standardised = []

    for _ in range(len(tensor)):
        
        mod = ( tensor[_] - m[_] ) / std[_]
        standardised.append(mod)


# In[19]:


ccy = ["USDZAR","USDTRY","USDMXN"]
path = "/Users/julienraffaud/Documents/FX/"
spot_list = []
for _ in ccy:
    df = pd.read_excel(path+_+".xlsx",header=None)
    df = df[df.columns[:2]]
    df.columns = ["time",_]
    df = df.set_index("time")
    spot_list.append(df)
spot = pd.concat(spot_list,axis=1,join="inner")
spot['HOUR'] = spot.index.hour
spot["MINUTE"] = spot.index.minute
grp = spot.groupby(['HOUR', 'MINUTE'])
minute = [16,30]
spot= spot[(spot["HOUR"]==minute[0]) & (spot["MINUTE"]==minute[1])][spot.columns[:3]]

