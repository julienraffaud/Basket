import pandas as pd
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


def cholesky_single_simulation(array,steps):
    c = correlation_matrix(array)
    cholesky = np.linalg.cholesky(c)
    z = np.random.randn(len(spot.columns),steps+1)
    x = np.matmul(cholesky,z).T
    return x


def cholesky_tensor(array,T,trials,steps):
    assets = array.columns ; d = drift(array) ; vol = std(array) ; s1 = array.iloc[0].values
    individual_asset_brownians = {}
    for asset in assets:
        individual_asset_brownians[asset] = []
    for simulation in range(trials):
        cholesky_sim = cholesky_single_simulation(array,steps)
        for nb in range(len(assets)):
            individual_asset_brownians[assets[nb]].append(cholesky_sim[:,nb].tolist())
    independent_simulations = []
    for _ in range(len(assets)):
        r = d[_] ; S0 = s1[_] ; sig = vol[_] ; T = 1  
        M = steps ; dt = T/M ; I = trials
        S = S0*np.exp(np.cumsum((r - 0.5*sig**2)*dt+sig*math.sqrt(dt)*np.array(individual_asset_brownians[assets[nb]]).T,axis=0)) ; S[0] = S0
        independent_simulations.append(S.T)
    tensor = np.array(independent_simulations)
    return tensor



#test loading data
ccy = ["USDZAR","USDTRY","USDMXN"]
path = "/Users/julienraffaud/Documents/FX/"
spot_list = []
for _ in ccy:
    df = pd.read_excel(path+_+".xlsx",header=None)
    df.columns = ["time",_]
    df = df.set_index("time")
    spot_list.append(df)
spot = pd.concat(spot_list,axis=1,join="inner")
spot['HOUR'] = spot.index.hour
spot["MINUTE"] = spot.index.minute
grp = spot.groupby(['HOUR', 'MINUTE'])
minute = [16,30]
spot= spot[(spot["HOUR"]==minute[0]) & (spot["MINUTE"]==minute[1])][spot.columns[:3]]

