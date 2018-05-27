import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import math
from time import time
import seaborn as sns
import copy


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


def joint_dist(dtf):
    sns.set()
    assets = dtf.columns
    cmp = []
    for _ in assets:
        for o in [x for x in assets if x!=_]:
            if (str(_+o) not in cmp) & (str(o+_) not in cmp):
                u = sns.jointplot(x=_,y=o,data= np.log(dtf).diff(),kind="kde")
                cmp.append(str(_+o))
            else:
                pass
    return None


def cholesky_simulation(array,T,steps):
    assets = array.columns ; r = drift(array) ; sig = std(array) 
    s1 = array.iloc[0].values ; M = steps ; dt = T/M
    c = correlation_matrix(array)
    cholesky = np.linalg.cholesky(c)
    z = np.random.randn(len(spot.columns),steps+1)
    x = np.matmul(cholesky,z).T
    S = s1*np.exp(np.cumsum((r - 0.5*sig**2)*dt+sig*math.sqrt(dt)*x, axis = 0)) ; S[0] = s1
    S = pd.DataFrame(S,columns = array.columns)
    return S


def basket_path(array,T,trials,steps):
    paths = []
    for simulation in range(trials):
        path = cholesky_simulation(array,T,steps)
        paths.append(path.mean(axis=1).tolist())
    basket = np.array(paths)
    return basket


def basket_price(array,strike_percentage,Call=True):
    strike = strike_percentage*array[0][0]
    display(strike)
    payoff = copy.deepcopy(array)
    if Call:
        payoff = np.maximum(payoff - strike,0)
    else:
        payoff = np.maximum(strike - payoff,0)
    return payoff


def polynomial_regression(independent,dependent):
    return None

