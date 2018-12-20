import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from time import time
import math
import seaborn as sns
import copy


def drift(array):
    
    "Compute risk-neutral drift vector of the multivariate sequence."
    
    log = np.log(array.tail(252)).diff()
    mu =  np.mean(log)
    v = log.var()
    d = mu - (.5*v)
    
    return d.values*252


def correlation_matrix(array):
    
    "Compute correlation matrix."
    
    return np.array(np.log(array.tail(252)).diff().corr())


def std(array):
    
    "Compute the standard deviation vector."
    
    std = np.log(array.tail(252)).diff().std()*np.sqrt(252)
    
    return std.values


def plot_joint_dist(dtf):
    
    "Plot the joint distribution."
    
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


def cholesky_simulation(assets,s1,r,sig,correlation_matrix,T,steps):
    
    "Generate correlated paths using the Cholesky algorithm."
    
    dt = T/steps
    cholesky = np.linalg.cholesky(correlation_matrix)
    z = np.random.randn(len(assets),steps+1)
    x = np.matmul(cholesky,z).T
    S = s1*np.exp(np.cumsum((r - 0.5*sig**2)*dt+sig*math.sqrt(dt)*x, axis = 0)) ; S[0] = s1
    S = pd.DataFrame(S,columns = assets)
    
    return S


def basket_path(assets,s1,r,sig,correlation_matrix,T,trials,steps):
    
    "Compute the basket paths as arithmetic averages of the simulated multivariate 
    " sequences."
    
    paths = []
    for simulation in range(trials):
        
        path = cholesky_simulation(assets,s1,r,sig,
                                   correlation_matrix,T,
                                   steps)
        paths.append(path.mean(axis=1).tolist())
    basket = np.array(paths)
    
    return basket


def least_squares_price(simulations,strike_percentage,r,Call=True):
    
    "Algorithm to recursively regress price from expiry to today, obtaining an American Basket Option's
    "value."
    
    strike = simulations[0][0]*strike_percentage
    
    c_p = 1
    if not Call:
        c_p *=-1
        
    paths = simulations
    for _ in range(simulations.shape[1] - 1):
        
        if _ ==0:
            
            m = paths[:,-2:]
            m[:,-1] = np.maximum((m[:,-1] - strike)*c_p,0)
            
        else:
            
            m = paths[:,-(2+_):-_]
            
        m[:,-2] = [x if np.maximum((x - strike)*c_p,0)>0 else 0 for x in m[:,-2]]
        adjusted = m[(m[:,-2]!=0),:]
        
        X = adjusted[:,-2]
        Y = adjusted[:,-1]*np.exp(-r)
        
        coeff = np.polyfit(X,Y,2)
        
        m[:,-2] = [coeff[0]*x**2 + coeff[1]*x + coeff[2] 
                  if np.maximum((x - strike)*c_p,0)>0 else 0 for x in m[:,-2]]
        m[:,-2] = [x if x>0 else 0 for x in m[:,-2]]
        
    paths =  paths[:,1:]
    
    for _ in range(len(paths)):
        
        paths[_,:] = [x if x==max(paths[_,:]) else 0 for x in paths[_,:]]
        
    for _ in range(paths.shape[1]):
        
        paths[:,_] = paths[:,_]*np.exp(-r*(_+1))
        
    return np.mean(paths.max(1))
