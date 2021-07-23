import numpy as np
import pandas as pd

from datetime import datetime

import pymc3 as pm
from pymc3.ode import DifferentialEquation
import arviz as az
import itertools


class SGNCompleteModel:
    
    @staticmethod
    def hill_equation(x, K, n):
        return x**n / (K**n + x**n)
    
    @staticmethod
    def gfp_only_model(y, t, p):
        
        #dependent variables
        Auto, OD = y[0], y[1]
        #a = p[0]
        #alpha, beta = extra
        a, alpha, beta = p[0], p[1], p[2]
        
        gamma = SGNCompleteModel.growth_rate(t, OD, alpha, beta)
        #differential equations
        dOD = gamma * OD
        dAuto = a - gamma * Auto
        return [dAuto, dOD]
    
    @staticmethod
    def gate_model_no_auto(y, t, p):
        
        bn, bc, bg, syn_ECFn, syn_ECFc, syn_ECF, deg, syn_GFP, deg_GFP, K, n = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10]
        ECFn, ECFc, ECF, GFP = y[0], y[1], y[2], y[3]
        ind1, ind2 = 1, 1
        
        #gamma = growth_rate(t, OD, alpha, beta)
        #differential equations
        #dOD = gamma * OD
        dECFn = bn + syn_ECFn * ind1 - deg * ECFn
        dECFc = bc + syn_ECFc * ind2 - deg * ECFc
        dECF = syn_ECF * ECFn * ECFc - deg * ECF
        
        dGFP = bg + syn_GFP * SGNCompleteModel.hill_equation(ECF, K, n) - deg_GFP * GFP

        return [dECFn, dECFc, dECF, dGFP]

fluos = pd.read_csv('marionette_fluo.csv', index_col='time')
gate = 'e11x32STPhoRadA'
fluo = fluos.loc[:, fluos.columns.str.startswith(gate)].iloc[:,3]
pars = ['bn', 'bc', 'bg', 'syn_ECFn', 'syn_ECFc', 'syn_ECF', 'deg', 'syn_GFP', 'deg_GFP', 'K', 'n']

beginning = datetime.now()
print('Started at:', beginning)

with pm.Model() as od_model:
    
    bn = pm.Uniform('bn', 0, 1e1)
    bc = pm.Uniform('bc', 0, 1e1)
    bg = pm.Uniform('bg', 0, 1e1)
    syn_ECFn = pm.Uniform('syn_ECFn', 0, 1e2)
    syn_ECFc = pm.Uniform('syn_ECFc', 0, 1e2)
    syn_ECF = pm.Uniform('syn_ECF', 0, 1e-4)
    syn_GFP = pm.Uniform('syn_GFP', 0, 1e5)
    deg = pm.Uniform('deg', 0, 1e-1)
    deg_GFP = pm.Uniform('deg_GFP', 0, 1e0)
    K = pm.Uniform('K', 0, 1e2)
    n = pm.Uniform('n', 0, 4)
    
    y_hat = pm.ode.DifferentialEquation(
        func=SGNCompleteModel.gate_model_no_auto, times=fluo.index, n_states=4, n_theta=11
    )(y0=[0, 0, 0, 0], theta=[bn, bc, bg, syn_ECFn, syn_ECFc, syn_ECF, deg, syn_GFP, deg_GFP, K, n])
    
    fluo_est = pm.Normal('fluo', mu=y_hat.T[3], sd=0.25, observed=fluo)
    
    step = pm.Metropolis()
    trace = pm.sample(1000, tune=2000, cores=-1, chains=2, step=step)

    data = az.from_pymc3(trace=trace)
    data.to_netcdf('Marionette-Complete-' + gate + '-' + datetime.now().strftime('%Y%m%d') + '.nc')
    
ending = datetime.now()
print('Finished at:', ending)
print('Execution time:', ending-beginning)