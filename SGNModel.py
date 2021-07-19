import numpy as np
import pandas as pd

from datetime import datetime

import pymc3 as pm
from pymc3.ode import DifferentialEquation
import arviz as az

class SGNModel:
    
    @staticmethod
    def hill_activation(x, K, n):
        return x**n / (K**n + x**n)
    
    @staticmethod
    def growth_rate(t, y, a, b):
        return (a * (1 - (y/b)))
    
    @staticmethod
    def gate_model_auto_only(y, t, p):
        
        Auto, OD = y[0], y[1]
        a, alpha, beta = p[0], p[1], p[2]
        
        gamma = SGNModel.growth_rate(t, OD, alpha, beta)
        
        dOD = gamma * OD
        dAuto = a - gamma * Auto
        return [dAuto, dOD]
    
    @staticmethod
    def gate_model_no_auto(y, t, p):
        
        bn, bc, bg, syn_ECFn, syn_ECFc, syn_ECF, deg, syn_GFP, deg_GFP, K, n = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10]
        ECFn, ECFc, ECF, GFP = y[0], y[1], y[2], y[3]
        ind1, ind2 = 1, 1
        
        dECFn = bn + syn_ECFn * ind1 - deg * ECFn
        dECFc = bc + syn_ECFc * ind2 - deg * ECFc
        dECF = syn_ECF * ECFn * ECFc - deg * ECF
        
        dGFP = bg + syn_GFP * SGNModel.hill_activation(ECF, K, n) - deg_GFP * GFP
        return [dECFn, dECFc, dECF, dGFP]

fluos = pd.read_csv('marionette_fluo.csv', index_col='time')
ods = pd.read_csv('marionette_od.csv', index_col='time')
gates = list(set([i[:-3] for i in fluos.columns.tolist()]))
gate = 'e42x32STIMPDH1'
fluo_sel = fluos.loc[:, fluos.columns.str.startswith(gate)]
od_sel = ods.loc[:, ods.columns.str.startswith(gate)]
fluo = fluo_sel.iloc[:,3]
od = od_sel.iloc[:,3]

beginning = datetime.now()
print('Started at:', beginning)
with pm.Model() as bayesian_model:
    
    bn = pm.Uniform('bn', 0, 1e1)
    bc = pm.Uniform('bc', 0, 1e1)
    bg = pm.Uniform('bg', 0, 1e1)
    syn_ECFn = pm.Uniform('syn_ECFn', 0, 1e2)
    syn_ECFc = pm.Uniform('syn_ECFc', 0, 1e2)
    syn_ECF = pm.Uniform('syn_ECF', 0, 1e-4)
    syn_GFP = pm.Uniform('syn_GFP', 0, 1e-1)
    deg = pm.Uniform('deg', 0, 1e5)
    deg_GFP = pm.Uniform('deg_GFP', 0, 1e0)
    K = pm.Uniform('K', 0, 1e2)
    n = pm.Uniform('n', 0, 4)
    
    y_hat = DifferentialEquation(
        func=SGNModel.gate_model_no_auto, times=fluo.index, n_states=4, n_theta=11
    )(y0=[0, 0, 0, 0], theta=[bn, bc, bg, syn_ECFn, syn_ECFc, syn_ECF, deg, syn_GFP, deg_GFP, K, n])
    
    fluo_est = pm.Normal('fluo', mu=y_hat.T[0], sd=0.25, observed=fluo)
    
    step = pm.Metropolis()
    trace = pm.sample(5000, tune=3000, cores=1, step=step)
    
with bayesian_model:
    data = az.from_pymc3(trace=trace)
    data.to_netcdf(gate + '-' + datetime.now().strftime('%Y%m%d') + '.nc')
    
ending = datetime.now()

print('Finished at:', ending)
print('Execution time:', ending-beginning)