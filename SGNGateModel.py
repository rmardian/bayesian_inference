import numpy as np
import pandas as pd

from datetime import datetime

import pymc3 as pm
from pymc3.ode import DifferentialEquation
import arviz as az

class SGNGateModel:
    
    @staticmethod
    def hill_activation(x, K, n):
        return x**n / (K**n + x**n)
    
    @staticmethod
    def growth_rate(y, a, b):
        return (a * (1 - (y/b)))
    
    @staticmethod
    def gate_model(y, t, p):
        
        #dependent variables
        ECFn, ECFc, ECF, GFP, OD = y[0], y[1], y[2], y[3], y[4]
        bn, bc, bg, syn_ECFn, syn_ECFc, syn_ECF, deg, syn_GFP, deg_GFP, K, n, alpha, beta, _ = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13]
        ind1, ind2 = 1, 1
        
        gamma = SGNGateModel.growth_rate(OD, alpha, beta)
        #differential equations
        dOD = gamma * OD
        dECFn = bn + syn_ECFn * ind1 - (deg + gamma) * ECFn
        dECFc = bc + syn_ECFc * ind2 - (deg + gamma) * ECFc
        dECF = syn_ECF * ECFn * ECFc - (deg + gamma) * ECF
        
        dGFP = bg + syn_GFP * SGNGateModel.hill_activation(ECF, K, n) - (deg_GFP + gamma) * GFP

        return [dECFn, dECFc, dECF, dGFP, dOD]

fluos = pd.read_csv('marionette_fluo.csv', index_col='time')
gate = 'e42x32STIMPDH1'
fluo = fluos.loc[:, fluos.columns.str.startswith(gate)].iloc[:,3]

od_params = {
    'e38x32gp418': [0.014832545686692651, 1.206999008636522, 0.05586756964448041],
    'e20x32gp411': [0.014610032725458771, 1.2772684500977902, 0.04458182544478886],
    'e11x32STPhoRadA': [0.007844653484011118, 1.1134153588014837, 0.10602396117757999],
    'e34x30MjaKlbA': [0.013408372081617236, 1.14628949032757, 0.052259068950788926],
    'e32x30SspGyrB': [0.015133693377991284, 1.151731463297364, 0.04901732064094526],
    'e15x32NpuSspS2': [0.015778141196678447, 1.192983296185405, 0.049950021825183724],
    'e16x33NrdA2': [0.014463894907212425, 1.2421974135452174, 0.055150036439220534],
    'e41x32NrdJ1': [0.013145465488589058, 1.2385872370025641, 0.05941283831303974],
    'e42x32STIMPDH1': [0.012125122999895207, 1.2815026045141547, 0.0524312129470823],
}

beginning = datetime.now()
print('Started at:', beginning)
with pm.Model() as fluo_model:
    
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
    r, cc, c0 = od_params[gate]
        
    y_hat = pm.ode.DifferentialEquation(
        func=SGNGateModel.gate_model, times=fluos.index, n_states=5, n_theta=14
    )(y0=[0, 0, 0, 0, c0], theta=[bn, bc, bg, syn_ECFn, syn_ECFc, syn_ECF, deg, syn_GFP, deg_GFP, K, n, r, cc, c0])
    
    fluo_est = pm.Normal('fluo', mu=y_hat.T[3], sd=0.25, observed=fluo)
    
    step = pm.NUTS()
    trace = pm.sample(1000, tune=1000, cores=-1, chains=2, step=step)

    data = az.from_pymc3(trace=trace)
    data.to_netcdf('Marionette-NUTS' + datetime.now().strftime('%Y%m%d') + '.nc')
    
ending = datetime.now()
print('Finished at:', ending)
print('Execution time:', ending-beginning)