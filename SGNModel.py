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
    def growth_rate(y, a, b):
        return (a * (1 - (y/b)))
    
    @staticmethod
    def gate_model(y, t, p):
        
        #dependent variables
        ECFn, ECFc, ECF, GFP, OD = y[0], y[1], y[2], y[3], y[4]
        bn, bc, bg, syn_ECFn, syn_ECFc, syn_ECF, deg, syn_GFP, deg_GFP, K, n, alpha, beta, _ = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13]
        ind1, ind2 = 1, 1
        
        gamma = SGNModel.growth_rate(OD, alpha, beta)
        #differential equations
        dOD = gamma * OD
        dECFn = bn + syn_ECFn * ind1 - (deg + gamma) * ECFn
        dECFc = bc + syn_ECFc * ind2 - (deg + gamma) * ECFc
        dECF = syn_ECF * ECFn * ECFc - (deg + gamma) * ECF
        
        dGFP = bg + syn_GFP * SGNModel.hill_activation(ECF, K, n) - (deg_GFP + gamma) * GFP

        return [dECFn, dECFc, dECF, dGFP, dOD]

fluos = pd.read_csv('datasets/marionette_fluo.csv', index_col='time')
gates = [g for g in list(set([i[:-3] for i in fluos.columns.tolist()])) if g not in ['positive_control', 'negative_control', 'blank']]
fluos_arr = [fluos.loc[:, fluos.columns.str.startswith(gate)].iloc[:,3] for gate in gates]

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
    
    y_hats = [None] * len(gates)
    fluo_ests = [None] * len(gates)
    ff = [0] * len(gates)
    hh = [0] * len(gates)
    
    aa = [2 for i in range(9)]
    bb = [2 for i in range(9)]
    c = 2
    dd = [50 for i in range(9)]
    ee = [50 for i in range(9)]
    #ff = [f0, f1, f2, f3, f4, f5, f6, f7, f8]
    gg = [0.05 for i in range(9)]
    #hh = [h0, h1, h2, h3, h4, h5, h6, h7, h8]
    i = 0.05
    jj = [20 for i in range(9)]
    kk = [4 for i in range(9)]
    #r, cc, c0 = 0.007844653484011118, 1.1134153588014837, 0.10602396117757999
    
    for idx in range(len(gates)):
        
        ff[idx] = pm.Uniform('syn_ECF_' + gates[idx], 0, 1e2)
        hh[idx] = pm.Uniform('syn_GFP_' + gates[idx], 0, 1e5)
        
    for idx in range(len(gates)):
        
        r, cc, c0 = od_params[gates[idx]]
        
        y_hats[idx] = pm.ode.DifferentialEquation(
            func=SGNModel.gate_model, times=fluos.index, n_states=5, n_theta=14
        )(y0=[0, 0, 0, 0, c0], theta=[aa[idx], bb[idx], c, dd[idx], ee[idx], ff[idx], gg[idx], hh[idx], i, jj[idx], kk[idx], r, cc, c0])
    
        fluo_ests[idx] = pm.Normal('fluo_' + gates[idx], mu=y_hats[idx].T[0], sd=0.25, observed=fluos_arr[idx])
    
    step = pm.Metropolis()
    trace = pm.sample(1000, tune=1000, cores=-1, chains=2, step=step)

    data = az.from_pymc3(trace=trace)
    data.to_netcdf('Marionette-' + datetime.now().strftime('%Y%m%d') + '.nc')
    
ending = datetime.now()
print('Finished at:', ending)
print('Execution time:', ending-beginning)