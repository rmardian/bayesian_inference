import numpy as np
import pandas as pd

from datetime import datetime

import pymc3 as pm
from pymc3.ode import DifferentialEquation
import arviz as az

hill_params = {
    'e11x32STPhoRadA': [8.19215086e+00, 2.77324618e+00, 6.00000000e+00, 1.24643956e+00, 5.95605924e+00, 3.46957143e-09, 2.99296531e+02, 5.36128135e+02],
    'e15x32NpuSspS2': [9.8458614, 8.7957319, 2.89598247, 1., 36.84876916, 30.33289918, 156.72063863, 156.06142356],
    'e16x33NrdA2': [10.13165313, 6.30849837, 5.02963422, 1.26435637, 15.99371476, 6.8959355, 380.68087314, 387.7069378],
    'e20x32gp411': [12.53828935, 1., 2.78451886, 1.49241488, 8.39069666, 27.77880598, 314.77726787, 311.65818168],
    'e32x30SspGyrB': [10.77387795, 8.56616674, 3.12574014, 1.07032582, 15.2982435, 11.91592347, 342.96437349, 343.5314864],
    'e34x30MjaKlbA': [8.42632247, 13.40974257, 3.28555513, 1.81757507, 3.4673668, 20.82148359, 307.13693296, 290.48137472],
    'e38x32gp418': [7.61231223, 1.51099399, 5.04169259, 1.4068252, 26.05989294, 20.59322098, 218.62810381, 218.64413488],
    'e41x32NrdJ1': [9.59574651, 12.05238497, 3.84271899, 1., 21.72980962, 19.19063999, 277.09322359, 275.39980816],
    'e42x32STIMPDH1': [10.41225458, 5.87647366, 4.30770405, 1.01184319, 22.82771137, 18.70845616, 228.18083668, 227.98611955]
}

class AlternativeModel:

    @staticmethod
    def hill_activation(x, K, n, ymin, ymax):
    
        return ymin + (ymax - ymin) * (x**n / (K**n + x**n))
    
    @staticmethod
    def hill_activation_and(x1, x2, K1, K2, n1, n2, ymin1, ymin2, ymax1, ymax2):

        return AlternativeModel.hill_activation(x1, K1, n1, ymin1, ymax1) * AlternativeModel.hill_activation(x2, K2, n2, ymin2, ymax2)
    
    @staticmethod
    def gate_model(y, t, p):

        y1, GFP = y[0], y[1]
        r, synGFP, degGFP, x1, x2, k1, k2, n1, n2, ymin1, ymin2, ymax1, ymax2 = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12]
        ymax = AlternativeModel.hill_activation_and(x1, x2, k1, k2, n1, n2, ymin1, ymin2, ymax1, ymax2)
        dy1 = r * (1 - (y1/ymax)) * y1
        dGFP = synGFP * y1 - degGFP * GFP
        return [dy1, dGFP]

fluos = pd.read_csv('induction_fluo.csv', index_col='time')
gates = ['e11x32STPhoRadA', 'e15x32NpuSspS2', 'e16x33NrdA2', 'e20x32gp411', 'e32x30SspGyrB',
         'e34x30MjaKlbA', 'e38x32gp418', 'e41x32NrdJ1', 'e42x32STIMPDH1']
cumas = [0, 6.25, 12.5, 25, 50, 100]
aras = [0, 0.8125, 3.25, 13, 52, 208]
gate = 'e11x32STPhoRadA'
a, b = 5, 5
cuma, ara = cumas[a], aras[b]
fluo = fluos['{}_{}{}'.format(gate, a, b)]

beginning = datetime.now()
print('Started at:', beginning)
with pm.Model() as bayesian_model:
    
    #sigma = pm.Normal('sigma', mu=y.mean(), sigma=0.5*ymin)
    r = pm.Normal('r', mu=1e2, sigma=5e1)
    y0 = pm.Normal('y0', mu=1e2, sigma=5e1)
    synGFP = pm.Normal('synGFP', mu=1e2, sigma=5e1)
    degGFP = 0.01#pm.Normal('n1', mu=2, sigma=1)
    
    k1, k2, n1, n2, ymin1, ymin2, ymax1, ymax2 = hill_params[gate]
        
    y_hat = pm.ode.DifferentialEquation(
        func=AlternativeModel.gate_model, times=fluos.index, n_states=2, n_theta=13
    )(y0=[y0, 0], theta=[r, synGFP, degGFP, cuma, ara, k1, k2, n1, n2, ymin1, ymin2, ymax1, ymax2])
    
    fluo_est = pm.Normal('fluo', mu=y_hat.T[1], sd=0.5, observed=fluo)
    
    step = pm.NUTS()
    trace = pm.sample(1000, tune=1000, cores=-1, chains=2, step=step)

    data = az.from_pymc3(trace=trace)
    data.to_netcdf('Metropolis-Alternative-{}-{}{}.nc'.format(gate, a, b))
    
ending = datetime.now()
print('Finished at:', ending)
print('Execution time:', ending-beginning)