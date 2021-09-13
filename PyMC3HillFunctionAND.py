import numpy as np
import pandas as pd

from datetime import datetime

import pymc3 as pm
from pymc3.ode import DifferentialEquation
import arviz as az

class HillFunction:
    
    @staticmethod
    def hill_activation(x, K, n, ymin):
    
        return ymin + (1 - ymin) * (x**n / (K**n + x**n))
    
    @staticmethod
    def hill_activation_and(x, K1, K2, n1, n2, ymin1, ymin2):
        
        x1, x2 = x
        return HillFunction.hill_activation(x1, K1, n1, ymin1) * HillFunction.hill_activation(x2, K2, n2, ymin2)

    @staticmethod
    def hill_activation_combined(x, K1, K2, n1, n2, ymin1, ymin2):
        
        result = []
        for i in range(0, 1460, 20):
            temp = HillFunction.hill_activation_and(x, K1, K2, n1, n2, ymin1, ymin2)
            result = np.append(result, temp)   
        return result

fluos = pd.read_csv('induction_fluo.csv', index_col='time')
gates = ['e11x32STPhoRadA', 'e15x32NpuSspS2', 'e16x33NrdA2', 'e20x32gp411', 'e32x30SspGyrB',
         'e34x30MjaKlbA', 'e38x32gp418', 'e41x32NrdJ1', 'e42x32STIMPDH1']
cumas = [0, 6.25, 12.5, 25, 50, 100]
aras = [0, 0.8125, 3.25, 13, 52, 208]
gate = 'e11x32STPhoRadA'
#t = 1440
x1, x2 = np.meshgrid(cumas, aras)
x = np.vstack((x1.ravel(), x2.ravel()))

fluo = fluos[filter(lambda x: x.startswith(gate), fluos.columns)]
fluo_t = fluo.transpose().reset_index().rename(columns={'index': 'gate'})

y = pd.Series()
start, end, gap = 0, 1460, 20
for t in range(start, end, gap):
    y_ = fluo_t[t]
    y_[y_ < 0] = 0.01 #replace negative values
    y_ = y_ / y_.max()
    y = y.append(y_)

beginning = datetime.now()
print('Started at:', beginning)
with pm.Model() as bayesian_model:
    
    sigma = pm.Normal('sigma', mu=0, sigma=1)
    k1 = pm.Normal('K1', mu=1e2, sigma=5e1)
    k2 = pm.Normal('K2', mu=1e2, sigma=5e1)
    n1 = pm.Normal('n1', mu=3, sigma=1)
    n2 = pm.Normal('n2', mu=3, sigma=1)
    ymin1 = pm.Normal('ymin1', mu=ymin, sigma=0.5*ymin)
    ymin2 = pm.Normal('ymin2', mu=ymin, sigma=0.5*ymin)
    
    y_hat = HillFunction.hill_activation_combined(x, k1, k2, n1, n2, ymin1, ymin2)
    y_pred = pm.Normal('y_hat', mu=y_hat, sigma=sigma, observed=y.ravel())
    
    step = pm.Metropolis()
    trace = pm.sample(5000, tune=5000, cores=-1, chains=10, step=step)

    data = az.from_pymc3(trace=trace)
    data.to_netcdf('hill-complete-model/Hill-{}-{}.nc'.format(gate))
    
ending = datetime.now()
print('Finished at:', ending)
print('Execution time:', ending-beginning)