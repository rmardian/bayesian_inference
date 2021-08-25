import numpy as np
import pandas as pd

from datetime import datetime

import pymc3 as pm
from pymc3.ode import DifferentialEquation
import arviz as az

class HillFunction:
    
    @staticmethod
    def hill_activation(x, K, n, ymin, ymax):
    
        return ymin + (ymax - ymin) * (x**n / (K**n + x**n))
    
    @staticmethod
    def hill_activation_and(x1, x2, K1, K2, n1, n2, ymin1, ymin2, ymax1, ymax2):

        #x1, x2 = x
        return HillFunction.hill_activation(x1, K1, n1, ymin1, ymax1) * HillFunction.hill_activation(x2, K2, n2, ymin2, ymax2)
    
fluos = pd.read_csv('induction_fluo.csv', index_col='time')
gates = ['e11x32STPhoRadA', 'e15x32NpuSspS2', 'e16x33NrdA2', 'e20x32gp411', 'e32x30SspGyrB',
         'e34x30MjaKlbA', 'e38x32gp418', 'e41x32NrdJ1', 'e42x32STIMPDH1']
cumas = [0, 6.25, 12.5, 25, 50, 100]
aras = [0, 0.8125, 3.25, 13, 52, 208]
gate = 'e11x32STPhoRadA'
t = 1440
x1, x2 = np.meshgrid(cumas, aras)
#x = np.vstack((x1.ravel(), x2.ravel()))

fluo = fluos[filter(lambda x: x.startswith(gate), fluos.columns)]
fluo_t = fluo.transpose().reset_index().rename(columns={'index': 'gate'})
y = fluo_t[t]
ymin = y.min()
ymax = y.min()

beginning = datetime.now()
print('Started at:', beginning)
with pm.Model() as bayesian_model:
    
    #sigma = pm.Normal('sigma', mu=y.mean(), sigma=0.5*ymin)
    k1 = pm.Normal('K1', mu=1e2, sigma=5e1)
    k2 = pm.Normal('K2', mu=1e2, sigma=5e1)
    n1 = pm.Normal('n1', mu=2, sigma=1)
    n2 = pm.Normal('n2', mu=2, sigma=1)
    ymin1 = pm.Normal('ymin1', mu=100, sigma=10)
    ymin2 = pm.Normal('ymin2', mu=100, sigma=10)
    ymax1 = pm.Normal('ymax1', mu=198393, sigma=10000)
    ymax2 = pm.Normal('ymax2', mu=198393, sigma=10000)
    
    y_hat = HillFunction.hill_activation_and(x1.ravel(), x2.ravel(), k1, k2, n1, n2, ymin1, ymin2, ymax1, ymax2)
    y_pred = pm.Normal('y_hat', mu=y_hat, sigma=1000, observed=y.ravel())
    
    step = pm.NUTS()
    trace = pm.sample(1000, tune=1000, cores=-1, chains=2, step=step)

    data = az.from_pymc3(trace=trace)
    data.to_netcdf('NUTS' + datetime.now().strftime('%Y%m%d') + '.nc')
    
ending = datetime.now()
print('Finished at:', ending)
print('Execution time:', ending-beginning)