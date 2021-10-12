import numpy as np
import pandas as pd

from datetime import datetime

import pymc3 as pm
import arviz as az

class HillFunction:
    
    @staticmethod
    def hill_activation(x, K, n, ymin, ymax):
    
        return ymin + (ymax - ymin) * (x**n / (K**n + x**n))

fluos = pd.read_csv('responses.csv')
inducers = pd.read_csv('inducers.csv')
columns = fluos.columns.tolist()

beginning = datetime.now()
print('Started at:', beginning)

for col in columns[1:]:

    print('***************************{}'.format(col))
    
    fluo = fluos[col].values
    inducer = inducers[col].values
    ymin0 = fluo.min()
    ymax0 = fluo.max()

    with pm.Model() as bayesian_model:
        
        sigma = pm.Normal('sigma', mu=0, sigma=1)
        k = pm.Normal('K', mu=1e1, sigma=5)
        n = pm.Normal('n', mu=5, sigma=1)
        ymin = pm.Normal('ymin', mu=ymin0, sigma=0.1*ymin0)
        ymax = pm.Normal('ymax', mu=ymax0, sigma=0.1*ymax0)
        
        y_hat = HillFunction.hill_activation(inducer, k, n, ymin, ymax)
        y_pred = pm.Normal('y_hat', mu=y_hat, sigma=sigma, observed=fluo)
        
        step = pm.Metropolis()
        trace = pm.sample(5000, tune=5000, cores=-1, chains=10, step=step)

        data = az.from_pymc3(trace=trace)
        data.to_netcdf('response-functions/resp-{}.nc'.format(col))
    
ending = datetime.now()
print('Finished at:', ending)
print('Execution time:', ending-beginning)