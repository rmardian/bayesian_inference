import numpy as np
import pandas as pd

from datetime import datetime

import pymc3 as pm
import arviz as az

class HillFunction:
    
    @staticmethod
    def hill_activation(x, K, n, ymin, ymax):
    
        return ymin + (ymax - ymin) * (x**n / (K**n + x**n))
    
    @staticmethod
    def hill_activation_and(x, K1, K2, n1, n2, ymin1, ymin2, ymax1, ymax2):
        
        x1, x2 = x
        return HillFunction.hill_activation(x1, K1, n1, ymin1, ymax1) * HillFunction.hill_activation(x2, K2, n2, ymin2, ymax2)

fluos = pd.read_csv('fluos_rpu_600.csv')
gates = ['e11x32STPhoRadA', 'e15x32NpuSspS2', 'e16x33NrdA2', 'e20x32gp411', 'e32x30SspGyrB',
         'e34x30MjaKlbA', 'e38x32gp418', 'e41x32NrdJ1', 'e42x32STIMPDH1']
cumas_rpu = [0.0024930475754170005, 0.012183419457922659, 0.12362677605742128, 0.8546102179257526, 1.5521602964092271, 1.652939765595156],
aras_rpu = [0.024326008944639297, 2.2200495754596083, 8.398392174929793, 12.221625319031114, 12.8752073213337, 12.952376203722466]

beginning = datetime.now()
print('Started at:', beginning)

x1, x2 = np.meshgrid(cumas_rpu, aras_rpu)
x = np.vstack((x1.ravel(), x2.ravel()))

for gate in gates:

    print('******************{}'.format(gate))

    fluo = fluos[fluos['gate'].str.startswith(gate)]
    y = fluo['fluo'].values

    with pm.Model() as bayesian_model:
        
        #sigma = pm.Normal('sigma', mu=0, sigma=1)
        k1 = pm.Normal('K1', mu=1e2, sigma=5e1)
        k2 = pm.Normal('K2', mu=1e2, sigma=5e1)
        n1 = pm.Normal('n1', mu=3, sigma=1)
        n2 = pm.Normal('n2', mu=3, sigma=1)
        ymin1 = pm.Normal('ymin1', mu=y.min(), sigma=1)
        ymin2 = pm.Normal('ymin2', mu=y.min(), sigma=1)
        ymax1 = pm.Normal('ymax1', mu=y.max(), sigma=1)
        ymax2 = pm.Normal('ymax2', mu=y.max(), sigma=1)
        
        y_hat = HillFunction.hill_activation(x, k1, k2, n1, n2, ymin1, ymin2, ymax1, ymax2)
        y_pred = pm.Normal('y_hat', mu=y_hat, sigma=1, observed=y)
        
        step = pm.Metropolis()
        trace = pm.sample(5000, tune=5000, cores=-1, chains=10, step=step)

        data = az.from_pymc3(trace=trace)
        data.to_netcdf('response-functions/hill-{}-{}.nc'.format(gate))
    
ending = datetime.now()
print('Finished at:', ending)
print('Execution time:', ending-beginning)