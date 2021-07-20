import numpy as np
import pandas as pd
from datetime import datetime

import pymc3 as pm
from pymc3.ode import DifferentialEquation
import arviz as az

class ODModel:
    
    @staticmethod
    def growth_rate(y, r, c):
        return (r * (1 - (y/c)))
    
    @staticmethod
    def od_model(y, t, p):
        dOD = growth_rate(y[0], p[0], p[1]) * y[0]
        return [dOD]

ods = pd.read_csv('datasets/marionette_od.csv', index_col='time')
gates = list(set([i[:-3] for i in ods.columns.tolist()]))

for gate in gates:
    
    if gate=='blank':
        continue
    
    od = ods.loc[:, ods.columns.str.startswith(gate)].iloc[:,3] #take only 11 state

    beginning = datetime.now()
    print('Started at:', beginning)
    with pm.Model() as od_model:
    
        r = pm.Uniform('r', 0, 1)
        c = pm.Uniform('c', 0, 2)
        y0 = pm.Uniform('y0', 0, 0.2)

        y_hat = pm.ode.DifferentialEquation(
            func=ODModel.od_model, times=od.index, n_states=1, n_theta=2
        )(y0=[y0], theta=[r, c])

        od_est = pm.Normal('od', mu=y_hat.T[0], sd=0.3, observed=od)

        step = pm.Metropolis()
        trace = pm.sample(5000, tune=3000, cores=1, chains=3, step=step)
        
        data = az.from_pymc3(trace=trace)
        data.to_netcdf(gate + '-' + datetime.now().strftime('%Y%m%d') + '.nc')
    
ending = datetime.now()
print('Finished at:', ending)
print('Execution time:', ending-beginning)