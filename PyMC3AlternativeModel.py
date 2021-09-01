import numpy as np
import pandas as pd

from datetime import datetime

import pymc3 as pm
from pymc3.ode import DifferentialEquation
import arviz as az

hill_params = {
    'e11x32STPhoRadA': [7.645346632875684, 3.6278561616938205, 9.2171235388524, 1.1001985686622204, 14.8593116571963, 5.743762670085866, 458.5136836256326, 360.3202790463802],
    'e15x32NpuSspS2': [10.427864870911893, 9.059619137810259, 2.2173651514145054, 0.8959265079247906, 38.20872424607038, 26.812977822048122, 171.8535154764809, 149.69297721244843],
    'e16x33NrdA2': [9.991312723754868, 6.5897949388608605, 4.998239806913596, 1.2397846739565033, 7.416887642587476, 15.220098324908523, 196.10124234936688, 756.8276701181695],
    'e20x32gp411': [14.070212065863386, 0.7915636408282356, 2.0945754070850655, 1.0928383754262643, 10.826010849280056, 11.452710743807668, 511.61206006462345, 205.14895215235003],
    'e32x30SspGyrB': [10.653207289112212, 8.478594476251368, 3.11756946790781, 1.0683764045459927, 16.72019154901677, 9.778231993280107, 405.3316974141839, 289.8767336384693],
    'e34x30MjaKlbA': [1374.6349039619042, 12.144747077620854, 0.7451430715403362, 5.146751203285993, 11.715294696653377, 66.35185714414995, 1201.9421334190895, 805.9118807569571],
    'e38x32gp418': [7.395490239623599, 1.5509958154810697, 5.670357037879099, 1.3597918973710645, 19.06629063316735, 27.75559921258502, 163.83497173432002, 291.32238950469014],
    'e41x32NrdJ1': [9.79737497627455, 16.568623568277534, 3.407741906734099, 0.7844447687490096, 23.59776411698231, 13.14939309181156, 322.7098655707045, 264.1390343569163],
    'e42x32STIMPDH1': [10.336009368125339, 6.000475416768916, 4.477739982311845, 0.9757489564383387, 18.905781228897766, 21.953530170906987, 187.57210700503205, 278.66102832941397]
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
    degGFP = pm.Normal('n1', mu=2, sigma=1)
    
    k1, k2, n1, n2, ymin1, ymin2, ymax1, ymax2 = hill_params[gate]
        
    y_hat = pm.ode.DifferentialEquation(
        func=AlternativeModel.gate_model, times=fluos.index, n_states=2, n_theta=13
    )(y0=[y0, 0], theta=[r, synGFP, degGFP, cuma, ara, k1, k2, n1, n2, ymin1, ymin2, ymax1, ymax2])
    
    fluo_est = pm.Normal('fluo', mu=y_hat.T[1], sd=0.5, observed=fluo)
    
    step = pm.Metropolis()
    trace = pm.sample(1000, tune=1000, cores=-1, chains=2, step=step)

    data = az.from_pymc3(trace=trace)
    data.to_netcdf('Metropolis-Alternative-{}-{}{}.nc'.format(gate, a, b))
    
ending = datetime.now()
print('Finished at:', ending)
print('Execution time:', ending-beginning)