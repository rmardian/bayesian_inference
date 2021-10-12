import numpy as np
import pandas as pd
import stan
import arviz as az
from datetime import datetime

model = """
    functions {
        real hill_activation(real x, real K, real n, real ymin, real ymax) {
            real hill;
            hill = ymin + (ymax - ymin) * (pow(x, n) / (pow(K, n) + pow(x, n)));
            return hill;
        }
        real[] hill_activation_and(real[] x1, real[] x2, real K1, real K2, real n1, real n2, real ymin1, real ymin2, real ymax1, real ymax2, int T) {
            real hill[T];
            for (t in 1:T) {
                hill[t] = hill_activation(x1[t], K1, n1, ymin1, ymax1) * hill_activation(x2[t], K2, n2, ymin2, ymax2);
            }
            return hill;
        }
    }
    data {
        int<lower=1> T;
        real x1[T];
        real x2[T];
        real y[T];
    }
    transformed data {
        real ymin;
        real ymax;
        ymin = min(y);
        ymax = max(y);
    }
    parameters {
        real<lower=0> sigma;
        real<lower=0> K1;
        real<lower=0> K2;
        real<lower=0> n1;
        real<lower=0> n2;
        real<lower=0> ymin1;
        real<lower=0> ymin2;
        real<lower=0> ymax1;
        real<lower=0> ymax2;
    }
    model {
        real y_hat[T];
        sigma ~ normal(0, 1);
        K1 ~ normal(1e2, 5e1);
        K2 ~ normal(1e2, 5e1);
        n1 ~ normal(3, 1);
        n2 ~ normal(3, 1);
        ymin1 ~ normal(ymin, 0.5*ymin);
        ymin2 ~ normal(ymin, 0.5*ymin);
        ymax1 ~ normal(ymax, 0.5*ymax);
        ymax2 ~ normal(ymax, 0.5*ymax);
        y_hat = hill_activation_and(x1, x2, K1, K2, n1, n2, ymin1, ymin2, ymax1, ymax2, T);
        y ~ normal(y_hat, sigma);
    }
"""

fluos = pd.read_csv('induction_fluo.csv', index_col='time')
gates = ['e11x32STPhoRadA', 'e15x32NpuSspS2', 'e16x33NrdA2', 'e20x32gp411', 'e32x30SspGyrB',
         'e34x30MjaKlbA', 'e38x32gp418', 'e41x32NrdJ1', 'e42x32STIMPDH1']
cuma_list = [0, 6.25, 12.5, 25, 50, 100]
ara_list = [0, 0.8125, 3.25, 13, 52, 208]
minutes = [900]

t = len(cuma_list) * len(ara_list)
x1, x2 = np.meshgrid(cuma_list, ara_list)

beginning = datetime.now()

for gate in gates:

    fluo = fluos[filter(lambda x: x.startswith(gate), fluos.columns)]
    fluo_t = fluo.transpose().reset_index().rename(columns={'index': 'gate'})

    for at_m in minutes:

        print('***************************{}-{}'.format(gate, at_m))
        y = fluo_t[at_m].values
        y[y < 0] = 0.01
        data = {
            'T': t,
            'x1': x1.ravel(),
            'x2': x2.ravel(),
            'y': y,
        }
        # Compile the model
        posterior = stan.build(model, data=data)
        fit = posterior.sample(num_chains=10, num_warmup=5000, num_samples=5000)
        df = fit.to_frame()
        df.to_csv('hill-model-complete/Hill-{}-{}.csv'.format(gate, at_m))

        data = az.from_pystan(posterior=fit)
        data.to_netcdf('hill-model-complete/Hill-{}-{}.nc'.format(gate, at_m))

ending = datetime.now()
print('Started at:', beginning)
print('Finished at:', ending)
print('Execution time:', ending-beginning)
