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
        vector hill_activation_and(vector x1, vector x2, vector theta, int T) {
            real K1 = theta[1];
            real K2 = theta[2];
            real n1 = theta[3];
            real n2 = theta[4];
            real ymin1 = theta[5];
            real ymin2 = theta[6];
            real ymax1 = theta[7];
            real ymax2 = theta[8];
            vector[T] hill;
            for (t in 1:T) {
                hill[t] = hill_activation(x1[t], K1, n1, ymin1, ymax1) + hill_activation(x2[t], K2, n2, ymin2, ymax2);
            }
            return hill;
        }
    }
    data {
        int<lower=1> T;
        vector[T] x1;
        vector[T] x2;
        vector[T] y;
        //real ymin;
        //real ymax;
    }
    transformed data {
        vector[T] x1_std;
        vector[T] x2_std;
        vector[T] y_std;
        real ymin_std;
        real ymax_std;
        x1_std = (x1 - mean(x1)) / sd(x1);
        x2_std = (x2 - mean(x2)) / sd(x2);
        y_std = (y - mean(y)) / sd(y);
        ymin_std = min(y_std);
        ymax_std = max(y_std);
    }
    parameters {
        real<lower=0> sigma;
        vector<lower=0>[8] theta;
    }
    model {
        vector[T] y_hat;
        sigma ~ normal(0, 1);
        theta[1] ~ normal(1e2, 5e1);
        theta[2] ~ normal(1e2, 5e1);
        theta[3] ~ normal(2, 1);
        theta[4] ~ normal(2, 1);
        theta[5] ~ normal(ymin_std, 0.1*ymin_std);
        theta[6] ~ normal(ymin_std, 0.1*ymin_std);
        theta[7] ~ normal(ymax_std, 0.1*ymax_std);
        theta[8] ~ normal(ymax_std, 0.1*ymax_std);
        y_hat = hill_activation_and(x1, x2, theta, T);
        y ~ normal(y_hat, sigma);
    }
"""

fluos = pd.read_csv('induction_fluo.csv', index_col='time')
gates = ['e11x32STPhoRadA', 'e15x32NpuSspS2', 'e16x33NrdA2', 'e20x32gp411', 'e32x30SspGyrB',
         'e34x30MjaKlbA', 'e38x32gp418', 'e41x32NrdJ1', 'e42x32STIMPDH1']
cuma_list = [0, 6.25, 12.5, 25, 50, 100]
ara_list = [0, 0.8125, 3.25, 13, 52, 208]
minutes = [960, 1440]

t = len(cuma_list) * len(ara_list)
x1, x2 = np.meshgrid(cuma_list, ara_list)

beginning = datetime.now()

for gate in gates:

    fluo = fluos[filter(lambda x: x.startswith(gate), fluos.columns)]
    fluo_t = fluo.transpose().reset_index().rename(columns={'index': 'gate'})

    for at_m in minutes:
        
        print('***************************{}-{}'.format(gate, at_m))
        y = fluo_t[at_m].values
        data = {
            'T': t,
            'x1': x1.ravel(),
            'x2': x2.ravel(),
            'y': y
        }
        # Compile the model
        posterior = stan.build(model, data=data)
        fit = posterior.sample(num_chains=10, num_warmup=5000, num_samples=10000)
        df = fit.to_frame()
        df.to_csv('hill-model-complete/P3-Hill-{}-{}.csv'.format(gate, at_m))

        data = az.from_pystan(posterior=fit)
        data.to_netcdf('hill-model-complete/P3-Hill-{}-{}.nc'.format(gate, at_m))

ending = datetime.now()
print('Started at:', beginning)
print('Finished at:', ending)
print('Execution time:', ending-beginning)
