import numpy as np
import pandas as pd
import pystan
import arviz as az
from datetime import datetime

fluos = pd.read_csv('datasets/induction_matrix/induction_fluo.csv', index_col='time')
gates = ['e11x32STPhoRadA', 'e15x32NpuSspS2', 'e16x33NrdA2', 'e20x32gp411', 'e32x30SspGyrB',
         'e34x30MjaKlbA', 'e38x32gp418', 'e41x32NrdJ1', 'e42x32STIMPDH1']
cuma_list = [0, 6.25, 12.5, 25, 50, 100]
ara_list = [0, 0.8125, 3.25, 13, 52, 208]

gate = 'e42x32STIMPDH1'
at_m = 1440

fluo = fluos[filter(lambda x: x.startswith(gate), fluos.columns)]
fluo_t = fluo.transpose().reset_index().rename(columns={'index': 'gate'})
fluo_t['cuma'] = fluo_t['gate'].str[-2]
fluo_t['ara'] = fluo_t['gate'].str[-1]
fluo_pvt = fluo_t.drop('gate', axis=1).pivot('cuma', 'ara', at_m)

model = """
    functions {
        real hill_activation(real x, real K, real n, real ymin, real ymax) {
            real hill;
            hill = ymin + (ymax - ymin) * (pow(x, n) / (pow(K, n) + pow(x, n)));
            return hill;
        }
        real hill_activation_and(real x, real theta[]) {
            real x1 = x[1];
            real x2 = x[2];
            real K1 = theta[1];
            real K2 = theta[2];
            real n1 = theta[3];
            real n2 = theta[4];
            real ymin1 = theta[5];
            real ymin2 = theta[6];
            real ymax1 = theta[7];
            real ymax2 = theta[8];
            real hill;
            hill = hill_activation(x1, K1, n1, ymin1, ymax1) * hill_activation(x2, K2, n2, ymin2, ymax2)
            return hill;
        }
    }
    data {
        int<lower=1> T;
        real x1[T];
        real x2[T];
        real y[T, T];
        real ymin;
        real ymax;
    }
    transformed data {
        real h = ymax - ymin;
    }
    parameters {
        real<lower=0> sigma;
        real<lower=0> theta[8];
    }
    model {
        real y;
        real x[2] = [x1, x2];
        sigma ~ normal(0, 0.1);
        theta[1] ~ uniform(1, 1e5);
        theta[2] ~ uniform(1, 4);
        theta[3] ~ uniform(0, ymin-0.5*h);
        theta[4] ~ uniform(0, ymax+0.5*h);
        theta[5] ~ uniform(1, 1e5);
        theta[6] ~ uniform(1, 4);
        theta[7] ~ uniform(0, ymin-0.5*h);
        theta[8] ~ uniform(0, ymax+0.5*h);
        y_hat = hill_activation_and(x, theta);
        y ~ normal(y_hat, sigma);
    }
"""

data = {
    'T': len(cuma_list),
    'x1': cuma_list,
    'x2': ara_list,
    'y': fluo_pvt.values.ravel(),
    'ymin': fluo_pvt.values.ravel().min(),
    'ymax': fluo_pvt.values.ravel().max()
}

beginning = datetime.now()
print('Started at:', beginning)
# Compile the model
sm = pystan.StanModel(model_code=model)
# Train the model and generate samples
fit = sm.sampling(data=data, iter=5000, warmup=2500, thin=2, chains=2, n_jobs=-1, control=dict(adapt_delta=0.9), verbose=True)
print(fit)
#with open('Stan-Fluo-' + gate + '.pkl', 'wb') as f:
#    pickle.dump({'model': sm, 'fit': fit}, f)
summary_dict = fit.summary()
df = pd.DataFrame(summary_dict['summary'], 
                columns=summary_dict['summary_colnames'], 
                index=summary_dict['summary_rownames'])
                df.to_csv('Hill-' + gate +  '.csv')

data = az.from_pystan(posterior=fit)
data.to_netcdf('Hill-' + gate + '.nc')
ending = datetime.now()
print('Finished at:', ending)
print('Execution time:', ending-beginning)
