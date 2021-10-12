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
    }
    data {
        int<lower=1> T;
        real x[T];
        real y[T];
    }
    transformed data {
        real ymin0;
        real ymax0;
        ymin0 = min(y);
        ymax0 = max(y);
    }
    parameters {
        real<lower=0> sigma;
        real<lower=0> K;
        real<lower=0> n;
        real<lower=0> ymin;
        real<lower=0> ymax;
    }
    model {
        real y_hat[T];
        sigma ~ normal(0, 1);
        K ~ normal(1e2, 5e1);
        n ~ normal(3, 1);
        ymin ~ normal(ymin0, 0.5*ymin0);
        ymax ~ normal(ymax0, 0.5*ymax0);
        y_hat = hill_activation(x, K, n, ymin, ymax, T);
        y ~ normal(y_hat, sigma);
    }
"""

fluos = pd.read_csv('responses.csv')
inducers = pd.read_csv('inducers.csv')
columns = fluos.columns.tolist()

beginning = datetime.now()

for col in columns:

    print('***************************{}'.format(col))
    
    fluo = fluos[col]
    inducer = inducers[col]
    
    data = {
        'T': len(fluo),
        'x': inducer.values,
        'y': fluo.values,
    }
    # Compile the model
    posterior = stan.build(model, data=data)
    fit = posterior.sample(num_chains=10, num_warmup=5000, num_samples=5000)
    df = fit.to_frame()
    df.to_csv('response-functions/resp-{}.csv'.format(col))

    data = az.from_pystan(posterior=fit)
    data.to_netcdf('response-functions/resp-{}.nc'.format(col))

ending = datetime.now()
print('Started at:', beginning)
print('Finished at:', ending)
print('Execution time:', ending-beginning)
