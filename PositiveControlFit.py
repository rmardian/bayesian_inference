import numpy as np
import pandas as pd
import stan
import arviz as az
from datetime import datetime

od_params = {
    'positive_control_4AE_00': [0.02019822, 1.1707921 , 0.01132236],
    'positive_control_4AE_01': [0.01652411, 1.3044008 , 0.02164124],
    'positive_control_4AE_02': [0.01533817, 1.3701453 , 0.02625717],
    'positive_control_4AE_00.1': [0.01753164, 1.29914223, 0.01971833],
    'positive_control_4AE_01.1': [0.01596627, 1.36364788, 0.02891259],
    'positive_control_4AE_02.1': [0.01548979, 1.37373082, 0.03470967],
    'positive_control_4AE_00.2': [0.01738461, 1.28283639, 0.02055534],
    'positive_control_4AE_01.2': [0.01568295, 1.3532864 , 0.02918193],
    'positive_control_4AE_02.2': [0.0146497 , 1.39611788, 0.03517179],
    'positive_control_4AE_00.3': [0.01838002, 1.26159703, 0.01908134],
    'positive_control_4AE_01.3': [0.01628566, 1.3291291 , 0.02815582],
    'positive_control_4AE_02.3': [0.01544879, 1.38785372, 0.03312078],
    'positive_control_4AE_00.4': [0.01963368, 1.19889495, 0.0164443 ],
    'positive_control_4AE_01.4': [0.01748235, 1.25934   , 0.02392224],
    'positive_control_4AE_02.4': [0.01521494, 1.35037272, 0.03731348]
}

model = """
    functions {
        real growth_rate(real od, real r, real c) {
            real g;
            g = r * (1 - (od/c));
            return g;
        }
        real[] pos_ctrl(real t,
                    real[] y,
                    real[] theta,
                    real[] x_r,
                    int[] x_i
                    ) {
            real dydt[2];
            real gamma;
            gamma = growth_rate(y[2], x_r[1], x_r[1]);
            dydt[1] = theta[1] - (theta[2] + gamma) * y[1];
            dydt[2] = gamma * y[2];
            return dydt;
        }
    }
    data {
        int<lower=1> T;
        real y[T, 2];
        real t0;
        real ts[T];
        real params[3];
    }
    transformed data {
        real x_r[2];
        int x_i[0];
        x_r[1] = params[1];
        x_r[2] = params[2];
    }
    parameters {
        real<lower=0> sigma;
        real<lower=0> theta[2];
    }
    model {
        real y_hat[T, 2];
        real y0[2];
        theta[1] ~ normal(10, 2);
        theta[2] ~ normal(1, 0.5);
        sigma ~ normal(0, 1);
        y0[1] = 0;
        y0[2] = params[3];
        y_hat = integrate_ode_rk45(pos_ctrl, y0, t0, ts, theta, x_r, x_i);
        y[, 1] ~ normal(y_hat[, 1], sigma);
    }
"""

fluos = pd.read_csv('fluos.csv', index_col='time')
gfp_fluos = fluos[filter(lambda x: x.startswith('positive_control_4AE'), fluos.columns)]

beginning = datetime.now()

for control in gfp_fluos.columns:

    print('******************{}'.format(control))
    fluo = gfp_fluos[control]

    data = {
        'T': len(fluo),
        'y': fluo.values.reshape(-1, 1),
        't0': -20,
        'ts': fluo.index.values,
        'params': od_params[control]
    }

    # compile the model
    posterior = stan.build(model, data=data)
    fit = posterior.sample(num_chains=10, num_warmup=5000, num_samples=5000)
    df = fit.to_frame()
    df.to_csv('alternative-model-complete/PosCtrl-{}.csv'.format(control))
    data = az.from_pystan(posterior=fit)
    data.to_netcdf('alternative-model-complete/PosCtrl-{}.nc'.format(control))

ending = datetime.now()
print('Started at:', beginning)
print('Finished at:', ending)
print('Execution time:', ending-beginning)
