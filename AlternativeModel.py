import numpy as np
import pandas as pd
import stan
import arviz as az
from datetime import datetime

hill_params = {
    'e11x32STPhoRadA': [8.19215086e+00, 2.77324618e+00, 6.00000000e+00, 1.24643956e+00, 5.95605924e+00, 3.46957143e-09, 2.99296531e+02, 5.36128135e+02],
    'e15x32NpuSspS2': [9.8458614, 8.7957319, 2.89598247, 1., 36.84876916, 30.33289918, 156.72063863, 156.06142356],
    'e16x33NrdA2': [10.13165313, 6.30849837, 5.02963422, 1.26435637, 15.99371476, 6.8959355, 380.68087314, 387.7069378],
    'e20x32gp411': [12.53828935, 1., 2.78451886, 1.49241488, 8.39069666, 27.77880598, 314.77726787, 311.65818168],
    'e32x30SspGyrB': [10.77387795, 8.56616674, 3.12574014, 1.07032582, 15.2982435, 11.91592347, 342.96437349, 343.5314864],
    'e34x30MjaKlbA': [8.42632247, 13.40974257, 3.28555513, 1.81757507, 3.4673668, 20.82148359, 307.13693296, 290.48137472],
    'e38x32gp418': [7.61231223, 1.51099399, 5.04169259, 1.4068252, 26.05989294, 20.59322098, 218.62810381, 218.64413488],
    'e41x32NrdJ1': [9.59574651, 12.05238497, 3.84271899, 1., 21.72980962, 19.19063999, 277.09322359, 275.39980816],
    'e42x32STIMPDH1': [10.41225458, 5.87647366, 4.30770405, 1.01184319, 22.82771137, 18.70845616, 228.18083668, 227.98611955]
}

model = """
    functions {
        real hill_activation(real x, real K, real n, real ymin, real ymax) {
            real hill;
            hill = ymin + (ymax - ymin) * (pow(x, n) / (pow(K, n) + pow(x, n)));
            return hill;
        }
        real hill_activation_and(real x1, real x2, real K1, real K2, real n1, real n2, real ymin1, real ymin2, real ymax1, real ymax2) {
            real hill;
            hill = hill_activation(x1, K1, n1, ymin1, ymax1) + hill_activation(x2, K2, n2, ymin2, ymax2);
            return hill;
        }
        real[] alternative_dynamics(real t,
                    real[] y,
                    real[] theta,
                    real[] x_r,
                    int[] x_i
                    ) {
            real dydt[2];
            real ymax;
            ymax = hill_activation_and(x_r[1], x_r[2], x_r[3], x_r[4], x_r[5], x_r[6], x_r[7], x_r[8], x_r[9], x_r[10]);
            dydt[1] = theta[1] * y[1] * (1-y[1]/ymax);
            dydt[2] = theta[2] * y[1] - x_r[11] * y[2];
            return dydt;
        }
    }
    data {
        int<lower=1> T;
        real x1;
        real x2;
        real y[T, 1];
        real t0;
        real ts[T];
        real params[8];
        real degGFP;
    }
    transformed data {
        real x_r[11];
        int x_i[0];
        x_r[1] = x1;
        x_r[2] = x2;
        x_r[3] = params[1];
        x_r[4] = params[2];
        x_r[5] = params[3];
        x_r[6] = params[4];
        x_r[7] = params[5];
        x_r[8] = params[6];
        x_r[9] = params[7];
        x_r[10] = params[8];
        x_r[11] = degGFP;
    }
    parameters {
        real<lower=0> sigma;
        real<lower=0> theta[2];
        real<lower=0.01> y0;
    }
    model {
        real y_hat[T, 1];
        real y0_[2];
        theta[1] ~ normal(1, 0.2);
        theta[2] ~ normal(10, 5);
        y0 ~ normal(1e3, 5e2);
        sigma ~ normal(0, 1);
        y0_[1] = y0;
        y0_[2] = 0;
        y_hat = integrate_ode_rk45(alternative_dynamics, y0_, t0, ts, theta, x_r, x_i);
        y[,1] ~ normal(y_hat[,2], sigma);
    }
"""

fluos = pd.read_csv('induction_fluo.csv', index_col='time')
gates = ['e11x32STPhoRadA', 'e15x32NpuSspS2', 'e16x33NrdA2', 'e20x32gp411', 'e32x30SspGyrB',
         'e34x30MjaKlbA', 'e38x32gp418', 'e41x32NrdJ1', 'e42x32STIMPDH1']
cuma_list = [0, 6.25, 12.5, 25, 50, 100]
ara_list = [0, 0.8125, 3.25, 13, 52, 208]

beginning = datetime.now()

for gate in gates:
    for a in range(5, 6):

        print('******************{}_{}{}'.format(gate, a, a))
        fluo = fluos['{}_{}{}'.format(gate, a, a)]
        cuma = cuma_list[a]
        ara = ara_list[a]

        data = {
            'T': len(fluo),
            'x1': cuma,
            'x2': ara,
            'y': fluo.values.reshape(-1, 1),
            't0': -20,
            'ts': fluo.index.values,
            'params': hill_params[gate],
            'degGFP': 0.01
        }

        # Compile the model
        posterior = stan.build(model, data=data)
        fit = posterior.sample(num_chains=10, num_warmup=5000, num_samples=10000)
        df = fit.to_frame()
        df.to_csv('alternative-model-complete/P3-Alternative-{}-{}{}.csv'.format(gate, a, a))
        data = az.from_pystan(posterior=fit)
        data.to_netcdf('alternative-model-complete/P3-Alternative-{}-{}{}.nc'.format(gate, a, a))

ending = datetime.now()
print('Started at:', beginning)
print('Finished at:', ending)
print('Execution time:', ending-beginning)
