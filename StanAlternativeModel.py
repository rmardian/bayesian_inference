import numpy as np
import pandas as pd
import pystan
import arviz as az
from datetime import datetime

hill_params = {
    'e11x32STPhoRadA': [8.19215086e+00, 2.77324618e+00, 6.00000000e+00, 1.24643956e+00, 3.56423426e+00, 1.50252702e-10, 3.55907011e+02, 4.95959256e+02],
    'e15x32NpuSspS2': [9.8458614, 8.7957319, 2.89598247, 1.0 , 33.3864939, 26.68735596, 157.80098788, 157.51127152],
    'e16x33NrdA2': [10.13165313, 6.30849837, 5.02963422, 1.26435637, 12.31034403, 5.56520849, 389.61315225, 399.52903402],
    'e20x32gp411': [12.53828935, 1.0, 2.78451886, 1.49241488, 8.42038183, 19.770931, 316.74400613, 314.41729987],
    'e32x30SspGyrB': [10.77387795, 8.56616674, 3.12574014, 1.07032582, 12.42641636, 10.01199595, 353.07383233, 353.36938984],
    'e34x30MjaKlbA': [8.42632247, 13.40974257, 3.28555513, 1.81757507, 8.91852601, 20.20877259, 289.53290172, 287.92030858],
    'e38x32gp418': [7.61231223, 1.51099399, 5.04169259, 1.4068252, 22.75331651, 18.846011, 222.80791219, 223.15092773],
    'e41x32NrdJ1': [9.59574651, 12.05238497, 3.84271899, 1.0, 20.50936546, 14.68953094, 279.98024852, 280.45758993],
    'e42x32STIMPDH1': [10.41225458, 5.87647366, 4.30770405, 1.01184319, 19.08872036, 15.87715881, 232.88219568, 232.90886374]
}
gate_params = {
    "e11x32STPhoRadA": [1.46415109e-02, 6.63450682e+02, 1.58327452e-02, 0.02],
    "e15x32NpuSspS2": [0.03165863, 4.7853412,  0.02219695, 0.02],
    "e16x33NrdA2": [2.21403462e-02, 1.44320142e+02, 2.20095167e-02, 0.02],
    "e20x32gp411": [2.33100887e-02, 9.93028856e+01, 2.26403674e-02, 0.02],
    "e32x30SspGyrB": [ 0.03359541, 10.92812258,  0.02336038, 0.02],
    "e34x30MjaKlbA": [2.05466494e-02, 1.61293627e+02, 2.15819156e-02, 0.02],
    "e38x32gp418": [2.34074609e-02, 9.02122673e+01, 1.89467403e-02, 0.02],
    "e41x32NrdJ1": [2.31185209e-02, 1.45064667e+02, 2.17997477e-02, 0.02],
    "e42x32STIMPDH1": [0.03508648, 0.2788824,  0.02389252, 0.02]
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
            ymax = hill_activation_and(x_r[1], x_r[2], x_r[3], x_r[4], x_r[5], x_r[6], x_r[7], x_r[8], x_r[9], x_r[10], x_r[11]);
            dydt[1] = theta[1] * y[1] * (1-y[1]/ymax);
            dydt[2] = theta[2] * y[1] - x_r[3] * y[2];
            return dydt;
        }
    }
    data {
        int<lower=1> T;
        int<lower=1> num_params;
        int<lower=1> num_states;
        real x1;
        real x2;
        real y[T, 1];
        real t0;
        real ts[T];
        real params[num_params];
        real degGFP;
    }
    transformed data {
        real x_r[11];
        int x_i[0];
        x_r[1] = x1;
        x_r[2] = x2;
        x_r[3] = degGFP;
        x_r[4] = params[1];
        x_r[5] = params[2];
        x_r[6] = params[3];
        x_r[7] = params[4];
        x_r[8] = params[5];
        x_r[9] = params[6];
        x_r[10] = params[7];
        x_r[11] = params[8];
    }
    parameters {
        real<lower=0> sigma;
        real<lower=0> theta[3];
        real<lower=0.01> y0;
    }
    model {
        real y_hat[T, 1];
        real y0_[num_states];
        theta[1] ~ uniform(0, 1);
        theta[2] ~ uniform(0, 2);
        sigma ~ normal(0, 0.1);
        y0 ~ uniform(0, 1);
        y0_[1] = y0;
        y0_[2] = 0;
        y_hat = integrate_ode_rk45(alternative_dynamics, y0_, t0, ts, theta, x_r, x_i);
        y[,1] ~ normal(y_hat[,1], sigma);
    }
"""

degGFP = 0.02 #fixed deg param

fluos = pd.read_csv('induction_fluo.csv', index_col='time')
gates = ['e11x32STPhoRadA', 'e15x32NpuSspS2', 'e16x33NrdA2', 'e20x32gp411', 'e32x30SspGyrB',
         'e34x30MjaKlbA', 'e38x32gp418', 'e41x32NrdJ1', 'e42x32STIMPDH1']

a = 5
b = 5
gate = 'e42x32STIMPDH1'
fluo = fluos['{}_{}{}'.format(gate, a, b)]

cuma_list = [0, 6.25, 12.5, 25, 50, 100]
ara_list = [0, 0.8125, 3.25, 13, 52, 208]
cuma = cuma_list[a]
ara = ara_list[b]
t = len(cuma_list) * len(ara_list)
x1, x2 = np.meshgrid(cuma_list, ara_list)

data = {
    'T': t,
    'num_params': 3,
    'num_states': 2,
    'x1': cuma,
    'x2': ara,
    'y': fluo.values.reshape(-1, 1),
    't0': -20,
    'ts': fluo.index,
    'params': hill_params[gate],
    'degGFP': 0.02
}

beginning = datetime.now()
print('Started at:', beginning)
# Compile the model
sm = pystan.StanModel(model_code=model)
# Train the model and generate samples
fit = sm.sampling(data=data, iter=5000, warmup=2500, thin=2, chains=2, n_jobs=-1, control=dict(adapt_delta=0.9), verbose=True)
print(fit)

summary_dict = fit.summary()
df = pd.DataFrame(summary_dict['summary'], 
                columns=summary_dict['summary_colnames'], 
                index=summary_dict['summary_rownames'])
#df.to_csv('Hill-' + gate +  '.csv')

#data = az.from_pystan(posterior=fit)
#data.to_netcdf('Hill-' + gate + '.nc')
ending = datetime.now()
print('Finished at:', ending)
print('Execution time:', ending-beginning)