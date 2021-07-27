import numpy as np
import pandas as pd
import pystan
import pickle
fluos = pd.read_csv('marionette_fluo.csv', index_col='time')
ods = pd.read_csv('marionette_od.csv', index_col='time')
all_gates = list(set([i[:-3] for i in fluos.columns.tolist()]))
gates = [g for g in all_gates if g not in ['blank', 'positive_control', 'negative_control']]
#sel = 0
#gate = gates[sel]
gate = "e42x32STIMPDH1"
print(gate)
fluo_sel = fluos.loc[:, fluos.columns.str.startswith(gate)]
od_sel = ods.loc[:, fluos.columns.str.startswith(gate)]
fluo = fluo_sel.iloc[:,3]
od = od_sel.iloc[:,3]
model = """
    functions {
        real[] growth(real t,
                      real[] y,
                      real[] theta,
                      real[] x_r,
                      int[] x_i
                      ) {
            real dydt[1];
            dydt[1] = theta[1] * y[1] * (1-y[1]/theta[2]);
            return dydt;
        }
    }
    data {
        int<lower=1> T;
        //real y0[1];
        real y[T, 1];
        real t0;
        real ts[T];
    }
    transformed data {
        real x_r[0];
        int x_i[0];
    }
    parameters {
        real<lower=0> theta[2];
        real<lower=0> sigma;
        real<lower=0> y0[1];
    }
    model {
        real y_hat[T, 1];
        theta[1] ~ uniform(0, 1);
        theta[2] ~ uniform(0, 2);
        y0[1] ~ uniform(0, 1);
        sigma ~ normal(0, 0.1);
        y_hat = integrate_ode_rk45(growth, y0, t0, ts, theta, x_r, x_i);
        y[,1] ~ normal(y_hat[,1], sigma);
    }
"""

data = {
    'T': len(od),
    #'n_wells': 1,
    #'y0': [0.02],
    'y': od.values.reshape(-1, 1),
    't0': -20,
    'ts': od.index
}
# Compile the model
sm = pystan.StanModel(model_code=model)
# Train the model and generate samples
fit = sm.sampling(data=data, iter=1000, chains=2, n_jobs=-1, verbose=True)
print(fit)
with open('Stan-' + gate + '.pkl', 'wb') as f:
    pickle.dump({'model': sm, 'fit': fit}, f)



