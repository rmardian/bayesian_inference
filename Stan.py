import numpy as np
import pandas as pd
import pystan

fluos = pd.read_csv('marionette_fluo.csv', index_col='time')
ods = pd.read_csv('marionette_od.csv', index_col='time')
all_gates = list(set([i[:-3] for i in fluos.columns.tolist()]))
gates = [g for g in all_gates if g not in ['blank', 'positive_control', 'negative_control']]

sel = 0
gate = gates[sel]
print(sel, gate)
fluo_sel = fluos.loc[:, fluos.columns.str.startswith(gate)]
od_sel = ods.loc[:, fluos.columns.str.startswith(gate)]
fluo = fluo_sel.iloc[:,3]
od = od_sel.iloc[:,3]

model = """
    functions {
        real[] logisticgrowth(real t,
                      real[] y,
                      real[] theta,
                      real[] x_r,
                      int[] x_i
                      ) {
            real dydt[x_i[1]];
            for (i in 1:x_i[1]){
                dydt[i] = theta[1] * y[i] * (1-y[i]/theta[2]);
            }
            return dydt;
        }
    }
    data {
        int<lower=1> T;
        int<lower=1> n_wells;
        real y0[n_wells];
        real z[T, n_wells];
        real t0;
        real ts[T];
    }
    transformed data {
        real x_r[0];
        int x_i[1];
        x_i[1] = n_wells;
    }
    parameters {
        real<lower=0> theta[2];
        real<lower=0> sigma;
    }
    model {
        real y_hat[T, n_wells];
        theta ~ cauchy(0,2.5);
        sigma ~ normal(0,0.01);
        y_hat = integrate_ode_rk45(logisticgrowth, y0, t0, ts, theta, x_r, x_i);
        for (t in 1:T) {
            for (i in 1:n_wells) {
                z[t,i] ~ normal(y_hat[t,i], sigma);
            }
        }
    }
"""

data = {
    'T': len(od),
    'n_wells': 1,
    'y0': [0.02],
    'z': od.values.reshape(-1, 1),
    't0': 0,
    'ts': od.index
}

# Compile the model
sm = pystan.StanModel(model_code=model)

# Train the model and generate samples
fit = sm.sampling(data=data, iter=100, chains=2, n_jobs=1, verbose=True)
print(fit)

