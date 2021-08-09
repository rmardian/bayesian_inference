import numpy as np
import pandas as pd
import pystan
import arviz as az
from datetime import datetime
fluos = pd.read_csv('marionette_fluo.csv', index_col='time')
all_gates = list(set([i[:-3] for i in fluos.columns.tolist()]))
gates = [g for g in all_gates if g not in ['blank', 'positive_control', 'negative_control']]
gate = "e42x32STIMPDH1"
print(gate)
fluo = fluos.loc[:, fluos.columns.str.startswith(gate)].iloc[:,3]
#from PyMC3
#od_params = {
#    'e38x32gp418': [0.014832545686692651, 1.206999008636522, 0.05586756964448041],
#    'e20x32gp411': [0.014610032725458771, 1.2772684500977902, 0.04458182544478886],
#    'e11x32STPhoRadA': [0.007844653484011118, 1.1134153588014837, 0.10602396117757999],
#    'e34x30MjaKlbA': [0.013408372081617236, 1.14628949032757, 0.052259068950788926],
#    'e32x30SspGyrB': [0.015133693377991284, 1.151731463297364, 0.04901732064094526],
#    'e15x32NpuSspS2': [0.015778141196678447, 1.192983296185405, 0.049950021825183724],
#    'e16x33NrdA2': [0.014463894907212425, 1.2421974135452174, 0.055150036439220534],
#    'e41x32NrdJ1': [0.013145465488589058, 1.2385872370025641, 0.05941283831303974],
#    'e42x32STIMPDH1': [0.012125122999895207, 1.2815026045141547, 0.0524312129470823],
#}
#from PyStan
od_params = {
    'e16x33NrdA2': [0.0148194554039306, 1.2408917618889663, 0.0251889022569271],
    'e42x32STIMPDH1': [0.013165608798406, 1.278935730092542, 0.0216016431163906],
    'e32x30SspGyrB': [0.0159760283755244, 1.150249628464946, 0.0196547733645749],
    'e11x32STPhoRadA': [0.0070462352099844, 1.1229375761211622, 0.0973103761235306],
    'e34x30MjaKlbA': [0.0139910399333654, 1.147632550566836, 0.0235340568336425],
    'e41x32NrdJ1': [0.0136188162357974, 1.2375665371324622, 0.0300644005159601],
    'e20x32gp411': [0.0155212086766198, 1.2780552798586051, 0.0193773015388865],
    'e38x32gp418': [0.0155869734557929, 1.2060333912302554, 0.0247131234412902],
    'e15x32NpuSspS2': [0.0158706098949703, 1.1961969417584624, 0.0235672266874314]
}

model = """
    functions {
        real hill_activation(real x, real K, real n) {
            real hill;
            hill = pow(x, n) / (pow(K, n) + pow(x, n));
            return hill;
        }
        real growth_rate(real y, real a, real b) {
            real growth;
            growth = (a * (1 - (y/b)));
            return growth;
        }
        real[] gate_model(real t, real[] y, real[] theta, real[] x_r, int[] x_i) {
            real OD = y[1];
            real ECFn = y[2];
            real ECFc = y[3];
            real ECF = y[4];
            real GFP = y[5];
            real syn_ECF = theta[1];
            real syn_GFP = theta[2];
            real alpha = x_r[1];
            real beta = x_r[2];
            real bn = x_r[3];
            real bc = x_r[4];
            real bg = x_r[5];
            real syn_ECFn = x_r[6];
            real syn_ECFc = x_r[7];
            real deg = x_r[8];
            real deg_GFP = x_r[9];
            real K = x_r[10];
            real n = x_r[11];
            int ind1 = x_i[1];
            int ind2 = x_i[2];
            real gamma = growth_rate(OD, alpha, beta);
            real dOD = gamma * OD;
            real dECFn = bn + syn_ECFn * ind1 - (deg + gamma) * ECFn;
            real dECFc = bc + syn_ECFc * ind2 - (deg + gamma) * ECFc;
            real dECF = syn_ECF * ECFn * ECFc - (deg + gamma) * ECF;
            real dGFP = bg + syn_GFP * hill_activation(ECF, K, n) - (deg_GFP + gamma) * GFP;
            return { dOD, dECFn, dECFc, dECF, dGFP };
        }
    }
    data {
        int<lower=1> T;
        int<lower=1> num_states;
        real y0[num_states];
        real y[T, 1];
        real t0;
        real ts[T];
        real od_params[2];
        real fixed_params[9];
        int inducers[2];
    }

    transformed data {
        real x_r[11];
        int x_i[2];
        x_r[1] = od_params[1];
        x_r[2] = od_params[2];
        x_r[3] = fixed_params[1];
        x_r[4] = fixed_params[2];
        x_r[5] = fixed_params[3];
        x_r[6] = fixed_params[4];
        x_r[7] = fixed_params[5];
        x_r[8] = fixed_params[6];
        x_r[9] = fixed_params[7];
        x_r[10] = fixed_params[8];
        x_r[11] = fixed_params[9];
        x_i[1] = inducers[1];
        x_i[2] = inducers[2];
    }
    parameters {
        real<lower=0> theta[2];
        real<lower=0> sigma;}
    model {
        real y_hat[T, num_states];
        theta[1] ~ uniform(0, 1e1);
        theta[2] ~ uniform(0, 1e5);
        sigma ~ normal(0, 0.1);
        y_hat = integrate_ode_rk45(gate_model, y0, t0, ts, theta, x_r, x_i);
        y[,1] ~ normal(y_hat[,5], sigma);
    }
"""

data = {
    'T': len(fluo),
    'num_states': 5,
    'y0': [od_params[gate][2], 0, 0, 0, 0],
    'y': fluo.values.reshape(-1, 1),
    't0': -20,
    'ts': fluo.index,
    'od_params': [od_params[gate][0], od_params[gate][1]],
    'fixed_params': [2, 2, 2, 50, 50, 0.05, 0.05, 2, 4],
    'inducers': [1, 1]
}
beginning = datetime.now()
print('Started at:', beginning)
# Compile the model
sm = pystan.StanModel(model_code=model)
# Train the model and generate samples
fit = sm.sampling(data=data, iter=5000, warmup=2500, thin=2, chains=2, n_jobs=-1, control=dict(adapt_delta=0.9), v$
print(fit)
#with open('Stan-Fluo-' + gate + '.pkl', 'wb') as f:
#    pickle.dump({'model': sm, 'fit': fit}, f)
summary_dict = fit.summary()
df = pd.DataFrame(summary_dict['summary'], 
                columns=summary_dict['summary_colnames'], 
                index=summary_dict['summary_rownames'])
                df.to_csv('Fluo-' + gate +  '.csv')

data = az.from_pystan(posterior=fit)
data.to_netcdf('Fluo-' + gate + '.nc')
ending = datetime.now()
print('Finished at:', ending)
print('Execution time:', ending-beginning)
