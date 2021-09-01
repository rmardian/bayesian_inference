import numpy as np
import pandas as pd
import stan
import arviz as az
from datetime import datetime

hill_params = {
    'e11x32STPhoRadA': [7.645346632875684, 3.6278561616938205, 9.2171235388524, 1.1001985686622204, 14.8593116571963, 5.743762670085866, 458.5136836256326, 360.3202790463802],
    'e15x32NpuSspS2': [10.427864870911893, 9.059619137810259, 2.2173651514145054, 0.8959265079247906, 38.20872424607038, 26.812977822048122, 171.8535154764809, 149.69297721244843],
    'e16x33NrdA2': [9.991312723754868, 6.5897949388608605, 4.998239806913596, 1.2397846739565033, 7.416887642587476, 15.220098324908523, 196.10124234936688, 756.8276701181695],
    'e20x32gp411': [14.070212065863386, 0.7915636408282356, 2.0945754070850655, 1.0928383754262643, 10.826010849280056, 11.452710743807668, 511.61206006462345, 205.14895215235003],
    'e32x30SspGyrB': [10.653207289112212, 8.478594476251368, 3.11756946790781, 1.0683764045459927, 16.72019154901677, 9.778231993280107, 405.3316974141839, 289.8767336384693],
    'e34x30MjaKlbA': [1374.6349039619042, 12.144747077620854, 0.7451430715403362, 5.146751203285993, 11.715294696653377, 66.35185714414995, 1201.9421334190895, 805.9118807569571],
    'e38x32gp418': [7.395490239623599, 1.5509958154810697, 5.670357037879099, 1.3597918973710645, 19.06629063316735, 27.75559921258502, 163.83497173432002, 291.32238950469014],
    'e41x32NrdJ1': [9.79737497627455, 16.568623568277534, 3.407741906734099, 0.7844447687490096, 23.59776411698231, 13.14939309181156, 322.7098655707045, 264.1390343569163],
    'e42x32STIMPDH1': [10.336009368125339, 6.000475416768916, 4.477739982311845, 0.9757489564383387, 18.905781228897766, 21.953530170906987, 187.57210700503205, 278.66102832941397]
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
            hill = hill_activation(x1, K1, n1, ymin1, ymax1) * hill_activation(x2, K2, n2, ymin2, ymax2);
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
        real y_hat[T, 2];
        real y0_[2];
        theta[1] ~ normal(1, 0.2);
        theta[2] ~ normal(10, 5);
        y0 ~ normal(1e3, 5e2);
        sigma ~ normal(0, 1);
        y0_[1] = y0;
        y0_[2] = 0;
        y_hat = integrate_ode_rk45(alternative_dynamics, y0_, t0, ts, theta, x_r, x_i);
        y[, 1] ~ normal(y_hat[, 2], sigma);
    }
"""

fluos = pd.read_csv('induction_fluo.csv', index_col='time')
gates = ['e11x32STPhoRadA', 'e15x32NpuSspS2', 'e16x33NrdA2', 'e20x32gp411', 'e32x30SspGyrB',
         'e34x30MjaKlbA', 'e38x32gp418', 'e41x32NrdJ1', 'e42x32STIMPDH1']
cuma_list = [0, 6.25, 12.5, 25, 50, 100]
ara_list = [0, 0.8125, 3.25, 13, 52, 208]

beginning = datetime.now()

for gate in gates:
    for a in range(6):

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
        fit = posterior.sample(num_chains=10, num_warmup=5000, num_samples=5000)
        df = fit.to_frame()
        df.to_csv('alternative-model-complete/P3-Alternative-{}-{}{}.csv'.format(gate, a, a))
        data = az.from_pystan(posterior=fit)
        data.to_netcdf('alternative-model-complete/P3-Alternative-{}-{}{}.nc'.format(gate, a, a))

ending = datetime.now()
print('Started at:', beginning)
print('Finished at:', ending)
print('Execution time:', ending-beginning)
