import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import arviz as az
az.style.use("arviz-darkgrid")
from IPython.display import display
import pymc3 as pm
import theano.tensor as tt
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# generate n=100 random observations from t = [0,1]
n=5000
#percent = 0.1
percent = 20
mult = 1 + np.random.uniform(-percent/100,percent/100, n)
t = np.zeros(n)
for i in range(n): t[i] = random.uniform(0,1)
# True Values
f_e = np.sin(2*t+np.pi/2)*mult
plt.plot(t,f_e,'.')
plt.gca().update(dict(title='f_noise_20', xlabel='t', ylabel='f'))
plt.savefig('./fig/f44.png')
plt.show()

# define model
basic_model = pm.Model()
with basic_model:
    # Priors for unknown model parameters
    c = pm.Uniform("c_noise_20", lower=1, upper=10)
    phi = pm.Uniform("phi_noise_20", lower=0, upper=2*np.pi)

    # Expected & Observed values
    f = np.sin(c*t+phi)
    observed = f_e

    # Likelihood (sampling distribution) of observations
    sigma = pm.HalfNormal("sigma1", sd=0.5)
    func = pm.Normal("func", mu=f,sd=sigma,observed=observed)

# trace/sample model
with basic_model:
    trace = pm.sample(1000, chains=1)

# plot outputs
with basic_model:
    pm.plot_trace(trace) # pm.plot_trace(trace,['nu','sigma'])
    plt.show()
    az.plot_posterior(trace,['c_noise_20'],ref_val=2)
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(0,-500,600,400) #(left,top,width,height)
    plt.xticks(fontsize=10)
    plt.savefig('./fig/c44.png')
    plt.show()
    az.plot_posterior(trace,['phi_noise_20'],ref_val=np.pi/2)
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(0,-500,600,400) #(left,top,width,height)
    plt.xticks(fontsize=10)
    plt.savefig('./fig/p44.png')
    plt.show()

with basic_model:
    display(az.summary(trace, round_to=2))
