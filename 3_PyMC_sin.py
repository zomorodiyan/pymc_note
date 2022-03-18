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

def model(t,c,phi):
    x = np.sin(c*t+phi)
    return x

# generate data
n=20
sig = 0.2
t = np.linspace(0,1,n)

# True Values
np.random.seed(seed=1)
f_e = np.sin(2*t+np.pi/2) + np.random.normal(0,sig,t.shape)
#np.save('./dat/fn1',f_e)
plt.plot(t,f_e,'.')
plt.gca().update(dict(title=('f sigma:'+str(n)+"_"+str(sig)), xlabel='t', ylabel='f'))
plt.savefig('./Plots/f'+str(n)+"_"+str(sig)+'.png')
#plt.show()

# define model
basic_model = pm.Model()
with basic_model:
    # Priors for unknown model parameters
    c = pm.Uniform("c", lower=1, upper=10)
    phi = pm.Uniform("phi", lower=0, upper=2*np.pi)

    # Expected & Observed values
    f = np.sin(c*t+phi)
    observed = f_e

    # Likelihood (sampling distribution) of observations
    sigma = pm.HalfNormal("sigma", sd=0.1)
    func = pm.Normal("func", mu=f,sd=sigma,observed=observed)

# trace/sample model
with basic_model:
    trace = pm.sample(1000, chains=1)

# plot outputs
with basic_model:
    pm.plot_trace(trace) # pm.plot_trace(trace,['nu','sigma'])
    #plt.show()
    az.plot_posterior(trace,['c'],ref_val=2)
    #mngr = plt.get_current_fig_manager()
    #mngr.window.setGeometry(0,-500,600,400) #(left,top,width,height)
    plt.xticks(fontsize=10)
    plt.savefig('./Plots/c'+str(n)+"_"+str(sig)+'.png')
    #plt.show()
    az.plot_posterior(trace,['phi'],ref_val=np.pi/2)
    #mngr = plt.get_current_fig_manager()
    #mngr.window.setGeometry(0,-500,600,400) #(left,top,width,height)
    plt.xticks(fontsize=10)
    plt.savefig('./Plots/p'+str(n)+"_"+str(sig)+'.png')
    #plt.show()

with basic_model:
    display(az.summary(trace, round_to=2))
