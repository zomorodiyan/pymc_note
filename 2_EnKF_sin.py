# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 2022

EnKF for the parameter estimation in sin(ct+phi)

For coding questions and/or suggestions, please contact Shady Ahmed at shady.ahmed@okstate.edu
"""

#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import os, sys


#%% Define functions

#%% Define Functions

def model(t,c,phi):
    x = np.sin(c*t+phi)
    return x
    
# Jacobian of the  model
def Jac(t,c,phi):
    dM = np.array([[0,t*np.cos(c*t+phi),np.cos(c*t+phi)],
                   [0,1,0],
                   [0,0,1]])
    return dM
    
# Observational map
def h(x):
    z = x
    return z

# Jacobian of observational map
def Dh(x):
    D = np.array([ [1,0,0] ])
    return D
    

def EnKF(ubi,z,h,Dh,R,B):
    
    # The analysis step for the (stochastic) ensemble Kalman filter 
    # with virtual observations

    n,N = ubi.shape # n is the state dimension and N is the size of ensemble
    m = z.shape[0] # m is the size of measurement vector

    # compute the mean of forecast ensemble
    ub = np.mean(ubi,1)    
    # compute Jacobian of observation operator at ub
    H = Dh(ub)
    # compute Kalman gain
    D = H@B@H.T + R
    K = B @ H.T @ np.linalg.inv(D)
            
    zi = np.zeros([m,N])
    uai = np.zeros([n,N])
    for i in range(N):
        # create virtual observations
        zi[:,i] = z + np.random.multivariate_normal(np.zeros(m), R)
        # compute analysis ensemble
        uai[:,i] = ubi[:,i] + K @ (zi[:,i]-h(ubi[0,i]))
        
    # compute the mean of analysis ensemble
    ua = np.mean(uai,1)    
    # compute analysis error covariance matrix
    P = (1/(N-1)) * (uai - ua.reshape(-1,1)) @ (uai - ua.reshape(-1,1)).T
    return uai, P

#%% Main script

n = 1 #dimension of state u
p = 2 #dimension of parameters
m = 1 #dimension of measurement z

  
tm = 1
nt = 1000 #number of timesteps
dt = tm/nt
t = np.linspace(0,tm,nt+1)

c = 2
phi = np.pi/2

par0 = np.array([1.1,1]) #some initial guess for model's parameters

#%% Time integration -- Truth
xtrue = np.zeros([nt+1])
xtrue[:] =  model(t,c,phi)
        
    
#%% Different levels of noise -- Fixed number of measurements
    
# measurement indices
nobs = 100 #number of observations
freq = int(nt/nobs)
tind1 = np.arange(freq,nt+1,freq)   
zobs1 = [] 
param1 = []  
xpred1 = []
sig_values = [0.05,0.10,0.15,0.20]

#define augmented state vector
X1 = np.zeros(3)

#define background covariance matrix
P0 = 1e-4*np.eye(3) #uncertainty in initial condition
P0[1,1] = 1e0 #some value to reflect the uncerainty in the prior estimate of c
P0[2,2] = 1e0 #some value to reflect the uncerainty in the prior estimate of phi
    
for sig in sig_values:    

    
    R = sig**2 * np.eye(m)
    Ri = np.linalg.inv(R)

    # Generate measurements [twin experiment]
    np.random.seed(seed=1)
    z = xtrue[tind1] + np.random.normal(0,sig,xtrue[tind1].shape)
    
    #Parameter Estimation
    par = par0
    P = P0

    max_iter= 100
    for jj in range(max_iter):
            
        X1[1:] = np.copy(par)
        k = 0
        c1 = par[0]
        phi1 = par[1]


        ens_size = 10
        X1i = np.zeros([3,ens_size])
        for ii in range(ens_size):
            X1i[:,ii] = X1 + np.random.multivariate_normal(np.zeros(3), P)


        for i in range(1,tind1[-1]+1):
                
            for ii in range(ens_size):
                X1i[0,ii] = model(t[i],X1i[1,ii],X1i[2,ii])
                

            if i == tind1[k]:
                X1i, P = EnKF(X1i,(z[k]).reshape(-1,1),h,Dh,R,P)
                X1 = np.mean(X1i,axis=1)      
                k = k+1
           
        dpar = X1[1:] - par        
        par = par + dpar
        c1 = par[0]
        phi1 = par[1]

        if np.linalg.norm(dpar/par) <= 1e-6:
            print(jj)
            print(par)
            c1 = par[0]
            phi1 = par[1]
            break
    
    zobs1.append(z)
    param1.append(par)
    xpred1.append(model(t,c1,phi1))
   
    
#%% Fixed level of noise -- Different number of measurements

sig = 0.10
R = sig**2 * np.eye(m)
Ri = np.linalg.inv(R)
zobs2 = []
param2 = []  
xpred2 = []
tind2 = []
nobs_values = [100,40,20,10]

#define augmented state vector
X1 = np.zeros(3)

#define background covariance matrix
P0 = 1e-4*np.eye(3) #uncertainty in initial condition
P0[1,1] = 1e0 #some value to reflect the uncerainty in the prior estimate of c
P0[2,2] = 1e0 #some value to reflect the uncerainty in the prior estimate of phi
    
for nobs in nobs_values: #number of observations


    # measurement indices
    freq = int(nt/nobs)
    tind = np.arange(freq,nt+1,freq)   

    # Generate measurements [twin experiment]
    np.random.seed(seed=1)
    z = xtrue[tind] + np.random.normal(0,sig,xtrue[tind].shape)
    
    #Parameter Estimation
    par = par0
    P = P0

    max_iter= 100
    for jj in range(max_iter):
            
        X1[1:] = np.copy(par)
        k = 0
        c1 = par[0]
        phi1 = par[1]


        ens_size = 10
        X1i = np.zeros([3,ens_size])
        for ii in range(ens_size):
            X1i[:,ii] = X1 + np.random.multivariate_normal(np.zeros(3), P)


        for i in range(1,tind[-1]+1):
                
            for ii in range(ens_size):
                X1i[0,ii] = model(t[i],X1i[1,ii],X1i[2,ii])
                

            if i == tind[k]:
                X1i, P = EnKF(X1i,(z[k]).reshape(-1,1),h,Dh,R,P)
                X1 = np.mean(X1i,axis=1)      
                k = k+1
           
        dpar = X1[1:] - par        
        par = par + dpar
        c1 = par[0]
        phi1 = par[1]

        if np.linalg.norm(dpar/par) <= 1e-6:
            print(jj)
            print(par)
            c1 = par[0]
            phi1 = par[1]
            break
    
    tind2.append(tind)
    zobs2.append(z)
    param2.append(par)
    xpred2.append(model(t,c1,phi1))


#%% Compare
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 3


mpl.rc('text', usetex=True)

mpl.rcParams['text.latex.preamble']= r"\usepackage{amsmath}"
mpl.rcParams['text.latex.preamble'] = r"\boldmath"

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

mpl.rc('font', **font)
mpl.rcParams['mathtext.default'] = 'it'
#matplotlib.rcParams['text.usetex'] = False

# create plot folder
if os.path.isdir("./Plots"):
    print('Plots folder already exists')
else: 
    print('Creating Plots folder')
    os.makedirs("./Plots")
    
#%%
fig, ax = plt.subplots(nrows=3,ncols=2, figsize=(12,12))
ax = ax.flat

ax[0].plot(sig_values,np.array(param1)[:,0],'*', color='C0', markersize=8, markeredgewidth=2, fillstyle='none', label=r'\textbf{Estimated}')  
ax[0].plot(sig_values,np.ones(len(sig_values))*c,'--',color='k',linewidth=2, label=r'\textbf{True}')  
ax[0].set_xlabel(r'$\sigma$', fontsize=22)
ax[0].set_ylabel(r'$c$', fontsize=18)
ax[0].legend(loc="lower left", fontsize=14)

ax[1].plot(sig_values,np.array(param1)[:,1],'*', color='C0', markersize=8, markeredgewidth=2, fillstyle='none', label=r'\textbf{Estimated}')  
ax[1].plot(sig_values,np.ones(len(sig_values))*phi,'--',color='k',linewidth=2, label=r'\textbf{True}')  
ax[1].set_xlabel(r'$\sigma$', fontsize=22)
ax[1].set_ylabel(r'$\phi$', fontsize=18)
ax[1].legend(loc="upper left", fontsize=14)

for i in range(len(sig_values)):
    ax[i+2].plot(t,xtrue,'--',color='k',linewidth=2, label=r'\textbf{True}')  
    ax[i+2].plot(t,xpred1[i],'-', color='C0',linewidth=2, label=r'\textbf{Predicted}')
    ax[i+2].plot(t[tind1],zobs1[i],'o', color='C2', markersize=6, markeredgewidth=1, fillstyle='none', label=r'\textbf{Observations}')
    ax[i+2].set_xlabel(r'$t$', fontsize=20)
    ax[i+2].set_ylabel(r'$f(t;c,\phi)$', fontsize=20)
    ax[i+2].set_title(r'$\sigma='+str(np.round(sig_values[i],decimals=2))+'$', fontsize=20)
    ax[i+2].legend(loc="lower left", fontsize=14)


fig.subplots_adjust(hspace=0.6,wspace=0.35)
plt.savefig('./Plots/EnKF_varying_noise.png', dpi = 300, bbox_inches = 'tight')
plt.savefig('./Plots/EnKF_varying_noise.pdf', dpi = 300, bbox_inches = 'tight')


#%%
fig, ax = plt.subplots(nrows=3,ncols=2, figsize=(12,12))
ax = ax.flat

ax[0].plot(nobs_values,np.array(param2)[:,0],'*', color='C0', markersize=8, markeredgewidth=2, fillstyle='none', label=r'\textbf{Estimated}')  
ax[0].plot(nobs_values,np.ones(len(nobs_values))*c,'--',color='k',linewidth=2, label=r'\textbf{True}')  
ax[0].set_xlabel(r'$N_{Obs}$', fontsize=20)
ax[0].set_ylabel(r'$c$', fontsize=18)
ax[0].legend(loc="lower right", fontsize=14)

ax[1].plot(nobs_values,np.array(param1)[:,1],'*', color='C0', markersize=8, markeredgewidth=2, fillstyle='none', label=r'\textbf{Estimated}')  
ax[1].plot(nobs_values,np.ones(len(nobs_values))*phi,'--',color='k',linewidth=2, label=r'\textbf{True}')  
ax[1].set_xlabel(r'$N_{Obs}$', fontsize=20)
ax[1].set_ylabel(r'$\phi$', fontsize=18)
ax[1].legend(loc="upper right", fontsize=14)

for i in range(len(nobs_values)):
    ax[i+2].plot(t,xtrue, '--',color='k',linewidth=3, label=r'\textbf{True}')  
    ax[i+2].plot(t,xpred2[i],'-', color='C0',linewidth=3, label=r'\textbf{Predicted}')
    ax[i+2].plot(t[tind2[i]],zobs2[i],'o', color='C2', markersize=6, markeredgewidth=1, fillstyle='none', label=r'\textbf{Observations}')
    ax[i+2].set_xlabel(r'$t$', fontsize=20)
    ax[i+2].set_ylabel(r'$f(t;c,\phi)$', fontsize=20)
    ax[i+2].set_title(r'$N_{Obs}='+str(np.round(nobs_values[i],decimals=0))+'$', fontsize=20)
    ax[i+2].legend(loc="lower left", fontsize=14)


fig.subplots_adjust(hspace=0.6,wspace=0.35)
plt.savefig('./Plots/EnKF_varying_nobs.png', dpi = 300, bbox_inches = 'tight')
plt.savefig('./Plots/EnKF_varying_nobs.pdf', dpi = 300, bbox_inches = 'tight')
