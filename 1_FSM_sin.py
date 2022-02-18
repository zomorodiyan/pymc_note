# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 2022

Forward Sensitivity Method for the parameter estimation in sin(ct+phi)

For coding questions and/or suggestions, please contact Shady Ahmed at shady.ahmed@okstate.edu
"""


#%% Import libraries
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

np.random.seed(seed=0)
import os, sys


#%% Define Functions

def model(t,c,phi):
    x = np.sin(c*t+phi)
    return x
    
# Jacobian of the  model
def Jac(t,c,phi):
    dM = np.array([[0,t*np.cos(c*t+phi),np.cos(c*t+phi)]])
    dMx = dM[:,:1]
    dMa = dM[:,1:]
    return dMx, dMa
    
# Observational map
def h(x):
    z = x
    return z

# Jacobian of observational map
def Dh(x):
    D = np.eye(1)
    return D
    

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

par0 = np.array([1,1]) #some initial guess for model's parameters

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
sig_values = [1e-10,0.05,0.10,0.20]
for sig in sig_values:
    R = sig**2 * np.eye(m)
    Ri = np.linalg.inv(R)

    # Generate measurements [twin experiment]
    np.random.seed(seed=1)
    z = xtrue[tind1] + np.random.normal(0,sig,xtrue[tind1].shape)
    #sys.exit()
    
    # FSM Parameter Estimation
    par = par0
    
    max_iter= 100
    for jj in range(max_iter):
        U = np.eye(1,1)
        V = np.zeros((1,2))
    
        H = np.zeros((1,2))
        e = np.zeros((1,1))
        W = np.zeros((1,1)) #weighting matrix
        k = 0
        c1 = par[0]
        phi1 = par[1]
        for i in range(1,tind1[-1]+1):
            x1 = model(t[i],c1,phi1)
            dMx , dMa = Jac(t[i],c1,phi1)
            V = dMx @ V + dMa
            
            if i == tind1[k]:
                Hk = Dh(x1) @ V
                H = np.vstack((H,Hk))
                ek = (h(x1) - z[k]).reshape(-1,1)
                e = np.vstack((e,ek))
                W = block_diag(W,Ri)
                k = k+1
                
        H = np.delete(H, (0), axis=0)
        e = np.delete(e, (0), axis=0)
        W = np.delete(W, (0), axis=0)
        W = np.delete(W, (0), axis=1)
        
        # solve weighted least-squares
        W1 = np.sqrt(W) 
        dpar = np.linalg.lstsq(W1@H, -W1@e, rcond=None)[0]
        par = par + dpar.ravel()#/np.linalg.norm(dc)
        if np.linalg.norm(dpar) <= 1e-6:
            print(sig)
            print(jj)
            print(par)
            c1 = par[0]
            phi1 = par[1]
            break
    zobs1.append(z)
    param1.append(par)
    xpred1.append(model(t,c1,phi1))
   
    
#%% Fixed level of noise -- Different number of measurements

sig = 0.20
R = sig**2 * np.eye(m)
Ri = np.linalg.inv(R)
zobs2 = []
param2 = []  
xpred2 = []
tind2 = []
nobs_values = [100,40,20,10]
for nobs in nobs_values: #number of observations
         
    # measurement indices
    freq = int(nt/nobs)
    tind = np.arange(freq,nt+1,freq)   

    # Generate measurements [twin experiment]
    np.random.seed(seed=1)
    z = xtrue[tind] + np.random.normal(0,sig,xtrue[tind].shape)
    #sys.exit()
    
    # FSM Parameter Estimation
    par = par0
    
    max_iter= 100
    for jj in range(max_iter):
        U = np.eye(1,1)
        V = np.zeros((1,2))
    
        H = np.zeros((1,2))
        e = np.zeros((1,1))
        W = np.zeros((1,1)) #weighting matrix
        k = 0
        c1 = par[0]
        phi1 = par[1]
        for i in range(1,tind[-1]+1):
            x1 = model(t[i],c1,phi1)
            dMx , dMa = Jac(t[i],c1,phi1)
            V = dMx @ V + dMa
            
            if i == tind[k]:
                Hk = Dh(x1) @ V
                H = np.vstack((H,Hk))
                ek = (h(x1) - z[k]).reshape(-1,1)
                e = np.vstack((e,ek))
                W = block_diag(W,Ri)
                k = k+1
                
        H = np.delete(H, (0), axis=0)
        e = np.delete(e, (0), axis=0)
        W = np.delete(W, (0), axis=0)
        W = np.delete(W, (0), axis=1)
        
        # solve weighted least-squares
        W1 = np.sqrt(W) 
        dpar = np.linalg.lstsq(W1@H, -W1@e, rcond=None)[0]
        par = par + dpar.ravel()#/np.linalg.norm(dpar)
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
plt.savefig('./Plots/FSM_varying_noise.png', dpi = 300, bbox_inches = 'tight')
plt.savefig('./Plots/FSM_varying_noise.pdf', dpi = 300, bbox_inches = 'tight')


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
plt.savefig('./Plots/FSM_varying_nobs.png', dpi = 300, bbox_inches = 'tight')
plt.savefig('./Plots/FSM_varying_nobs.pdf', dpi = 300, bbox_inches = 'tight')

#%%
# ax[0].plot(t,X1[0,:], '--', color='C1')  
# ax[0].plot(t[0],X[0,0],'o', color='k', markersize=8, markeredgewidth=3, fillstyle='none', label=r'\textbf{Estimated}')
# ax[0].plot(t[tind],Z[0,:],'*', color='C0', markersize=8, markeredgewidth=2, label=r'\textbf{True}')
# ax[0].set_xlabel(r'$t$', fontsize=22)
# ax[0].set_ylabel(r'$v(t)$', fontsize=18, labelpad=0)

# ax[1].plot(t,X[1,:], color='k')
# ax[1].plot(t,X1[1,:], '--', color='C1')
# ax[1].plot(t[0],X[1,0], 'o', color='k', markersize=8, markeredgewidth=3, fillstyle='none')   
# ax[1].plot(t[tind],Z[1,:],'*', color='C0', markersize=8, markeredgewidth=2, label=r'\textbf{True}')
# ax[1].set_xlabel(r'$t$', fontsize=22)
# ax[1].set_ylabel(r'$w(t)$', fontsize=18, labelpad=0)

# # Nullclines
# # V-curve [dV/dt=0]
# V = np.linspace(-3,3,100)
# W1 = V - (1/3)*V**3+I

# # W-curve [dV/dt=0]
# W2 = (V+a)/b
# ax[2].plot(V,W1,'-.', color='gray', label=r'\textbf{Nullclines}')
# ax[2].plot(V,W2,'-.', color='gray')#, label=r'$W-$\textbf{Nullcline}')

# ax[2].plot(X[0,:],X[1,:], color='k', label=r'\textbf{True}')  
# ax[2].plot(X1[0,:],X1[1,:], '--', color='C1', label=r'\textbf{Predicted}')  

# ax[2].plot(X[0,0],X[1,0], 'o', color='k', markersize=8, markeredgewidth=3, fillstyle='none')  
# ax[2].plot(Z[0,:],Z[1,:],'*', color='C0', markersize=8, markeredgewidth=2, label=r'\textbf{Measurements}')

# ax[2].set_xlabel(r'$v(t)$', fontsize=18)
# ax[2].set_ylabel(r'$w(t)$', fontsize=18, labelpad=0)
# #ax[2].set_xlim([-2.5,2.5])
# #ax[2].set_ylim([-2,2])
# #ax[2].set_ylim([np.min(W1), np.max(W1)])

# fig.subplots_adjust(wspace=0.45)
# ax[2].legend(loc="center", bbox_to_anchor=(-0.95,-0.4), ncol=5, fontsize=16)

# #plt.savefig('./Plots/I5_5s.png', dpi = 300, bbox_inches = 'tight')

# plt.show()
# #%%%
    
# V = np.zeros([2,2,nt+1])
# G = np.zeros([2,2,nt+1])

# for k in range(nt):
#     dMx , dMa = JRK4(X[:,k],a,b,tau,I,dt)
#     #U = DM_a(u,mu) @ U
#     V[:,:,k+1] = dMx @ V[:,:,k] + dMa
#     G[:,:,k+1] = (V[:,:,k+1]).T @ V[:,:,k+1]
# #%%    
# fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(9,9))
# ax = ax.flat

# ax[0].plot(t,V[0,0,:]**1)    
# ax[0].set_ylabel(r'$V_{11}$', fontsize=14)

# ax[1].plot(t,V[0,1,:]**1)  
# ax[1].set_ylabel(r'$V_{12}$', fontsize=14)
  
# ax[2].plot(t,V[1,0,:]**1)  
# ax[2].set_ylabel(r'$V_{21}$', fontsize=14)
  
# ax[3].plot(t,V[1,1,:]**1)    
# ax[3].set_ylabel(r'$V_{22}$', fontsize=14)

# for i in range(4):
#     ax[i].set_xlabel(r'$t$', fontsize=14)

# fig.subplots_adjust(hspace=0.35, wspace=0.35)


# plt.show()

# #%%    
# mpl.rc('text', usetex=True)

# mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
# mpl.rcParams['text.latex.preamble'] = [r'\boldmath']

# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 24}

# mpl.rc('font', **font)



# fig, ax = plt.subplots(nrows=1,ncols=4, figsize=(16,3))
# ax = ax.flat

# ax[0].plot(t,V[0,0,:]**2,linewidth=3)    
# ax[0].plot(t[tind],V[0,0,tind]**2,'o', markersize=8, markeredgewidth=3, fillstyle='none')    
# ax[0].set_ylabel(r'$V_{11}^2$', fontsize=20)

# ax[1].plot(t,V[0,1,:]**2,linewidth=3)  
# ax[1].plot(t[tind],V[0,1,tind]**2,'o', markersize=8, markeredgewidth=3, fillstyle='none')    
# ax[1].set_ylabel(r'$V_{12}^2$', fontsize=20)


# ax[2].plot(t,V[1,0,:]**2,linewidth=3)  
# ax[2].plot(t[tind],V[1,0,tind]**2,'o', markersize=8, markeredgewidth=3, fillstyle='none')    
# ax[2].set_ylabel(r'$V_{21}^2$', fontsize=20)


# ax[3].plot(t,V[1,1,:]**2,linewidth=3)    
# ax[3].plot(t[tind],V[1,1,tind]**2,'o', markersize=8, markeredgewidth=3, fillstyle='none')    
# ax[3].set_ylabel(r'$V_{22}^2$', fontsize=20)

# for i in range(4):
#     ax[i].set_xlabel(r'$t$', fontsize=26)

# fig.subplots_adjust(wspace=0.4)

# #ax[2].legend(loc="center", bbox_to_anchor=(-0.95,-0.4), ncol=5, fontsize=16)

# #plt.savefig('./Plots/I5sens.png', dpi = 300, bbox_inches = 'tight')

# plt.show()

# #%%    


# fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(8,3))
# ax = ax.flat


# ax[0].plot(t,V[0,0,:]**2+V[1,0,:]**2,linewidth=2)    
# ax[0].plot(t[tind],V[0,0,tind]**2+V[1,0,tind]**2,'o', markersize=8, markeredgewidth=3, fillstyle='none')    
# ax[0].set_ylabel(r'$V_{11}^2+V_{21}^2$', fontsize=16)


# ax[1].plot(t,V[0,1,:]**2+V[1,1,:]**2,linewidth=2)  
# ax[1].plot(t[tind],V[0,1,tind]**2+V[1,1,tind]**2,'o', markersize=8, markeredgewidth=3, fillstyle='none')    
# ax[1].set_ylabel(r'$V_{12}^2+V_{22}^2$', fontsize=16)


# for i in range(2):
#     ax[i].set_xlabel(r'$t$', fontsize=18)

# fig.subplots_adjust(hspace=0.35, wspace=0.35)

# #plt.tight_layout()


# plt.show()

# #%%    
# fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(9,4))
# ax.plot(t,V[0,0,:]*V[0,1,:]+V[1,0,:]*V[1,1,:])    
# ax.plot(t[tind],V[0,0,tind]*V[0,1,tind]+V[1,0,tind]*V[1,1,tind],'o')    
# ax.set_ylabel(r'$V_{11}V_{12}+V_{21}V_{22}$', fontsize=14)




# ax.set_xlabel(r'$t$', fontsize=14)

# fig.subplots_adjust(hspace=0.35, wspace=0.35)


# plt.show()

