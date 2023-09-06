#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   analysis.py
@Time    :   2023/09/05 15:14:11
@Author  :   Brandon Chen 
@Version :   1.0
@Contact :   bc1693@nyu.edu
@License :   None
@Desc    :   None
'''
#%% Import Packages
import autograd.numpy as np
from scipy.linalg import qr
from functools import partial

from cvxopt import solvers, matrix
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.optimizers import ConjugateGradient
import pickle5 as pickle
import pandas as pd
import matplotlib.pyplot as plt

# Try to import manifold analysis code locally 
import sys
import os
sys.path.append(os.getcwd())
from manana import manifold_analysis_AC as ma


#%% Load activations 
fpath = os.getcwd() + '/results/activations.pkl'
with open(fpath, 'rb') as f:
    data = pickle.load(f)
    activations = data['activation']
#%%

# Reshape activations into list of manifolds 
locs = [1,2, 3, 4]
ranks = [1,2]

actExpand = np.dstack(activations)

# Parameters for manifold analysis
kappa = 0 
n_t = 100
manifolds =  [actExpand[i,:,:] for i in range(0,4)]


# Initialize dictionaries for storing manifold metrics
capacities = {(rank, loc) : [] for rank in ranks for loc in locs}
radii = {(rank, loc) : [] for rank in ranks for loc in locs}
dims = {(rank, loc) : [] for rank in ranks for loc in locs}

# Parameters for bootstrap 
nReps = 100
nSamples = 150
rng = np.random.default_rng()

αM = []
rM = []
DM = [] 

for i in range(nReps):
    #Subsample trials for each manifold 
    subManifolds = []
    for manifold in manifolds:
        #Randomly sample data points from each manifold of size nSamples, with replacement
        subSample = rng.choice(manifold, axis = 1, size = nSamples, replace = True)
        # Append each sub-sampled manifold to list
        subManifolds.append(subSample)
        
    αSub, rSub, DSub = ma.manifold_analysis(subManifolds, kappa, n_t, t_vecs=None, n_reps=10)
    
    αM.append(αSub)
    rM.append(rSub)
    DM.append(DSub)

#%% Plot first ten activations 

g,ax = plt.subplots()

plt.plot(activations[0][:,:10])
plt.xticks(np.arange(0, 4, 1))
ax.set_xticklabels(['1', '2', '3', '4'])
plt.legend()


#%% Plot Metrics 

fig, ax = plt.subplots(1,3)


#%%
