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
import torch

# Try to import manifold analysis code locally 
import sys
import os
sys.path.append(os.getcwd())
from manana import manifold_analysis_AC as ma


#%% Load activations & Reshape Data
fpath = os.getcwd() + '/results/activations.pkl'
with open(fpath, 'rb') as f:
    data = pickle.load(f)

# Change correct action from tensor to list 
# Assuming the last element in the list is the correct location for that rank 
# Convert Tensor to List
data.corrected_action = data.corrected_action.apply(lambda x : x.tolist())
# Extract Correct Location for each rank
locs = [1,2, 3, 4]
ranks = [1,2]
timeSteps = [1,2,3,4]
numRanks = len(ranks)
numLocs = len(locs)
numTimeSteps = len(timeSteps)

for rank in range(numRanks):
    data[f'rank{rank+1}Cor'] = data.corrected_action.apply(lambda x : x[rank][-1])

# Nested Dict, Time step : Rank, Location : Activations
manifoldDict = {timeStep : {(rank, loc) : [] for rank in ranks for loc in locs} for timeStep in timeSteps}
for rank in ranks:
    for loc in locs:
        # Extract activations for each rank and location
        activations = data.loc[data[f'rank{rank}Cor'] == loc].activation
        # Concatenate activations into 3D array
        activations = np.dstack(activations)
        for timeStep in timeSteps:
            # Extract activations for each time step
            activationsTime = activations[timeStep-1, :, :]
            # Append activations to dictionary 
            manifoldDict[timeStep][(rank, loc)] = activationsTime

# Parameters for manifold analysis
kappa = 0 
n_t = 100
# Turn dict into list, order of dict w.r.t keys should be time, rank, location
# Not sure if it makes sense to run analysis on all manifolds at once
# or to do it for each time step individually 
# Num of manifolds = num of time steps * num of ranks * num of locations
manifolds = [manifoldDict[time][(rank,loc)] for time in manifoldDict.keys() for (rank, loc) in manifoldDict[time].keys()]

            
# Parameters for bootstrap 
nReps = 100
nSamples = 50
rng = np.random.default_rng()
# Initialize dictionaries for storing manifold metrics
capacities = {timeStep : {(rank, loc) : [] for rank in ranks for loc in locs} for timeStep in timeSteps}
radii = {timeStep : {(rank, loc) : [] for rank in ranks for loc in locs} for timeStep in timeSteps}
dims = {timeStep : {(rank, loc) : [] for rank in ranks for loc in locs} for timeStep in timeSteps}

for i in range(nReps):
    #Subsample trials for each manifold 
    subManifolds = []
    for manifold in manifolds:
        #Randomly sample data points from each manifold of size nSamples, with replacement
        subSample = rng.choice(manifold, axis = 1, size = nSamples, replace = True)
        # Append each sub-sampled manifold to list
        subManifolds.append(subSample)
        
    αSub, rSub, DSub = ma.manifold_analysis(subManifolds, kappa, n_t, t_vecs=None, n_reps=10)
    manifoldNum = 0 
    for timeStep in manifoldDict.keys():
        for (rank, loc) in manifoldDict[timeStep].keys():
            capacities[timeStep][(rank, loc)].append(αSub[manifoldNum])
            radii[timeStep][(rank, loc)].append(rSub[manifoldNum])
            dims[timeStep][(rank, loc)].append(DSub[manifoldNum])
            
            # increment counter for manifold number in list
            manifoldNum += 1

# Take average across bootstrap samples
αM = {timeStep : {(rank, loc) : np.array(capacities[timeStep][(rank, loc)]).mean() for rank in ranks for loc in locs} for timeStep in timeSteps}
rM = {timeStep : {(rank, loc) : np.array(radii[timeStep][(rank, loc)]).mean() for rank in ranks for loc in locs} for timeStep in timeSteps}
DM = {timeStep : {(rank, loc) : np.array(dims[timeStep][(rank, loc)]).mean() for rank in ranks for loc in locs} for timeStep in timeSteps}

# Save manifold metrics
manifoldMetrics = [αM, rM, DM]
network = '4tRNN'
with open(f'manifoldMetrics_{network}.pkl', 'wb') as file:
    pickle.dump(manifoldMetrics, file)
    
#%% Plot first ten activations 

g,ax = plt.subplots()

plt.plot(activations[0][:,:10])
plt.xticks(np.arange(0, 4, 1))
ax.set_xticklabels(['1', '2', '3', '4'])
plt.legend()


#%% Plot Metrics 

fig, ax = plt.subplots(1,3)


#%%
