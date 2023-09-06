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
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import pickle5 as pickle
# Try to import manifold analysis code locally 
import sys
import os
sys.path.append(os.getcwd())
from manana import manifold_analysis_AC as ma

from numba import jit, typeof, typed, types, cuda

#%% Define Functions    
def reshapeData(data, locs, ranks, timeSteps):
    # Change correct action from tensor to list 
    # Assuming the last element in the list is the correct location for that rank 
    # Convert Tensor to List
    data.corrected_action = data.corrected_action.apply(lambda x : x.tolist())
    # Extract Correct Location for each rank
    locs = locs
    ranks = ranks
    timeSteps = timeSteps
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
    return manifoldDict

# @jit(target_backend='cuda')
def bootstrapManifoldMetrics(data, locs, ranks, timeSteps, network, nReps, nSamples):
    locs = locs
    ranks = ranks
    timeSteps = timeSteps
    numRanks = len(ranks)
    numLocs = len(locs)
    numTimeSteps = len(timeSteps)
    
    # Parameters for manifold analysis
    kappa = 0 
    n_t = 100
    # Turn dict into list, order of dict w.r.t keys should be time, rank, location
    # Not sure if it makes sense to run analysis on all manifolds at once
    # or to do it for each time step individually 
    # Num of manifolds = num of time steps * num of ranks * num of locations
    manifolds = [manifoldDict[time][(rank,loc)] for time in manifoldDict.keys() for (rank, loc) in manifoldDict[time].keys()]

    # Parameters for bootstrap 
    nReps = nReps
    nSamples = nSamples
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
        # Run manifold analysis on subsampled manifolds
        αSub, rSub, DSub = ma.manifold_analysis(subManifolds, kappa, n_t, t_vecs=None, n_reps=10)
        # Initialize counter for manifold number in list
        manifoldNum = 0 
        for timeStep in manifoldDict.keys():
            for (rank, loc) in manifoldDict[timeStep].keys():
                capacities[timeStep][(rank, loc)].append(αSub[manifoldNum])
                radii[timeStep][(rank, loc)].append(rSub[manifoldNum])
                dims[timeStep][(rank, loc)].append(DSub[manifoldNum])
                
                # increment counter for manifold number in list
                manifoldNum += 1

    # Take average across bootstrap samples
    αM = {timeStep : {(rank, loc) : np.array(capacities[timeStep][(rank, loc)]) for rank in ranks for loc in locs} for timeStep in timeSteps}
    rM = {timeStep : {(rank, loc) : np.array(radii[timeStep][(rank, loc)]) for rank in ranks for loc in locs} for timeStep in timeSteps}
    DM = {timeStep : {(rank, loc) : np.array(dims[timeStep][(rank, loc)]) for rank in ranks for loc in locs} for timeStep in timeSteps}

    return αM, rM, DM
#%% Run analysis 
# Load activations & Reshape Data
fpath = os.getcwd() + '/results/seq2_frame_rep2_activations.pkl'
with open(fpath, 'rb') as f:
    data = pickle.load(f)
    
locs = [1,2,3,4]
ranks = [1,2]
timeSteps = [1,2,3,4,5,6,7,8]
nReps = 20
nSamples = 100
network = '8tRNNSeq2'
manifoldDict = reshapeData(data, locs, ranks, timeSteps)
αM, rM, DM = bootstrapManifoldMetrics(manifoldDict, locs, ranks, timeSteps,network, nReps, nSamples)


# Save manifold metrics
manifoldMetrics = [αM, rM, DM]
network = network
with open(f'manifoldMetrics_{network}.pkl', 'wb') as file:
    pickle.dump(manifoldMetrics, file)
#%%
# Load manifold metrics
network = '4tRNNSeq2'
with open(f'manifoldMetrics_{network}.pkl', 'rb') as file:
    αM, rM, DM = pickle.load(file)
    
# Convert dictionaries to DF
def convertDictToDF(metricDict):
    
    df = pd.DataFrame.from_records(
        [
            (level1, level2, level3, leaf)
            for level1, level2_dict in metricDict.items()
            for (level2, level3), leaf in level2_dict.items()
        ],
        columns=['TimeStep', 'Rank', 'Loc', 'value']
    )
    df = df.explode('value').reset_index(drop=True)
    return df

αM = convertDictToDF(αM)
rM = convertDictToDF(rM)
DM = convertDictToDF(DM)

metrics = pd.merge(αM, rM, on = ['TimeStep', 'Rank', 'Loc'], suffixes = ('α', 'r'))
metrics = pd.merge(metrics, DM, on = ['TimeStep', 'Rank', 'Loc'])
#%% Plot Metrics 

fig, ax = plt.subplots(1,3, figsize = (15,5))

sns.lineplot(x = 'TimeStep', y = 'value', hue = 'Loc', style = 'Rank', label = 'αM', data = αM, ax = ax[0])
sns.lineplot(x = 'TimeStep', y = 'value', hue = 'Loc', style = 'Rank', label = 'RM', data = rM, ax = ax[1])
sns.lineplot(x = 'TimeStep', y = 'value', hue = 'Loc', style = 'Rank', label = 'DM', data = DM, ax = ax[2])


# for ax in ax:
    # plt.xticks(np.arange(0, 4, 1))
    # ax.set_xticklabels(['1', '2', '3', '4'])
       
#%%

#%% Garbage Plots 
# Plot first ten activations 

g,ax = plt.subplots()

plt.plot(activations[0][:,:10])
plt.xticks(np.arange(0, 4, 1))
ax.set_xticklabels(['1', '2', '3', '4'])
plt.legend()
