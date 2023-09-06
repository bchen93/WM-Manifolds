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
#%% 
from numba.types import Tuple, int64, float64  
from numba import types
from numba.typed import Dict  
import numba
from numba.experimental import jitclass
key_ty = Tuple((int64, int64)) 

inner_dict_type = types.DictType(key_ty, types.float64[:,:])
@jitclass([('dt', types.int32),
            ('last_index_tick', types.DictType(types.int32, types.DictType(key_ty,types.float64[:,:])))])
class TickFeature:
    def __init__(self, dt: int):
        self.dt = dt
        #self.last_index_tick = defaultdict(dict)
        #tmp = numba.typed.Dict.empty(key_type=types.unicode_type, value_type=types.int64)
        self.last_index_tick = numba.typed.Dict.empty(
            key_type= types.int32,
            value_type=inner_dict_type
        )
        # self.init_index_tick()


test = TickFeature(4)
#%% Run analysis 
# Load activations & Reshape Data
fpath = os.getcwd() + '/results/seq3_activations.pkl'
with open(fpath, 'rb') as f:
    data = pickle.load(f)
    
locs = [1,2,3,4]
ranks = [1,2,3]
timeSteps = [1,2,3,4]
nReps = 100
nSamples = 100
network = '4tRNNSeq3'
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
    manifoldMetrics = pickle.load(file)

# Convert dicts to dataframe
αM = pd.DataFrame.from_records(manifoldMetrics[0], orient = 'index')
rM = pd.DataFrame.from_dict(manifoldMetrics[1], orient = 'index')
DM = pd.DataFrame.from_dict(manifoldMetrics[2], orient = 'index')

#%% Plot first ten activations 

g,ax = plt.subplots()

plt.plot(activations[0][:,:10])
plt.xticks(np.arange(0, 4, 1))
ax.set_xticklabels(['1', '2', '3', '4'])
plt.legend()


#%% Plot Metrics 

fig, ax = plt.subplots(1,3)

sns.lineplot(αM, ax = ax[0])
sns.lineplot(rM, ax = ax[1])
sns.lineplot(DM, ax = ax[2])

plt.xticks(np.arange(0, 4, 1))
ax.set_xticklabels(['1', '2', '3', '4'])

#%%
