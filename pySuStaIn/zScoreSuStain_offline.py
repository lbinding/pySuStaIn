#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 16:29:55 2024

@author: lawrencebinding
"""

#%% Import libraries 
import warnings
from tqdm.auto import tqdm
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import pathos
import pandas as pd
import os
import matplotlib.colors as mcolors
from pathlib import Path
import pickle
from functools import partial#, partialmethod


# =============================================================================
#  Below are packages SuStaIn typically requires... 
# =============================================================================
#   As We've taken it 'offline' in one script it doesn't need them...
#       maybe...

#from pySuStaIn.AbstractSustain import AbstractSustainData
#from pySuStaIn.AbstractSustain import AbstractSustain
#from multiprocessing import Value
#from abc import ABC, abstractmethod
#from tqdm.auto import tqdm
#import scipy.stats as stats
#import csv
#import os
#import multiprocessing
#import time

#%% Setup dict format to mirror what var.self does 

#*******************************************
#The data structure class for ZscoreSustain. It holds the z-scored data that gets passed around and re-indexed in places.
class ZScoreSustainData():
    def __init__(self, data, numStages):
        self.data        = data
        self.__numStages    = numStages

    def getNumSamples(self):
        return self.data.shape[0]

    def getNumBiomarkers(self):
        return self.data.shape[1]

    def getNumStages(self):
        return self.__numStages

    def reindex(self, index):
        return ZScoreSustainData(self.data[index,], self.__numStages)




#*******************************************
#An implementation of the AbstractSustain class with multiple events for each biomarker based on deviations from normality, measured in z-scores.
#There are a fixed number of thresholds for each biomarker, specified at initialization of the ZscoreSustain object.
class ZscoreSustain():
    def __init__(self,
                 data,
                 Z_vals,
                 Z_max,
                 biomarker_labels,
                 N_startpoints,
                 N_S_max,
                 N_iterations_MCMC,
                 output_folder,
                 dataset_name,
                 use_parallel_startpoints,
                 seed=1):
        # The initializer for the z-score based events implementation of AbstractSustain
        # Parameters:
        #   data                        - !important! needs to be (positive) z-scores!
        #                                 dim: number of subjects x number of biomarkers
        #   Z_vals                      - a matrix specifying the z-score thresholds for each biomarker
        #                                 for M biomarkers and 3 thresholds (1,2 and 3 for example) this would be a dim: M x 3 matrix
        #   Z_max                       - a vector specifying the maximum z-score for each biomarker
        #                                 when using z-score thresholds of 1,2,3 this would typically be 5.
        #                                 for M biomarkers this would be a dim: M x 1 vector
        #   biomarker_labels            - the names of the biomarkers as a list of strings
        #   N_startpoints               - number of startpoints to use in maximum likelihood step of SuStaIn, typically 25
        #   N_S_max                     - maximum number of subtypes, should be 1 or more
        #   N_iterations_MCMC           - number of MCMC iterations, typically 1e5 or 1e6 but can be lower for debugging
        #   output_folder               - where to save pickle files, etc.
        #   dataset_name                - for naming pickle files
        #   use_parallel_startpoints    - boolean for whether or not to parallelize the maximum likelihood loop
        #   seed                        - random number seed

        N                               = data.shape[1]  # number of biomarkers
        assert (len(biomarker_labels) == N), "number of labels should match number of biomarkers"

        stage_zscore            = Z_vals.T.flatten()    #np.array([y for x in Z_vals.T for y in x])
        stage_zscore            = stage_zscore.reshape(1,len(stage_zscore))
        IX_select               = stage_zscore>0
        stage_zscore            = stage_zscore[IX_select]
        stage_zscore            = stage_zscore.reshape(1,len(stage_zscore))

        num_zscores             = Z_vals.shape[1]
        IX_vals                 = np.array([[x for x in range(N)]] * num_zscores).T
        stage_biomarker_index   = IX_vals.T.flatten()   #np.array([y for x in IX_vals.T for y in x])
        stage_biomarker_index   = stage_biomarker_index.reshape(1,len(stage_biomarker_index))
        stage_biomarker_index   = stage_biomarker_index[IX_select]
        stage_biomarker_index   = stage_biomarker_index.reshape(1,len(stage_biomarker_index))

        self.Z_vals                     = Z_vals
        self.stage_zscore               = stage_zscore
        self.stage_biomarker_index      = stage_biomarker_index

        self.min_biomarker_zscore       = [0] * N
        self.max_biomarker_zscore       = Z_max
        self.std_biomarker_zscore       = [1] * N

        self.biomarker_labels           = biomarker_labels

        numStages                       = stage_zscore.shape[1]
        self.sustainData              = ZScoreSustainData(data, numStages)
        
        self.output_folder = output_folder
        self.seed = seed
        self.global_rng = np.random.default_rng(self.seed)
        self.N_S_max = N_S_max
        self.dataset_name = dataset_name
        self.N_startpoints = N_startpoints
        self.N_iterations_MCMC = N_iterations_MCMC

        self.pool                   = pathos.serial.SerialPool()
        
        
        # super().__init__(self.__sustainData,
        #                  N_startpoints,
        #                  N_S_max,
        #                  N_iterations_MCMC,
        #                  output_folder,
        #                  dataset_name,
        #                  use_parallel_startpoints,
        #                  seed)

#%% zSuStaIn functions 

def _initialise_sequence(SuStaIn_inputs_dict, sustainData, rng):
    # Randomly initialises a linear z-score model ensuring that the biomarkers
    # are monotonically increasing
    #
    #
    # OUTPUTS:
    # S - a random linear z-score model under the condition that each biomarker
    # is monotonically increasing

    N                                   = np.array(SuStaIn_inputs_dict['stage_zscore']).shape[1]
    S                                   = np.zeros(N)
    for i in range(N):

        IS_min_stage_zscore             = np.array([False] * N)
        possible_biomarkers             = np.unique(SuStaIn_inputs_dict['stage_biomarker_index'])
        for j in range(len(possible_biomarkers)):
            IS_unselected               = [False] * N
            for k in set(range(N)) - set(S[:i]):
                IS_unselected[k]        = True

            this_biomarkers             = np.array([(np.array(SuStaIn_inputs_dict['stage_biomarker_index'])[0] == possible_biomarkers[j]).astype(int) +
                                                    (np.array(IS_unselected) == 1).astype(int)]) == 2
            if not np.any(this_biomarkers):
                this_min_stage_zscore   = 0
            else:
                this_min_stage_zscore   = min(SuStaIn_inputs_dict['stage_zscore'][this_biomarkers])
            if (this_min_stage_zscore):
                temp                    = ((this_biomarkers.astype(int) + (SuStaIn_inputs_dict['stage_zscore'] == this_min_stage_zscore).astype(int)) == 2).T
                temp                    = temp.reshape(len(temp), )
                IS_min_stage_zscore[temp] = True

        events                          = np.array(range(N))
        possible_events                 = np.array(events[IS_min_stage_zscore])
        this_index                      = np.ceil(rng.random() * ((len(possible_events)))) - 1
        S[i]                            = possible_events[int(this_index)]

    S                                   = S.reshape(1, len(S))
    return S


def linspace_local2(a, b, N, arange_N):
    return a + (b - a) / (N - 1.) * arange_N


def _calculate_likelihood_stage(SuStaIn_inputs_dict, sustainData, S):
    '''
     Computes the likelihood of a single linear z-score model using an
     approximation method (faster)
    Outputs:
    ========
     p_perm_k - the probability of each subjects data at each stage of a particular subtype
     in the SuStaIn model
    '''

    N                                   = SuStaIn_inputs_dict['stage_biomarker_index'].shape[1]
    S_inv                               = np.array([0] * N)
    S_inv[S.astype(int)]                = np.arange(N)
    possible_biomarkers                 = np.unique(SuStaIn_inputs_dict['stage_biomarker_index'])
    B                                   = len(possible_biomarkers)
    point_value                         = np.zeros((B, N + 2))

    # all the arange you'll need below
    arange_N                            = np.arange(N + 2)

    for i in range(B):
        b                               = possible_biomarkers[i]
        event_location                  = np.concatenate([[0], S_inv[(SuStaIn_inputs_dict['stage_biomarker_index'] == b)[0]], [N]])
        event_value                     = np.concatenate([[SuStaIn_inputs_dict['min_biomarker_zscore'][i]], SuStaIn_inputs_dict['stage_zscore'][SuStaIn_inputs_dict['stage_biomarker_index'] == b], [SuStaIn_inputs_dict['max_biomarker_zscore'][i]]])
        for j in range(len(event_location) - 1):

            if j == 0:  # FIXME: nasty hack to get Matlab indexing to match up - necessary here because indices are used for linspace limits

                # original
                #temp                   = np.arange(event_location[j],event_location[j+1]+2)
                #point_value[i,temp]    = np.linspace(event_value[j],event_value[j+1],event_location[j+1]-event_location[j]+2)

                # fastest by a bit
                temp                    = arange_N[event_location[j]:(event_location[j + 1] + 2)]
                N_j                     = event_location[j + 1] - event_location[j] + 2
                point_value[i, temp]    = linspace_local2(event_value[j], event_value[j + 1], N_j, arange_N[0:N_j])

            else:
                # original
                #temp                   = np.arange(event_location[j] + 1, event_location[j + 1] + 2)
                #point_value[i, temp]   = np.linspace(event_value[j],event_value[j+1],event_location[j+1]-event_location[j]+1)

                # fastest by a bit
                temp                    = arange_N[(event_location[j] + 1):(event_location[j + 1] + 2)]
                N_j                     = event_location[j + 1] - event_location[j] + 1
                point_value[i, temp]    = linspace_local2(event_value[j], event_value[j + 1], N_j, arange_N[0:N_j])

    stage_value                         = 0.5 * point_value[:, :point_value.shape[1] - 1] + 0.5 * point_value[:, 1:]

    M                                   = sustainData.getNumSamples()   #data_local.shape[0]
    p_perm_k                            = np.zeros((M, N + 1))

    # optimised likelihood calc - take log and only call np.exp once after loop
    sigmat = np.array(SuStaIn_inputs_dict['std_biomarker_zscore'])

    factor                              = np.log(1. / np.sqrt(np.pi * 2.0) * sigmat)
    coeff                               = np.log(1. / float(N + 1))

    # original
    """
    for j in range(N+1):
        x                   = (data-np.tile(stage_value[:,j],(M,1)))/sigmat
        p_perm_k[:,j]       = coeff+np.sum(factor-.5*x*x,1)
    """
    # faster - do the tiling once
    # stage_value_tiled                   = np.tile(stage_value, (M, 1))
    # N_biomarkers                        = stage_value.shape[0]
    # for j in range(N + 1):
    #     stage_value_tiled_j             = stage_value_tiled[:, j].reshape(M, N_biomarkers)
    #     x                               = (sustainData.Tau_data - stage_value_tiled_j) / sigmat  #(data_local - stage_value_tiled_j) / sigmat
    #     p_perm_k[:, j]                  = coeff + np.sum(factor - .5 * np.square(x), 1)
    # p_perm_k                            = np.exp(p_perm_k)

    # even faster - do in one go
    x = (sustainData.Tau_data[:, :, None] - stage_value) / sigmat[None, :, None]
    p_perm_k = coeff + np.sum(factor[None, :, None] - 0.5 * np.square(x), 1)
    p_perm_k = np.exp(p_perm_k)

    return p_perm_k


def _optimise_parameters(SuStaIn_inputs_dict, sustainData, S_init, f_init, rng):
    # Optimise the parameters of the SuStaIn model

    M                                   = sustainData.getNumSamples()   #data_local.shape[0]
    N_S                                 = S_init.shape[0]
    N                                   = SuStaIn_inputs_dict['stage_zscore'].shape[1]

    S_opt                               = S_init.copy()  # have to copy or changes will be passed to S_init
    f_opt                               = np.array(f_init).reshape(N_S, 1, 1)
    f_val_mat                           = np.tile(f_opt, (1, N + 1, M))
    f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))
    p_perm_k                            = np.zeros((M, N + 1, N_S))

    for s in range(N_S):
        p_perm_k[:, :, s]               = _calculate_likelihood_stage(SuStaIn_inputs_dict,sustainData, S_opt[s])

    p_perm_k_weighted                   = p_perm_k * f_val_mat
    p_perm_k_norm                       = p_perm_k_weighted / np.sum(p_perm_k_weighted + 1e-250, axis=(1, 2), keepdims=True)
    f_opt                               = (np.squeeze(sum(sum(p_perm_k_norm))) / sum(sum(sum(p_perm_k_norm)))).reshape(N_S, 1, 1)
    f_val_mat                           = np.tile(f_opt, (1, N + 1, M))
    f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))
    order_seq                           = rng.permutation(N_S)  # this will produce different random numbers to Matlab

    for s in order_seq:
        order_bio                       = rng.permutation(N)  # this will produce different random numbers to Matlab
        for i in order_bio:
            current_sequence            = S_opt[s]
            current_location            = np.array([0] * len(current_sequence))
            current_location[current_sequence.astype(int)] = np.arange(len(current_sequence))

            selected_event              = i

            move_event_from             = current_location[selected_event]

            this_stage_zscore           = SuStaIn_inputs_dict['stage_zscore'][0, selected_event]
            selected_biomarker          = SuStaIn_inputs_dict['stage_biomarker_index'][0, selected_event]
            possible_zscores_biomarker  = SuStaIn_inputs_dict['stage_zscore'][SuStaIn_inputs_dict['stage_biomarker_index'] == selected_biomarker]

            # slightly different conditional check to matlab version to protect python from calling min,max on an empty array
            min_filter                  = possible_zscores_biomarker < this_stage_zscore
            max_filter                  = possible_zscores_biomarker > this_stage_zscore
            events                      = np.array(range(N))
            if np.any(min_filter):
                min_zscore_bound        = max(possible_zscores_biomarker[min_filter])
                min_zscore_bound_event  = events[((SuStaIn_inputs_dict['stage_zscore'][0] == min_zscore_bound).astype(int) + (SuStaIn_inputs_dict['stage_biomarker_index'][0] == selected_biomarker).astype(int)) == 2]
                move_event_to_lower_bound = current_location[min_zscore_bound_event] + 1
            else:
                move_event_to_lower_bound = 0
            if np.any(max_filter):
                max_zscore_bound        = min(possible_zscores_biomarker[max_filter])
                max_zscore_bound_event  = events[((SuStaIn_inputs_dict['stage_zscore'][0] == max_zscore_bound).astype(int) + (SuStaIn_inputs_dict['stage_biomarker_index'][0] == selected_biomarker).astype(int)) == 2]
                move_event_to_upper_bound = current_location[max_zscore_bound_event]
            else:
                move_event_to_upper_bound = N
                # FIXME: hack because python won't produce an array in range (N,N), while matlab will produce an array (N)... urgh
            if move_event_to_lower_bound == move_event_to_upper_bound:
                possible_positions      = np.array([0])
            else:
                possible_positions      = np.arange(move_event_to_lower_bound, move_event_to_upper_bound)
            possible_sequences          = np.zeros((len(possible_positions), N))
            possible_likelihood         = np.zeros((len(possible_positions), 1))
            possible_p_perm_k           = np.zeros((M, N + 1, len(possible_positions)))
            for index in range(len(possible_positions)):
                current_sequence        = S_opt[s]

                #choose a position in the sequence to move an event to
                move_event_to           = possible_positions[index]

                # move this event in its new position
                current_sequence        = np.delete(current_sequence, move_event_from, 0)  # this is different to the Matlab version, which call current_sequence(move_event_from) = []
                new_sequence            = np.concatenate([current_sequence[np.arange(move_event_to)], [selected_event], current_sequence[np.arange(move_event_to, N - 1)]])
                possible_sequences[index, :] = new_sequence

                possible_p_perm_k[:, :, index] = _calculate_likelihood_stage(SuStaIn_inputs_dict,sustainData, new_sequence)

                p_perm_k[:, :, s]       = possible_p_perm_k[:, :, index]
                total_prob_stage        = np.sum(p_perm_k * f_val_mat, 2)
                total_prob_subj         = np.sum(total_prob_stage, 1)
                possible_likelihood[index] = np.sum(np.log(total_prob_subj + 1e-250))

            possible_likelihood         = possible_likelihood.reshape(possible_likelihood.shape[0])
            max_likelihood              = max(possible_likelihood)
            this_S                      = possible_sequences[possible_likelihood == max_likelihood, :]
            this_S                      = this_S[0, :]
            S_opt[s]                    = this_S
            this_p_perm_k               = possible_p_perm_k[:, :, possible_likelihood == max_likelihood]
            p_perm_k[:, :, s]           = this_p_perm_k[:, :, 0]

        S_opt[s]                        = this_S

    p_perm_k_weighted                   = p_perm_k * f_val_mat
    #adding 1e-250 fixes divide by zero problem that happens rarely
    #p_perm_k_norm                       = p_perm_k_weighted / np.tile(np.sum(np.sum(p_perm_k_weighted, 1), 1).reshape(M, 1, 1), (1, N + 1, N_S))  # the second summation axis is different to Matlab version
    p_perm_k_norm                       = p_perm_k_weighted / np.sum(p_perm_k_weighted + 1e-250, axis=(1, 2), keepdims=True)

    f_opt                               = (np.squeeze(sum(sum(p_perm_k_norm))) / sum(sum(sum(p_perm_k_norm)))).reshape(N_S, 1, 1)
    f_val_mat                           = np.tile(f_opt, (1, N + 1, M))
    f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))

    f_opt                               = f_opt.reshape(N_S)
    total_prob_stage                    = np.sum(p_perm_k * f_val_mat, 2)
    total_prob_subj                     = np.sum(total_prob_stage, 1)

    likelihood_opt                      = np.sum(np.log(total_prob_subj + 1e-250))

    return S_opt, f_opt, likelihood_opt

def calc_coeff(sig):
    return 1. / np.sqrt(np.pi * 2.0) * sig

def calc_exp(x, mu, sig):
    x = (x - mu) / sig
    return np.exp(-.5 * x * x)

def _calculate_likelihood(SuStaIn_inputs_dict, sustainData, S, f):
    # Computes the likelihood of a mixture of models
    #
    #
    # OUTPUTS:
    # loglike               - the log-likelihood of the current model
    # total_prob_subj       - the total probability of the current SuStaIn model for each subject
    # total_prob_stage      - the total probability of each stage in the current SuStaIn model
    # total_prob_cluster    - the total probability of each subtype in the current SuStaIn model
    # p_perm_k              - the probability of each subjects data at each stage of each subtype in the current SuStaIn model

    M                                   = sustainData.getNumSamples()  #data_local.shape[0]
    N_S                                 = S.shape[0]
    N                                   = sustainData.getNumStages()    #self.stage_zscore.shape[1]

    f                                   = np.array(f).reshape(N_S, 1, 1)
    f_val_mat                           = np.tile(f, (1, N + 1, M))
    f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))

    p_perm_k                            = np.zeros((M, N + 1, N_S))

    for s in range(N_S):
        p_perm_k[:, :, s]               = _calculate_likelihood_stage(SuStaIn_inputs_dict,sustainData, S[s])  #self.__calculate_likelihood_stage_linearzscoremodel_approx(data_local, S[s])


    total_prob_cluster                  = np.squeeze(np.sum(p_perm_k * f_val_mat, 1))
    total_prob_stage                    = np.sum(p_perm_k * f_val_mat, 2)
    total_prob_subj                     = np.sum(total_prob_stage, 1)

    loglike                             = np.sum(np.log(total_prob_subj + 1e-250))

    return loglike, total_prob_subj, total_prob_stage, total_prob_cluster, p_perm_k

def _perform_mcmc(SuStaIn_inputs_dict, sustainData, seq_init, f_init, n_iterations, seq_sigma, f_sigma):
    # Take MCMC samples of the uncertainty in the SuStaIn model parameters

    N                                   = SuStaIn_inputs_dict['stage_zscore'].shape[1]
    N_S                                 = seq_init.shape[0]

    if isinstance(f_sigma, float):  # FIXME: hack to enable multiplication
        f_sigma                         = np.array([f_sigma])

    samples_sequence                    = np.zeros((N_S, N, n_iterations))
    samples_f                           = np.zeros((N_S, n_iterations))
    samples_likelihood                  = np.zeros((n_iterations, 1))
    samples_sequence[:, :, 0]           = seq_init  # don't need to copy as we don't write to 0 index
    samples_f[:, 0]                     = f_init

    # Reduce frequency of tqdm update to 0.1% of total for larger iteration numbers
    tqdm_update_iters = int(n_iterations/1000) if n_iterations > 100000 else None 

    for i in tqdm(range(n_iterations), "MCMC Iteration", n_iterations, miniters=tqdm_update_iters):
        if i > 0:
            seq_order                   = SuStaIn_inputs_dict['global_rng'].permutation(N_S)  # this function returns different random numbers to Matlab
            for s in seq_order:
                move_event_from         = int(np.ceil(N * SuStaIn_inputs_dict['global_rng'].random())) - 1
                current_sequence        = samples_sequence[s, :, i - 1]

                current_location        = np.array([0] * N)
                current_location[current_sequence.astype(int)] = np.arange(N)

                selected_event          = int(current_sequence[move_event_from])
                this_stage_zscore       = SuStaIn_inputs_dict['stage_zscore'][0, selected_event]
                selected_biomarker      = SuStaIn_inputs_dict['stage_biomarker_index'][0, selected_event]
                possible_zscores_biomarker = SuStaIn_inputs_dict['stage_zscore'][SuStaIn_inputs_dict['stage_biomarker_index'] == selected_biomarker]

                # slightly different conditional check to matlab version to protect python from calling min,max on an empty array
                min_filter              = possible_zscores_biomarker < this_stage_zscore
                max_filter              = possible_zscores_biomarker > this_stage_zscore
                events                  = np.array(range(N))
                if np.any(min_filter):
                    min_zscore_bound            = max(possible_zscores_biomarker[min_filter])
                    min_zscore_bound_event      = events[((SuStaIn_inputs_dict['stage_zscore'][0] == min_zscore_bound).astype(int) + (SuStaIn_inputs_dict['stage_biomarker_index'][0] == selected_biomarker).astype(int)) == 2]
                    move_event_to_lower_bound   = current_location[min_zscore_bound_event] + 1
                else:
                    move_event_to_lower_bound   = 0

                if np.any(max_filter):
                    max_zscore_bound            = min(possible_zscores_biomarker[max_filter])
                    max_zscore_bound_event      = events[((SuStaIn_inputs_dict['stage_zscore'][0] == max_zscore_bound).astype(int) + (SuStaIn_inputs_dict['stage_biomarker_index'][0] == selected_biomarker).astype(int)) == 2]
                    move_event_to_upper_bound   = current_location[max_zscore_bound_event]
                else:
                    move_event_to_upper_bound   = N

                # FIXME: hack because python won't produce an array in range (N,N), while matlab will produce an array (N)... urgh
                if move_event_to_lower_bound == move_event_to_upper_bound:
                    possible_positions          = np.array([0])
                else:
                    possible_positions          = np.arange(move_event_to_lower_bound, move_event_to_upper_bound)

                distance                = possible_positions - move_event_from

                if isinstance(seq_sigma, int):  # FIXME: change to float
                    this_seq_sigma      = seq_sigma
                else:
                    this_seq_sigma      = seq_sigma[s, selected_event]

                # use own normal PDF because stats.norm is slow
                weight                  = calc_coeff(this_seq_sigma) * calc_exp(distance, 0., this_seq_sigma)
                weight                  /= np.sum(weight)
                index                   = SuStaIn_inputs_dict['global_rng'].choice(range(len(possible_positions)), 1, replace=True, p=weight)  # FIXME: difficult to check this because random.choice is different to Matlab randsample

                move_event_to           = possible_positions[index]

                current_sequence        = np.delete(current_sequence, move_event_from, 0)
                new_sequence            = np.concatenate([current_sequence[np.arange(move_event_to)], [selected_event], current_sequence[np.arange(move_event_to, N - 1)]])
                samples_sequence[s, :, i] = new_sequence

            new_f                       = samples_f[:, i - 1] + f_sigma * SuStaIn_inputs_dict['global_rng'].standard_normal()
            new_f                       = (np.fabs(new_f) / np.sum(np.fabs(new_f)))
            samples_f[:, i]             = new_f

        S                               = samples_sequence[:, :, i]
        f                               = samples_f[:, i]
        likelihood_sample, _, _, _, _   = _calculate_likelihood(SuStaIn_inputs_dict,sustainData, S, f)
        samples_likelihood[i]           = likelihood_sample

        if i > 0:
            ratio                           = np.exp(samples_likelihood[i] - samples_likelihood[i - 1])
            if ratio < SuStaIn_inputs_dict['global_rng'].random():
                samples_likelihood[i]       = samples_likelihood[i - 1]
                samples_sequence[:, :, i]   = samples_sequence[:, :, i - 1]
                samples_f[:, i]             = samples_f[:, i - 1]

    perm_index                          = np.where(samples_likelihood == max(samples_likelihood))
    perm_index                          = perm_index[0]
    ml_likelihood                       = max(samples_likelihood)
    ml_sequence                         = samples_sequence[:, :, perm_index]
    ml_f                                = samples_f[:, perm_index]

    return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood

def _plot_sustain_model(SuStaIn_inputs_dict, *args, **kwargs):
    return ZscoreSustain.plot_positional_var(*args, Z_vals=SuStaIn_inputs_dict['Z_vals'], **kwargs)

def subtype_and_stage_individuals(SuStaIn_inputs_dict, sustainData, samples_sequence, samples_f, N_samples):
    # Subtype and stage a set of subjects. Useful for subtyping/staging subjects that were not used to build the model

    nSamples                            = sustainData.getNumSamples()  #data_local.shape[0]
    nStages                             = sustainData.getNumStages()    #self.stage_zscore.shape[1]

    n_iterations_MCMC                   = samples_sequence.shape[2]
    select_samples                      = np.round(np.linspace(0, n_iterations_MCMC - 1, N_samples))
    N_S                                 = samples_sequence.shape[0]
    temp_mean_f                         = np.mean(samples_f, axis=1)
    ix                                  = np.argsort(temp_mean_f)[::-1]

    prob_subtype_stage                  = np.zeros((nSamples, nStages + 1, N_S))
    prob_subtype                        = np.zeros((nSamples, N_S))
    prob_stage                          = np.zeros((nSamples, nStages + 1))

    for i in range(N_samples):
        sample                          = int(select_samples[i])

        this_S                          = samples_sequence[ix, :, sample]
        this_f                          = samples_f[ix, sample]

        _,                  \
        _,                  \
        total_prob_stage,   \
        total_prob_subtype, \
        total_prob_subtype_stage        = _calculate_likelihood(SuStaIn_inputs_dict,sustainData, this_S, this_f)

        total_prob_subtype              = total_prob_subtype.reshape(len(total_prob_subtype), N_S)
        total_prob_subtype_norm         = total_prob_subtype        / np.tile(np.sum(total_prob_subtype, 1).reshape(len(total_prob_subtype), 1),        (1, N_S))
        total_prob_stage_norm           = total_prob_stage          / np.tile(np.sum(total_prob_stage, 1).reshape(len(total_prob_stage), 1),          (1, nStages + 1)) #removed total_prob_subtype

        #total_prob_subtype_stage_norm   = total_prob_subtype_stage  / np.tile(np.sum(np.sum(total_prob_subtype_stage, 1), 1).reshape(nSamples, 1, 1),   (1, nStages + 1, N_S))
        total_prob_subtype_stage_norm   = total_prob_subtype_stage / np.tile(np.sum(np.sum(total_prob_subtype_stage, 1, keepdims=True), 2).reshape(nSamples, 1, 1),(1, nStages + 1, N_S))

        prob_subtype_stage              = (i / (i + 1.) * prob_subtype_stage)   + (1. / (i + 1.) * total_prob_subtype_stage_norm)
        prob_subtype                    = (i / (i + 1.) * prob_subtype)         + (1. / (i + 1.) * total_prob_subtype_norm)
        prob_stage                      = (i / (i + 1.) * prob_stage)           + (1. / (i + 1.) * total_prob_stage_norm)

    ml_subtype                          = np.nan * np.ones((nSamples, 1))
    prob_ml_subtype                     = np.nan * np.ones((nSamples, 1))
    ml_stage                            = np.nan * np.ones((nSamples, 1))
    prob_ml_stage                       = np.nan * np.ones((nSamples, 1))

    for i in range(nSamples):
        this_prob_subtype               = np.squeeze(prob_subtype[i, :])
        # if not np.isnan(this_prob_subtype).any()
        if (np.sum(np.isnan(this_prob_subtype)) == 0):
            # this_subtype = this_prob_subtype.argmax(
            this_subtype                = np.where(this_prob_subtype == np.max(this_prob_subtype))

            try:
                ml_subtype[i]           = this_subtype
            except:
                ml_subtype[i]           = this_subtype[0][0]
            if this_prob_subtype.size == 1 and this_prob_subtype == 1:
                prob_ml_subtype[i]      = 1
            else:
                try:
                    prob_ml_subtype[i]  = this_prob_subtype[this_subtype]
                except:
                    prob_ml_subtype[i]  = this_prob_subtype[this_subtype[0][0]]

        this_prob_stage                 = np.squeeze(prob_subtype_stage[i, :, int(ml_subtype[i])])
        
        if (np.sum(np.isnan(this_prob_stage)) == 0):
            # this_stage = 
            this_stage                  = np.where(this_prob_stage == np.max(this_prob_stage))
            ml_stage[i]                 = this_stage[0][0]
            prob_ml_stage[i]            = this_prob_stage[this_stage[0][0]]
    # NOTE: The above loop can be replaced with some simpler numpy calls
    # May need to do some masking to avoid NaNs, or use `np.nanargmax` depending on preference
    # E.g. ml_subtype == prob_subtype.argmax(1)
    # E.g. ml_stage == prob_subtype_stage[np.arange(prob_subtype_stage.shape[0]), :, ml_subtype].argmax(1)
    return ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype, prob_stage, prob_subtype_stage

def subtype_and_stage_individuals_newData(SuStaIn_inputs_dict, data_new, samples_sequence, samples_f, N_samples):

    numStages_new                   = SuStaIn_inputs_dict['sustainData'].getNumStages() #data_new.shape[1]
    sustainData_newData             = ZScoreSustainData(data_new, numStages_new)

    ml_subtype,         \
    prob_ml_subtype,    \
    ml_stage,           \
    prob_ml_stage,      \
    prob_subtype,       \
    prob_stage,         \
    prob_subtype_stage          = subtype_and_stage_individuals(sustainData_newData, samples_sequence, samples_f, N_samples)

    return ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype, prob_stage, prob_subtype_stage

# ********************* STATIC METHODS
def linspace_local2(a, b, N, arange_N):
    return a + (b - a) / (N - 1.) * arange_N

def check_biomarker_colours(biomarker_colours, biomarker_labels):
    if isinstance(biomarker_colours, dict):
        # Check each label exists
        assert all(i in biomarker_labels for i in biomarker_colours.keys()), "A label doesn't match!"
        # Check each colour exists
        assert all(mcolors.is_color_like(i) for i in biomarker_colours.values()), "A proper colour wasn't given!"
        # Add in any colours that aren't defined, allowing for partial colouration
        for label in biomarker_labels:
            if label not in biomarker_colours:
                biomarker_colours[label] = "black"
    elif isinstance(biomarker_colours, (list, tuple)):
        # Check each colour exists
        assert all(mcolors.is_color_like(i) for i in biomarker_colours), "A proper colour wasn't given!"
        # Check right number of colours given
        assert len(biomarker_colours) == len(biomarker_labels), "The number of colours and labels do not match!"
        # Turn list of colours into a label:colour mapping
        biomarker_colours = {k:v for k,v in zip(biomarker_labels, biomarker_colours)}
    else:
        raise TypeError("A dictionary mapping label:colour or list/tuple of colours must be given!")
    return biomarker_colours


# LB: Includes fix on plots when trying to plot unique abnormal z_vals for each variable 
def plot_positional_var(samples_sequence, samples_f, n_samples, Z_vals, biomarker_labels=None, ml_f_EM=None, cval=False, subtype_order=None, biomarker_order=None, title_font_size=12, stage_font_size=10, stage_label='SuStaIn Stage', stage_rot=0, stage_interval=1, label_font_size=10, label_rot=0, cmap="original", biomarker_colours=None, figsize=None, subtype_titles=None, separate_subtypes=False, save_path=None, save_kwargs={}):
    # Get the number of subtypes
    N_S = samples_sequence.shape[0]
    # Get the number of features/biomarkers
    N_bio = Z_vals.shape[0]
    # Check that the number of labels given match
    if biomarker_labels is not None:
        assert len(biomarker_labels) == N_bio
    # Set subtype order if not given
    if subtype_order is None:
        # Determine order if info given
        if ml_f_EM is not None:
            subtype_order = np.argsort(ml_f_EM)[::-1]
        # Otherwise determine order from samples_f
        else:
            subtype_order = np.argsort(np.mean(samples_f, 1))[::-1]
    elif isinstance(subtype_order, tuple):
        subtype_order = list(subtype_order)
    # Unravel the stage zscores from Z_vals
    stage_zscore = Z_vals.T.flatten()
    IX_select = np.nonzero(stage_zscore)[0]
    stage_zscore = stage_zscore[IX_select][None, :]
    # Get the z-scores and their number
    zvalues = np.unique(stage_zscore)
    N_z = len(zvalues)
    # Extract which biomarkers have which zscores/stages
    stage_biomarker_index = np.tile(np.arange(N_bio), (N_z,))
    stage_biomarker_index = stage_biomarker_index[IX_select]
    # Warn user of reordering if labels and order given
    if biomarker_labels is not None and biomarker_order is not None:
        warnings.warn(
            "Both labels and an order have been given. The labels will be reordered according to the given order!"
        )
    if biomarker_order is not None:
        # SuStaIn_inputs_dict_plot_biomarker_order is not suited to zscore version
        # Ignore for compatability, for now
        # One option is to reshape, sum position, and lowest->highest determines order
        if len(biomarker_order) > N_bio:
            biomarker_order = np.arange(N_bio)
    # Otherwise use default order
    else:
        biomarker_order = np.arange(N_bio)
    # If no labels given, set dummy defaults
    if biomarker_labels is None:
        biomarker_labels = [f"Biomarker {i}" for i in range(N_bio)]
    # Otherwise reorder according to given order (or not if not given)
    else:
        biomarker_labels = [biomarker_labels[i] for i in biomarker_order]
    # Check number of subtype titles is correct if given
    if subtype_titles is not None:
        assert len(subtype_titles) == N_S
    # Z-score colour definition
    if cmap == "original":
        # Hard-coded colours: hooray!
        colour_mat = np.array([[1, 0, 0], [1, 0, 1], [0, 0, 1], [0.5, 0, 1], [0, 1, 1], [0, 1, 0.5]])[:N_z]
        # We only have up to 5 default colours, so double-check
        if colour_mat.shape[0] > N_z:
            raise ValueError(f"Colours are only defined for {len(colour_mat)} z-scores!")
    else:
        raise NotImplementedError
    '''
    Note for future self/others: The use of any arbitrary colourmap is problematic, as when the same stage can have the same biomarker with different z-scores of different certainties, the colours need to mix in a visually informative way and there can be issues with RGB mixing/interpolation, particulary if there are >2 z-scores for the same biomarker at the same stage. It may be possible, but the end result may no longer be useful to look at.
    '''

    # Check biomarker label colours
    # If custom biomarker text colours are given
    if biomarker_colours is not None:
        biomarker_colours = check_biomarker_colours(
        biomarker_colours, biomarker_labels
    )
    # Default case of all-black colours
    # Unnecessary, but skips a check later
    else:
        biomarker_colours = {i:"black" for i in biomarker_labels}

    # Flag to plot subtypes separately
    if separate_subtypes:
        nrows, ncols = 1, 1
    else:
        # Determine number of rows and columns (rounded up)
        if N_S == 1:
            nrows, ncols = 1, 1
        elif N_S < 3:
            nrows, ncols = 1, N_S
        elif N_S < 7:
            nrows, ncols = 2, int(np.ceil(N_S / 2))
        else:
            nrows, ncols = 3, int(np.ceil(N_S / 3))
    # Total axes used to loop over
    total_axes = nrows * ncols
    # Create list of single figure object if not separated
    if separate_subtypes:
        subtype_loops = N_S
    else:
        subtype_loops = 1
    # Container for all figure objects
    figs = []
    # Loop over figures (only makes a diff if separate_subtypes=True)
    for i in range(subtype_loops):
        # Create the figure and axis for this subtype loop
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        figs.append(fig)
        # Loop over each axis
        for j in range(total_axes):
            # Normal functionality (all subtypes on one plot)
            if not separate_subtypes:
                i = j
            # Handle case of a single array
            if isinstance(axs, np.ndarray):
                ax = axs.flat[i]
            else:
                ax = axs
            # Check if i is superfluous
            if i not in range(N_S):
                ax.set_axis_off()
                continue

            this_samples_sequence = samples_sequence[subtype_order[i],:,:].T
            N = this_samples_sequence.shape[1]

            # Construct confusion matrix (vectorized)
            # We compare `this_samples_sequence` against each position
            # Sum each time it was observed at that point in the sequence
            # And normalize for number of samples/sequences
            confus_matrix = (this_samples_sequence==np.arange(N)[:, None, None]).sum(1) / this_samples_sequence.shape[0]

            # Define the confusion matrix to insert the colours
            # Use 1s to start with all white
            confus_matrix_c = np.ones((N_bio, N, 3))

            # Loop over each z-score event
            for j, z in enumerate(zvalues):
                # Determine which colours to alter
                # I.e. red (1,0,0) means removing green & blue channels
                # according to the certainty of red (representing z-score 1)
                alter_level = colour_mat[j] == 0
                # Extract the uncertainties for this z-score
                confus_matrix_zscore = confus_matrix[(stage_zscore==z)[0]]
                # Subtract the certainty for this colour
                confus_matrix_c[
                    np.ix_(
                        stage_biomarker_index[(stage_zscore==z)[0]], range(N),
                        alter_level
                    )
                ] -= np.tile(
                    confus_matrix_zscore.reshape((stage_zscore==z).sum(), N, 1),
                    (1, 1, alter_level.sum())
                )
            if subtype_titles is not None:
                title_i = subtype_titles[i]
            else:
                # Add axis title
                if cval == False:
                    temp_mean_f = np.mean(samples_f, 1)
                    # Shuffle vals according to subtype_order
                    # This defaults to previous method if custom order not given
                    vals = temp_mean_f[subtype_order]

                    if n_samples != np.inf:
                        title_i = f"Subtype {i+1} (f={vals[i]:.2f}, n={np.round(vals[i] * n_samples):n})"
                    else:
                        title_i = f"Subtype {i+1} (f={vals[i]:.2f})"
                else:
                    title_i = f"Subtype {i+1} cross-validated"
            # Plot the colourized matrix
            ax.imshow(
                confus_matrix_c[biomarker_order, :, :],
                interpolation='nearest'
            )
            # Add the xticks and labels
            stage_ticks = np.arange(0, N, stage_interval)
            ax.set_xticks(stage_ticks)
            ax.set_xticklabels(stage_ticks+1, fontsize=stage_font_size, rotation=stage_rot)
            # Add the yticks and labels
            ax.set_yticks(np.arange(N_bio))
            # Add biomarker labels to LHS of every row only
            if (i % ncols) == 0:
                ax.set_yticklabels(biomarker_labels, ha='right', fontsize=label_font_size, rotation=label_rot)
                # Set biomarker label colours
                for tick_label in ax.get_yticklabels():
                    tick_label.set_color(biomarker_colours[tick_label.get_text()])
            else:
                ax.set_yticklabels([])
            # Make the event label slightly bigger than the ticks
            ax.set_xlabel(stage_label, fontsize=stage_font_size+2)
            ax.set_title(title_i, fontsize=title_font_size)
        # Tighten up the figure
        fig.tight_layout()
        # Save if a path is given
        if save_path is not None:
            # Modify path for specific subtype if specified
            # Don't modify save_path!
            if separate_subtypes:
                save_name = f"{save_path}_subtype{i}"
            else:
                save_name = f"{save_path}_all-subtypes"
            # Handle file format, avoids issue with . in filenames
            if "format" in save_kwargs:
                file_format = save_kwargs.pop("format")
            # Default to png
            else:
                file_format = "png"
            # Save the figure, with additional kwargs
            fig.savefig(
                f"{save_name}.{file_format}",
                **save_kwargs
            )
    return figs, axs

# ********************* TEST METHODS
@classmethod
def test_sustain(cls, n_biomarkers, n_samples, n_subtypes, 
ground_truth_subtypes, sustain_kwargs, seed=42):
    # Set a global seed to propagate
    np.random.seed(seed)
    # Create Z values
    Z_vals = np.tile(np.arange(1, 4), (n_biomarkers, 1))
    Z_vals[0, 2] = 0

    Z_max = np.full((n_biomarkers,), 5)
    Z_max[2] = 2

    ground_truth_sequences = cls.generate_random_model(Z_vals, n_subtypes)
    N_stages = np.sum(Z_vals > 0) + 1

    ground_truth_stages_control = np.zeros((int(np.round(n_samples * 0.25)), 1))
    ground_truth_stages_other = np.random.randint(1, N_stages+1, (int(np.round(n_samples * 0.75)), 1))
    ground_truth_stages = np.vstack((ground_truth_stages_control, ground_truth_stages_other)).astype(int)

    data, data_denoised, stage_value = cls.generate_data(
        ground_truth_subtypes,
        ground_truth_stages,
        ground_truth_sequences,
        Z_vals,
        Z_max
    )

    return cls(
        data, Z_vals, Z_max,
        **sustain_kwargs
    )

@staticmethod
def generate_random_model(Z_vals, N_S, seed=None):
    num_biomarkers = Z_vals.shape[0]

    stage_zscore = Z_vals.T.flatten()#[np.newaxis, :]

    IX_select = np.nonzero(stage_zscore)[0]
    stage_zscore = stage_zscore[IX_select]#[np.newaxis, :]
    num_zscores = Z_vals.shape[0]

    stage_biomarker_index = np.tile(np.arange(num_biomarkers), (num_zscores,))
    stage_biomarker_index = stage_biomarker_index[IX_select]#[np.newaxis, :]

    N = stage_zscore.shape[0]
    S = np.zeros((N_S, N))
    # Moved outside loop, no need
    possible_biomarkers = np.unique(stage_biomarker_index)

    for s in range(N_S):
        for i in range(N):

            IS_min_stage_zscore = np.full(N, False)

            for j in possible_biomarkers:
                IS_unselected = np.full(N, False)
                # I have no idea what purpose this serves, so leaving for now
                for k in set(range(N)) - set(S[s][:i]):
                    IS_unselected[k] = True

                this_biomarkers = np.logical_and(
                    stage_biomarker_index == possible_biomarkers[j],
                    np.array(IS_unselected) == 1
                )
                if not np.any(this_biomarkers):
                    this_min_stage_zscore = 0
                else:
                    this_min_stage_zscore = np.min(stage_zscore[this_biomarkers])
                
                if this_min_stage_zscore:
                    IS_min_stage_zscore[np.logical_and(
                        this_biomarkers,
                        stage_zscore == this_min_stage_zscore
                    )] = True

            events = np.arange(N)
            possible_events = events[IS_min_stage_zscore]
            this_index = np.ceil(np.random.rand() * len(possible_events)) - 1
            
            S[s][i] = possible_events[int(this_index)]
    return S

# TODO: Refactor this as above
@staticmethod
def generate_data(subtypes, stages, gt_ordering, Z_vals, Z_max):
    B = Z_vals.shape[0]
    stage_zscore = np.array([y for x in Z_vals.T for y in x])
    stage_zscore = stage_zscore.reshape(1,len(stage_zscore))
    IX_select = stage_zscore>0
    stage_zscore = stage_zscore[IX_select]
    stage_zscore = stage_zscore.reshape(1,len(stage_zscore))

    num_zscores = Z_vals.shape[1]
    IX_vals = np.array([[x for x in range(B)]] * num_zscores).T
    stage_biomarker_index = np.array([y for x in IX_vals.T for y in x])
    stage_biomarker_index = stage_biomarker_index.reshape(1,len(stage_biomarker_index))
    stage_biomarker_index = stage_biomarker_index[IX_select]
    stage_biomarker_index = stage_biomarker_index.reshape(1,len(stage_biomarker_index))

    min_biomarker_zscore = [0]*B
    max_biomarker_zscore = Z_max
    std_biomarker_zscore = [1]*B

    N = stage_biomarker_index.shape[1]
    N_S = gt_ordering.shape[0]

    possible_biomarkers = np.unique(stage_biomarker_index)
    stage_value = np.zeros((B,N+2,N_S))

    for s in range(N_S):
        S = gt_ordering[s,:]
        S_inv = np.array([0]*N)
        S_inv[S.astype(int)] = np.arange(N)
        for i in range(B):
            b = possible_biomarkers[i]
            event_location = np.concatenate([[0], S_inv[(stage_biomarker_index == b)[0]], [N]])

            event_value = np.concatenate([[min_biomarker_zscore[i]], stage_zscore[stage_biomarker_index == b], [max_biomarker_zscore[i]]])

            for j in range(len(event_location)-1):

                if j == 0: # FIXME: nasty hack to get Matlab indexing to match up - necessary here because indices are used for linspace limits
                    index = np.arange(event_location[j],event_location[j+1]+2)
                    stage_value[i,index,s] = np.linspace(event_value[j],event_value[j+1],event_location[j+1]-event_location[j]+2)
                else:
                    index = np.arange(event_location[j] + 1, event_location[j + 1] + 2)
                    stage_value[i,index,s] = np.linspace(event_value[j],event_value[j+1],event_location[j+1]-event_location[j]+1)

    M = stages.shape[0]
    data_denoised = np.zeros((M,B))
    for m in range(M):
        data_denoised[m,:] = stage_value[:,int(stages[m]),subtypes[m]]
    data = data_denoised + norm.ppf(np.random.rand(B,M).T)*np.tile(std_biomarker_zscore,(M,1))

    return data, data_denoised, stage_value


def _perform_em(SuStaIn_inputs_dict, sustainData, current_sequence, current_f, rng):

    # Perform an E-M procedure to estimate parameters of SuStaIn model
    MaxIter                             = 100

    N                                   = sustainData.getNumStages()    #self.stage_zscore.shape[1]
    N_S                                 = current_sequence.shape[0]
    current_likelihood, _, _, _, _      = _calculate_likelihood(SuStaIn_inputs_dict,sustainData, current_sequence, current_f)

    terminate                           = 0
    iteration                           = 0
    samples_sequence                    = np.nan * np.ones((MaxIter, N, N_S))
    samples_f                           = np.nan * np.ones((MaxIter, N_S))
    samples_likelihood                  = np.nan * np.ones((MaxIter, 1))

    samples_sequence[0, :, :]           = current_sequence.reshape(current_sequence.shape[1], current_sequence.shape[0])
    current_f                           = np.array(current_f).reshape(len(current_f))
    samples_f[0, :]                     = current_f
    samples_likelihood[0]               = current_likelihood
    while terminate == 0:

        candidate_sequence,     \
        candidate_f,            \
        candidate_likelihood            = _optimise_parameters(SuStaIn_inputs_dict,sustainData, current_sequence, current_f, rng)
        HAS_converged                   = np.fabs((candidate_likelihood - current_likelihood) / max(candidate_likelihood, current_likelihood)) < 1e-6

        if HAS_converged:
            #print('EM converged in', iteration + 1, 'iterations')
            terminate                   = 1
        else:
            if candidate_likelihood > current_likelihood:
                current_sequence        = candidate_sequence
                current_f               = candidate_f
                current_likelihood      = candidate_likelihood
                
        samples_sequence[iteration, :, :] = current_sequence.T.reshape(current_sequence.T.shape[0], N_S)
        samples_f[iteration, :]         = current_f
        samples_likelihood[iteration]   = current_likelihood
            

        if iteration == (MaxIter - 1):
            terminate                   = 1
        iteration                       = iteration + 1

    ml_sequence                         = current_sequence
    ml_f                                = current_f
    ml_likelihood                       = current_likelihood
    
    return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood

def calculate_z_max(data):
    Z_max  = np.array([np.nan]*data.shape[1])           # maximum z-score

    for iter_n in range(data.shape[1]):            
        #Extract biomarker data 
        biomarker_data = np.asarray(data[:,iter_n],dtype=float).reshape(-1,1)
        # Fit a Gaussian Mixture Model with 3 components
        ci_lower, ci_upper = np.percentile(biomarker_data, [5, 95])
        Z_max[iter_n] = ci_upper
    
    return Z_max


def _find_ml_mixture_iteration(SuStaIn_inputs_dict, sustainData, seq_init, f_init, seed_seq):
    #Convenience sub-function for above

    # Get process-appropriate Generator
    rng = np.random.default_rng(seed_seq)

    ml_sequence,        \
    ml_f,               \
    ml_likelihood,      \
    samples_sequence,   \
    samples_f,          \
    samples_likelihood = _perform_em(SuStaIn_inputs_dict,sustainData, seq_init, f_init, rng)

    return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood



def _find_ml_mixture(SuStaIn_inputs_dict, sustainData, seq_init, f_init):
    # Fit a mixture of models
    #
    #
    # OUTPUTS:
    # ml_sequence   - the ordering of the stages for each subtype for the next SuStaIn model in the hierarchy
    # ml_f          - the most probable proportion of individuals belonging to each subtype for the next SuStaIn model in the hierarchy
    # ml_likelihood - the likelihood of the most probable SuStaIn model for the next SuStaIn model in the hierarchy

    N_S                                 = seq_init.shape[0]

    partial_iter                        = partial(_find_ml_mixture_iteration, SuStaIn_inputs_dict, sustainData, seq_init, f_init)
    seed_sequences = np.random.SeedSequence(SuStaIn_inputs_dict['global_rng'].integers(1e10))
    pool_output_list                    = SuStaIn_inputs_dict['pool'].map(partial_iter, seed_sequences.spawn(SuStaIn_inputs_dict['N_startpoints']))

    if ~isinstance(pool_output_list, list):
        pool_output_list                = list(pool_output_list)

    ml_sequence_mat                     = np.zeros((N_S, sustainData.getNumStages(), SuStaIn_inputs_dict['N_startpoints']))
    ml_f_mat                            = np.zeros((N_S, SuStaIn_inputs_dict['N_startpoints']))
    ml_likelihood_mat                   = np.zeros((SuStaIn_inputs_dict['N_startpoints'], 1))

    for i in range(SuStaIn_inputs_dict['N_startpoints']):
        ml_sequence_mat[:, :, i]        = pool_output_list[i][0]
        ml_f_mat[:, i]                  = pool_output_list[i][1]
        ml_likelihood_mat[i]            = pool_output_list[i][2]

    ix                                  = np.where(ml_likelihood_mat == max(ml_likelihood_mat))
    ix                                  = ix[0]

    ml_sequence                         = ml_sequence_mat[:, :, ix]
    ml_f                                = ml_f_mat[:, ix]
    ml_likelihood                       = ml_likelihood_mat[ix]

    return ml_sequence, ml_f, ml_likelihood, ml_sequence_mat, ml_f_mat, ml_likelihood_mat



def _find_ml_iteration(SuStaIn_inputs_dict, sustainData, seed_seq):
    #Convenience sub-function for above

    # Get process-appropriate Generator
    rng = np.random.default_rng(seed_seq)

    # randomly initialise the sequence of the linear z-score model
    seq_init                        = _initialise_sequence(SuStaIn_inputs_dict,sustainData, rng)
    f_init                          = [1]

    this_ml_sequence,   \
    this_ml_f,          \
    this_ml_likelihood, \
    _,                  \
    _,                  \
    _                   = _perform_em(SuStaIn_inputs_dict,sustainData, seq_init, f_init, rng)
    

    return this_ml_sequence, this_ml_f, this_ml_likelihood

def _find_ml(SuStaIn_inputs_dict, sustainData):
    # Fit the maximum likelihood model
    #
    # OUTPUTS:
    # ml_sequence   - the ordering of the stages for each subtype
    # ml_f          - the most probable proportion of individuals belonging to each subtype
    # ml_likelihood - the likelihood of the most probable SuStaIn model

    partial_iter                        = partial(_find_ml_iteration, SuStaIn_inputs_dict,sustainData)
    seed_sequences = np.random.SeedSequence(SuStaIn_inputs_dict['global_rng'].integers(1e10))
    pool_output_list                    = SuStaIn_inputs_dict['pool'].map(partial_iter, seed_sequences.spawn(SuStaIn_inputs_dict['N_startpoints']))

    if ~isinstance(pool_output_list, list):
        pool_output_list                = list(pool_output_list)

    ml_sequence_mat                     = np.zeros((1, sustainData.getNumStages(), SuStaIn_inputs_dict['N_startpoints'])) #np.zeros((1, self.stage_zscore.shape[1], self.N_startpoints))
    ml_f_mat                            = np.zeros((1, SuStaIn_inputs_dict['N_startpoints']))
    ml_likelihood_mat                   = np.zeros(SuStaIn_inputs_dict['N_startpoints'])

    for i in range(SuStaIn_inputs_dict['N_startpoints']):
        ml_sequence_mat[:, :, i]        = pool_output_list[i][0]
        ml_f_mat[:, i]                  = pool_output_list[i][1]
        ml_likelihood_mat[i]            = pool_output_list[i][2]


    ix                                  = np.argmax(ml_likelihood_mat)

    ml_sequence                         = ml_sequence_mat[:, :, ix]
    ml_f                                = ml_f_mat[:, ix]
    ml_likelihood                       = ml_likelihood_mat[ix]

    return ml_sequence, ml_f, ml_likelihood, ml_sequence_mat, ml_f_mat, ml_likelihood_mat


def _find_ml_split_iteration(SuStaIn_inputs_dict, sustainData, seed_seq):
    #Convenience sub-function for above

    # Get process-appropriate Generator
    rng = np.random.default_rng(seed_seq)

    N_S                                 = 2

    # randomly initialise individuals as belonging to one of the two subtypes (clusters)
    min_N_cluster                       = 0
    while min_N_cluster == 0:
        vals = rng.random(sustainData.getNumSamples())
        cluster_assignment = np.ceil(N_S * vals).astype(int)
        # Count cluster sizes
        # Guarantee 1s and 2s counts with minlength=3
        # Ignore 0s count with [1:]
        cluster_sizes = np.bincount(cluster_assignment, minlength=3)[1:]
        # Get the minimum cluster size
        min_N_cluster = cluster_sizes.min()

    # initialise the stages of the two models by fitting a single model to each of the two sets of individuals
    seq_init                            = np.zeros((N_S, sustainData.getNumStages()))

    for s in range(N_S):
        index_s                         = cluster_assignment.reshape(cluster_assignment.shape[0], ) == (s + 1)
        temp_sustainData                = sustainData.reindex(index_s)

        temp_seq_init                   = _initialise_sequence(SuStaIn_inputs_dict,sustainData, rng)
        seq_init[s, :], _, _, _, _, _  = _perform_em(SuStaIn_inputs_dict,temp_sustainData, temp_seq_init, [1], rng)

    f_init                              = np.array([1.] * N_S) / float(N_S)

    # optimise the mixture of two models from the initialisation
    this_ml_sequence, \
    this_ml_f, \
    this_ml_likelihood, _, _, _      = _perform_em(SuStaIn_inputs_dict,sustainData, seq_init, f_init, rng)

    return this_ml_sequence, this_ml_f, this_ml_likelihood


def _find_ml_split(SuStaIn_inputs_dict, sustainData):
    # Fit a mixture of two models
    #
    #
    # OUTPUTS:
    # ml_sequence   - the ordering of the stages for each subtype
    # ml_f          - the most probable proportion of individuals belonging to each subtype
    # ml_likelihood - the likelihood of the most probable SuStaIn model

    N_S                                 = 2

    partial_iter                        = partial(_find_ml_split_iteration, SuStaIn_inputs_dict, sustainData)
    seed_sequences = np.random.SeedSequence(SuStaIn_inputs_dict['global_rng'].integers(1e10))
    pool_output_list                    = SuStaIn_inputs_dict['pool'].map(partial_iter, seed_sequences.spawn(SuStaIn_inputs_dict['N_startpoints']))

    if ~isinstance(pool_output_list, list):
        pool_output_list                = list(pool_output_list)

    ml_sequence_mat                     = np.zeros((N_S, sustainData.getNumStages(), SuStaIn_inputs_dict['N_startpoints']))
    ml_f_mat                            = np.zeros((N_S, SuStaIn_inputs_dict['N_startpoints']))
    ml_likelihood_mat                   = np.zeros((SuStaIn_inputs_dict['N_startpoints'], 1))

    for i in range(SuStaIn_inputs_dict['N_startpoints']):
        ml_sequence_mat[:, :, i]        = pool_output_list[i][0]
        ml_f_mat[:, i]                  = pool_output_list[i][1]
        ml_likelihood_mat[i]            = pool_output_list[i][2]

    ix                                  = [np.where(ml_likelihood_mat == max(ml_likelihood_mat))[0][0]] #ugly bit of code to get first index where likelihood is maximum

    ml_sequence                         = ml_sequence_mat[:, :, ix]
    ml_f                                = ml_f_mat[:, ix]
    ml_likelihood                       = ml_likelihood_mat[ix]

    return ml_sequence, ml_f, ml_likelihood, ml_sequence_mat, ml_f_mat, ml_likelihood_mat


def _estimate_ml_sustain_model_nplus1_clusters(SuStaIn_inputs_dict, sustainData, ml_sequence_prev, ml_f_prev):
    # Given the previous SuStaIn model, estimate the next model in the
    # hierarchy (i.e. number of subtypes goes from N to N+1)
    #
    #
    # OUTPUTS:
    # ml_sequence       - the ordering of the stages for each subtype for the next SuStaIn model in the hierarchy
    # ml_f              - the most probable proportion of individuals belonging to each subtype for the next SuStaIn model in the hierarchy
    # ml_likelihood     - the likelihood of the most probable SuStaIn model for the next SuStaIn model in the hierarchy

    N_S = len(ml_sequence_prev) + 1
    if N_S == 1:
        # If the number of subtypes is 1, fit a single linear z-score model
        print('Finding ML solution to 1 cluster problem')
        ml_sequence,        \
        ml_f,               \
        ml_likelihood,      \
        ml_sequence_mat,    \
        ml_f_mat,           \
        ml_likelihood_mat = _find_ml(SuStaIn_inputs_dict,sustainData)


    else:
        # If the number of subtypes is greater than 1, go through each subtype
        # in turn and try splitting into two subtypes
        _, _, _, p_sequence, _          = _calculate_likelihood(SuStaIn_inputs_dict,sustainData, ml_sequence_prev, ml_f_prev)

        ml_sequence_prev                = ml_sequence_prev.reshape(ml_sequence_prev.shape[0], ml_sequence_prev.shape[1])
        p_sequence                      = p_sequence.reshape(p_sequence.shape[0], N_S - 1)
        p_sequence_norm                 = p_sequence / np.tile(np.sum(p_sequence, 1).reshape(len(p_sequence), 1), (N_S - 1))

        # Assign individuals to a subtype (cluster) based on the previous model
        ml_cluster_subj                 = np.zeros((sustainData.getNumSamples(), 1))   #np.zeros((len(data_local), 1))
        for m in range(sustainData.getNumSamples()):                                   #range(len(data_local)):
            ix                          = np.argmax(p_sequence_norm[m, :]) + 1

            #TEMP: MATLAB comparison
            #ml_cluster_subj[m]          = ix*np.ceil(np.random.rand())
            ml_cluster_subj[m]          = ix  # FIXME: should check this always works, as it differs to the Matlab code, which treats ix as an array

        ml_likelihood                   = -np.inf
        for ix_cluster_split in range(N_S - 1):
            this_N_cluster              = sum(ml_cluster_subj == int(ix_cluster_split + 1))

            if this_N_cluster > 1:

                # Take the data from the individuals belonging to a particular
                # cluster and fit a two subtype model
                print('Splitting cluster', ix_cluster_split + 1, 'of', N_S - 1)
                ix_i                    = (ml_cluster_subj == int(ix_cluster_split + 1)).reshape(sustainData.getNumSamples(), )
                sustainData_i           = sustainData.reindex(ix_i)

                print(' + Resolving 2 cluster problem')
                this_ml_sequence_split, _, _, _, _, _ = _find_ml_split(SuStaIn_inputs_dict,sustainData_i)

                # Use the two subtype model combined with the other subtypes to
                # inititialise the fitting of the next SuStaIn model in the
                # hierarchy
                this_seq_init           = ml_sequence_prev.copy()  # have to copy or changes will be passed to ml_sequence_prev

                #replace the previous sequence with the first (row index zero) new sequence
                this_seq_init[ix_cluster_split] = (this_ml_sequence_split[0]).reshape(this_ml_sequence_split.shape[1])

                #add the second new sequence (row index one) to the stack of sequences, 
                #so that you now have N_S sequences instead of N_S-1
                this_seq_init           = np.hstack((this_seq_init.T, this_ml_sequence_split[1])).T
                
                #initialize fraction of subjects in each subtype to be uniform
                this_f_init             = np.array([1.] * N_S) / float(N_S)

                print(' + Finding ML solution from hierarchical initialisation')
                this_ml_sequence,       \
                this_ml_f,              \
                this_ml_likelihood,     \
                this_ml_sequence_mat,   \
                this_ml_f_mat,          \
                this_ml_likelihood_mat  = _find_ml_mixture(SuStaIn_inputs_dict,sustainData, this_seq_init, this_f_init)

                # Choose the most probable SuStaIn model from the different
                # possible SuStaIn models initialised by splitting each subtype
                # in turn
                # FIXME: these arrays have an unnecessary additional axis with size = N_startpoints - remove it further upstream
                if this_ml_likelihood[0] > ml_likelihood:
                    ml_likelihood       = this_ml_likelihood[0]
                    ml_sequence         = this_ml_sequence[:, :, 0]
                    ml_f                = this_ml_f[:, 0]
                    ml_likelihood_mat   = this_ml_likelihood_mat[0]
                    ml_sequence_mat     = this_ml_sequence_mat[:, :, 0]
                    ml_f_mat            = this_ml_f_mat[:, 0]
                print('- ML likelihood is', this_ml_likelihood[0])
            else:
                print(f'Cluster {ix_cluster_split + 1} of {N_S - 1} too small for subdivision')
        print(f'Overall ML likelihood is', ml_likelihood)

    return ml_sequence, ml_f, ml_likelihood, ml_sequence_mat, ml_f_mat, ml_likelihood_mat


def _optimise_mcmc_settings(SuStaIn_inputs_dict, sustainData, seq_init, f_init):

    # Optimise the perturbation size for the MCMC algorithm
    n_iterations_MCMC_optimisation      = int(1e4)  # FIXME: set externally

    n_passes_optimisation               = 3

    seq_sigma_currentpass               = 1
    f_sigma_currentpass                 = 0.01  # magic number

    N_S                                 = seq_init.shape[0]

    for i in range(n_passes_optimisation):

        _, _, _, samples_sequence_currentpass, samples_f_currentpass, _ = _perform_mcmc(SuStaIn_inputs_dict,   
                                                                                                sustainData,
                                                                                                 seq_init,
                                                                                                 f_init,
                                                                                                 n_iterations_MCMC_optimisation,
                                                                                                 seq_sigma_currentpass,
                                                                                                 f_sigma_currentpass)

        samples_position_currentpass    = np.zeros(samples_sequence_currentpass.shape)
        for s in range(N_S):
            for sample in range(n_iterations_MCMC_optimisation):
                temp_seq                        = samples_sequence_currentpass[s, :, sample]
                temp_inv                        = np.array([0] * samples_sequence_currentpass.shape[1])
                temp_inv[temp_seq.astype(int)]  = np.arange(samples_sequence_currentpass.shape[1])
                samples_position_currentpass[s, :, sample] = temp_inv

        seq_sigma_currentpass           = np.std(samples_position_currentpass, axis=2, ddof=1)  # np.std is different to Matlab std, which normalises to N-1 by default
        seq_sigma_currentpass[seq_sigma_currentpass < 0.01] = 0.01  # magic number

        f_sigma_currentpass             = np.std(samples_f_currentpass, axis=1, ddof=1)         # np.std is different to Matlab std, which normalises to N-1 by default

    seq_sigma_opt                       = seq_sigma_currentpass
    f_sigma_opt                         = f_sigma_currentpass

    return seq_sigma_opt, f_sigma_opt




def _estimate_uncertainty_sustain_model(SuStaIn_inputs_dict,sustainData, seq_init, f_init):
    # Estimate the uncertainty in the subtype progression patterns and
    # proportion of individuals belonging to the SuStaIn model
    #
    #
    # OUTPUTS:
    # ml_sequence       - the most probable ordering of the stages for each subtype found across MCMC samples
    # ml_f              - the most probable proportion of individuals belonging to each subtype found across MCMC samples
    # ml_likelihood     - the likelihood of the most probable SuStaIn model found across MCMC samples
    # samples_sequence  - samples of the ordering of the stages for each subtype obtained from MCMC sampling
    # samples_f         - samples of the proportion of individuals belonging to each subtype obtained from MCMC sampling
    # samples_likeilhood - samples of the likelihood of each SuStaIn model sampled by the MCMC sampling
        
    # Perform a few initial passes where the perturbation sizes of the MCMC uncertainty estimation are tuned
    seq_sigma_opt, f_sigma_opt          = _optimise_mcmc_settings(SuStaIn_inputs_dict,sustainData, seq_init, f_init)

    # Run the full MCMC algorithm to estimate the uncertainty
    ml_sequence,        \
    ml_f,               \
    ml_likelihood,      \
    samples_sequence,   \
    samples_f,          \
    samples_likelihood                  = _perform_mcmc(SuStaIn_inputs_dict,SuStaIn_inputs_dict['sustainData'], seq_init, f_init, SuStaIn_inputs_dict['N_iterations_MCMC'], seq_sigma_opt, f_sigma_opt)

    return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood

#%% Data Setup 
#   LB: This is where we actually run SuStaIn using the functions defined as above 
#       This section is where you define your data 


#Setup some initial variables 
plot=False
plot_format="png"


# =============================================================================
# REPLACE BELOW WITH YOUR OWN DATA 
# =============================================================================
#REPLACE THESE WITH YOUR OWN DATA 
roi_names=['BLAH1','BLAH2','BLAH3']
data = np.random.uniform(0, 5, (1000, 3))
dataset_name='EXAMPLE'
data_dir='A/PATH/TO/WHEREEVER/YOUR/DATA/IS'

SuStaInLabels = roi_names
Z_vals = np.array([[np.nan,np.nan,np.nan]]*len(roi_names))     # Z-scores for each biomarker
Z_max  = np.array([np.nan]*len(roi_names))           # maximum z-score

Z_vals[:,0] = 1
Z_vals[:,1] = 2
Z_vals[:,2] = 4
Z_max[:]=5


data=data[roi_names].values
Z_vals=Z_vals
Z_max=Z_max
biomarker_labels=SuStaInLabels
N_startpoints = 2
N_S_max = 5
N_iterations_MCMC = int(1e5)
output_folder = os.path.join(data_dir, 'data_preproc/Tau_SuStaIn_week8_modified/')
dataset_name=dataset_name
use_parallel_startpoints=False
seed=None
# =============================================================================

SuStaIn_inputs = ZscoreSustain(
                              data.values,
                              Z_vals,
                              Z_max,
                              SuStaInLabels,
                              N_startpoints,
                              N_S_max, 
                              N_iterations_MCMC, 
                              output_folder, 
                              dataset_name, 
                              False)


#LB: I'm a silly goose and I've kept sustainData throughout and added the dict, 
#       instead of replacing it with the dict and added a line getting the data in the code 
#           Too deep to change now... 
SuStaIn_inputs_dict = SuStaIn_inputs.__dict__
sustainData = SuStaIn_inputs_dict['sustainData']



#%% Running SuStaIn

#Setup some empty seqeunces 
ml_sequence_prev_EM                 = []
ml_f_prev_EM                        = []

#Output pickle directory
pickle_dir                          = os.path.join(SuStaIn_inputs_dict['output_folder'], 'pickle_files')

#Make pickle directory 
if not os.path.isdir(pickle_dir):
    os.mkdir(pickle_dir)
    
#Check if we've set plot to true 
if plot:
    fig0, ax0                           = plt.subplots()
    
#Loop through user specified subtypes
for s in range(SuStaIn_inputs_dict['N_S_max']):
    pickle_filename_s               = os.path.join(pickle_dir, SuStaIn_inputs_dict['dataset_name'] + '_subtype' + str(s) + '.pickle')
    pickle_filepath                 = Path(pickle_filename_s)
    if pickle_filepath.exists():
        print("Found pickle file: " + pickle_filename_s + ". Using pickled variables for " + str(s) + " subtype.")

        pickle_file                 = open(pickle_filename_s, 'rb')

        loaded_variables            = pickle.load(pickle_file)

        #self.stage_zscore           = loaded_variables["stage_zscore"]
        #self.stage_biomarker_index  = loaded_variables["stage_biomarker_index"]
        #self.N_S_max                = loaded_variables["N_S_max"]

        samples_likelihood          = loaded_variables["samples_likelihood"]
        samples_sequence            = loaded_variables["samples_sequence"]
        samples_f                   = loaded_variables["samples_f"]

        ml_sequence_EM              = loaded_variables["ml_sequence_EM"]
        ml_sequence_prev_EM         = loaded_variables["ml_sequence_prev_EM"]
        ml_f_EM                     = loaded_variables["ml_f_EM"]
        ml_f_prev_EM                = loaded_variables["ml_f_prev_EM"]

        pickle_file.close()
    else:
        print("Failed to find pickle file: " + pickle_filename_s + ". Running SuStaIn model for " + str(s) + " subtype.")
        
        #Get Sequence 
        ml_sequence_EM,     \
        ml_f_EM,            \
        ml_likelihood_EM,   \
        ml_sequence_mat_EM, \
        ml_f_mat_EM,        \
        ml_likelihood_mat_EM  =  _estimate_ml_sustain_model_nplus1_clusters(SuStaIn_inputs_dict,SuStaIn_inputs_dict['sustainData'], ml_sequence_prev_EM, ml_f_prev_EM) #self.__estimate_ml_sustain_model_nplus1_clusters(self.__data, ml_sequence_prev_EM, ml_f_prev_EM)
        
        #Output the sequences to particular varaibles 
        ml_sequence_prev_EM         = ml_sequence_EM
        ml_f_prev_EM                = ml_f_EM

        #Perform MCMC on the SuStaIn Sequence
        seq_init                    = ml_sequence_EM
        f_init                      = ml_f_EM

        ml_sequence,        \
        ml_f,               \
        ml_likelihood,      \
        samples_sequence,   \
        samples_f,          \
        samples_likelihood          = _estimate_uncertainty_sustain_model(SuStaIn_inputs_dict,SuStaIn_inputs_dict['sustainData'], seq_init, f_init)           #self.__estimate_uncertainty_sustain_model(self.__data, seq_init, f_init)

    # max like subtype and stage / subject
    N_samples                       = 1000
    ml_subtype,             \
    prob_ml_subtype,        \
    ml_stage,               \
    prob_ml_stage,          \
    prob_subtype,           \
    prob_stage,             \
    prob_subtype_stage               = subtype_and_stage_individuals(SuStaIn_inputs_dict,SuStaIn_inputs_dict['sustainData'], samples_sequence, samples_f, N_samples)   #self.subtype_and_stage_individuals(self.__data, samples_sequence, samples_f, N_samples)
    
    
    if not pickle_filepath.exists():

        if not os.path.exists(SuStaIn_inputs_dict['output_folder']):
            os.makedirs(SuStaIn_inputs_dict['output_folder'])

        save_variables                          = {}
        save_variables["samples_sequence"]      = samples_sequence
        save_variables["samples_f"]             = samples_f
        save_variables["samples_likelihood"]    = samples_likelihood

        save_variables["ml_subtype"]            = ml_subtype
        save_variables["prob_ml_subtype"]       = prob_ml_subtype
        save_variables["ml_stage"]              = ml_stage
        save_variables["prob_ml_stage"]         = prob_ml_stage
        save_variables["prob_subtype"]          = prob_subtype
        save_variables["prob_stage"]            = prob_stage
        save_variables["prob_subtype_stage"]    = prob_subtype_stage

        save_variables["ml_sequence_EM"]        = ml_sequence_EM
        save_variables["ml_sequence_prev_EM"]   = ml_sequence_prev_EM
        save_variables["ml_f_EM"]               = ml_f_EM
        save_variables["ml_f_prev_EM"]          = ml_f_prev_EM
        

        pickle_file                 = open(pickle_filename_s, 'wb')
        pickle_output               = pickle.dump(save_variables, pickle_file)
        pickle_file.close()

    n_samples                       = SuStaIn_inputs_dict['sustainData'].getNumSamples() #self.__data.shape[0]

    #order of subtypes displayed in positional variance diagrams plotted by _plot_sustain_model
    SuStaIn_inputs_dict['_plot_subtype_order']        = np.argsort(ml_f_EM)[::-1]
    #order of biomarkers in each subtypes' positional variance diagram
    SuStaIn_inputs_dict['_plot_biomarker_order']      = ml_sequence_EM[SuStaIn_inputs_dict['_plot_subtype_order'][0], :].astype(int)

    # plot results
    if plot:
        figs, ax = SuStaIn_inputs_dict['_plot_sustain_model'](
            samples_sequence=samples_sequence,
            samples_f=samples_f,
            n_samples=n_samples,
            biomarker_labels=SuStaIn_inputs_dict['biomarker_labels'],
            subtype_order=SuStaIn_inputs_dict['_plot_subtype_order'],
            biomarker_order=SuStaIn_inputs_dict['_plot_biomarker_order'],
            save_path=Path(SuStaIn_inputs_dict['output_folder']) / f"{SuStaIn_inputs_dict['dataset_name']}_subtype{s}_PVD.{plot_format}",
        )
        for fig in figs:
            fig.show()

        ax0.plot(range(SuStaIn_inputs_dict['N_iterations_MCMC']), samples_likelihood, label="Subtype " + str(s+1))

# save and show this figure after all subtypes have been calculcated
if plot:
    ax0.legend(loc='upper right')
    fig0.tight_layout()
    fig0.savefig(Path(SuStaIn_inputs_dict['output_folder']) / f"MCMC_likelihoods.{plot_format}", bbox_inches='tight')
    fig0.show()

#%% LB CODE: Plot positional variance of particular sequence run 
s = 0# 1 split = 2 subtypes

#(If you haven't been saving the sequence you don't need this code )
# get the sample sequences and f
pickle_filename_s = output_folder + '/pickle_files/' + dataset_name + '_subtype' + str(s) + '.pickle'
pk = pd.read_pickle(pickle_filename_s)
samples_sequence = pk["samples_sequence"]
samples_f = pk["samples_f"]
#(/)


#Plot 
plot_positional_var(samples_sequence=samples_sequence,
            samples_f=samples_f,
            n_samples=n_samples,
            Z_vals=SuStaIn_inputs_dict['Z_vals'])
