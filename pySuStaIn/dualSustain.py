#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:19:25 2024

@author: lawrencebinding
"""

#%% Libraries 
import numpy as np
from scipy.stats import norm


#%% Setup Class 
class DualSuStaIn_data_generator():
    """ Dual SuStaIn Data Generator
        ---------------------------------------
        This works by generating an initial sequence which is applied to the X data. We then expect that this sequence is present 
        in the Y data but to a smaller degree (determined by hidden_preportion). e.g. if we set the hidden preprortion to be 0.25
        we expect that the X sequence is present in at a degree of 0.75 of the data and there is a small hidden signal of 0.25. 
        We then calculate a second sequence with the main sequence max value and zAbnormality multiplied by hidden preportion and
        add this to the Y data. This results in an X and Y data with the same max abnormality.
        
        Usage
        ---------------------------------------
        
        data_gen = DualSuStaIn_data_generator(n_subtypes, n_biomarkers, n_samples, zVal, zMax, hidden_preportion)
        
        x_data, x_data_noise, \
        y_data, y_data_noise, \
        ground_truth_stages_combined, \
        ground_truth_subtypes_combined, \
        zVals_combined, \
        ground_truth_sequences_combined, \
        ground_truth_fractions = data_gen.generate_data()
        
        Options  
        ---------------------------------------
        n_subtypes: Number of subtypes (e.g., 2) 
        n_biomarkers: Number of biomarkers (e.g., 10)
        n_samples: Number of samples (e.g., 1000)
        zVal: Threshold for abnormality (e.g., 2)
        zMax: Max value of clean data (e.g., 5)
        hidden_preportion: Preportion of main/hidden signal (e.g., 0.5        
        """
    def __init__(self, n_subtypes, n_biomarkers, n_samples, zVal, zMax, hidden_preportion=0.5):
        self.n_subtypes = n_subtypes
        self.n_biomarkers = n_biomarkers
        self.n_samples = n_samples        
        self.hidden_preportion = hidden_preportion
        self.z_vals = np.array([[zVal]] * n_biomarkers) 
        self.z_vals_hidden = np.array([[(zVal*hidden_preportion)]] * n_biomarkers) 
        self.z_max = np.array([zMax] * n_biomarkers) 
        self.z_max_hidden = np.array([(zMax*hidden_preportion)] * n_biomarkers)
                
        

    def generate_random_Zscore_sustain_model(self):
    
        B                                   = self.z_vals.shape[0]
        stage_zscore                        = np.array([y for x in self.z_vals.T for y in x])
        stage_zscore                        = stage_zscore.reshape(1, len(stage_zscore))
    
        IX_select                           = stage_zscore > 0
        stage_zscore                        = stage_zscore[IX_select]
        stage_zscore                        = stage_zscore.reshape(1, len(stage_zscore))
    
        num_zscores                         = self.z_vals.shape[1]
        IX_vals                             = np.array([[x for x in range(B)]] * num_zscores).T
        stage_biomarker_index               = np.array([y for x in IX_vals.T for y in x])
        stage_biomarker_index               = stage_biomarker_index.reshape(1, len(stage_biomarker_index))
        stage_biomarker_index               = stage_biomarker_index[IX_select]
        stage_biomarker_index               = stage_biomarker_index.reshape(1, len(stage_biomarker_index))
    
        N                                   = np.array(stage_zscore).shape[1]
        S                                   = np.zeros((self.n_subtypes, N))
        for s in range(self.n_subtypes):
            for i in range(N):
                IS_min_stage_zscore         = np.array([False] * N)
                possible_biomarkers         = np.unique(stage_biomarker_index)
    
                for j in range(len(possible_biomarkers)):
                    IS_unselected           = [False] * N
    
                    for k in set(range(N)) - set(S[s][:i]):
                        IS_unselected[k]    = True
    
                    this_biomarkers         = np.array([(np.array(stage_biomarker_index)[0] == possible_biomarkers[j]).astype(int) + (np.array(IS_unselected) == 1).astype(int)]) == 2
                    if not np.any(this_biomarkers):
                        this_min_stage_zscore = 0
                    else:
                        this_min_stage_zscore = min(stage_zscore[this_biomarkers])
                    if (this_min_stage_zscore):
                        temp                = ((this_biomarkers.astype(int) + (stage_zscore == this_min_stage_zscore).astype(int)) == 2).T
                        temp                = temp.reshape(len(temp), )
                        IS_min_stage_zscore[temp] = True
    
                events                      = np.array(range(N))
                possible_events             = np.array(events[IS_min_stage_zscore])
                this_index                  = np.ceil(np.random.rand() * ((len(possible_events)))) - 1
                S[s][i]                     = possible_events[int(this_index)]
    
        return S
    
    
    def generate_data_Zscore_sustain_x_y(self, subtypes, stages, gt_ordering):
        
        B                                   = self.z_vals.shape[0]
        stage_zscore                        = np.array([y for x in self.z_vals.T for y in x])
        stage_zscore                        = stage_zscore.reshape(1,len(stage_zscore))
        IX_select                           = stage_zscore>0
        stage_zscore                        = stage_zscore[IX_select]
        stage_zscore                        = stage_zscore.reshape(1,len(stage_zscore))
    
        num_zscores                         = self.z_vals.shape[1]
        IX_vals                             = np.array([[x for x in range(B)]] * num_zscores).T
        stage_biomarker_index               = np.array([y for x in IX_vals.T for y in x])
        stage_biomarker_index               = stage_biomarker_index.reshape(1,len(stage_biomarker_index))
        stage_biomarker_index               = stage_biomarker_index[IX_select]
        stage_biomarker_index               = stage_biomarker_index.reshape(1,len(stage_biomarker_index))
    
        min_biomarker_zscore                = [0]*B
        std_biomarker_zscore                = [1]*B
    
        N                                   = stage_biomarker_index.shape[1]
    
        possible_biomarkers                 = np.unique(stage_biomarker_index)
        stage_value                         = np.zeros((B,N+2,self.n_subtypes))
        M                                   = stages.shape[0]
    
        # Initialize an empty NumPy array to hold the simulated data
        x_data = np.zeros((M, B))
        y_data = np.zeros((M, B))
    
        for s in range(self.n_subtypes):
            S                               = gt_ordering[s,:]
            S_inv                           = np.array([0]*N)
            S_inv[S.astype(int)]            = np.arange(N)
            for i in range(B):
                b                           = possible_biomarkers[i]
                event_location              = np.concatenate([[0], S_inv[(stage_biomarker_index == b)[0]], [N]])
                event_value                 = np.concatenate([[min_biomarker_zscore[i]], stage_zscore[stage_biomarker_index == b], [self.z_max[i]]])
    
                for j in range(len(event_location)-1):
    
                    if j == 0: # FIXME: nasty hack to get Matlab indexing to match up - necessary here because indices are used for linspace limits
                        index               = np.arange(event_location[j],event_location[j+1]+2)
                        stage_value[i,index,s] = np.linspace(event_value[j],event_value[j+1],event_location[j+1]-event_location[j]+2)
                    else:
                        index               = np.arange(event_location[j] + 1, event_location[j + 1] + 2)
                        stage_value[i,index,s] = np.linspace(event_value[j],event_value[j+1],event_location[j+1]-event_location[j]+1)
    
        M                                   = stages.shape[0]
        
        #Add the stage to existing y_data values
        #
        for m in range(M):
            x_data[m,:] = x_data[m,:] + stage_value[:,int(stages[m]),subtypes[m]]
            y_data[m,:] = y_data[m,:] + (stage_value[:,int(stages[m]),subtypes[m]] * abs(1-self.hidden_preportion))
    
    
        return x_data, y_data, stage_value
    
    def add_noise(self, x_data, y_data):
        y_data_noise = y_data.copy()
        x_data_noise = x_data.copy()
    
        #for i in range(N):
            # Randomly select a 'mu' and 'std' from the list
            #selected_index = np.random.randint(len(column_CT_distribution_HC))
            # Generate x_data from the sample distribution 
        generate_noise = norm.ppf(np.random.rand(x_data.shape[1],x_data.shape[0]).T)*np.tile([1]*x_data.shape[1],(x_data.shape[0],1))
        x_data_noise = x_data + generate_noise #np.random.normal(column_Tau_distribution_HC[selected_index]['mu'], column_Tau_distribution_HC[selected_index]['std'], M)
            # Generate x_data from the sample distribution 
        y_data_noise = y_data + generate_noise #+ np.random.normal(column_CT_distribution_HC[selected_index]['mu'], column_CT_distribution_HC[selected_index]['std'], M)
        return x_data_noise, y_data_noise

    def generate_data_Zscore_sustain_y_hidden_sequence(self, y_data, subtypes, stages, gt_ordering):
        B                                   = self.z_vals_hidden.shape[0]
        stage_zscore                        = np.array([y for x in self.z_vals.T for y in x])
        stage_zscore                        = stage_zscore.reshape(1,len(stage_zscore))
        IX_select                           = stage_zscore>0
        stage_zscore                        = stage_zscore[IX_select]
        stage_zscore                        = stage_zscore.reshape(1,len(stage_zscore))
    
        num_zscores                         = self.z_vals_hidden.shape[1]
        IX_vals                             = np.array([[x for x in range(B)]] * num_zscores).T
        stage_biomarker_index               = np.array([y for x in IX_vals.T for y in x])
        stage_biomarker_index               = stage_biomarker_index.reshape(1,len(stage_biomarker_index))
        stage_biomarker_index               = stage_biomarker_index[IX_select]
        stage_biomarker_index               = stage_biomarker_index.reshape(1,len(stage_biomarker_index))
    
        min_biomarker_zscore                = [0]*B
        max_biomarker_zscore                = self.z_max_hidden
        std_biomarker_zscore                = [1]*B
    
        N                                   = stage_biomarker_index.shape[1]
    
        possible_biomarkers                 = np.unique(stage_biomarker_index)
        stage_value                         = np.zeros((B,N+2,self.n_subtypes))
        M                                   = stages.shape[0]
    
        # Initialize an empty NumPy array to hold the simulated data
    
        for s in range(self.n_subtypes):
            S                               = gt_ordering[s,:]
            S_inv                           = np.array([0]*N)
            S_inv[S.astype(int)]            = np.arange(N)
            for i in range(B):
                b                           = possible_biomarkers[i]
                event_location              = np.concatenate([[0], S_inv[(stage_biomarker_index == b)[0]], [N]])
                event_value                 = np.concatenate([[min_biomarker_zscore[i]], stage_zscore[stage_biomarker_index == b], [max_biomarker_zscore[i]]])
    
                for j in range(len(event_location)-1):
    
                    if j == 0: # FIXME: nasty hack to get Matlab indexing to match up - necessary here because indices are used for linspace limits
                        index               = np.arange(event_location[j],event_location[j+1]+2)
                        stage_value[i,index,s] = np.linspace(event_value[j],event_value[j+1],event_location[j+1]-event_location[j]+2)
                    else:
                        index               = np.arange(event_location[j] + 1, event_location[j + 1] + 2)
                        stage_value[i,index,s] = np.linspace(event_value[j],event_value[j+1],event_location[j+1]-event_location[j]+1)
    
        M                                   = stages.shape[0]
        
        #Add the stage to existing y_data values
        for m in range(M):
            y_data[m,:] = y_data[m,:] + stage_value[:,int(stages[m]),subtypes[m]]
        
        return y_data, stage_value
    
    
    def generate_data(self):
        #Randomly generate ground truth 
        N_S_ground_truth = self.n_subtypes#random.randint(1, 5)
        #Number of subjects
        M = self.n_samples
        #Number of biomarkers
        #N = random.randint(5, 128)
        N = self.n_biomarkers
        #generate ground truth fractions 
        random_values = np.random.rand(N_S_ground_truth)
        ground_truth_fractions = (random_values) / random_values.sum()
        ground_truth_fractions[-1] += 1 - ground_truth_fractions.sum()
    
        #   Generate data for main Tau sequence 
        # ---------------------------------------
        #Generate some ground truth sequence for each subtype 
        ground_truth_sequences = self.generate_random_Zscore_sustain_model()
        
        # randomly generate the ground truth subtype and stage assignment for every one of the M subjects
        ground_truth_subtypes   = np.random.choice(range(N_S_ground_truth), M, replace=True, p=ground_truth_fractions).astype(int)
        # Get the number of stages
        N_stages                = np.sum(self.z_vals > 0) + 1

        #Calcualt the proportion of controls and patients        
        control_proportion = 0.10 # (Can vary by replacing 0.10 with random.uniform(0.01, 0.3))
        patient_proportion = 0.90 #Varied: 1 - control_proportion
    
        # #Here we make sure that there are at least 5 number of patients at each stage 
        # initial_assignments = []
        # for stage in range(1, N_stages + 1):
        #     initial_assignments.extend([stage] * 5)
        
        # remaining_subjects = int(np.round(M * patient_proportion)) - len(initial_assignments)
        
        # if remaining_subjects > 0:
        #     additional_assignments = np.random.choice(range(1, N_stages + 1), remaining_subjects)
        #     final_assignments = initial_assignments + additional_assignments.tolist()
        # else:
        #     final_assignments = initial_assignments[:int(np.round(M * patient_proportion))]  # In case total_subjects < initial_assignments length
        
        # np.random.shuffle(final_assignments)
    
        # # Convert to numpy array with shape (total_subjects, 1)
        # ground_truth_stages_other = np.array(final_assignments).reshape((int(np.round(M * patient_proportion)), 1))
    
        # #Controls are assigned to stage zero of the disease progression
        # ground_truth_stages_control = np.zeros((int(np.round(M * control_proportion)), 1))
        
        # #Combine patients and controls         
        # ground_truth_stages         = np.vstack((ground_truth_stages_control, ground_truth_stages_other)).astype(int)
        
        gt_stages_control = np.zeros((int(M*control_proportion),1))
        ground_truth_stages = np.concatenate((gt_stages_control, np.ceil(np.random.rand(M-int(M*control_proportion),1)*N_stages)), axis=0)

        
        #Generate X data and y data
        x_data, y_data, stage_value = self.generate_data_Zscore_sustain_x_y(ground_truth_subtypes,
                                                                         ground_truth_stages,
                                                                         ground_truth_sequences)            
        
        # Hidden Sequence 
        ground_truth_sequences_hidden = self.generate_random_Zscore_sustain_model()
        ground_truth_subtypes_hidden  = np.random.choice(range(N_S_ground_truth), M, replace=True, p=ground_truth_fractions).astype(int)
        N_stages = np.sum(self.z_vals_hidden > 0) + 1
        initial_assignments = []
        for stage in range(1, N_stages + 1):
            initial_assignments.extend([stage] * 5)
        remaining_subjects = int(np.round(M * patient_proportion)) - len(initial_assignments)
        if remaining_subjects > 0:
            additional_assignments = np.random.choice(range(1, N_stages + 1), remaining_subjects)
            final_assignments = initial_assignments + additional_assignments.tolist()
        else:
            final_assignments = initial_assignments[:int(np.round(M * patient_proportion))]  # In case total_subjects < initial_assignments length
    
        np.random.shuffle(final_assignments)
    
        #ground_truth_stages_other_hidden = np.array(final_assignments).reshape((int(np.round(M * patient_proportion)), 1))
        #ground_truth_stages_control_hidden = np.zeros((int(np.round(M * control_proportion)), 1))
        #ground_truth_stages_hidden         = np.vstack((ground_truth_stages_control_hidden, ground_truth_stages_other_hidden)).astype(int)
        
        y_data_hidden, stage_value_hidden = self.generate_data_Zscore_sustain_y_hidden_sequence(y_data, ground_truth_subtypes_hidden,
                                                                         ground_truth_stages,
                                                                         ground_truth_sequences_hidden)         
        #Add noise to both of these 
        x_data_noise, y_data_noise = self.add_noise(x_data, y_data_hidden)
        
        #Combine data for output 
        ground_truth_stages_combined = np.vstack((ground_truth_stages.squeeze(), ground_truth_stages.squeeze()))
        ground_truth_subtypes_combined = np.vstack((ground_truth_subtypes, ground_truth_subtypes_hidden))
        zVals_combined =     np.vstack((self.z_vals.squeeze(), self.z_vals_hidden.squeeze()))
        ground_truth_sequences_combined =     np.stack((ground_truth_sequences, ground_truth_sequences_hidden))
        ground_truth_stageVal_combined =     np.stack((stage_value.squeeze(), stage_value_hidden.squeeze()))

        return x_data, x_data_noise, y_data_hidden, y_data_noise, ground_truth_stages_combined, ground_truth_subtypes_combined, zVals_combined, ground_truth_sequences_combined, ground_truth_fractions, ground_truth_stageVal_combined
