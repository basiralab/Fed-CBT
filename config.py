import numpy as np
import torch
import uuid


"""
   This repository contains the implementation of Fed-CBT algorithm, its training and testing procedure.
   After executing simulate_data.py and obtaining the simulated dataset, one needs to execute demo.py.
   PS: If one wants to set hyperparameters, they should go to config.py before executing demo.py
    Inputs:
        model of any device:        Deep Graph Normalizer (DGN). It can be found at model.py
        whole dataset: (N_Subjects, N_Nodes, N_Nodes, N_views)
            every fold's dataset: (N_Subjects//n_fold, N_Nodes, N_Nodes, N_views)
                every fold's test dataset:  ((N_Subjects//n_fold), N_Nodes, N_Nodes, N_views)
                every fold's train dataset: (N_Subjects-(N_Subjects//n_fold), N_Nodes, N_Nodes, N_views)
                    every device's train dataset in every fold: (N_Subjects-(N_Subjects//n_fold), N_Nodes, N_Nodes, N_views)
                        
    Outputs:
        for every fold
            for every client
                loss plot of every device individually in a png form
                loss plot of every device after the device is updated with global model in a png form
                saved model in a torch model form
                all connectomic brain templates (cbt) in a numpy form
                fused cbt in a numpy form
            final loss of all clients in a txt form
    In order to evaluate Fed-CBT 3-fold cross-validation strategy is used.
    ---------------------------------------------------------------------
    Hızır Can Bayram
"""

# HYPERPARAMETERS #
Dataset_name = 'ASD RH'
Setup_name = 'no_federation' # it's either no_federation, federation or temp_federation(temporary weighted)
early_stop_limit = 10 # tells if how many rounds a model doesn't improve, it's stopped to train
N_max_epochs = 500 #500
n_folds = 3 # cross validation fold number
number_of_samples = 3 # how many device we want to use for federated learning
numEpoch = 1 # how many round we want to train in an epoch
random_sample_size = 11 # 
lr = 0.00084

N_views = 6
N_Nodes = 35
early_stop = True
model_name = "DGN_TEST"
CONV1 = 36
CONV2 = 24
CONV3 = 5
# HYPERPARAMETERS #






N_Subjects = None
if 'ASD' in Dataset_name:
    N_Subjects = 155
else:
    N_Subjects = 186

temporal_weighting = None
if 'temp' in Setup_name:
    temporal_weighting = True
else:
    temporal_weighting = False

C_sgd = None # 1/3 # 0.91
if 'temp' in Setup_name:
    C_sgd = 1/3
else:
    C_sgd = 0.91

isFederated = None
if 'temp' in Setup_name:
    isFederated = True
elif 'no' in Setup_name:
    isFederated = False
else:
    isFederated = True
    
average_all = None
if 'temp' in Setup_name:
    average_all = True
elif 'no' in Setup_name:
    average_all = None
else:
    average_all = False
    

Path_input = 'inputs/' + Dataset_name + '/'
Path_output = 'output/' + Dataset_name + '/' + Setup_name + '/'
TEMP_FOLDER = "./temp"
model_id = str(uuid.uuid4())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PARAMS = {
        "N_ROIs": N_Nodes,
        "learning_rate" : lr,
        "n_attr": N_views,
        "Linear1" : {"in": N_views, "out": CONV1},
        "conv1": {"in" : 1, "out": CONV1},

        "Linear2" : {"in": N_views, "out": CONV1*CONV2},
        "conv2": {"in" : CONV1, "out": CONV2},

        "Linear3" : {"in": N_views, "out": CONV2*CONV3},
        "conv3": {"in" : CONV2, "out": CONV3} 
}



