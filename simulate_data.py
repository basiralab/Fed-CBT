import pickle
import numpy as np
import os
from config import *
from helper import antiVectorize, simulate_dataset



if not os.path.exists('inputs/'):
    os.mkdir('inputs')
if not os.path.exists('inputs/' + Dataset_name):
    os.mkdir('inputs/' + Dataset_name)
dataset = np.random.randint(0,1,(N_Subjects,N_Nodes,N_Nodes,N_views))
np.save('inputs/' + Dataset_name + '/' + Dataset_name + '.npy', dataset)
    
for i in range(n_folds):
    if not os.path.exists('inputs/' + Dataset_name + '/fold' + str(i)):
        os.mkdir('inputs/' + Dataset_name + '/fold' + str(i))

    #test_data = np.random.randint(0,1,(N_Subjects//n_folds,N_Nodes,N_Nodes,N_views))
    #train_data = np.random.randint(0,1,(N_Subjects-(N_Subjects//n_folds),N_Nodes,N_Nodes,N_views))
    test_data = simulate_dataset(N_Subjects//n_folds, N_Nodes, N_views)
    train_data = simulate_dataset(N_Subjects-(N_Subjects//n_folds), N_Nodes, N_views)
    
    np.save('inputs/' + Dataset_name + '/fold' + str(i) + '/' + 'fold' + str(i) + '_test_data.npy', test_data)
    np.save('inputs/' + Dataset_name + '/fold' + str(i) + '/' + 'fold' + str(i) + '_train_data.npy', train_data)
    
    the_dict = dict()
    
    for k in range(number_of_samples):
        the_dict['x_train' + str(k)] = train_data[k * len(train_data) // number_of_samples:(k+1) * len(train_data) // number_of_samples, :,:,:]
        #np.random.randint(0,1,((N_Subjects-(N_Subjects//n_folds)) // number_of_samples,N_Nodes,N_Nodes,N_views))
        
    with open('inputs/' + Dataset_name + '/fold' + str(i) + '/client_data_fold_' + str(i) + '.pkl', 'wb') as handle:
        pickle.dump(the_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)