from config import *
import torch
import numpy as np
import pickle
import random
from model import DGN
from helper import *

def create_model_optimizer_criterion_dict(number_of_samples):
    '''
    This function creates dictionaries that contain optimize function, loss function, gnn model for every device introduced
    to the federated pipeline.
    '''
    model_dict = dict()
    optimizer_dict= dict()
    criterion_dict = dict()
    
    for i in range(number_of_samples):
        model_name="model"+str(i)
        model_info=DGN(MODEL_PARAMS).to(device) 
        model_dict.update({model_name : model_info })
        
        optimizer_name="optimizer"+str(i)
        optimizer_info = torch.optim.AdamW(model_info.parameters(), lr=MODEL_PARAMS["learning_rate"], weight_decay= 0.00)
        optimizer_dict.update({optimizer_name : optimizer_info })
        
        criterion_name = "criterion"+str(i)
        criterion_info = []
        criterion_dict.update({criterion_name : criterion_info})
        
    return model_dict, optimizer_dict, criterion_dict 



def get_averaged_weights(model_dict, name_of_models, number_of_samples, clients_with_access, \
                         average_all=True, last_updated_dict=None, current_epoch=-1):
    '''
    This function averages model weights after a designated number of round so that we can have the weights of the global model
    that takes full advantage of introduced devices in the federated pipeline.
    '''
    conv1_nn_mean_weight = torch.zeros(size=model_dict[name_of_models[0]].conv1.nn[0].weight.data.shape).to(device)
    conv1_nn_mean_bias = torch.zeros(size=model_dict[name_of_models[0]].conv1.nn[0].bias.data.shape).to(device)
    conv1_bias = torch.zeros(size=model_dict[name_of_models[0]].conv1.bias.data.shape).to(device)
    conv1_root = torch.zeros(size=model_dict[name_of_models[0]].conv1.root.data.shape).to(device)
    
    conv2_nn_mean_weight = torch.zeros(size=model_dict[name_of_models[0]].conv2.nn[0].weight.data.shape).to(device)
    conv2_nn_mean_bias = torch.zeros(size=model_dict[name_of_models[0]].conv2.nn[0].bias.data.shape).to(device)
    conv2_bias = torch.zeros(size=model_dict[name_of_models[0]].conv2.bias.data.shape).to(device)
    conv2_root = torch.zeros(size=model_dict[name_of_models[0]].conv2.root.data.shape).to(device)    
    
    conv3_nn_mean_weight = torch.zeros(size=model_dict[name_of_models[0]].conv3.nn[0].weight.data.shape).to(device)
    conv3_nn_mean_bias = torch.zeros(size=model_dict[name_of_models[0]].conv3.nn[0].bias.data.shape).to(device)
    conv3_bias = torch.zeros(size=model_dict[name_of_models[0]].conv3.bias.data.shape).to(device)
    conv3_root = torch.zeros(size=model_dict[name_of_models[0]].conv3.root.data.shape).to(device)
    
    cls = None # 
    if average_all:
        cls = range(number_of_samples)
    else:
        cls = clients_with_access

    with torch.no_grad():
        def getWeight_i(i):
            return ((np.e / 2) ** (- (current_epoch - last_updated_dict['client'+str(i)])))
        for i in cls: # cls
            if temporal_weighting == True:
                all_weights = getWeight_i(0) + getWeight_i(1) + getWeight_i(2)
                client_weight = getWeight_i(i) / all_weights
            else:
                client_weight = 1/len(cls)

            conv1_nn_mean_weight += (client_weight * model_dict[name_of_models[i]].conv1.nn[0].weight.data.clone())
            conv1_nn_mean_bias += (client_weight * model_dict[name_of_models[i]].conv1.nn[0].bias.data.clone())
            conv1_bias += (client_weight * model_dict[name_of_models[i]].conv1.bias.data.clone())
            conv1_root += (client_weight * model_dict[name_of_models[i]].conv1.root.data.clone())
            
            conv2_nn_mean_weight += (client_weight * model_dict[name_of_models[i]].conv2.nn[0].weight.data.clone())
            conv2_nn_mean_bias += (client_weight * model_dict[name_of_models[i]].conv2.nn[0].bias.data.clone())
            conv2_bias += (client_weight * model_dict[name_of_models[i]].conv2.bias.data.clone())
            conv2_root += (client_weight * model_dict[name_of_models[i]].conv2.root.data.clone())
            
            conv3_nn_mean_weight += (client_weight * model_dict[name_of_models[i]].conv3.nn[0].weight.data.clone())
            conv3_nn_mean_bias += (client_weight * model_dict[name_of_models[i]].conv3.nn[0].bias.data.clone())
            conv3_bias += (client_weight * model_dict[name_of_models[i]].conv3.bias.data.clone())
            conv3_root += (client_weight * model_dict[name_of_models[i]].conv3.root.data.clone())

    return conv1_nn_mean_weight, conv1_nn_mean_bias, conv1_bias, conv1_root, \
            conv2_nn_mean_weight, conv2_nn_mean_bias, conv2_bias, conv2_root, \
            conv3_nn_mean_weight, conv3_nn_mean_bias, conv3_bias, conv3_root
 
    
    
    
def set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, model_dict, name_of_models, \
                                                                     number_of_samples, clients_with_access, \
                                                                    last_updated_dict, current_epoch):
    '''
    This function takes combined weights for global model and assigns them to the global model.
    '''
    conv1_nn_mean_weight, conv1_nn_mean_bias, conv1_bias, conv1_root, \
    conv2_nn_mean_weight, conv2_nn_mean_bias, conv2_bias, conv2_root, \
    conv3_nn_mean_weight, conv3_nn_mean_bias, conv3_bias, conv3_root = get_averaged_weights(model_dict, \
                                                    name_of_models, number_of_samples, clients_with_access, \
                                                    average_all, last_updated_dict, current_epoch)
    
    with torch.no_grad():
        main_model.conv1.nn[0].weight.data = conv1_nn_mean_weight.clone()
        main_model.conv1.nn[0].bias.data = conv1_nn_mean_bias.clone()
        main_model.conv1.bias.data = conv1_bias.clone()
        main_model.conv1.root.data = conv1_root.clone()
        
        main_model.conv2.nn[0].weight.data = conv2_nn_mean_weight.clone()
        main_model.conv2.nn[0].bias.data = conv2_nn_mean_bias.clone()
        main_model.conv2.bias.data = conv2_bias.clone()
        main_model.conv2.root.data = conv2_root.clone()
        
        main_model.conv3.nn[0].weight.data = conv3_nn_mean_weight.clone()
        main_model.conv3.nn[0].bias.data = conv3_nn_mean_bias.clone()
        main_model.conv3.bias.data = conv3_bias.clone()
        main_model.conv3.root.data = conv3_root.clone()

    return main_model



def send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, name_of_models, \
                                                   number_of_samples, clients_with_access):
    '''
    This function updates clients with the newly global model so that all the devices can take advantage of the common
    information.
    '''
    with torch.no_grad():
        for i in range(number_of_samples): # clients_with_access 

            model_dict[name_of_models[i]].conv1.nn[0].weight.data = main_model.conv1.nn[0].weight.data.clone()
            model_dict[name_of_models[i]].conv1.nn[0].bias.data = main_model.conv1.nn[0].bias.data.clone()
            model_dict[name_of_models[i]].conv1.bias.data = main_model.conv1.bias.data.clone() 
            model_dict[name_of_models[i]].conv1.root.data = main_model.conv1.root.data.clone() 

            model_dict[name_of_models[i]].conv2.nn[0].weight.data = main_model.conv2.nn[0].weight.data.clone()
            model_dict[name_of_models[i]].conv2.nn[0].bias.data = main_model.conv2.nn[0].bias.data.clone()
            model_dict[name_of_models[i]].conv2.bias.data = main_model.conv2.bias.data.clone() 
            model_dict[name_of_models[i]].conv2.root.data = main_model.conv2.root.data.clone() 
            
            model_dict[name_of_models[i]].conv3.nn[0].weight.data = main_model.conv3.nn[0].weight.data.clone()
            model_dict[name_of_models[i]].conv3.nn[0].bias.data = main_model.conv3.nn[0].bias.data.clone()
            model_dict[name_of_models[i]].conv3.bias.data = main_model.conv3.bias.data.clone() 
            model_dict[name_of_models[i]].conv3.root.data = main_model.conv3.root.data.clone() 
            
    return model_dict



def train_kfold(Path_input, loss_table_list, number_of_samples):
    '''
    This function is the main loop. Executes all the functions mentioned above and federated pipeline.
    '''
    for i in range(n_folds):
        
        loss_table = dict()
        for m in range(number_of_samples):
            loss_table.update({'local_loss_global_data_'+str(m): [] })
            loss_table.update({'combining_local_loss_global_data_'+str(m): [] })
        
        client_access = [True] * number_of_samples
        
        torch.cuda.empty_cache() 
        
        all_train_data = np.load('{}fold{}/fold{}_train_data.npy'.format(Path_input, i, i))  
        all_test_data = np.load('{}fold{}/fold{}_test_data.npy'.format(Path_input, i, i)) 
        all_train_data, all_test_data = map(torch.tensor, (all_train_data, all_test_data))
        
        a_file = open("{}fold{}/client_data_fold_{}.pkl".format(Path_input, i, i), "rb")
        x_train_dict = pickle.load(a_file)
        a_file.close()
        # dictionaries are ready to be involved into the federated pipeline.
        model_dict, optimizer_dict, criterion_dict = create_model_optimizer_criterion_dict(number_of_samples)

        name_of_x_train_sets = list(x_train_dict.keys())
        name_of_models = list(model_dict.keys())
        name_of_optimizers = list(optimizer_dict.keys())
        name_of_criterions = list(criterion_dict.keys())
        # main model creation
        main_model = DGN(MODEL_PARAMS)
        main_model = main_model.to(device)
        main_optimizer = torch.optim.AdamW(main_model.parameters(), lr=MODEL_PARAMS["learning_rate"], weight_decay= 0.00)
       
        #train_casted = [d.to(device) for d in cast_data(all_train_data)]
        test_casted = [d.to(device) for d in cast_data(all_test_data)]
        # all models updated with the initial global model.
        send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, name_of_models, \
                                                       number_of_samples, list(range(0, number_of_samples)))
  
        for ith_model in range(number_of_samples): # for every device in the federated pipeline, compute initial validation values
            train_data_ith_model = x_train_dict[name_of_x_train_sets[ith_model]]
            train_mean_ith_model = np.mean(train_data_ith_model, axis=(0,1,2))
            train_casted_ith_model = [d.to(device) for d in cast_data(train_data_ith_model)]       
            model_ith = model_dict[name_of_models[ith_model]].to(device)
            kth_model_rep_loss = validation(model_ith, train_casted_ith_model, test_casted)
            print(ith_model, 'th sample loss:', "{:.4f}".format(kth_model_rep_loss), end=' ')

        last_updated_dict = {'client0': 0, 'client1': 0, 'client2': 0}
        
        for n in range(N_max_epochs): # every epoch's loop
            client_indices = [i for i, x in enumerate(client_access) if x]
            clients_with_access = None
            if len(client_indices) > 1:
                if int(len(client_indices)*C_sgd) < 1:
                    clients_with_access = sorted(random.sample(client_indices, 1))
                else:
                    clients_with_access = sorted(random.sample(client_indices, int(len(client_indices)*C_sgd)))
            elif len(client_indices) == 1:
                clients_with_access = sorted(random.sample(client_indices, 1))
            else:
                break
                
            for gga in clients_with_access:
                last_updated_dict['client'+str(gga)] = n+1
            # models learning
            start_train_end_node_process(model_dict, criterion_dict, optimizer_dict, x_train_dict, \
                                         name_of_x_train_sets, name_of_models, name_of_criterions, \
                                         name_of_optimizers, loss_table, test_casted, n, client_access, \
                                         clients_with_access)
            

            for ith_model in range(number_of_samples): # loop of every device of every epoch 
                train_data_ith_model = x_train_dict[name_of_x_train_sets[ith_model]] # data that the device has
                train_mean_ith_model = np.mean(train_data_ith_model, axis=(0,1,2))
                train_casted_ith_model = [d.to(device) for d in cast_data(train_data_ith_model)]       
                model_ith = model_dict[name_of_models[ith_model]].to(device)
                kth_model_rep_loss = validation(model_ith, train_casted_ith_model, test_casted)
                torch.save(model_ith.state_dict(), TEMP_FOLDER + "/weight_" + model_id + "_" + str(kth_model_rep_loss)[:5]  + "_" + str(ith_model) + ".model")
                loss_table['local_loss_global_data_'+str(ith_model)].append(kth_model_rep_loss)

                if not isFederated: # if no federation during the experiment, simply train every model with its own dataset
                    if len(loss_table['local_loss_global_data_'+str(ith_model)]) > early_stop_limit and early_stop:
                        last_6 = loss_table['local_loss_global_data_'+str(ith_model)][-early_stop_limit:]
                        if(all(last_6[i] <= last_6[i + 1] for i in range(early_stop_limit-1))):
                            client_access[ith_model] = False

            if isFederated:
                main_model = set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, \
                                      model_dict, name_of_models, number_of_samples, \
                                        clients_with_access, last_updated_dict, n+1)           
                
                send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, name_of_models, \
                                                               number_of_samples, clients_with_access)
            
                for ith_model in range(number_of_samples):
                    train_data_ith_model = x_train_dict[name_of_x_train_sets[ith_model]]
                    train_mean_ith_model = np.mean(train_data_ith_model, axis=(0,1,2))
                    train_casted_ith_model = [d.to(device) for d in cast_data(train_data_ith_model)]       
                    model_ith = model_dict[name_of_models[ith_model]].to(device)
                    kth_model_rep_loss = validation(model_ith, train_casted_ith_model, test_casted)
                    torch.save(model_ith.state_dict(), TEMP_FOLDER + "/weight_" + model_id + "_" + str(kth_model_rep_loss)[:5]  + "_" + str(ith_model) + ".model")
                    loss_table['combining_local_loss_global_data_'+str(ith_model)].append(kth_model_rep_loss)
                    if len(loss_table['combining_local_loss_global_data_'+str(ith_model)]) > early_stop_limit and early_stop:
                        last_6 = loss_table['combining_local_loss_global_data_'+str(ith_model)][-early_stop_limit:]
                        if(all(last_6[i] <= last_6[i + 1] for i in range(early_stop_limit-1))):
                            client_access[ith_model] = False
        loss_table_list.append(loss_table)
        
        for k in range(number_of_samples): # saving models, logging operations etc for every deviced introduced to the federated pipeline.
            cur_model = DGN(MODEL_PARAMS).to(device)
            restore = None
            if isFederated:
                restore = TEMP_FOLDER + "/weight_" + model_id + "_" + str(min(loss_table['combining_local_loss_global_data_'+str(k)]))[:5] + "_" + str(k) + ".model"
            else:
                restore = TEMP_FOLDER + "/weight_" + model_id + "_" + str(min(loss_table['local_loss_global_data_'+str(k)]))[:5] + "_" + str(k) + ".model"            
            cur_model.load_state_dict(torch.load(restore))
            torch.save(cur_model.state_dict(), '{}fold{}_cli_{}_{}.model'.format(Path_output, i, i, k, Setup_name))
            
            kth_train_data = x_train_dict[name_of_x_train_sets[k]]
            kth_train_casted = [d.to(device) for d in cast_data(kth_train_data)]  
            
            cbt = generate_cbt_median(cur_model, kth_train_casted)
            rep_loss = mean_frobenious_distance(cbt, test_casted)
            cbt = cbt.cpu().numpy()            
            print(k, 'th model', 'final loss based on kth_train_casted-cbt:', rep_loss)
            np.save('{}fold{}_cli_{}_{}_cbt'.format(Path_output, i, i, k, Setup_name), cbt)
            all_cbts = generate_subject_biased_cbts(cur_model, kth_train_casted)
            np.save('{}fold{}_cli_{}_{}_all_cbts'.format(Path_output, i, i, k, Setup_name), all_cbts)
            with open('{}fold{}_loss'.format(Path_output, i, i), "a+") as file:
                file.write('Loss for {} Client {} is {}\n'.format(Setup_name, k, rep_loss))
        print('------------------------------End of the fold------------------------------')
        
        
        
def train(model, train_casted, targets, losses, optimizer, loss_weightes):
    '''
    Every model is trained with its own dataset in this function.
    '''
    model.train()
    losses = [] 
    for data in train_casted:
        cbt = model(data)
        views_sampled = random.sample(targets, random_sample_size) 
        sampled_targets = torch.cat(views_sampled, axis = 2).permute((2,1,0))
        expanded_cbt = cbt.expand((sampled_targets.shape[0],MODEL_PARAMS["N_ROIs"],MODEL_PARAMS["N_ROIs"]))
        diff = torch.abs(expanded_cbt - sampled_targets) #Absolute difference
        sum_of_all = torch.mul(diff, diff).sum(axis = (1,2)) #Sum of squares
        l = torch.sqrt(sum_of_all)  #Square root of the sum
        losses.append((l * loss_weightes[:random_sample_size * MODEL_PARAMS["n_attr"]]).sum()) 
    
    optimizer.zero_grad()
    loss = torch.mean(torch.stack(losses))
    loss.backward()
    optimizer.step()            
    
    
    
def validation(model, train_casted, test_casted):
    '''
    Model is tested with frobenious distance in this function so that we can benchmark.
    '''
    model.eval()
    cbt = generate_cbt_median(model, train_casted) # (35x35) : shape of cbt
    rep_loss = mean_frobenious_distance(cbt, test_casted) 
    rep_loss = float(rep_loss)
    return rep_loss



def start_train_end_node_process(model_dict, criterion_dict, optimizer_dict,\
                                x_train_dict, name_of_x_train_sets, name_of_models, name_of_criterions, \
                                 name_of_optimizers, loss_table, test_casted, nth_iter, client_access, \
                                 clients_with_access):
    for k in clients_with_access: # clients allowed to be trained in the current round, are trained(designated clients if temporary weighted strategy, all clients otherwise
        train_data = x_train_dict[name_of_x_train_sets[k]]
        train_mean = np.mean(train_data, axis=(0,1,2))
        train_std =   np.std(train_data, axis=(0,1,2))
        train_casted = [d.to(device) for d in cast_data(train_data)]       
        loss_weightes = torch.tensor(np.array(list((1 / train_mean) / np.max(1 / train_mean))*len(train_data)), dtype = torch.float32)
        loss_weightes = loss_weightes.to(device)
             
        model = model_dict[name_of_models[k]].to(device)
        loss = criterion_dict[name_of_criterions[k]]
        optimizer = optimizer_dict[name_of_optimizers[k]]

        targets = [torch.tensor(tensor, dtype = torch.float32).to(device) for tensor in train_data] 

        kth_model_rep_loss = 0
        for epoch in range(numEpoch):
            train(model, train_casted, targets, loss, optimizer, loss_weightes)


            
            
if __name__ == "__main__":
    if not os.path.exists('output/'):
        os.mkdir('output')
    if not os.path.exists('output/' + Dataset_name):
        os.mkdir('output/' + Dataset_name)
    if not os.path.exists('output/' + Dataset_name + '/' + Setup_name):
        os.mkdir('output/' + Dataset_name + '/' + Setup_name)


    loss_table_list = []
    train_kfold(Path_input, loss_table_list, number_of_samples)    
    clear_dir(TEMP_FOLDER)


    plotLosses(loss_table_list)


    for i in range(n_folds):
        for k in range(number_of_samples):
            arr = np.load('{}fold{}_cli_{}_{}_cbt.npy'.format(Path_output, i, i, k, Setup_name))
            show_image(arr, i, k)
        