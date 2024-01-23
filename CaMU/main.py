import torch
import torch.nn.functional as F
import random
import os
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
import dataset
import Model
import train
import config as cf
import prepare
import remove
import utils
import copy
import time


    
def main(): 
    results_lists = [[] for i in range(7)]
    
    # Initialize target label for class unlearning
    # The label information is not used for random data unlearning
    for label in [0]:
        
        UNLEARN_LABEL = [label]
        OUTPUT_LABEL = [i for i in range(cf.CLASS_COUNT) if i != label]
        UNLEARNED_DISTRIBUTION = [0 for i in range(cf.CLASS_COUNT)] 
        UNLEARNED_DISTRIBUTION[label] = 1
            
        for seed in range(5):   
        
            # Experiments on different seeds   
            utils.random_setting(seed)
            
            print('***********Building data loaders************')
            data = dataset.data_construction(cf.DATASET)        
            loaders = data.construct_data(cf.DATATYPE, UNLEARNED_DISTRIBUTION)
            
            # Initialize a model
            if cf.DATASET == 'Digit' or cf.DATASET == 'Fashion':
                trained_model = Model.CNN()
            elif cf.DATASET == 'CIFAR10' or cf.DATASET == 'CIFAR100':
                trained_model = Model.ResNet()
            
            print('***********Start Initial Training************')
            # Load well-trained model   
            model_savepath = '../original.pt'
            trained_model = torch.load(model_savepath)

            # Train a well-trained model
            # train.train(trained_model, loaders['train'])
            
            new_model = copy.deepcopy(trained_model)
            
            print('***********Start Constructing Counterfactual Dataset************')   
            time_0 = time.time()
            close_dataset = prepare.counterfactual_generation(trained_model, loaders['unlearn'], loaders['sample'])
            close_loader = torch.utils.data.DataLoader(close_dataset, batch_size=32, shuffle=True, num_workers=3)
            
            print('***********Finish Constructing Counterfactual Dataset************')

            print('***********Start Removing************')
            
            remove.causal_effect_removel(trained_model, new_model, close_loader, loaders['sample'])
            
            time_taken = time.time() - time_0
            
            add_model = copy.deepcopy(new_model)
            
            print('***********Finish Unlearning************')
            
            print('***********Start Test************')
            
            attack_model = train.train_attack_model(add_model, loaders['sample'], loaders['test'])    
            acc_re= train.attack(add_model, attack_model, loaders['unlearn'], loaders['test'])
            acc_re_tr = train.test(add_model, loaders['source'])
            acc_fr_tr = train.test(add_model, loaders['unlearn'])
            acc_ts = train.test(add_model, loaders['test'])
            acc_re_ts = train.test(add_model, loaders['test'], OUTPUT_LABEL)
            acc_fr_ts = train.test(add_model, loaders['test'], UNLEARN_LABEL)
            
            print('***********Finish Test************')
            
            results_lists[0].append(acc_re_tr)
            results_lists[1].append(acc_fr_tr)
            results_lists[2].append(acc_ts)
            results_lists[3].append(acc_re_ts)
            results_lists[4].append(acc_fr_ts)
            results_lists[5].append(time_taken)
            results_lists[6].append(acc_re)
        
        utils.write_results(results_lists)
        
    
if __name__ == "__main__": 
    

    main()






