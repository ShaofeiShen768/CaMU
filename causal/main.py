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
import add
import copy
import time
'''
Random settings
'''
def random_setting(seed = 1):
    # random seed setting
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
def write_results(results_lists):
    
    f = open('./Results15/' + cf.DATASET + cf.DATATYPE + 'unlearn' + '.txt', 'w')

    f.write('*****acc re tr mean= {}-std= {} *****\n'.format(np.mean(results_lists[0]), np.std(results_lists[0])))
    f.write('\n')
    f.write('*****acc fr tr mean= {}-std= {} *****n'.format(np.mean(results_lists[1]), np.std(results_lists[1])))
    f.write('\n')
    f.write('*****acc ts mean= {}-std= {} *****\n'.format(np.mean(results_lists[2]), np.std(results_lists[2])))
    f.write('\n')
    f.write('*****acc re ts Acc mean= {}-std= {} *****n'.format(np.mean(results_lists[3]), np.std(results_lists[3])))
    f.write('\n')
    f.write('*****acc fr ts mean= {}-std= {} *****\n'.format(np.mean(results_lists[4]), np.std(results_lists[4])))
    f.write('\n')
    f.write('*****Retrain time mean= {}-std= {} *****\n'.format(np.mean(results_lists[5]), np.std(results_lists[5])))
    f.write('\n')
    f.write('*****MIA ACC retrain = {}-std= {} *****\n'.format(np.mean(results_lists[6]), np.std(results_lists[6])))
    f.write('\n')

    
def main(): 
    results_lists = [[] for i in range(7)]
    
    # Experiments on different seeds
    for label in range(1):
        
        UNLEARN_LABEL = [label]
        OUTPUT_LABEL = [i for i in range(cf.CLASS_COUNT) if i != label]
        UNLEARNED_DISTRIBUTION = [0 for i in range(cf.CLASS_COUNT)] 
        UNLEARNED_DISTRIBUTION[label] = 1
            
        for seed in range(5):   
        
            # Experiments on different seeds   
            random_setting(seed)
            
            #build data loaders
            data = dataset.data_construction(cf.DATASET)        
            loaders = data.construct_data(cf.DATATYPE, UNLEARNED_DISTRIBUTION)
            random_setting(seed + 1)
            # train a model using the whole training data
            if cf.DATASET == 'Digit' or cf.DATASET == 'Fashion':
                trained_model = Model.CNN()
            elif cf.DATASET == 'CIFAR10' or cf.DATASET == 'CIFAR100':
                trained_model = Model.ResNet()
            
            print('***********Start Initial Training************')
                
            model_savepath = '../Original_model/' + cf.DATASET + 'seed' + str(seed) + 'original' + '.pt'
            
            trained_model = torch.load(model_savepath)

            
            print('***********Start Constructing Counterfactual Dataset************')   
            
            close_dataset = prepare.dist_calculation(trained_model, loaders['unlearn'], loaders['sample'])
            close_loader = torch.utils.data.DataLoader(close_dataset, batch_size=cf.BATCH_SIZE, shuffle=True, num_workers=cf.NUM_WORKERS)
            
            print('***********Finish Constructing Counterfactual Dataset************')
            new_model = copy.deepcopy(trained_model)
            
            print('***********Start Removing************')
            time_0 = time.time()
            remove.causal_effect_removel(trained_model, new_model, close_loader, loaders['sample'])
            
            time_taken = time.time() - time_0
            
            add_model = copy.deepcopy(new_model)
            
            torch.save(add_model, '../Causal_model15/' + cf.DATASET + cf.DATATYPE + 'seed' + str(seed) + 'causal' + '.pt')
            
            print('***********Finish Unlearning************')
            
            print('***********Start Test************')
            
            attack_model = train.train_attack_model(add_model, loaders['sample'], loaders['test'])
            
            acc_re= train.attack(add_model, attack_model, loaders['unlearn'], loaders['test'])
        
            acc_re_tr = train.test(add_model, loaders['source'])
            acc_fr_tr = train.test(add_model, loaders['unlearn'])
            acc_tr = train.test(add_model, loaders['test'])
            acc_re_ts = train.test(add_model, loaders['test'], OUTPUT_LABEL)
            acc_fr_ts = train.test(add_model, loaders['test'], UNLEARN_LABEL)
            
            print('***********Finish Test************')
            
            results_lists[0].append(acc_re_tr)
            results_lists[1].append(acc_fr_tr)
            results_lists[2].append(acc_tr)
            results_lists[3].append(acc_re_ts)
            results_lists[4].append(acc_fr_ts)
            results_lists[5].append(time_taken)
            results_lists[6].append(acc_re)
        
        write_results(results_lists)
        
                   
    
if __name__ == "__main__": 
    

    main()






