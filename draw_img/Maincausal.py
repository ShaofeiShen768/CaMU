import torch
from torch import optim
import torch.nn.functional as F
import random
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
import torch.nn as nn
from torch.nn.modules.sparse import Embedding
import torchsummary
import torchvision.transforms as T
import Data
import Model
import Train
import Config as cf
import matplotlib.pyplot as plt
from matplotlib import pyplot
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

def draw_line(name_of_alg,color_index,datas):
    palette = pyplot.get_cmap('Set1')
    iters=list(range(0, cf.CLASSIFIER_UNLEARNING_EPOCH+1))
    color=palette(color_index)
    avg=np.mean(datas,axis=0)
    std=np.std(datas,axis=0)
    r1 = list(map(lambda x: x[0]-x[1], zip(avg, std)))
    r2 = list(map(lambda x: x[0]+x[1], zip(avg, std)))  
    plt.plot(iters, avg, color=color,label=name_of_alg,linewidth=3.5)
    plt.fill_between(iters, r1, r2, color=color, alpha=0.2)
            
def plot_results(img_results_lists):
    
    fig = plt.figure(figsize = (8,6.4),dpi = 80)
    palette = pyplot.get_cmap('Set1')

    iters=list(range(0, len(img_results_lists[0])))
    plt.plot(iters, img_results_lists[0], color=palette(1),label="Remained Train Data",linewidth=3.5)
    plt.plot(iters, img_results_lists[1], color=palette(2),label="Unlearned Train Data",linewidth=3.5)
    plt.plot(iters, img_results_lists[2], color=palette(3),label="Test Data",linewidth=3.5)
    #plt.plot(iters, img_results_lists[3], color=palette(4),label="Remained Test Data",linewidth=3.5)
    #plt.plot(iters, img_results_lists[4], color=palette(7),label="Unlearned Test Data",linewidth=3.5)

    plt.xticks(np.arange(0, len(img_results_lists[0]), 1))
    plt.yticks()
    plt.xlabel('Batches')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(0.8, 0), loc=3, borderaxespad=0)
    plt.title("Relearning Performances")
    plt.grid(color = '#e9ecef', which='both', linestyle = '-', linewidth = 1)
    plt.show()
    plt.savefig('../Img/' + cf.DATASET + cf.DATATYPE + 'causal' + '.png')
  

    
    
    
def MAIN(): 
    # Build list storing results in experiments   
    results_lists = [[] for i in range(5)]
    img_results_lists = [[] for i in range(5)]
    # Experiments on different seeds
    for label in range(1):
        
        UNLEARN_LABEL = [label]
        OUTPUT_LABEL = [i for i in range(cf.CLASS_COUNT) if i != label]
        UNLEARNED_DISTRIBUTION = [0 for i in range(cf.CLASS_COUNT)] 
        UNLEARNED_DISTRIBUTION[label] = 1
            
        for seed in range(5):   
            
            random_setting(seed)
            
            #build data loaders
            data = Data.data_construction(cf.DATASET)        
            loaders = data.construct_data(cf.DATATYPE, UNLEARNED_DISTRIBUTION)
            
            # train a model using the whole training data
            if cf.DATASET == 'Digit' or cf.DATASET == 'Fashion':
                model = Model.CNN()
            elif cf.DATASET == 'CIFAR10' or cf.DATASET == 'CIFAR100':
                model = Model.ResNet()
                
            

            model_savepath1 = '../Causal_model/' + cf.DATASET + cf.DATATYPE + 'seed' + str(seed) + 'causal' + '.pt'
            model = torch.load(model_savepath1)
            
            acc_re_tr_list, acc_fr_tr_list, acc_tr_list, acc_re_ts_list, acc_fr_ts_list = Train.train(model, loaders, UNLEARN_LABEL, OUTPUT_LABEL)
            
                
            results_lists[0].append(acc_re_tr_list)
            results_lists[1].append(acc_fr_tr_list)
            results_lists[2].append(acc_tr_list)
            results_lists[3].append(acc_re_ts_list)
            results_lists[4].append(acc_fr_ts_list)
            
    results_lists[0] = np.array(results_lists[0])
    results_lists[1] = np.array(results_lists[1])
    results_lists[2] = np.array(results_lists[2])
    results_lists[3] = np.array(results_lists[3])
    results_lists[4] = np.array(results_lists[4])
    results_lists = np.array(results_lists)
    
    np.save('../npy/' + cf.DATASET + cf.DATATYPE + 'causal' + '.npy', results_lists)

    for i in range(len(results_lists[2][0])):
        img_results_lists[0].append(np.mean([results_lists[0][j][i] for j in range(5)]))
        img_results_lists[1].append(np.mean([results_lists[1][j][i] for j in range(5)]))
        img_results_lists[2].append(np.mean([results_lists[2][j][i] for j in range(5)]))
        img_results_lists[3].append(np.mean([results_lists[3][j][i] for j in range(5)]))
        img_results_lists[4].append(np.mean([results_lists[4][j][i] for j in range(5)]))
    
    plot_results(img_results_lists)
    
    
if __name__ == "__main__": 
    
    MAIN()






