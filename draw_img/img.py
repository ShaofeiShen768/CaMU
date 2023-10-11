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
    
    fig = plt.figure(figsize = (8,7),dpi = 80)
    palette = pyplot.get_cmap('Set1')
    a = len(img_results_lists[0])
    
    iters=list(range(0, len(img_results_lists[0])))
    plt.plot(iters, img_results_lists[5], color=palette(8),label="Retrain",linewidth=3.5)
    
    plt.plot(iters, img_results_lists[0], color=palette(3),label="TS",linewidth=3.5)
    plt.plot(iters, img_results_lists[1], color=palette(2),label="Boundary",linewidth=3.5)
    
    plt.plot(iters, img_results_lists[3], color=palette(4),label="Unrolling",linewidth=3.5)
    plt.plot(iters, img_results_lists[4], color=palette(7),label="NegGrad",linewidth=3.5)
    plt.plot(iters, img_results_lists[2], color=palette(1),label="Causal",linewidth=3.5)
    
    fig.subplots_adjust(right=0.7)

    plt.xticks(np.arange(0, len(img_results_lists[0]), 2),size = 14)
    plt.yticks(size = 18)
    plt.xlabel('Batches',size = 18)
    plt.ylabel('Changes of Accuracy Differences',size = 18)
    plt.legend(bbox_to_anchor=(1.03, 0), loc=3, borderaxespad=0,fontsize = 14)
    #plt.title(cf.DATASET)
    plt.grid(color = '#e9ecef', which='both', linestyle = '-', linewidth = 1)
    plt.show()
    plt.savefig('../Img/00222' + cf.DATASET + cf.DATATYPE + 'compare' + '.png')

    
def MAIN(): 
    
    ts_lists = np.load('../npy/' + cf.DATASET + cf.DATATYPE + 'ts' + '.npy',allow_pickle=True)
    boundary_lists = np.load('../npy/' + cf.DATASET + cf.DATATYPE + 'boundary' + '.npy',allow_pickle=True)
    causal_lists = np.load('../npy/' + cf.DATASET + cf.DATATYPE + 'causal' + '.npy',allow_pickle=True)
    unrolling_lists = np.load('../npy/' + cf.DATASET + cf.DATATYPE + 'unrolling' + '.npy',allow_pickle=True)
    neggrad_lists = np.load('../npy/' + cf.DATASET + cf.DATATYPE + 'neggrad' + '.npy',allow_pickle=True)
    retrain_lists = np.load('../npy/' + cf.DATASET + cf.DATATYPE + 'retrain' + '.npy',allow_pickle=True)
    print(ts_lists)
    ts_mean  = np.mean(ts_lists[0],axis=0) - np.mean(ts_lists[1],axis=0)
    boundary_mean  = np.mean(boundary_lists[0],axis=0) - np.mean(boundary_lists[1],axis=0)
    causal_mean  = np.mean(causal_lists[0],axis=0) - np.mean(causal_lists[1],axis=0)
    unrolling_mean  = np.mean(unrolling_lists[0],axis=0) - np.mean(unrolling_lists[1],axis=0)
    neggrad_mean  = np.mean(neggrad_lists[0],axis=0) - np.mean(neggrad_lists[1],axis=0)
    retrain_mean  = np.mean(retrain_lists[0],axis=0) - np.mean(retrain_lists[1],axis=0)
    
    img_results_lists = [ts_mean, boundary_mean, causal_mean, unrolling_mean, neggrad_mean, retrain_mean]
    
    plot_results(img_results_lists)
    
    
                
               
               
if __name__ == "__main__": 
    
    MAIN()






