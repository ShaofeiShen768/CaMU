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
from sklearn.metrics import confusion_matrix, classification_report
import Config as cf
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns 
import pandas as pd
from torch.nn.functional import softmax
from sklearn.model_selection import train_test_split
from sklearn import linear_model, model_selection    
from sklearn.metrics import accuracy_score, precision_score, recall_score
import xgboost as xgb
from xgboost import XGBClassifier


def train(model, loaders, UNLEARN_LABEL, OUTPUT_LABEL):
    
    model.cuda()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr = cf.PRETRAIN_LR, weight_decay=1e-4) 
    loss_func = nn.CrossEntropyLoss()  
    tr_time = 0 
    
    acc_re_tr_list = []
    acc_fr_tr_list = []
    acc_tr_list = []
    acc_re_ts_list = []
    acc_fr_ts_list = []
    
    for epoch in range(1):

        for i, (data, targets) in enumerate(loaders['source']):  
   
            data = data.cuda() 
            targets = targets.type(torch.LongTensor)  
            targets = targets.cuda()   
            output = model(data)[0]   
            
            loss = loss_func(output, targets)
            optimizer.zero_grad()           
            
            loss.backward()    
            
            optimizer.step() 
            
            if i%100 == 0:
                print (i)
                acc_re_tr_list.append(test(model, loaders['source']))
                acc_fr_tr_list.append(test(model, loaders['unlearn']))
                acc_tr_list.append(test(model, loaders['test']))
                acc_re_ts_list.append(test(model, loaders['test'], OUTPUT_LABEL))
                acc_fr_ts_list.append(test(model, loaders['test'], UNLEARN_LABEL))
    
    return acc_re_tr_list, acc_fr_tr_list, acc_tr_list, acc_re_ts_list, acc_fr_ts_list
         
def test(trained_model, test_loader, output_label = None):
    trained_model = trained_model.cuda()
    trained_model.eval()
    all_preds = torch.tensor([])
    targets = torch.tensor([])
    original_targets = torch.tensor([])
    total = 0
    correct = 0
    with torch.no_grad():

        for i, (x,y) in enumerate(test_loader):
            
            original_targets = torch.cat((original_targets, y.cpu()),dim=0)
                            
            x = x.cuda()
            y = y.type(torch.LongTensor)
            y = y.cuda()
            y_out, _ = trained_model(x)
            
            _, predicted = torch.max(y_out.data, dim=1)
            all_preds = torch.cat((all_preds, y_out.cpu()),dim=0)
            targets = torch.cat((targets, y.cpu()),dim=0)
            
            total += y.size(0)
            correct += (predicted == y).sum().item()   

        preds = all_preds.argmax(dim=1)
        targets = np.array(targets)
        acc = 100.0 * correct / total
        print('The accuracy of the all classes is: ', acc)
        if output_label is None:
            selected_acc = acc
        else:
            selected_preds = []
            selected_targets = []
            selected_total = 0
            for i in range(len(preds)):
                if original_targets[i] in output_label:
                    selected_preds.append(preds[i])
                    selected_targets.append(targets[i])
                    selected_total += 1  
                        
            selected_correct = (np.array(selected_preds) == np.array(selected_targets)).sum().item()                  
            selected_acc = 100.0 * selected_correct / selected_total
            print('The accuracy of the selected classes ', output_label, 'is: ', selected_acc)

    return selected_acc
     