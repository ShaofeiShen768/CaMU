import torch
from torch import optim
import torch.nn.functional as F
import random
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
import pandas as pd 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import config as cf
import time
from random import choice

'''
Construct counterfactual data for relation removing
'''
class CounterfactualData(Dataset):

    def __init__(self, data, targets, treatment_x, treatment_y):
        self.data = data
        self.targets = targets
        self.treatment_x = treatment_x
        self.treatment_y = treatment_y
        
    def __getitem__(self, index):
        return self.data[index], self.targets[index], self.treatment_x[index], self.treatment_y[index]

    def __len__(self):
        return len(self.data)

def prepare_counterfactual(model, train_loader):
    model.cuda()
    model.eval()
    batch_x = torch.tensor([]).cuda()
    batch_y = torch.tensor([]).cuda()
    for i, (data, targets) in enumerate(train_loader):

        data = data.cuda() 
        mask = torch.rand(data.shape).cuda()
        data = data + mask
        targets = targets.type(torch.LongTensor)  
        targets = targets.cuda()   
        batch_x = torch.cat((batch_x, data),dim=0)
        batch_y = torch.cat((batch_y, targets),dim=0)
    return batch_x.cpu().detach().numpy(), batch_y.cpu().detach().numpy() 

def prepare_factual(model, train_loader):
    model.cuda()
    model.eval()
    batch_x = torch.tensor([]).cuda()
    batch_y = torch.tensor([]).cuda()
    for i, (data, targets) in enumerate(train_loader):

        data = data.cuda() 
        targets = targets.type(torch.LongTensor)  
        targets = targets.cuda()   
        batch_x = torch.cat((batch_x, data),dim=0)
        batch_y = torch.cat((batch_y, targets),dim=0)
    return batch_x.cpu().detach().numpy(), batch_y.cpu().detach().numpy() 

# sample random data and add random noise to generate counterfactual data
def label_adjustment(target_y, batch_y):

    for i in range(len(target_y)):
        
        if batch_y[i] == target_y[i]:
            batch_y[i] = choice([i for i in range(cf.CLASS_COUNT) if i != target_y[i]])

    return target_y, batch_y


            
def counterfactual_generation(model, unlearn_loader, remain_loader):
    
    u_batch_x, u_batch_y = prepare_factual(model, unlearn_loader)
    treatment_x, treatment_y = prepare_counterfactual(model, remain_loader)
    u_batch_y, treatment_y = label_adjustment(u_batch_y, treatment_y)    
    close_dataset = CounterfactualData(u_batch_x, u_batch_y, treatment_x, treatment_y)
    
    return close_dataset  

























