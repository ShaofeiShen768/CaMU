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
class CounterfactualData(Dataset):

    def __init__(self, data, targets, remain_x, remain_y, treatment_x, treatment_y):
        self.data = data
        self.targets = targets
        self.remain_x = remain_x
        self.remain_y = remain_y
        self.treatment_x = treatment_x
        self.treatment_y = treatment_y
        
    def __getitem__(self, index):
        return self.data[index], self.targets[index], self.treatment_x[index], self.treatment_y[index] #self.remain_x[index], self.remain_y[index], 

    def __len__(self):
        return len(self.data)

def prepare(model, train_loader):
    
    model.cuda()
    model.eval()
    embedding_x = torch.tensor([]).cuda()
    batch_x = torch.tensor([]).cuda()
    batch_y = torch.tensor([]).cuda()
    for i, (data, targets) in enumerate(train_loader):

        data = data.cuda() 
        targets = targets.type(torch.LongTensor)  
        targets = targets.cuda()   
        batch_x = torch.cat((batch_x, data),dim=0)
        batch_y = torch.cat((batch_y, targets),dim=0)
        
    return batch_x.cpu().detach().numpy(), batch_y.cpu().detach().numpy() 

def random_data_sample(target_y, batch_x, batch_y):

    x_list, y_list  = batch_distance(batch_x, batch_y)

    remain_x, remain_y, treatment_x, treatment_y = target_data_remove(x_list, y_list)
    treatment_x = treatment_x + np.random.rand(remain_x.shape[0],remain_x.shape[1],remain_x.shape[2]).astype(np.float32)
    while treatment_y == target_y:
        treatment_y = np.random.randint(0,cf.CLASS_COUNT)         

    return remain_x, remain_y, treatment_x, treatment_y


def batch_distance(batch_x, batch_y):
    
    x_list, y_list = [], []
    for i in range(batch_x.shape[0]):
            x_list.append(batch_x[i])
            y_list.append(batch_y[i])
    return x_list, y_list        
            
def target_data_remove(x_list, y_list):
    
    df = pd.DataFrame({'x':x_list, 'y':y_list})

    i = random.randint(0,len(df)-1)
    remain_x = df.iloc[i]['x']
    remain_y = df.iloc[i]['y']

    i = random.randint(0,len(df)-1)
    treatment_x = df.iloc[i]['x']
    treatment_y = df.iloc[i]['y']
            
    return remain_x, remain_y, treatment_x, treatment_y

            
def dist_calculation(model, unlearn_loader, remain_loader):
    
    remain_x_list, remain_y_list, treatment_x_list, treatment_y_list = [], [], [], []
    u_batch_x, u_batch_y = prepare(model, unlearn_loader)
    r_batch_x, r_batch_y = prepare(model, remain_loader)
    for i in range(len(u_batch_x)):
        remain_x, remain_y, treatment_x, treatment_y = random_data_sample(u_batch_y[i], r_batch_x, r_batch_y)
        remain_x_list.append(remain_x)
        remain_y_list.append(remain_y)
        treatment_x_list.append(treatment_x)
        treatment_y_list.append(treatment_y)
        
    close_dataset = CounterfactualData(u_batch_x, u_batch_y, remain_x_list, remain_y_list, treatment_x_list, treatment_y_list)
    return close_dataset  

























