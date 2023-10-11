import torch
from torch import optim
import torch.nn.functional as F
import random
import os
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix, classification_report
import config as cf
import train
from torch.autograd import grad
from tqdm import tqdm
import time

def causal_effect_add(removed_model, new_model, add_loader):
    
    removed_model.cuda()
    removed_model.train()
    new_model.cuda()
    new_model.train()
    optimizer1 = optim.Adam(new_model.parameters(), lr = cf.KL_LR, weight_decay=1e-4) 
    optimizer3 = optim.Adam(new_model.parameters(), lr = cf.REPAIR_LR, weight_decay=1e-4) 
    loss_func1 = nn.KLDivLoss(reduction='sum')
    loss_func2 = nn.CrossEntropyLoss() 
    loss_func3 = nn.CrossEntropyLoss()
    max_acc = 0   

    for epoch in range(cf.REPAIR_EPOCHS):
        
        all_loss1 = 0
        all_loss2 = 0
        all_loss3 = 0
        count = 0
        
        for i, ((data, targets, treatment_x, treatment_y),(remain_x, remain_y)) in enumerate(zip(close_loader,sample_loader)):
            
            count += 1
            data = data.cuda() 
            targets = targets.type(torch.LongTensor)  
            targets = targets.cuda()   
            remain_x = remain_x.cuda()
            remain_y = remain_y.type(torch.LongTensor)
            remain_y = remain_y.cuda()
            treatment_x = treatment_x.cuda()
            treatment_y = treatment_y.type(torch.LongTensor)  
            treatment_y = treatment_y.cuda()
            
                        
            embedding1 = removed_model(remain_x)[1]   
            embedding2 = new_model(remain_x)[1]  
            embedding1 = F.softmax(embedding1,dim=1)
            embedding2 = F.log_softmax(embedding2,dim=1)        
            loss1 = loss_func1(embedding2, embedding1)
            # optimizer1.zero_grad()                       
            # loss1.backward()                
            # optimizer1.step() 
            all_loss1 += loss1.cpu().item()
            
            
            output3 = new_model(remain_x)[0]   
            output4 = removed_model(remain_x)[0]   
            _, predicts = torch.max(output4.data, dim=1)        
            loss3 = loss_func3(output3, remain_y) + loss_func1(embedding2, embedding1)
            optimizer3.zero_grad()                       
            loss3.backward()                
            optimizer3.step() 
            all_loss3 += loss3.cpu().item()
            
            
            
            
        epoch_loss1 = all_loss1/count
        epoch_loss2 = all_loss2/count
        epoch_loss3 = all_loss3/count    
                    
        print ('Epoch [{}/{}], Loss1: {:.4f}'.format(epoch + 1, cf.REPAIR_EPOCHS, epoch_loss1))
        print ('Epoch [{}/{}], Loss2: {:.4f}'.format(epoch + 1, cf.REPAIR_EPOCHS, epoch_loss2))
        print ('Epoch [{}/{}], Loss3: {:.4f}'.format(epoch + 1, cf.REPAIR_EPOCHS, epoch_loss3))
  







            
            
        