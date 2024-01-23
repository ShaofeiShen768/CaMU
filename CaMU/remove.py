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

def causal_effect_removel(trained_model, new_model, close_loader,sample_loader):
    
    trained_model.cuda()
    trained_model.eval()
    new_model.cuda()
    new_model.train()
    optimizer1 = optim.Adam(new_model.parameters(), lr = cf.KL_LR, weight_decay=1e-4) 
    optimizer2 = optim.Adam(new_model.parameters(), lr = cf.IMPAIR_LR, weight_decay=1e-4) 
    loss_func1 = nn.KLDivLoss(reduction='sum')
    loss_func2 = nn.KLDivLoss(reduction='sum')
    loss_func3 = nn.CrossEntropyLoss() 
    loss_func4 = nn.CrossEntropyLoss()
    max_acc = 0  
    for epoch in range(cf.IMPAIR_EPOCHS):
        torch.cuda.empty_cache()
        all_loss1 = 0
        all_loss2 = 0
        all_loss3 = 0
        all_loss4 = 0
        count = 0
        for i, ((data, targets, treatment_x, treatment_y),(remain_x, remain_y)) in enumerate(zip(close_loader,sample_loader)):
            
            count += 1
            data = data.cuda() 

            remain_x = remain_x.cuda()
            remain_y = remain_y.type(torch.LongTensor)
            remain_y = remain_y.cuda()
            treatment_x = treatment_x.type(torch.FloatTensor).cuda()
            treatment_y = treatment_y.type(torch.LongTensor)  
            treatment_y = treatment_y.cuda()

            with torch.no_grad():
                embedding1 = trained_model(remain_x)[1]   
            embedding2 = new_model(remain_x)[1]  
            embedding1 = F.softmax(embedding1,dim=1)
            embedding2 = F.log_softmax(embedding2,dim=1)        
            # loss1 = loss_func1(embedding2, embedding1)
            # all_loss1 += loss1.cpu().item()    
            with torch.no_grad():
                embedding4 = trained_model(treatment_x)[1]  
            embedding3 = new_model(data)[1] 
            
            embedding3 = F.log_softmax(embedding3,dim=1)
            embedding4 = F.softmax(embedding4,dim=1)              
            loss2 = 0.5 * loss_func1(embedding2, embedding1) + 0.5 * loss_func2(embedding3, embedding4)# + 
            optimizer1.zero_grad()                       
            loss2.backward()                
            optimizer1.step() 
            # all_loss2 += loss2.cpu().item()
            
            output1 = new_model(data)[0]       
            # loss3 = loss_func3(output1, treatment_y)
            # all_loss3 += loss3.cpu().item()
            output3 = new_model(remain_x)[0]         
            loss4 =  0.9 * loss_func4(output3, remain_y) + 0.01 * loss_func3(output1, treatment_y)
            optimizer2.zero_grad()                       
            loss4.backward()                
            optimizer2.step() 
            #all_loss4 += loss4.cpu().item()
            
                        
        # #epoch_loss1 = all_loss1/count
        # epoch_loss2 = all_loss2/count
        # epoch_loss3 = all_loss3/count    
        # epoch_loss4 = all_loss4/count  
                    
        # #print ('Epoch [{}/{}], Loss1: {:.4f}'.format(epoch + 1, cf.IMPAIR_EPOCHS, epoch_loss1))
        # print ('Epoch [{}/{}], Loss2: {:.4f}'.format(epoch + 1, cf.IMPAIR_EPOCHS, epoch_loss2))
        # print ('Epoch [{}/{}], Loss3: {:.4f}'.format(epoch + 1, cf.IMPAIR_EPOCHS, epoch_loss3))
        # print ('Epoch [{}/{}], Loss4: {:.4f}'.format(epoch + 1, cf.IMPAIR_EPOCHS, epoch_loss4))
    

