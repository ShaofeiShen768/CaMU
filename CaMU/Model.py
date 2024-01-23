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
import config as cf
from torchvision import models, transforms 

'''
CNN model for MNIST and Fashion-MNIST datasets
'''
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.fc1 = nn.Linear(32 * 7 * 7, 256)
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)       
        embedding = self.fc1(x)
        output = self.fc2(embedding)
        return output, embedding
'''
ResNet model for Cifar10 and Cifar100 datasets
'''    
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        
        resnet = models.resnet18() 
        num_features = resnet.fc.in_features 
        features = list(resnet.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*features)
        if cf.DATASET != 'CIFAR100':
            self.fc2 = nn.Linear(num_features, 10)
        else:
            self.fc2 = nn.Linear(num_features, 100)
        
    def forward(self, x):
        embedding = self.feature_extractor(x).squeeze(3).squeeze(2)
        output = self.fc2(embedding)
        return output, embedding       