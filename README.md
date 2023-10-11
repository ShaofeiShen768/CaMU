# CaMU: Disentangling Causal Effects in Deep Model Unlearning

***
Code for paper CaMU: Disentangling Causal Effects in Deep Model Unlearning
***

## Requirements:

First, install the camu environment and install all necessary packages:

    conda env create -f camu.yaml
    
## Dataset Download:  

The data folder includes the experiment data on Mnist and Fashion-Mnist datasets. As for the Cifar10 and Cifar100 datasets, you can run the code to download them via Torchvision.
   
## Results Reproduce:  

You can modify parameters in the config.py file for different models and datasets and then run the main.py file to reproduce the reported results:  

    python main.py
    
   
