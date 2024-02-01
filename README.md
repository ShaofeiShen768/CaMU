# [SDM 2024] CaMU: Disentangling Causal Effects in Deep Model Unlearning

## Abstract
This is a PyTorch implementation of [CaMU] Link: https://arxiv.org/abs/2401.17504

Machine unlearning requires removing the information of forgetting data while keeping the necessary information of remaining data. Despite recent advancements in this area, existing methodologies mainly focus on the effect removal of forgetting data without considering the negative impact this can have on the information of the remaining data, resulting in significant performance degradation after data removal. Although some methods try to repair the performance of remaining data after removal, the forgotten information can also return after repair. Such an issue is due to the intricate intertwining of the forgetting and remaining data. Without adequately differentiating the influence of these two kinds of data on the model, existing algorithms take the risk of either inadequate removal of the forgetting data or unnecessary loss of valuable information from the remaining data. To address this shortcoming, the present study undertakes a causal analysis of the unlearning and introduces a novel framework termed Causal Machine Unlearning (CaMU). This framework adds intervention on the information of remaining data to disentangle the causal effects between forgetting data and remaining data. Then CaMU eliminates the causal impact associated with forgetting data while concurrently preserving the causal relevance of the remaining data. Comprehensive empirical results on various datasets and models suggest that CaMU enhances performance on the remaining data and effectively minimizes the influences of forgetting data. Notably, this work is the first to interpret deep model unlearning tasks from a new perspective of causality and provide a solution based on causal analysis, which opens up new possibilities for future research in deep model unlearning. 

## Requirements:

First, install the camu environment and install all necessary packages:

    conda env create -f camu.yaml
    
## Dataset Download:  

The data folder includes the experiment data on Mnist and Fashion-Mnist datasets. As for the Cifar10 and Cifar100 datasets, you can run the code to download them via Torchvision.
   
## Results Reproduce:  

You can modify parameters in the config.py file for different models and datasets and then run the main.py file to reproduce the reported results:  

    python main.py

You can run the [CaMU.ipynb](CaMU.ipynb) file for easier debugging.

## License

This project is under the MIT license. See [LICENSE](License) for details.

## Citation

@misc{shen2024camu,<br> 
      title={CaMU: Disentangling Causal Effects in Deep Model Unlearning}, <br> 
      author={Shaofei Shen and Chenhao Zhang and Alina Bialkowski and Weitong Chen and Miao Xu},<br> 
      year={2024},<br> 
      eprint={2401.17504},<br> 
      archivePrefix={arXiv},<br> 
}
   
