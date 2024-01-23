import torch
from torch import optim
import torch.nn.functional as F
import random
import random
import os
import numpy as np
import config as cf
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
'''
Record final results
'''    
def write_results(results_lists):
    
    f = open('./Results/' + cf.DATASET + cf.DATATYPE + 'unlearn' + '.txt', 'w')

    f.write('*****acc re tr mean= {}-std= {} *****\n'.format(np.mean(results_lists[0]), np.std(results_lists[0])))
    f.write('\n')
    f.write('*****acc fr tr mean= {}-std= {} *****n'.format(np.mean(results_lists[1]), np.std(results_lists[1])))
    f.write('\n')
    f.write('*****acc ts mean= {}-std= {} *****\n'.format(np.mean(results_lists[2]), np.std(results_lists[2])))
    f.write('\n')
    f.write('*****acc re ts Acc mean= {}-std= {} *****n'.format(np.mean(results_lists[3]), np.std(results_lists[3])))
    f.write('\n')
    f.write('*****acc fr ts mean= {}-std= {} *****\n'.format(np.mean(results_lists[4]), np.std(results_lists[4])))
    f.write('\n')
    f.write('*****Retrain time mean= {}-std= {} *****\n'.format(np.mean(results_lists[5]), np.std(results_lists[5])))
    f.write('\n')
    f.write('*****MIA ACC retrain = {}-std= {} *****\n'.format(np.mean(results_lists[6]), np.std(results_lists[6])))
    f.write('\n')