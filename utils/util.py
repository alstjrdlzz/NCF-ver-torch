import torch
import torch.optim as optim
from model import loss
from model import metric


def build_model(config):

    return model

def prepare_device(config):
    '''
    To-do: multi GPU
    '''
    n_gpu_use = config['n_gpu']
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def get_criterion(config):
    name = config['loss']
    return getattr(loss, name)

def get_metrics(config):
    name_list = config['metrics']
    return [getattr(metric, name) for name in name_list]

def get_optimizer(config, params):
    name = config['optimizer']['name']
    options = config['optimizer']['options'] 
    return getattr(optim, name)(params, **options)

def get_lr_scheduler(config, optimizer):
    name = config['lr_scheduler']['name']
    options = config['lr_scheduler']['options']
    return getattr(optim.lr_scheduler, name)(optimizer, **options)