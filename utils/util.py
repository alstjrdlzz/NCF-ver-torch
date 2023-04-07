from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
from model import loss
from model import metric


def load_data(data_path):
    path = Path(data_path)
    raw_df = pd.read_csv(path/'u.data', sep='\t', encoding='latin-1', header=None)
    raw_df.columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    return raw_df

def split_data(df, cv_strategy):
    '''
    To-do: kfold cv
    '''
    name = cv_strategy['name']
    options = cv_strategy['options']

    if name == 'holdout':
        train_df, test_df = train_test_split(df, **options)
    else:
        raise NotImplementedError

    return train_df, test_df

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