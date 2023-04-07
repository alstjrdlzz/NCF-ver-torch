import torch
import numpy as np
from data.dataset import TrainDataset
from data.datamodule import TrainDataModule
from utils.util import prepare_device
from utils.util import get_criterion
from utils.util import get_metrics
from utils.util import get_optimizer
from utils.util import get_lr_scheduler
from trainer import Trainer

# To-do: config to json
config = {
    # data
    'data_path': '/content/data/ml-100k/',
    'batch_size': 4,
    'cv_startegy': {'name': 'holdout',
                    'options': {'test_size': 0.3}},
    # model
    'arch': ['GMF'],
    # train
    'n_gpu': 1,
    'loss': ['bce_loss'],
    'metrics': ['accuracy'],
    'optimizer': {'name': 'Adam',
                  'options': {'lr': 0.001, 'weight_decay': 0, 'amsgrad': True}},
    'lr_scheduler': {'name': 'StepLR',
                     'options': {'lr': 0.001, 'step_size': 50, 'gamma': 0.1}}
}

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    # setup dataloader
    train_dataloader = TrainDataModule(config).train_dataloader
    valid_dataloader = TrainDataModule(config).valid_dataloader
    
    # buld model
    '''
    To-do
    '''
    model = 

    # prepare device
    device = prepare_device(config)
    model = model.to(device)

    # get loss, metrics
    criterion = get_criterion(config)
    metrics = get_metrics(config)

    # get optimizer, lr_scheduler
    trainable_params = [p for p in model.parameters() if p.requires_gard]
    optimizer = get_optimizer(config, trainable_params)
    lr_scheduler = get_lr_scheduler(config, optimizer)

    # trainer
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      train_data_loader=train_dataloader,
                      valid_data_lodaer=valid_dataloader,
                      lr_scheduler=lr_scheduler)
    
    trainer.train()

if __name__ == '__main__':
    main(config)