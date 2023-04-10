import torch
import numpy as np
from data.datamodule import TrainDataModule
from utils.util import init_model
from utils.util import prepare_device
from utils.util import get_criterion
from utils.util import get_metrics
from utils.util import get_optimizer
from utils.util import get_lr_scheduler
from trainer.trainer import Trainer

# To-do: config to json
config = {
    # data
    'data_path': '/content/data/ml-100k/',
    'batch_size': 2048,
    'cv_startegy': {'name': 'holdout',
                    'options': {'test_size': 0.2}},
    # model
    'model': {'name': 'GMF',
              'options': {'M': 943,
                          'N': 1682,
                          'K': 512}},
    # train
    'n_gpu': 1,
    'loss': 'bce_loss',
    'metrics': ['accuracy'],
    'optimizer': {'name': 'Adam',
                  'options': {'lr': 0.001, 'weight_decay': 0, 'amsgrad': True}},
    'lr_scheduler': {'name': 'StepLR',
                     'options': {'step_size': 50, 'gamma': 0.1}},

    # wandb init args
    'project': 'test',
    'entity': 'alstjrdlzz',
    'name': '2023-04-10 test'
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
    
    # init model
    model = init_model(config)

    # prepare device
    device, device_ids = prepare_device(config)
    model = model.to(device)

    # get loss, metrics
    criterion = get_criterion(config)
    metrics = get_metrics(config)

    # get optimizer, lr_scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer(config, trainable_params)
    lr_scheduler = get_lr_scheduler(config, optimizer)

    # trainer
    trainer = Trainer(config=config,
                      train_dataloader=train_dataloader,
                      valid_dataloader=valid_dataloader,
                      model=model,
                      device=device,
                      criterion=criterion,
                      metrics=metrics,
                      optimizer=optimizer,
                      lr_scheduler=lr_scheduler)
    
    trainer.train()

if __name__ == '__main__':
    main(config)