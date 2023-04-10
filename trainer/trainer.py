import torch
import wandb


WANDB_API_KEY = 'd5105ed3392ca1fdbbdab3ee87c0724c7a1ec534'

class Trainer():
    def __init__(self,
                 config,
                 train_dataloader,
                 valid_dataloader,
                 model,
                 device,
                 criterion,
                 metrics,
                 optimizer,
                 lr_scheduler):
        self.config = config
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.epochs = config['epochs']

    def _train_epoch(self, epoch):

        tra_loss = 0
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            tra_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        tra_loss /= len(self.train_dataloader)
        return tra_loss
    
    @torch.no_grad()
    def _valid_epoch(self, epoch):

        val_loss = 0
        self.model.eval()
        for batch_idx, (data, target) in enumerate(self.valid_dataloader):

            data, target = data.to(self.device), target.to(self.device)
            
            output = self.model(data)
            loss = self.criterion(output, target)
            val_loss += loss.item()

        val_loss /= len(self.valid_dataloader)
        return val_loss

    def train(self):
        wandb.login(key=WANDB_API_KEY)
        wandb.init(project=self.config['project'], entity=self.config['entity'], name=self.config['name'])
        for epoch in range(1, self.epochs + 1):
            tra_loss = self._train_epoch(epoch)
            val_loss = self._valid_epoch(epoch)
            wandb.log({"epoch": epoch, "tra_loss": tra_loss, "val_loss": val_loss})