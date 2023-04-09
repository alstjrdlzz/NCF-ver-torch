import torch


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

        train_loss = 0
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        train_loss /= len(self.train_dataloader)
        return train_loss
    
    @torch.no_grad()
    def _valid_epoch(self, epoch):

        valid_loss = 0
        self.model.eval()
        for batch_idx, (data, target) in enumerate(self.valid_dataloader):

            data, target = data.to(self.device), target.to(self.device)
            
            output = self.model(data)
            loss = self.criterion(output, target)
            valid_loss += loss.item()

        valid_loss /= len(self.valid_dataloader)
        return valid_loss

    def train(self):
        train_loss = []
        valid_loss = []
        for epoch in range(1, self.epochs + 1):
            train_loss.append(self._train_epoch(epoch))
            valid_loss.append(self._valid_epoch(epoch))