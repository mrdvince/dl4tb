import math

import torch
from tqdm.auto import tqdm

from base.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        config,
        device,
        train_loader,
        valid_loader=None,
        lr_scheduler=None,
    ):
        super(Trainer, self).__init__(
            model,
            optimizer,
            criterion,
            metrics,
            config,
        )
        self.config = config
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.lr_scheduler = lr_scheduler
        self.log_step = int(math.sqrt(self.config.batch_size))

    def _train(self, epoch):
        train_loss = 0.0
        self.model.train()
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item() * data.size(0)
            if batch_idx % self.log_step == 0:
                pbar.set_postfix(
                    {
                        "Train Epoch": epoch,
                        "Train Loss": train_loss,
                    }
                )
        train_loss = train_loss / len(self.train_loader.dataset)
        val_loss = self._validate(epoch)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {
            "train_loss": train_loss,
            "val_loss": val_loss,
        }.update(self._metrics(output, target))

    def _metrics(self, output, target):
        log = {}
        for met in self.metrics:
            log[met.__name__] = met(output, target)
        return log

    def _validate(self, epoch):
        valid_loss = 0.0
        self.model.eval()
        pbar = tqdm(self.valid_loader, desc="Validation")
        with torch.no_grad():
            for _, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                pbar.set_postfix(
                    {
                        "Val Epoch": epoch,
                        "Val Loss": loss.item(),
                    }
                )
                valid_loss += loss.item() * data.size(0)
        return valid_loss / len(self.valid_loader.dataset)
