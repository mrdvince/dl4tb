import math
import numpy as np
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
        super().__init__(
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
        
        self.log = dict()
        def _train(self, epoch):
            self.model.train()
            pbar = tqdm(self.train_loader, desc="Training")
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output)
                loss.backward()
                self.optimizer.step()

                self.log.update(self._log(self.log, target, output, loss))
                
                if batch_idx % self.log_step == 0:
                    pbar.set_postfix(
                    {
                        "Train Epoch": epoch,
                        "Train Loss": loss.item(),
                    }
                )

        
            val_log = self._validate(epoch)
            self.log.update(**{"val_" + k: v for k, v in val_log.items()})

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            return self.log

        def _validate(self, epoch):
            self.model.eval()
            pbar = tqdm(self.valid_loader, desc="Validation")
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(pbar):
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = self.criterion(output)
                    self.log.update(self._log(self.log, target, output, loss))
                    pbar.set_postfix(
                        {
                            "Val Epoch": epoch,
                            "Val Loss": loss.item(),
                        }
                    )
            return self.log

        def _log(self, log,target, output, loss, valid=False):
            log.update("loss" ,loss.item())
            for met in self.metrics:
                log.update(met.__name__, met(output, target))
            return log