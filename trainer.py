import math

import torch
from tqdm.auto import tqdm

import wandb
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
        wandb.watch(self.model, self.criterion, log="all", log_freq=10)
        train_loss = []
        self.model.train()
        # pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            # metrics stuff
            train_loss.append(loss.item())

            if batch_idx % self.log_step == 0:
                # pbar.set_postfix(
                #     {
                #         "Train Epoch": epoch,
                #         "Train Loss": loss.item(),
                #     }
                # )
                print({
                        "Train Epoch": epoch,
                        "Train Loss": loss.item(),
                    })

        train_loss = sum(train_loss) / len(train_loss)
        # val_log = self._validate(epoch)
        print(train_loss)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        print("cows")
        return {**{
            "train_loss": round(train_loss, 4),
        }}

    def _metrics(self, output, target):
        log = {}
        for met in self.metrics:
            log[met.__name__] = round(met(output, target), 4)
        return log

    def _validate(self, epoch):
        valid_loss = []
        self.model.eval()
        pbar = tqdm(self.valid_loader, desc="Validation")
        with torch.no_grad():
            for _, (data, target) in enumerate(self.valid_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                pbar.set_postfix(
                    {
                        "Val Epoch": epoch,
                        "Val Loss": loss.item(),
                    }
                )
                valid_loss.append(loss.item())

        return {**self._metrics(output, target), **{
            "val_loss": round(sum(valid_loss) / len(valid_loss), 4),
        }}