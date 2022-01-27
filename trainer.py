import math

import torch
from tqdm.auto import tqdm

import wandb
from base.base_trainer import BaseTrainer

try:
    from habana_frameworks.torch.hpex import hmp
except ImportError:
    """"""


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
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()

            if self.device.type == "hpu":
                with hmp.disable_casts():
                    self.optimizer.step()
            else:
                self.optimizer.step()

            # metrics stuff
            train_loss.append(loss.item())

            if batch_idx % self.log_step == 0:
                pbar.set_postfix(
                    {
                        "Train Epoch": epoch,
                        "Train Loss": loss.item(),
                    }
                )
        train_loss = sum(train_loss) / len(train_loss)
        val_log = self._validate(epoch)

        if self.lr_scheduler is not None:
            if self.device.type == "hpu":
                with hmp.disable_casts():
                    self.lr_scheduler.step()
            else:
                self.lr_scheduler.step()

        return {
            **{
                "train_loss": round(train_loss, 4),
            },
            **val_log,
        }

    def _metrics(self, output, target):
        accuracy, pred, total_correct = self.metrics[0](output, target)
        topk, topk_pred, topk_total_correct = self.metrics[1](output, target, k=1)
        return (
            {
                self.metrics[0].__name__: round(accuracy, 4),
                self.metrics[1].__name__: round(topk, 4),
            },
            pred,
            topk_total_correct,
        )

    def _validate(self, epoch):
        valid_loss = []
        example_images = []
        log = {}
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
                met_dict, pred, total_correct = self._metrics(output, target)
                log.update(met_dict)

                # log images to wandb images
                classes = self.train_loader.dataset.classes
                if len(example_images) < 4:
                    example_images.append(
                        wandb.Image(
                            data[0],caption=f"Pred: {classes[pred[0]]} Target: {classes[target[0]]}",
                        )
                    )
        
                valid_loss.append(loss.item())
        wandb.log({"Images": example_images})
        return {
            **log,
            **{
                "val_loss": round(sum(valid_loss) / len(valid_loss), 4),
            },
        }
