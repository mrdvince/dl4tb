import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision
import wandb

from unet import UNET


class Dice(torchmetrics.Metric):
    def __init__(self):
        super(Dice, self).__init__()
        self.add_state("dice", default=torch.tensor(0))

    def update(self, preds, target, eps=1e-8):
        self.len_loader = len(preds)
        self.dice += 2.0 * (preds * target).sum() / (preds + target).sum() + eps

    def compute(self):
        return self.dice / self.len_loader


class UNETModel(pl.LightningModule):
    def __init__(self, lr):
        super(UNETModel, self).__init__()
        self.lr = lr
        self.save_hyperparameters()
        self.model = UNET(in_channels=3, out_channels=1)
        self.criterion = nn.BCEWithLogitsLoss()
        self.acc = torchmetrics.Accuracy()
        self.dice = Dice()

    def forward(self, x):
        self.model(x)

    def training_step(self, batch, batch_idx):
        image, mask = batch
        logits = self.model(image)
        loss = self.criterion(logits, mask.unsqueeze(1))
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        mask = mask.unsqueeze(1)
        logits = self.model(image)
        loss = self.criterion(logits, mask)
        acc = self.acc(logits, mask)
        dice_score = self.dice(logits, mask)

        preds = (torch.sigmoid(logits) > 0.5).float()

        torchvision.utils.save_image(preds, "pred.png")
        torchvision.utils.save_image(mask, "mask.png")

        # Logging metrics
        self.log("valid/acc", acc, prog_bar=True, on_step=True)
        self.log("valid/loss", loss, prog_bar=True, on_step=True)
        self.log("valid/dice_score", dice_score, prog_bar=True, on_step=True)
        return {"images": image, "mask": mask, "preds": preds}

    def validation_epoch_end(self, outputs):
        images = torch.cat([x["image"] for x in outputs])
        preds = torch.cat([x["preds"] for x in outputs])
        mask = torch.cat([x["mask"] for x in outputs])

        images = np.array(images.cpu())
        mask_data = np.array(mask.cpu())
        preds_data = np.array(preds.cpu())
        self.wb_mask(images, preds_data, mask_data)
        torchvision.utils.save_image(preds, "epoch_preds.png")
        torchvision.utils.save_image(mask, "epoch_mask.png")

    def wb_mask(self, bg_img, pred_mask, true_mask):
        return wandb.Image(
            bg_img,
            masks={
                "prediction": {"mask_data": pred_mask},
                "ground truth": {"mask_data": true_mask},
            },
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class CLSModel(pl.LightningModule):
    def __init__(self, num_classes, lr=0.001):
        super(CLSModel, self).__init__()
        self.lr = lr
        self.num_classes = num_classes

        self.save_hyperparameters()

        self.model = torchvision.models.resnet50(pretrained=False)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, self.num_classes),
        )
        self.criterion = nn.CrossEntropyLoss()
        # metrics

        self.train_accuracy_metric = torchmetrics.Accuracy()
        self.val_accuracy_metric = torchmetrics.Accuracy()
        self.f1_metric = torchmetrics.classification.f_beta.F1Score(
            num_classes=self.num_classes
        )
        self.precision_macro_metric = torchmetrics.Precision(
            average="macro", num_classes=self.num_classes
        )
        self.recall_macro_metric = torchmetrics.Recall(
            average="macro", num_classes=self.num_classes
        )
        self.precision_micro_metric = torchmetrics.Precision(average="micro")
        self.recall_micro_metric = torchmetrics.Recall(average="micro")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.criterion(logits, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)
        preds = torch.argmax(outputs, dim=1)
        loss = self.criterion(outputs, labels)

        # Metrics
        valid_acc = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # Logging metrics
        self.log("valid/loss", loss, prog_bar=True, on_step=True)
        self.log("valid/acc", valid_acc, prog_bar=True, on_step=True)
        self.log("valid/precision_macro", precision_macro, prog_bar=True)
        self.log("valid/recall_macro", recall_macro, prog_bar=True)
        self.log("valid/precision_micro", precision_micro, prog_bar=True)
        self.log("valid/recall_micro", recall_micro, prog_bar=True)
        self.log("valid/f1", f1, prog_bar=True)
        return {"labels": labels, "logits": outputs}

    def validation_epoch_end(self, outputs):
        labels = torch.cat([x["labels"] for x in outputs])
        logits = torch.cat([x["logits"] for x in outputs])

        self.logger.experiment.log(
            {
                "conf": wandb.plot.confusion_matrix(
                    probs=logits.cpu().numpy(), y_true=labels.cpu().numpy()
                )
            }
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
