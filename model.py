import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import accuracy_score


class Model(pl.LightningModule):
    def __init__(self, num_classes, lr=0.001):
        super(Model, self).__init__()
        self.lr = lr
        self.num_classes = num_classes

        self.save_hyperparameters()

        self.model = torchvision.models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, self.num_classes),
        )
        self.criterion = nn.CrossEntropyLoss()

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if "v_num" in tqdm_dict:
            del tqdm_dict["v_num"]
        return tqdm_dict

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.criterion(logits, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.criterion(logits, labels)
        _, preds = torch.max(logits, 1)
        val_acc = accuracy_score(labels.cpu(), preds.cpu())
        val_acc = torch.tensor(val_acc, dtype=torch.float)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", val_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
