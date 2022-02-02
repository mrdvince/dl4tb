from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project="dl4tb", save_dir="saved/")

import pandas as pd
import wandb

from data import DataModule
from model import Model


class SamplesVisualisationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()

        self.datamodule = datamodule

    def on_validation_end(self, trainer, pl_module):
        val_batch = next(iter(self.datamodule.val_dataloader()))
        val_images, val_labels = val_batch
        # get predictions
        outputs = pl_module(val_images)
        preds = torch.argmax(outputs, dim=1)
        # predictions and labels

        df = pd.DataFrame({"preds": preds.numpy(), "labels": val_labels.numpy()})
        # incorrect predictions
        wrong_df = df[df.preds != df.labels]

        # log wrong predictions to wandb as table
        trainer.logger.experiment.log(
            {
                "wrong_preds": wandb.Table(
                    dataframe=wrong_df,
                    columns=["preds", "labels"],
                    allow_mixed_types=True,
                ),
                "global_step": trainer.global_step,
            },
        )


if __name__ == "__main__":
    data: pl.LightningDataModule = DataModule()
    model = Model(num_classes=2)
    # prevent wandb from yelling at us -> the save_dir not existing
    Path("saved").mkdir(exist_ok=True)

    check_point = pl.callbacks.ModelCheckpoint(
        dirpath="saved/checkpoints",
        filename="model_checkpoint_{epoch}",
        save_top_k=3,
        verbose=True,
        monitor="valid/loss",
        mode="min",
    )
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="valid/loss", patience=10, verbose=True, mode="min"
    )
    trainer = pl.Trainer(
        log_every_n_steps=10,
        default_root_dir="saved/",
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=10,
        fast_dev_run=False,
        logger=wandb_logger,  # pl.loggers.TensorBoardLogger("saved/logs/", name="dltb", version=1),
        callbacks=[
            check_point,
            early_stopping_callback,
            SamplesVisualisationLogger(data),
        ],
    )
    trainer.fit(model, data)
