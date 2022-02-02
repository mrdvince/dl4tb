import pytorch_lightning as pl
import torch

from data import DataModule
from model import Model

if __name__ == "__main__":
    data: pl.LightningDataModule = DataModule()
    print(data.train_dataloader)
    model = Model(num_classes=2)

    check_point = pl.callbacks.ModelCheckpoint(
        dirpath="saved/checkpoints",
        filename="model_checkpoint_{epoch}",
        save_top_k=3,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, verbose=True, mode="min"
    )
    trainer = pl.Trainer(
        log_every_n_steps=10,
        default_root_dir="saved/",
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=10,
        fast_dev_run=False,
        logger=pl.loggers.TensorBoardLogger("saved/logs/", name="dltb", version=1),
        callbacks=[check_point, early_stopping_callback],
    )
    trainer.fit(model, data)
