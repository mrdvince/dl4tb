import random

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger

from data import ClassifierDataModule, UNETDataModule
from model import Model, UNETModel


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


def permute_params(model, to_filters_last, lazy_mode):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim == 4:
                if to_filters_last:
                    param.data = param.data.permute(
                        (2, 3, 1, 0)
                    )  # permute KCRS to RSCK
                else:
                    param.data = param.data.permute(
                        (3, 2, 0, 1)
                    )  # permute RSCK to KCRS
    if lazy_mode:
        import habana_frameworks.torch.core as htcore

        htcore.mark_step()


def permute_momentum(optimizer, to_filters_last, lazy_mode):
    # Permute the momentum buffer before using for checkpoint
    for group in optimizer.param_groups:
        for p in group["params"]:
            param_state = optimizer.state[p]
            if "momentum_buffer" in param_state:
                buf = param_state["momentum_buffer"]
                if buf.ndim == 4:
                    if to_filters_last:
                        buf = buf.permute((2, 3, 1, 0))
                    else:
                        buf = buf.permute((3, 2, 0, 1))
                    param_state["momentum_buffer"] = buf

    if lazy_mode:
        import habana_frameworks.torch.core as htcore

        htcore.mark_step()


@hydra.main(config_path="configs", config_name="config")
def main(cfg):
    data: pl.LightningDataModule = ClassifierDataModule(config=cfg)
    model = Model(num_classes=cfg.model.num_classes)
    unet = UNETModel(lr=cfg.model.lr)
    unet_data = UNETDataModule()
    check_point = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="model_checkpoint_{epoch}",
        save_top_k=3,
        verbose=True,
        monitor="valid/loss",
        mode="min",
        save_on_train_epoch_end=True,
    )
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="valid/loss", patience=10, verbose=True, mode="min"
    )
    wandb_logger = WandbLogger(
        project="dl4tb", save_dir=hydra.utils.get_original_cwd() + "/saved/"
    )
    if cfg.training.device == "hpu":
        try:
            from habana_frameworks.torch.utils.library_loader import (
                load_habana_module,
            )  # type: ignore

            load_habana_module()  # type: ignore
            import habana_frameworks.torch.core  # type: ignore
            import habana_frameworks.torch.core.hccl  # type: ignore

            num_instances = cfg.training.num_instances
            parallel_hpus = [torch.device("hpu")] * num_instances
            hpus = True
            device = torch.device("hpu")
            import habana_frameworks.torch.core as htcore

            htcore.mark_step()

        except ModuleNotFoundError:
            device = torch.device("cpu")
            hpus = False
    trainer = pl.Trainer(
        max_epochs=1,
        fast_dev_run=True,
        default_root_dir=cfg.training.save_dir,
        deterministic=cfg.training.deterministic,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
    )

    trainer.fit(unet, unet_data)
    # trainer = pl.Trainer(
    #     # hpus=num_instances if hpus else None,
    #     gpus=(1 if torch.cuda.is_available() else 0),
    #     # strategy=pl.plugins.DDPPlugin(
    #     #     parallel_devices=parallel_hpus,
    #     #     bucket_cap_mb=cfg.training.bucket_cap_mb,
    #     #     gradient_as_bucket_view=True,
    #     #     static_graph=True,
    #     # )
    #     # if num_instances > 1
    #     # else None,
    #     log_every_n_steps=cfg.training.log_every_n_steps,
    #     default_root_dir=cfg.training.save_dir,
    #     max_epochs=cfg.training.max_epochs,
    #     fast_dev_run=False,
    #     logger=wandb_logger,
    #     callbacks=[
    #         check_point,
    #         early_stopping_callback,
    #         SamplesVisualisationLogger(data),
    #     ],
    #     deterministic=cfg.training.deterministic,
    #     limit_train_batches=cfg.training.limit_train_batches,
    #     limit_val_batches=cfg.training.limit_val_batches,
    # )
    # trainer.fit(model, data)


if __name__ == "__main__":
    seed = 69420
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    pl.seed_everything(seed)
    main()
