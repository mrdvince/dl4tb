from dataclasses import dataclass
import os
import yaml


@dataclass(repr=True)
class Config:
    name: str
    device: str
    arch: str
    dataloader: str
    data_dir: str
    batch_size: int
    shuffle: bool
    optimizer: str
    loss: str
    metrics: list
    lr_scheduler: str
    step_size: int
    gamma: float
    weight_decay: float
    amsgrad: bool
    monitor: list
    epochs: int = 10
    save_dir: str = "./checkpoints"
    save_period: int = 1
    verbosity: int = 2
    resume: str = False
    lr: float = 0.001
    validation_split: float = 0.2
    num_workers: int = 2
    early_stop: int = 5
    tensorboard: bool = True
    wandb: bool = False
    # wandb_project: str = 'default'
    # wandb_entity: str = 'default'


class LoadConfig:
    def __init__(self, config_path):

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f.read())

    def parse_config(self):
        dataloader = self.config.get("dataloader")
        optimizer = self.config.get("optimizer")
        lr_scheduler = self.config.get("lr_scheduler")
        trainer = self.config.get("trainer")
        config = Config(
            name=self.config.get("name"),
            device=self.config.get("device"),
            arch=self.config.get("arch").get("type"),
            dataloader=dataloader.get("type"),
            data_dir=dataloader.get("data_dir"),
            batch_size=dataloader.get("batch_size"),
            shuffle=dataloader.get("shuffle"),
            optimizer=optimizer.get("type"),
            lr=optimizer.get("lr"),
            loss=self.config.get("loss"),
            metrics=self.config.get("metrics"),
            lr_scheduler=lr_scheduler.get("type"),
            step_size=lr_scheduler.get("step_size"),
            gamma=lr_scheduler.get("gamma"),
            weight_decay=optimizer.get("weight_decay"),
            amsgrad=optimizer.get("amsgrad"),
            monitor=trainer.get("monitor"),
            epochs=trainer.get("epochs"),
            save_dir=trainer.get("save_dir"),
            save_period=trainer.get("save_period"),
            verbosity=trainer.get("verbosity"),
            resume=trainer.get("resume", False),
            validation_split=dataloader.get("validation_split"),
            num_workers=dataloader.get("num_workers"),
            early_stop=trainer.get("early_stop"),
            tensorboard=trainer.get("tensorboard"),
            wandb=trainer.get("wandb"),
        )
        return config


if __name__ == "__main__":
    lc = LoadConfig(os.path.join("config.yaml"))
    conf = lc.parse_config()
    print(conf.wandb)
