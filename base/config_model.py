from dataclasses import dataclass


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
    log_dir: str = "./log_dir"
    save_period: int = 1
    verbosity: int = 2
    resume: str = False
    lr: float = 0.001
    validation_split: float = 0.2
    num_workers: int = 2
    early_stop: int = 5
    tensorboard: bool = True
    wandb: bool = False
