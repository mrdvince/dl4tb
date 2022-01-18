import shutil
from datetime import datetime
from pathlib import Path

import yaml

from base.config_model import Config


class LoadConfig:
    def __init__(self, config_path):
        self.config_path = config_path

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f.read())

    def parse_config(self):
        dataloader = self.config.get("dataloader")
        optimizer = self.config.get("optimizer")
        lr_scheduler = self.config.get("lr_scheduler")
        trainer = self.config.get("trainer")
        base_save_dir = Path(trainer.get("save_dir"))
        run_id = datetime.now().strftime(r"%m%d_%H%M%S")
        save_dir = base_save_dir / "models" / self.config.get("name") / run_id
        log_dir = base_save_dir / "logs" / self.config.get("name") / run_id
        save_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
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
            save_dir=str(save_dir),
            log_dir=str(log_dir),
            save_period=trainer.get("save_period"),
            verbosity=trainer.get("verbosity"),
            resume=trainer.get("resume", False),
            validation_split=dataloader.get("validation_split"),
            num_workers=dataloader.get("num_workers"),
            early_stop=trainer.get("early_stop"),
            tensorboard=trainer.get("tensorboard"),
            wandb=trainer.get("wandb"),
        )
        shutil.copy(self.config_path, save_dir / "config.yaml")
        return config
