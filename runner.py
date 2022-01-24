import argparse
import logging
import os
from pathlib import Path

import torch

from base.parse_config import LoadConfig
from data_loaders import DataLoader
from logger import logger
from logger.logger import get_logger
from model import criterion
from model import metrics as met  # avoiding name collison
from model import model as arch
from trainer import Trainer


def get_device(config):
    logger = get_logger("train", config.verbosity)

    if config.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            logger.info("No accelerator found, defaulting to using the CPU")
    
    if device == 'hpu':
        try:
            from habana_frameworks.torch.utils.library_loader import load_habana_module
            load_habana_module()
            device = "hpu"
        except Exception as e:
           device = "cuda" if torch.cuda.is_available() else "cpu" 
        
    if device == 'cpu': os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    logger.info(f"Using device: {device}") 
    
    return torch.device(device)

def main(config):
    # dataloaders
    dl = DataLoader(
        config.data_dir,
        config.batch_size,
        config.shuffle,
        config.validation_split,
        config.num_workers,
    )
    train_loader, valid_loader = dl.train_loader, dl.valid_loader
    #  device
    device = get_device(config)
    loss = getattr(criterion, config.loss)
    metrics = [getattr(met, metric) for metric in config.metrics]
    model = getattr(arch, config.arch)(len(train_loader.dataset.classes))
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optim_args = {
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "amsgrad": config.amsgrad,
    }
    optimizer = getattr(torch.optim, config.optimizer)(
        **optim_args, params=trainable_params
    )
    scheduler_args = {
        "step_size": config.step_size,
        "gamma": config.gamma,
    }
    scheduler = getattr(torch.optim.lr_scheduler, config.lr_scheduler)(
        **scheduler_args, optimizer=optimizer
    )

    trainer = Trainer(
        model,
        loss,
        metrics,
        optimizer,
        config,
        device,
        train_loader,
        valid_loader,
        scheduler,
    )
    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Train the model")
    args.add_argument(
        "-c", "--config", type=str, default="config.yaml", help="config file"
    )
    args.add_argument(
        "-m", "--model", type=str, default="None", help="config file"
    )
    
    args = args.parse_args()
    lc = LoadConfig(os.path.join(args.config))
    config = lc.parse_config()
    # update model if name provided
    if args.model != "None":
        config.model = args.model
        
    # setup logger
    logger.setup_logging(Path(config.log_dir))
    try:
        main(config=config)
    except KeyboardInterrupt:
        print("-" * 30)
        print("Exiting from training early")
