    if config.device == "cpu":
        device = "cpu"
    elif config.device == "hpu":
        try:
            from habana_frameworks.torch.utils.library_loader import load_habana_module

            load_habana_module()
            device = "hpu"
        except Exception:
            logging.warning(
                "Habana module not found, checking for other acceptable devices"
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                logging.info("No accelerator found, defaulting to using the CPU")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            logging.info("No accelerator found, defaulting to using the CPU")
    device = torch.device(device)
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

def get_logger(name, verbosity):
    log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logger = logging.getLogger(name)
    assert (
        verbosity in log_levels
    ), f"Invalid verbosity level {verbosity}. Options are {log_levels.keys()}"
    logger.setLevel(log_levels[verbosity])
    return logger

