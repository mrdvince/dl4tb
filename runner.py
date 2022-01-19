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

def get_logger(name, verbosity):
    log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logger = logging.getLogger(name)
    assert (
        verbosity in log_levels
    ), f"Invalid verbosity level {verbosity}. Options are {log_levels.keys()}"
    logger.setLevel(log_levels[verbosity])
    return logger

