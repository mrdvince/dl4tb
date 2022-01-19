
def get_logger(name, verbosity):
    log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logger = logging.getLogger(name)
    assert (
        verbosity in log_levels
    ), f"Invalid verbosity level {verbosity}. Options are {log_levels.keys()}"
    logger.setLevel(log_levels[verbosity])
    return logger

