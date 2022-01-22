import json
import logging
import logging.config
from collections import OrderedDict
from pathlib import Path


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def setup_logging(
    save_dir, log_config="logger/logger.json", default_level=logging.INFO
):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config["handlers"].items():
            if "filename" in handler:
                handler["filename"] = str(save_dir / handler["filename"])

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level, format="%(levelname)s-%(message)s")
        logging.warning(
            "Logging configuration file is not found in {}.".format(log_config)
        )


def get_logger(name, verbosity):
    log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logger = logging.getLogger(name)
    assert (
        verbosity in log_levels
    ), f"Invalid verbosity level {verbosity}. Options are {log_levels.keys()}"
    logger.setLevel(log_levels[verbosity])
    return logger
