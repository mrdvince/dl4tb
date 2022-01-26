
import os

import torch
from logger.logger import get_logger

def get_device(config):
    logger = get_logger("train", config.verbosity)

    if config.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            logger.info("No accelerator found, defaulting to using the CPU")

    if config.device == "hpu":
        try:
            from habana_frameworks.torch.utils.library_loader import \
                load_habana_module

            load_habana_module()
            device = "hpu"
        except Exception as e:
            device = "cuda" if torch.cuda.is_available() else "cpu"

    if config.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = "cpu"

    logger.info(f"Using device: {device}")

    return torch.device(device)


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