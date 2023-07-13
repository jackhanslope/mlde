import logging
import os
import torch.nn as nn

from ..unet import unet


def create_model(config, num_predictors):
    if config.model.name == "u-net":
        return unet.UNet(num_predictors, 1)
    if config.model.name == "debug":
        return nn.Conv2d(num_predictors, 1, 3, stride=1, padding=1)
    raise NotImplementedError(f"Model {config.model.name} not supported yet!")


def restore_checkpoint(ckpt_dir, state, device):
    import torch

    if not os.path.exists(ckpt_dir):
        os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
        logging.warning(
            f"No checkpoint found at {ckpt_dir}." f"Returned the same state as input"
        )
        return state, False
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state["optimizer"].load_state_dict(loaded_state["optimizer"])
        state["model"].load_state_dict(loaded_state["model"], strict=False)
        state["step"] = loaded_state["step"]
        state["epoch"] = loaded_state["epoch"]
        logging.info(
            f"Checkpoint found at {ckpt_dir}. "
            f"Returned the state from {state['epoch']}/{state['step']}"
        )
        return state, True


def save_checkpoint(ckpt_dir, state):
    import torch

    saved_state = {
        "optimizer": state["optimizer"].state_dict(),
        "model": state["model"].state_dict(),
        "step": state["step"],
        "epoch": state["epoch"],
    }
    torch.save(saved_state, ckpt_dir)
