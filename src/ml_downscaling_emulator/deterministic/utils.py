import logging
import os


def restore_checkpoint(ckpt_dir, state, device):
    import torch

    if not os.path.exists(ckpt_dir):
        os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
        logging.warning(
            f"No checkpoint found at {ckpt_dir}. " f"Returned the same state as input"
        )
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state["optimizer"].load_state_dict(loaded_state["optimizer"])
        state["model"].load_state_dict(loaded_state["model"], strict=False)
        state["step"] = loaded_state["step"]
        state["epoch"] = loaded_state["epoch"]
        return state


def save_checkpoint(ckpt_dir, state):
    import torch

    saved_state = {
        "optimizer": state["optimizer"].state_dict(),
        "model": state["model"].state_dict(),
        "step": state["step"],
        "epoch": state["epoch"],
    }
    torch.save(saved_state, ckpt_dir)
