from typing import List
from codetiming import Timer
import glob
import logging
import os
from pathlib import Path
from knockknock import slack_sender
import shortuuid
import torch
import typer
import xarray as xr
import yaml

from mlde_utils import samples_path, DEFAULT_ENSEMBLE_MEMBER
from mlde_utils.training.dataset import load_raw_dataset_split
from ..deterministic import sampling
from ..deterministic.utils import restore_checkpoint
from ..torch import get_dataloader

from ..unet import unet

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(filename)s - %(asctime)s - %(message)s",
)
logger = logging.getLogger()
logger.setLevel("INFO")

app = typer.Typer()


@app.callback()
def callback():
    pass


@app.command()
@Timer(name="sample", text="{name}: {minutes:.1f} minutes", logger=logging.info)
@slack_sender(webhook_url=os.getenv("KK_SLACK_WH_URL"), channel="general")
def sample(
    workdir: Path,
    dataset: str = typer.Option(...),
    epoch: int = typer.Option(...),
    batch_size: int = typer.Option(...),
    num_samples: int = 1,
    input_transform_key: str = None,
    ensemble_member: str = DEFAULT_ENSEMBLE_MEMBER,
):

    config_path = os.path.join(workdir, "config.yml")
    logger.info(f"Loading config from {config_path}")
    with open(config_path) as f:
        config = yaml.unsafe_load(f)

    split = "val"

    if input_transform_key is not None:
        config["input_transform_key"] = input_transform_key

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    output_dirpath = samples_path(
        workdir=workdir,
        checkpoint=f"epoch-{epoch}",
        dataset=dataset,
        input_xfm=config["input_transform_key"],
        split=split,
        ensemble_member=ensemble_member,
    )
    os.makedirs(output_dirpath, exist_ok=True)

    transform_dir = os.path.join(workdir, "transforms")

    eval_dl, _, target_transform = get_dataloader(
        dataset,
        config.data.dataset_name,
        config.data.input_transform_key,
        config.data.target_transform_key,
        transform_dir,
        split=split,
        ensemble_members=[ensemble_member],
        evaluation=True,
        batch_size=config.eval.batch_size,
        shuffle=False,
    )

    ckpt_filename = os.path.join(workdir, "checkpoints", f"epoch_{epoch}.pth")
    num_predictors = eval_dl.dataset[0][0].shape
    model = unet.UNet(num_predictors, 1).to(device=device)
    optimizer = torch.optim.Adam(model.parameters())
    state = dict(step=0, epoch=0, optimizer=optimizer, model=model)
    state, loaded = restore_checkpoint(ckpt_filename, state, device)
    assert loaded, "Did not load state from checkpoint"

    for sample_id in range(num_samples):
        typer.echo(f"Sample run {sample_id}...")
        xr_samples = sampling.sample(state["model"], eval_dl, target_transform)

        output_filepath = os.path.join(
            output_dirpath, f"predictions-{shortuuid.uuid()}.nc"
        )

        logger.info(f"Saving predictions to {output_filepath}")
        os.makedirs(output_dirpath, exist_ok=True)
        xr_samples.to_netcdf(output_filepath)


@app.command()
@Timer(name="sample", text="{name}: {minutes:.1f} minutes", logger=logging.info)
@slack_sender(webhook_url=os.getenv("KK_SLACK_WH_URL"), channel="general")
def sample_id(
    workdir: Path,
    dataset: str = typer.Option(...),
    variable: str = "pr",
    split: str = "val",
    ensemble_member: str = "01",
):

    output_dirpath = samples_path(
        workdir=workdir,
        checkpoint=f"epoch-0",
        dataset=dataset,
        input_xfm="none",
        split=split,
        ensemble_member=ensemble_member,
    )
    os.makedirs(output_dirpath, exist_ok=True)

    eval_ds = load_raw_dataset_split(dataset, split).sel(
        ensemble_member=[ensemble_member]
    )
    samples = eval_ds[variable].values
    predictions = sampling.np_samples_to_xr(samples, eval_ds, target_transform=None)

    output_filepath = os.path.join(output_dirpath, f"predictions-{shortuuid.uuid()}.nc")

    logger.info(f"Saving predictions to {output_filepath}")
    predictions.to_netcdf(output_filepath)


@app.command()
def merge(
    input_dirs: List[Path],
    output_dir: Path,
):
    pred_file_globs = [
        glob.glob(os.path.join(samples_dir, "*.nc")) for samples_dir in input_dirs
    ]
    # there should be the same number of samples in each input dir
    assert 1 == len(set(map(len, pred_file_globs)))

    for pred_file_group in zip(*pred_file_globs):
        typer.echo(f"Concat {pred_file_group}")

        # take a bit of the random id in each sample file's name
        random_ids = [fn[-25:-20] for fn in pred_file_group]
        # join those partial random ids together for the output filepath in the train directory (rather than one of the subset train dirs)
        output_filepath = os.path.join(
            output_dir, f"predictions-{'-'.join(random_ids)}.nc"
        )

        typer.echo(f"save to {output_filepath}")
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        xr.concat([xr.open_dataset(f) for f in pred_file_group], dim="time").to_netcdf(
            output_filepath
        )
