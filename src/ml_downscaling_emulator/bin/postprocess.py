import logging
import os
from pathlib import Path
import typer
import xarray as xr

from mlde_utils import samples_path, samples_glob, TIME_PERIODS
from mlde_utils.training.dataset import open_raw_dataset_split

from ml_downscaling_emulator.postprocess import xrqm

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
def filter(
    workdir: Path,
    dataset: str = typer.Option(...),
    time_period: str = typer.Option(...),
    checkpoint: str = typer.Option(...),
    input_xfm: str = "stan",
    split: str = "val",
    ensemble_member: str = typer.Option(...),
):
    """Filter a set of samples based on time period."""

    new_dataset = f"{dataset}-{time_period}"
    filtered_samples_dirpath = samples_path(
        workdir,
        checkpoint=checkpoint,
        input_xfm=input_xfm,
        dataset=new_dataset,
        split=split,
        ensemble_member=ensemble_member,
    )
    os.makedirs(filtered_samples_dirpath, exist_ok=False)

    samples_filepaths_to_filter = samples_path(
        workdir,
        checkpoint=checkpoint,
        input_xfm=input_xfm,
        dataset=dataset,
        split=split,
        ensemble_member=ensemble_member,
    )

    logger.info(f"Found for filtering: {samples_filepaths_to_filter}")
    for sample_filepath in samples_glob(samples_filepaths_to_filter):
        logger.info(f"Working on {sample_filepath}")
        samples_ds = xr.open_dataset(sample_filepath)

        filtered_samples_filepath = filtered_samples_dirpath / sample_filepath.name

        logger.info(f"Saving to {filtered_samples_filepath}")
        samples_ds.sel(time=slice(*TIME_PERIODS[time_period])).to_netcdf(
            filtered_samples_filepath
        )


@app.command()
def qm(
    workdir: Path,
    checkpoint: str = typer.Option(...),
    train_dataset: str = typer.Option(...),
    train_input_xfm: str = "stan",
    eval_dataset: str = typer.Option(...),
    eval_input_xfm: str = "stan",
    split: str = "val",
    ensemble_member: str = typer.Option(...),
):
    pass
    # to compute the mapping, use train split data

    # open train split of dataset for the target_pr
    sim_train_da = open_raw_dataset_split(train_dataset, "train").sel(
        ensemble_member=ensemble_member
    )["target_pr"]

    # open sample of model from train split
    ml_train_da = xr.open_dataset(
        list(
            samples_glob(
                samples_path(
                    workdir,
                    checkpoint=checkpoint,
                    input_xfm=train_input_xfm,
                    dataset=train_dataset,
                    split="train",
                    ensemble_member=ensemble_member,
                )
            )
        )[0]
    )["pred_pr"]

    ml_eval_samples_dirpath = samples_path(
        workdir,
        checkpoint=checkpoint,
        input_xfm=eval_input_xfm,
        dataset=eval_dataset,
        split=split,
        ensemble_member=ensemble_member,
    )
    logger.info(f"QMapping samplesin {ml_eval_samples_dirpath}")
    for sample_filepath in samples_glob(ml_eval_samples_dirpath):
        logger.info(f"Working on {sample_filepath}")
        # open the samples to be qmapped
        ml_eval_ds = xr.open_dataset(sample_filepath)

        # do the qmapping
        qmapped_eval_da = xrqm(sim_train_da, ml_train_da, ml_eval_ds["pred_pr"])

        qmapped_eval_ds = ml_eval_ds.copy()
        qmapped_eval_ds["pred_pr"] = qmapped_eval_da

        # save output
        new_workdir = workdir / "postprocess" / "qm"

        qmapped_sample_filepath = (
            samples_path(
                new_workdir,
                checkpoint=checkpoint,
                input_xfm=eval_input_xfm,
                dataset=eval_dataset,
                split=split,
                ensemble_member=ensemble_member,
            )
            / sample_filepath.name
        )

        logger.info(f"Saving to {qmapped_sample_filepath}")
        qmapped_sample_filepath.parent.mkdir(parents=True, exist_ok=True)
        qmapped_eval_ds.to_netcdf(qmapped_sample_filepath)
