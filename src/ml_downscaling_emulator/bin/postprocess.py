import logging
import os
from pathlib import Path
import typer
import xarray as xr

from mlde_utils import samples_path, samples_glob, TIME_PERIODS

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

    for sample_filepath in samples_glob(
        samples_path(
            workdir,
            checkpoint=checkpoint,
            input_xfm=input_xfm,
            dataset=dataset,
            split=split,
            ensemble_member=ensemble_member,
        )
    ):
        samples_ds = xr.open_dataset(sample_filepath)

        filtered_samples_filepath = filtered_samples_dirpath / sample_filepath.name

        samples_ds.sel(time=slice(*TIME_PERIODS[time_period])).to_netcdf(
            filtered_samples_filepath
        )
