import xarray as xr

from ml_downscaling_emulator.deterministic.sampling import sample_id


def test_sample_id(dataset: xr.Dataset):
    """Ensure the sample_id function creates a set of predictions using the values of the given variable."""

    variable = "linpr"
    em_dataset = dataset.sel(ensemble_member=["01"])
    xr_samples = sample_id(variable, em_dataset)

    assert (xr_samples["pred_pr"].values == em_dataset["linpr"].values).all()
    for dim in ["time", "grid_latitude", "grid_longitude"]:
        assert (xr_samples[dim].values == em_dataset[dim].values).all()
