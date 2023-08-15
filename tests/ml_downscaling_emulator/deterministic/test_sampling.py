import cftime
import numpy as np
import pytest
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


@pytest.fixture
def dataset() -> xr.Dataset:
    """Create a dummy Dataset that can be used for sampling."""

    grid_latitude = xr.Variable(["grid_latitude"], np.linspace(-3, 3, 13), attrs={})

    grid_longitude = xr.Variable(["grid_longitude"], np.linspace(-4, 4, 17), attrs={})

    time = xr.Variable(
        ["time"],
        xr.cftime_range(
            cftime.Datetime360Day(1980, 12, 1, 12, 0, 0, 0, has_year_zero=True),
            periods=10,
            freq="D",
        ),
    )
    time_bnds_values = xr.cftime_range(
        cftime.Datetime360Day(1980, 12, 1, 0, 0, 0, 0, has_year_zero=True),
        periods=len(time) + 1,
        freq="D",
    ).values
    time_bnds_pairs = np.concatenate(
        [time_bnds_values[:-1, np.newaxis], time_bnds_values[1:, np.newaxis]], axis=1
    )

    time_bnds = xr.Variable(["time", "bnds"], time_bnds_pairs, attrs={})
    ensemble_member = xr.Variable(["ensemble_member"], np.array(["01", "02", "03"]))

    coords = {
        "ensemble_member": ensemble_member,
        "time": time,
        "grid_latitude": grid_latitude,
        "grid_longitude": grid_longitude,
    }

    data_vars = {
        "linpr": xr.Variable(
            ["ensemble_member", "time", "grid_latitude", "grid_longitude"],
            np.random.rand(
                len(ensemble_member), len(time), len(grid_latitude), len(grid_longitude)
            ),
        ),
        "target_pr": xr.Variable(
            ["ensemble_member", "time", "grid_latitude", "grid_longitude"],
            np.random.rand(
                len(ensemble_member), len(time), len(grid_latitude), len(grid_longitude)
            ),
        ),
        "time_bnds": time_bnds,
    }

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
    )

    return ds
