import cftime
import numpy as np
import pytest
import xarray as xr

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


@pytest.fixture
def samples_set() -> xr.Dataset:
    """Create a dummy Dataset that looks like a set of samples from the emulator."""

    ensemble_member = xr.Variable(["ensemble_member"], np.array(["01"]))

    coords = {
        "ensemble_member": ensemble_member,
        "time": time,
        "grid_latitude": grid_latitude,
        "grid_longitude": grid_longitude,
    }

    data_vars = {
        "pred_pr": xr.Variable(
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


@pytest.fixture
def dataset() -> xr.Dataset:
    """Create a dummy Dataset representing a split of a set of data for training and sampling."""

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
