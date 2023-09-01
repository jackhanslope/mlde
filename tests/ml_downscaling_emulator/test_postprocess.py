import numpy as np
import pytest
import xarray as xr

from ml_downscaling_emulator.postprocess import xrqm, qm_1d_dom_aware


def test_qm_applies_qm_at_each_gridbox(sim_train_da, ml_train_da, ml_eval_da):
    qm_ml_eval_da = xrqm(sim_train_da, ml_train_da, ml_eval_da)

    for ilat in range(len(ml_eval_da["grid_latitude"])):
        for ilon in range(len(ml_eval_da["grid_longitude"])):
            exp_value = qm_1d_dom_aware(
                sim_train_da.isel(grid_latitude=ilat, grid_longitude=ilon),
                ml_train_da.isel(grid_latitude=ilat, grid_longitude=ilon),
                ml_eval_da.isel(grid_latitude=ilat, grid_longitude=ilon),
                n_quantiles=250,
            )

            np.testing.assert_allclose(
                exp_value, qm_ml_eval_da.isel(grid_latitude=ilat, grid_longitude=ilon)
            )


def test_all_train_qm_match_sim_quantiles(sim_train_da, ml_train_da):
    qm_ml_train_da = xrqm(sim_train_da, ml_train_da, ml_train_da)

    np.testing.assert_allclose(
        sim_train_da.quantile([0.1, 0.25, 0.5, 0.75, 0.9], dim="time"),
        qm_ml_train_da.quantile([0.1, 0.25, 0.5, 0.75, 0.9], dim="time"),
        rtol=5e-2,
    )


def test_all_train_qm_match_sim_histogram(sim_train_da, ml_train_da):
    qm_ml_train_da = xrqm(sim_train_da, ml_train_da, ml_train_da)

    sim_ns, bins = np.histogram(sim_train_da, range=(-5.0, 5.0), bins=20)
    qm_ml_ns, bins = np.histogram(qm_ml_train_da, bins=bins)

    np.testing.assert_allclose(sim_ns, qm_ml_ns, atol=200)

    np.testing.assert_allclose(np.abs(sim_ns - qm_ml_ns).sum(), 0.0, atol=500)


@pytest.fixture
def time_range():
    return np.linspace(-2, 2, 20000)


@pytest.fixture
def lat_range():
    return np.linspace(-2, 2, 7)


@pytest.fixture
def lon_range():
    return np.linspace(-2, 2, 3)


@pytest.fixture
def sim_train_da(time_range, lat_range, lon_range):
    rng = np.random.default_rng()
    return xr.DataArray(
        data=rng.normal(
            loc=1.0, size=(len(time_range), len(lat_range), len(lon_range))
        ),
        dims=["time", "grid_latitude", "grid_longitude"],
        name="target_pr",
        coords=dict(
            time=(["time"], time_range),
            grid_latitude=(["grid_latitude"], lat_range),
            grid_longitude=(["grid_longitude"], lon_range),
        ),
    )


@pytest.fixture
def ml_train_da(time_range, lat_range, lon_range):
    rng = np.random.default_rng()
    return xr.DataArray(
        data=rng.normal(size=(len(time_range), len(lat_range), len(lon_range))),
        dims=["time", "grid_latitude", "grid_longitude"],
        name="pred_pr",
        coords=dict(
            time=(["time"], time_range),
            grid_latitude=(["grid_latitude"], lat_range),
            grid_longitude=(["grid_longitude"], lon_range),
        ),
    )


@pytest.fixture
def ml_eval_da(lat_range, lon_range):
    eval_time_range = np.linspace(3, 4, 50)
    rng = np.random.default_rng()
    return xr.DataArray(
        data=rng.normal(size=(len(eval_time_range), len(lat_range), len(lon_range))),
        dims=["time", "grid_latitude", "grid_longitude"],
        name="pred_pr",
        coords=dict(
            time=(["time"], eval_time_range),
            grid_latitude=(["grid_latitude"], lat_range),
            grid_longitude=(["grid_longitude"], lon_range),
        ),
    )
