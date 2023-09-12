import numpy as np
import pytest

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
def sim_train_da(dataset_factory):
    return dataset_factory(time_len=2000).sel(ensemble_member=["01"])["target_pr"]


@pytest.fixture
def ml_train_da(samples_factory):
    return samples_factory(time_len=2000)["pred_pr"]


@pytest.fixture
def ml_eval_da(samples_factory):
    return samples_factory(start_year=2060, time_len=1000)["pred_pr"]
