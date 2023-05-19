from cmethods import CMethods
import numpy as np
import xarray as xr


def qm(sim_train_da, ml_train_da, ml_eval_da):
    values = np.zeros(ml_eval_da.shape, float)
    for ilat in range(len(ml_eval_da["grid_latitude"])):
        for ilon in range(len(ml_eval_da["grid_longitude"])):
            values[:, ilat, ilon] = CMethods.quantile_mapping(
                sim_train_da.isel(grid_latitude=ilat, grid_longitude=ilon),
                ml_train_da.isel(grid_latitude=ilat, grid_longitude=ilon),
                ml_eval_da.isel(grid_latitude=ilat, grid_longitude=ilon),
                n_quantiles=250,
                kind="+",
            )

    qmapped = xr.zeros_like(ml_eval_da)
    qmapped.data = values

    return qmapped


def qm_vec(sim_train_da, ml_train_da, ml_eval_da):
    return (
        xr.apply_ufunc(
            CMethods.quantile_mapping,  # first the function
            sim_train_da,  # now arguments in the order expected by the function
            ml_train_da,
            ml_eval_da,
            kwargs=dict(n_quantiles=250, kind="+"),
            input_core_dims=[
                ["time"],
                ["time"],
                ["time"],
            ],  # list with one entry per arg
            output_core_dims=[["time"]],
            exclude_dims=set(
                ("time",)
            ),  # dimensions allowed to change size. Must be set!
            vectorize=True,
        )
        .transpose("time", "grid_latitude", "grid_longitude")
        .assign_coords(time=ml_eval_da["time"])
    )
