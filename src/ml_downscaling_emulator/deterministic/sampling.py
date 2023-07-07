import numpy as np
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import xarray as xr


def generate_np_samples(model, cond_batch):
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cond_batch = cond_batch.to(device)

    samples = model(cond_batch)
    # drop the feature channel dimension (only have target pr as output)
    samples = samples.squeeze(dim=1)
    # extract numpy array
    samples = samples.cpu().detach().numpy()
    return samples


def np_samples_to_xr(np_samples, coords, target_transform, cf_data_vars):
    coords = {**dict(coords)}

    pred_pr_dims = ["ensemble_member", "time", "grid_latitude", "grid_longitude"]
    pred_pr_attrs = {
        "grid_mapping": "rotated_latitude_longitude",
        "standard_name": "pred_pr",
        "units": "kg m-2 s-1",
    }
    # add ensemble member axis to np samples
    np_samples = np_samples[np.newaxis, :]
    pred_pr_var = (pred_pr_dims, np_samples, pred_pr_attrs)

    data_vars = {**cf_data_vars, "target_pr": pred_pr_var}

    pred_ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs={})

    if target_transform is not None:
        pred_ds = target_transform.invert(pred_ds)

    pred_ds = pred_ds.rename({"target_pr": "pred_pr"})

    return pred_ds


def sample(model, eval_dl, target_transform):
    cf_data_vars = {
        key: eval_dl.dataset.ds.data_vars[key]
        for key in [
            "rotated_latitude_longitude",
            "time_bnds",
            "grid_latitude_bnds",
            "grid_longitude_bnds",
        ]
    }
    preds = []
    with logging_redirect_tqdm():
        with tqdm(
            total=len(eval_dl.dataset), desc=f"Sampling", unit=" timesteps"
        ) as pbar:
            with torch.no_grad():
                for cond_batch, _, time_batch in eval_dl:
                    coords = eval_dl.dataset.ds.sel(time=time_batch).coords
                    batch_np_samples = generate_np_samples(model, cond_batch)

                    xr_samples = np_samples_to_xr(
                        batch_np_samples, coords, target_transform, cf_data_vars
                    )
                    preds.append(xr_samples)

                    pbar.update(cond_batch.shape[0])

    ds = xr.combine_by_coords(
        preds,
        compat="no_conflicts",
        combine_attrs="drop_conflicts",
        coords="all",
        join="inner",
        data_vars="all",
    )

    return ds
