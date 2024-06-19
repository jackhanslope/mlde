# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
# ---

# %%
import numpy as np
import torch
from hurricanes import utils
from hurricanes.utils import HURRICANE_DATA_DIR

# %%
i = 0

# %% Data loading
era5_test_dataset = utils.load_dataset("era5", 850, 1, "test")
hur_test_dataset = utils.load_dataset("hurricane", 850, 1, "test")

predicted_wind_field = np.load(
    "output/cncsnpp/2023-10-28T14:00:55UTC/samples/epoch-100/dataset_hurricanes/pixelmmsstanur/val/01/predictions-NXiHAY5jGvshdaCjVKcxDB.npy"
)


# %%
location = hur_test_dataset[i][1]

# %%
location_model = utils.load_model(HURRICANE_DATA_DIR / "runs" / "delta-0-epoch-100")
location_model.eval()

prediction_model = utils.load_model(HURRICANE_DATA_DIR / "runs" / "delta-1-epoch-100")
location_model.eval()
pass

# %%
pred_instance = predicted_wind_field[i]
pred_instance = torch.tensor(pred_instance)
pred_instance = pred_instance.unsqueeze(0)

pred_logits = location_model(pred_instance)

# %%
image_xr, label_xr = era5_test_dataset.get_xr_item(0)
image_arr = image_xr.to_array().values
shape = image_arr.shape

true_instance = era5_test_dataset[i][0]
true_instance = true_instance.view(*shape)
true_instance = true_instance.unsqueeze(0)

true_logits = prediction_model(true_instance)


# %%
true_logits.argmax()

# %%
pred_logits.argmax()

# %%
location

# %% [markdown]
# ---
