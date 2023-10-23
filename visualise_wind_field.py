import matplotlib.pyplot as plt
import numpy as np
import torch
from hurricanes.utils import ERA5_DATASETS_DIR

val_dataset = torch.load(ERA5_DATASETS_DIR / "delta-1" / "test_dataset_era5.pt")

predicted_wind_field = np.load(
    "output/cncsnpp/2023-10-18T20:56:02UTC/samples/epoch-100/dataset_hurricanes/pixelmmsstanur/val/01/predictions-9e6SMJpdgxsnnRcfjPt9UY.npy"
)

x_shape = predicted_wind_field.shape[3]
y_shape = predicted_wind_field.shape[2]

# `wind_field` shape is `(num_samples, channels, lat, long) = (1701, 2, 32, 56)`
#
# `channels` is `u200, v200`
#
# `u` is eastward
# `v` is northward

for i in np.random.choice(len(val_dataset), size=10, replace=False):
    plt.figure(figsize=(8, 12))

    u_pred = predicted_wind_field[i, 0]
    v_pred = predicted_wind_field[i, 1]

    x_pred = np.arange(x_shape)
    y_pred = np.arange(y_shape)

    X_pred, Y_pred = np.meshgrid(x_pred, y_pred)

    plt.subplot(211)
    plt.quiver(
        X_pred,
        Y_pred,
        u_pred,
        v_pred,
        scale=1000,
        scale_units="width",
        angles="uv",
        cmap="viridis",
    )

    plt.title(f"Predicted data for i={i}")

    _, (u_true, v_true) = val_dataset[i]

    x_true = np.arange(x_shape)
    y_true = np.arange(y_shape)

    X_true, Y_true = np.meshgrid(x_true, y_true)

    plt.subplot(212)
    plt.quiver(
        X_true,
        Y_true,
        u_true,
        v_true,
        scale=1000,
        scale_units="width",
        angles="uv",
        cmap="viridis",
    )

    plt.title(f"True data for i={i}")

    plt.savefig(f"figures/i-{i}.png")
