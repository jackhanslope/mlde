from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from hurricanes import utils

figure_dir = Path("figures/comparison")
figure_dir.mkdir(exist_ok=True)

# Data loading
hur_test_dataset = utils.load_dataset("hurricane", 850, 1, "test")

era5_test_dataset = utils.load_dataset("era5", 850, 1, "test")


predicted_wind_field = np.load(
    "output/cncsnpp/2023-10-28T14:00:55UTC/samples/epoch-100/dataset_hurricanes/pixelmmsstanur/val/01/predictions-NXiHAY5jGvshdaCjVKcxDB.npy"
)
x_shape = predicted_wind_field.shape[3]
y_shape = predicted_wind_field.shape[2]

# Config
size = 100
many_i = np.random.choice(len(hur_test_dataset), size=size, replace=False)

for count, i in enumerate(many_i):
    plt.figure(figsize=(8, 12))

    # Location
    _, location = hur_test_dataset[i]
    loc_lat, loc_lon = hur_test_dataset.unravel_index(location)

    # Predicted
    u_pred = predicted_wind_field[i, 0]
    v_pred = predicted_wind_field[i, 1]

    x_pred = np.arange(x_shape)
    y_pred = np.arange(y_shape)

    X_pred, Y_pred = np.meshgrid(x_pred, y_pred)

    plt.subplot(312)
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
    plt.scatter(
        loc_lon,
        loc_lat,
        c="C1",
        marker="x",
    )

    plt.title(f"Predicted data for i={i}")

    # True
    _, (u_true, v_true) = era5_test_dataset[i]

    x_true = np.arange(x_shape)
    y_true = np.arange(y_shape)

    X_true, Y_true = np.meshgrid(x_true, y_true)

    plt.subplot(311)
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
    plt.scatter(
        loc_lon,
        loc_lat,
        c="C1",
        marker="x",
    )
    plt.title(f"True data for i={i}")

    u_diff = u_true - u_pred
    v_diff = v_true - v_pred
    plt.subplot(313)
    plt.quiver(
        X_true,
        Y_true,
        u_diff,
        v_diff,
        scale=1000,
        scale_units="width",
        angles="uv",
        cmap="viridis",
    )
    plt.scatter(
        loc_lon,
        loc_lat,
        c="C1",
        marker="x",
    )
    plt.title(f"True data - predicted data for i={i}")

    # Saving and logging
    plt.savefig(f"{figure_dir}/i-{i:04d}.png")
    plt.close()
    print(f"{count + 1}/{size}")
