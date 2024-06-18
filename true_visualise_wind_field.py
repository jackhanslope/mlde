from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from hurricanes import utils
from hurricanes.hurricane_dataset import HurricaneDataset

figures_path = Path("figures/wind-850/delta-0")
figures_path.mkdir(parents=True, exist_ok=True)

# Data loading
test_dataset: HurricaneDataset = utils.load_dataset("hurricane", 850, 0, "test")

lat_shape, lon_shape = test_dataset.shape

# Config
size = 100
many_i = np.random.choice(len(test_dataset), size=size, replace=False)

for count, i in enumerate(many_i):
    plt.figure(figsize=(5.6, 3.2))

    plt.axis("equal")

    (u, v), location = test_dataset[i]

    u = u[-1]
    v = v[-1]

    lat = np.arange(lat_shape)
    lon = np.arange(lon_shape)

    lon, lat = np.meshgrid(lon, lat)

    plt.quiver(
        lon,
        lat,
        u,
        v,
        scale=1000,
        scale_units="width",
        angles="uv",
        cmap="viridis",
    )

    loc_lat, loc_lon = test_dataset.unravel_index(location)

    plt.scatter(
        loc_lon,
        loc_lat,
        c="C1",
        marker="x",
    )

    plt.title(f"True data for i={i}")

    # Saving and logging
    plt.savefig(figures_path / f"i-{i:04d}.png")
    plt.close()
    print(f"{count + 1}/{size}")
