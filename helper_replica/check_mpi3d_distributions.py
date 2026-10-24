from pathlib import Path
import numpy as np

from src.datasets.disentanglement import MPI3D
from src.helpers import read_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "mpi3d-cbm-res18-diagnostic"
SPLIT_SEED = 42

cfg = read_config(ROOT / "configs/mpi3d" / CONFIG)
data_cfg = cfg["data"].copy()

# Dataset direkt aufbauen; kein DataLoader und kein Modell nötig.
data_cfg.pop("dataset", None)
data_cfg.pop("batch_size", None)

FACTORS = {
    "object_shape": 6,
    "object_color": 6,
    "object_size": 2,
    "camera_height": 3,
    "background_color": 3,
}

for is_train in (True, False):
    dataset = MPI3D(
        train=is_train,
        seed=SPLIT_SEED,
        **data_cfg,
    )

    split = "TRAIN" if is_train else "TEST"
    print(f"\n=== {split}: {len(dataset)} Bilder ===")

    for factor, n_classes in FACTORS.items():
        idx = dataset.all_factors.index(factor)

        counts = np.bincount(
            dataset.labels[:, idx].astype(int),
            minlength=n_classes,
        )

        print(f"{factor}: {counts.tolist()}")