import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt


NPZ_PATH = "/pfs/work9/workspace/scratch/ma_iadam-mcbm/datasets/mpi3d/mpi3d_real.npz"
OUT_PATH = "helper_replica/mpi3d_factor_order_audit.png"

FACTOR_NAMES = [
    "object_color",
    "object_shape",
    "object_size",
    "camera_height",
    "background_color",
    "horizontal_axis",
    "vertical_axis",
]

FACTOR_SIZES = [6, 6, 2, 3, 3, 40, 40]



# Get images.npy without loading the 11+ GB array into RAM

extract_dir = os.path.dirname(NPZ_PATH)
npy_path = os.path.join(extract_dir, "images.npy")

if not os.path.exists(npy_path):
    print("Extracting images.npy from NPZ once...")
    with zipfile.ZipFile(NPZ_PATH, "r") as zf:
        zf.extract("images.npy", path=extract_dir)

print("Opening images.npy with mmap...")
images = np.load(npy_path, mmap_mode="r")

print("Image shape:", images.shape)
print("Image dtype:", images.dtype)


# ---------------------------------------------------------------------
# Convert factor tuple -> flat MPI3D index
#
# This is exactly the ordering assumed by:
#
# np.indices(FACTOR_SIZES).reshape(7, -1).T
# ---------------------------------------------------------------------

def factor_to_index(values):
    return np.ravel_multi_index(tuple(values), FACTOR_SIZES)


# Keep all factors at a controlled baseline.
# Positions are chosen near the middle so movement is easy to see.
baseline = [
    0,   # object_color
    0,   # object_shape
    0,   # object_size
    0,   # camera_height
    0,   # background_color
    20,  # horizontal_axis
    20,  # vertical_axis
]

# For each row:
# left  = baseline
# right = exactly ONE factor changed
comparison_values = {
    "object_color": 1,
    "object_shape": 1,
    "object_size": 1,
    "camera_height": 1,
    "background_color": 1,
    "horizontal_axis": 30,
    "vertical_axis": 30,
}


fig, axes = plt.subplots(
    nrows=len(FACTOR_NAMES),
    ncols=2,
    figsize=(7, 21)
)

print("\n=== FACTOR ORDER AUDIT ===")

for row, factor_name in enumerate(FACTOR_NAMES):

    factor_idx = FACTOR_NAMES.index(factor_name)

    left_values = baseline.copy()
    right_values = baseline.copy()
    right_values[factor_idx] = comparison_values[factor_name]

    left_index = factor_to_index(left_values)
    right_index = factor_to_index(right_values)

    print(f"\n{factor_name}")
    print(f"  baseline index : {left_index}")
    print(f"  changed index  : {right_index}")
    print(f"  baseline factors: {left_values}")
    print(f"  changed factors : {right_values}")

    axes[row, 0].imshow(images[left_index])
    axes[row, 0].set_title(
        f"baseline\nindex={left_index}",
        fontsize=9
    )
    axes[row, 0].axis("off")

    axes[row, 1].imshow(images[right_index])
    axes[row, 1].set_title(
        f"{factor_name}\n"
        f"{baseline[factor_idx]} -> {comparison_values[factor_name]}\n"
        f"index={right_index}",
        fontsize=9
    )
    axes[row, 1].axis("off")

    axes[row, 0].set_ylabel(
        factor_name,
        fontsize=10
    )

plt.suptitle(
    "MPI3D factor-order audit\n"
    "Only the named factor changes within each row",
    fontsize=14
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT_PATH, dpi=160)
plt.close()

print("\nSaved:")
print(OUT_PATH)