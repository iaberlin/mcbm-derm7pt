import random
import itertools
import numpy as np

SEED = 42

factor_names = [
    "object_color",
    "object_shape",
    "object_size",
    "camera_height",
    "background_color",
    "horizontal_axis",
    "vertical_axis",
]

factor_sizes = [6, 6, 2, 3, 3, 40, 40]

task = "object_shape"
concepts = [
    "object_color",
    "object_size",
    "camera_height",
    "background_color",
]
nuisances_task = ["horizontal_axis"]


# Reconstruct labels exactly as MPI3D loader does

labels = np.indices(factor_sizes).reshape(7, -1).T

print("=== RAW DATASET ===")
print("Samples:", len(labels))
print("Expected:", np.prod(factor_sizes))


# Reproduce _filter_samples_nuisances()

random.seed(SEED)

factors_task = concepts + nuisances_task

sizes_task = [
    factor_sizes[factor_names.index(k)]
    for k in factors_task
]

combinations = list(itertools.product(
    *[range(n) for n in sizes_task]
))

random.shuffle(combinations)


def split_into_groups(data, n):
    k, r = divmod(len(data), n)
    return [
        data[i * k + min(i, r):(i + 1) * k + min(i + 1, r)]
        for i in range(n)
    ]


n_task_classes = factor_sizes[factor_names.index(task)]
groups = split_into_groups(combinations, n_task_classes)

groups_task = [
    [i] + list(group)
    for i in range(n_task_classes)
    for group in groups[i]
]

idxs_filter = [
    factor_names.index(i)
    for i in [task] + factors_task
]


def find_row_indices(A, B):
    A_view = np.core.records.fromarrays(A.T)
    B_view = np.core.records.fromarrays(B.T)
    return np.nonzero(np.in1d(A_view, B_view))[0]


keep = find_row_indices(
    labels[:, idxs_filter],
    np.array(groups_task)
)

filtered_labels = labels[keep]

print("\n=== FILTER ===")
print("(c, n_y) combinations:", len(combinations))
print("Allowed (y, c, n_y) combinations:", len(groups_task))
print("Samples after filter:", len(filtered_labels))


# Verify (c, n_y) -> y

y_idx = factor_names.index(task)
cn_idx = [factor_names.index(k) for k in factors_task]

mapping = {}

for row in filtered_labels:
    cn = tuple(row[cn_idx])
    y = int(row[y_idx])

    mapping.setdefault(cn, set()).add(y)

violations = {
    cn: ys
    for cn, ys in mapping.items()
    if len(ys) != 1
}

print("\n=== UNIQUENESS CHECK ===")
print("Unique (c, n_y):", len(mapping))
print("Violations of (c, n_y) -> y:", len(violations))


# Reproduce _select_samples()

n_samples = len(filtered_labels)

sample_idxs = random.sample(
    range(n_samples),
    n_samples
)

n_train = n_samples * 75 // 100

train_idxs = sample_idxs[:n_train]
test_idxs = sample_idxs[n_train:]

print("\n=== SPLIT ===")
print("Train:", len(train_idxs))
print("Test:", len(test_idxs))
print(
    "Train/Test overlap:",
    len(set(train_idxs).intersection(test_idxs))
)


# Task-class distributions

train_y = filtered_labels[train_idxs, y_idx]
test_y = filtered_labels[test_idxs, y_idx]

print("\n=== TASK DISTRIBUTION ===")

for split_name, y in [
    ("train", train_y),
    ("test", test_y),
]:
    values, counts = np.unique(y, return_counts=True)

    print(split_name)
    for value, count in zip(values, counts):
        print(
            f"  shape {value}: "
            f"{count} ({count / len(y):.4f})"
        )

