import os
import numpy as np


fold_dir = (
    "/home/2023csz0012/AmitCode/code/stDiff_fullset/"
    "Result/merFISHBaselineFullset/gimVI/fold_results"
)

n_splits = 5


# ============================================================
# Load all fold-specific valid row indices
# ============================================================

all_fold_indices = []

for fold in range(1, n_splits + 1):

    path = os.path.join(
        fold_dir,
        f"fold_{fold}_row_indices.npy"
    )

    idx = np.load(path)

    idx = np.asarray(
        idx,
        dtype=np.int64
    )

    print(
        f"Fold {fold}: {len(idx)} cells"
    )

    all_fold_indices.append(idx)


# ============================================================
# Intersection across ALL folds
# ============================================================

common_idx = all_fold_indices[0]

for idx in all_fold_indices[1:]:

    common_idx = np.intersect1d(
        common_idx,
        idx
    )


common_idx = np.sort(common_idx)


print("\n======================================")
print("COMMON CELLS")
print("======================================")

print(
    "Number of common cells:",
    len(common_idx)
)

print(
    "First 20 indices:",
    common_idx[:20]
)


# ============================================================
# Save
# ============================================================

save_path = os.path.join(
    fold_dir,
    "common_row_indices.npy"
)

np.save(
    save_path,
    common_idx
)

print("\nSaved:")
print(save_path)