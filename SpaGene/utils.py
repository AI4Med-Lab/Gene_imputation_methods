"""SpaGene data preprocessing, batch loading, and benchmark metrics."""
import os
import random
import warnings
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from scipy.stats import pearsonr, spearmanr, ConstantInputWarning
from torch.utils.data import DataLoader, Dataset

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dense_float32(x):
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def matrix_min(x):
    if sp.issparse(x):
        if x.nnz == 0:
            return 0.0
        # Sparse matrices implicitly contain zeros.
        return float(min(0.0, np.min(x.data)))
    return float(np.min(np.asarray(x)))


def safe_corr(func, a, b):
    """Robust correlation helper used only for SpaGene checkpoint selection.

    This does NOT replace compute_metrics(). It is used by visible_gene_corr()
    during translator training to score the visible/train genes.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    if a.size < 2 or b.size < 2:
        return 0.0
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConstantInputWarning)
        try:
            value, _ = func(a, b)
        except (ConstantInputWarning, ValueError, FloatingPointError):
            return 0.0

    return float(value) if np.isfinite(value) else 0.0


def apply_transform(adata: ad.AnnData, transform: str) -> ad.AnnData:
    """Return a transformed copy. 'none' means use X exactly as stored."""
    out = adata.copy()

    if transform == "none":
        return out

    if matrix_min(out.X) < 0:
        raise ValueError(
            f"Transform '{transform}' requires non-negative expression values, "
            "but negative values were found."
        )

    if transform == "sqrt":
        if sp.issparse(out.X):
            out.X = out.X.copy()
            out.X.data = np.sqrt(out.X.data).astype(np.float32, copy=False)
        else:
            out.X = np.sqrt(np.asarray(out.X, dtype=np.float32))
        return out

    if transform == "log1p":
        sc.pp.normalize_total(out, target_sum=1e4)
        sc.pp.log1p(out)
        return out

    raise ValueError(f"Unknown transform: {transform}")


def filter_nonempty(adata: ad.AnnData, label: str) -> ad.AnnData:
    """Use the same min-count filtering convention when data are non-negative."""
    out = adata.copy()
    if matrix_min(out.X) < 0:
        print(f"[warning] {label}: negative values detected; skipping count-based filter_cells/filter_genes.")
        return out
    sc.pp.filter_cells(out, min_counts=1)
    sc.pp.filter_genes(out, min_counts=1)
    return out


def align_common_genes(adata_spatial, adata_seq, uppercase=False):
    if not adata_spatial.var_names.is_unique:
        print("[warning] ST var_names not unique; making unique.")
        adata_spatial.var_names_make_unique()
    if not adata_seq.var_names.is_unique:
        print("[warning] scRNA var_names not unique; making unique.")
        adata_seq.var_names_make_unique()
    if not adata_spatial.obs_names.is_unique:
        print("[warning] ST obs_names not unique; making unique.")
        adata_spatial.obs_names_make_unique()
    if not adata_seq.obs_names.is_unique:
        print("[warning] scRNA obs_names not unique; making unique.")
        adata_seq.obs_names_make_unique()

    if uppercase:
        adata_spatial.var_names = adata_spatial.var_names.astype(str).str.strip().str.upper()
        adata_seq.var_names = adata_seq.var_names.astype(str).str.strip().str.upper()
        if not adata_spatial.var_names.is_unique:
            adata_spatial.var_names_make_unique()
        if not adata_seq.var_names.is_unique:
            adata_seq.var_names_make_unique()

    # seq_gene_set = set(adata_seq.var_names)
    # common_genes = np.asarray(
    #     [g for g in adata_spatial.var_names if g in seq_gene_set], dtype=object
    # )

    common_genes = np.intersect1d(
    np.asarray(adata_spatial.var_names, dtype=str),
    np.asarray(adata_seq.var_names, dtype=str)
    )

    if len(common_genes) == 0:
        st_upper = {str(x).upper() for x in adata_spatial.var_names}
        sc_upper = {str(x).upper() for x in adata_seq.var_names}
        ci_overlap = len(st_upper.intersection(sc_upper))
        msg = "No exact common genes between ST and scRNA-seq."
        if ci_overlap > 0 and not uppercase:
            msg += (
                f" Found {ci_overlap} case-insensitive matches; rerun with --uppercase_genes."
            )
        raise ValueError(msg)

    return (
        adata_spatial[:, common_genes].copy(),
        adata_seq[:, common_genes].copy(),
        common_genes,
    )


class MatrixDataset(Dataset):
    def __init__(self, x):
        self.x = np.ascontiguousarray(x, dtype=np.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx])


def make_loader(x, batch_size, shuffle, drop_last):
    if x.shape[0] < 2:
        raise ValueError("SpaGene BatchNorm requires at least 2 cells per domain.")
    bs = min(int(batch_size), int(x.shape[0]))
    # Avoid a final 1-cell batch because BatchNorm cannot train on it.
    if drop_last and x.shape[0] % bs == 1 and bs > 2:
        bs -= 1
    return DataLoader(
        MatrixDataset(x),
        batch_size=bs,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
