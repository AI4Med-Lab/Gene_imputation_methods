"""Modular PASTA gene-holdout benchmarking implementation."""
from __future__ import annotations

import logging
from typing import List, Tuple
import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

LOGGER = logging.getLogger("PASTA-baseline")


def extract_coords(adata_sp: ad.AnnData, coord_key: str) -> np.ndarray:
    if coord_key != "auto":
        if coord_key in adata_sp.obsm:
            arr = np.asarray(adata_sp.obsm[coord_key])
            if arr.ndim != 2 or arr.shape[1] < 2:
                raise ValueError(f"obsm['{coord_key}'] must be [n_cells, >=2].")
            return np.asarray(arr[:, :2], dtype=np.float32)
        raise KeyError(f"Coordinate key '{coord_key}' not found in adata_sp.obsm.")

    for key in ("spatial", "coords"):
        if key in adata_sp.obsm:
            arr = np.asarray(adata_sp.obsm[key])
            if arr.ndim == 2 and arr.shape[1] >= 2:
                LOGGER.info("Using spatial coordinates from adata_sp.obsm['%s'].", key)
                return np.asarray(arr[:, :2], dtype=np.float32)

    for xk, yk in (("center_x", "center_y"), ("x", "y"), ("X", "Y")):
        if xk in adata_sp.obs and yk in adata_sp.obs:
            LOGGER.info("Using spatial coordinates from obs[%s, %s].", xk, yk)
            return adata_sp.obs[[xk, yk]].to_numpy(dtype=np.float32)

    raise KeyError(
        "No spatial coordinates found. Expected obsm['spatial'], obsm['coords'], "
        "or obs center_x/center_y. Use --coord_key if needed."
    )


def extract_celltypes(adata_sp: ad.AnnData, key: str) -> Tuple[np.ndarray, str]:
    if key != "auto":
        if key not in adata_sp.obs:
            raise KeyError(f"Cell-type key '{key}' not found in spatial obs.")
        vals = adata_sp.obs[key].astype(str).fillna("NA").to_numpy()
        return vals, key

    # subclass is a sensible first choice for cell-resolved brain ST datasets.
    candidates = (
        "subclass",
        "class_label",
        "label",
        "cell_type",
        "celltype",
        "celltype_major",
        "Cluster",
        "cluster",
        "leiden",
    )
    for candidate in candidates:
        if candidate in adata_sp.obs:
            vals = adata_sp.obs[candidate].astype(str).fillna("NA").to_numpy()
            LOGGER.info("Using ST cell types from adata_sp.obs['%s'] (%d groups).", candidate, len(np.unique(vals)))
            return vals, candidate

    LOGGER.warning(
        "No cell-type column was detected. All ST cells will be treated as one cell type; "
        "this is runnable but removes PASTA's cell-type-specific neighborhood assumption."
    )
    return np.repeat("ALL", adata_sp.n_obs), "ALL"


def build_same_type_neighbors(
    coords: np.ndarray,
    celltypes: np.ndarray,
    k: int,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Exact kNN within each cell type.

    Returns tuples of:
      (cell_type, global_cell_indices [n], global_neighbor_indices [n, k_type])
    Groups containing a single cell are retained with an empty neighbor matrix and
    are ignored by neighborhood losses.
    """
    groups: List[Tuple[str, np.ndarray, np.ndarray]] = []
    unique_types = pd.unique(celltypes)

    for ct in unique_types:
        idx = np.flatnonzero(celltypes == ct)
        n = len(idx)
        if n <= 1:
            groups.append((str(ct), idx.astype(np.int64), np.empty((n, 0), dtype=np.int64)))
            continue

        k_eff = min(k, n - 1)
        # +1 because the closest point is the cell itself.
        nn = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto", metric="euclidean", n_jobs=-1)
        nn.fit(coords[idx])
        local = nn.kneighbors(coords[idx], return_distance=False)[:, 1:]
        global_neighbors = idx[local]
        groups.append((str(ct), idx.astype(np.int64), global_neighbors.astype(np.int64)))

    return groups


def prepare_neighbor_stats(
    G_pathway: torch.Tensor,
    groups: List[Tuple[str, np.ndarray, np.ndarray]],
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Precompute statistics exactly equivalent to the official neighbor averages,
    while avoiding [n_cells, k, n_pathway_genes] tensors every epoch.

    For each cell type, returns:
      indices, E[neighbor expression], E[neighbor expression^2],
      E[L2-normalized neighbor expression]
    encoded as a tuple where the latter two expectations are packed separately.
    """
    prepared = []
    for _, idx_np, nbr_np in groups:
        if nbr_np.shape[1] == 0 or len(idx_np) == 0:
            continue
        idx = torch.as_tensor(idx_np, device=device, dtype=torch.long)
        nbr = torch.as_tensor(nbr_np, device=device, dtype=torch.long)
        nbr_expr = G_pathway[nbr]  # [n, k, p]
        mean_expr = nbr_expr.mean(dim=1)
        mean_sq = nbr_expr.square().mean(dim=1)
        nbr_unit = F.normalize(nbr_expr, p=2, dim=2, eps=1e-8)
        mean_unit = nbr_unit.mean(dim=1)
        prepared.append((idx, mean_expr, mean_sq, mean_unit))
    return prepared
