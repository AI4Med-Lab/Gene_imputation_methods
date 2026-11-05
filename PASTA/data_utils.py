"""Modular PASTA gene-holdout benchmarking implementation."""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch

LOGGER = logging.getLogger("PASTA-baseline")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dense_float32(x) -> np.ndarray:
    if sp.issparse(x):
        return x.toarray().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def nonzero_genes(x) -> np.ndarray:
    if sp.issparse(x):
        return np.asarray(x.getnnz(axis=0)).ravel() > 0
    return np.any(np.asarray(x) != 0, axis=0)


def make_uppercase_unique(adata: ad.AnnData, name: str) -> ad.AnnData:
    """Upper-case symbols and retain the first duplicate created by case folding."""
    a = adata.copy()
    names = pd.Index([str(g).upper() for g in a.var_names])
    keep = ~names.duplicated(keep="first")
    if not np.all(keep):
        LOGGER.warning("%s: dropping %d duplicated genes after upper-casing.", name, int((~keep).sum()))
        a = a[:, np.asarray(keep)].copy()
        names = names[keep]
    a.var_names = names.astype(str)
    return a


def maybe_normalize(adata: ad.AnnData, mode: str) -> ad.AnnData:
    if mode == "none":
        return adata
    if mode != "log1p":
        raise ValueError(f"Unknown normalization mode: {mode}")
    try:
        import scanpy as sc
    except ImportError as e:
        raise ImportError("scanpy is required for --normalize log1p") from e
    out = adata.copy()
    sc.pp.normalize_total(out, target_sum=1e4)
    sc.pp.log1p(out)
    return out


def intersect_and_filter_genes(
    adata_sc: ad.AnnData, adata_sp: ad.AnnData
) -> Tuple[ad.AnnData, ad.AnnData, List[str]]:
    # Preserve SC order for deterministic indexing.
    sp_set = set(adata_sp.var_names)
    # common = [g for g in adata_sp.var_names if g in adata_sc.var_names]
    common = np.intersect1d(
    np.asarray(adata_sp.var_names, dtype=str),
    np.asarray(adata_sc.var_names, dtype=str)
    )
    if len(common) < 2:
        raise ValueError(f"Only {len(common)} common genes found between scRNA and ST.")

    sc_common = adata_sc[:, common].copy()
    sp_common = adata_sp[:, common].copy()

    nz = nonzero_genes(sc_common.X) & nonzero_genes(sp_common.X)
    removed = int((~nz).sum())
    if removed:
        LOGGER.info("Removing %d common genes that are all-zero in SC or ST.", removed)
        common = list(np.asarray(common)[nz])
        sc_common = sc_common[:, nz].copy()
        sp_common = sp_common[:, nz].copy()

    return sc_common, sp_common, common


def load_pathway_genes(path: Optional[str], pathway_name: Optional[str]) -> Optional[set]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() in {".csv", ".tsv", ".txt"}:
        sep = "\t" if p.suffix.lower() == ".tsv" else ","
        try:
            df = pd.read_csv(p, sep=sep)
            lower_cols = {str(c).lower(): c for c in df.columns}
            if "gene" in lower_cols:
                if pathway_name is not None and "pathway" in lower_cols:
                    df = df[df[lower_cols["pathway"]].astype(str) == str(pathway_name)]
                genes = df[lower_cols["gene"]].astype(str).tolist()
                return {g.upper() for g in genes}
        except Exception:
            pass

    # Fallback: one gene per line; also accepts comma-delimited text.
    text = p.read_text().replace(",", "\n")
    genes = [x.strip() for x in text.splitlines() if x.strip() and not x.startswith("#")]
    return {g.upper() for g in genes}


def deterministic_subsample_sc(adata_sc: ad.AnnData, max_cells: int, seed: int) -> ad.AnnData:
    if max_cells <= 0 or adata_sc.n_obs <= max_cells:
        return adata_sc
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(adata_sc.n_obs, size=max_cells, replace=False))
    LOGGER.warning(
        "Subsampling scRNA reference from %d to %d cells because --max_sc_cells was set. "
        "This is a computational approximation, not the exact full-reference PASTA baseline.",
        adata_sc.n_obs,
        max_cells,
    )
    return adata_sc[idx].copy()
