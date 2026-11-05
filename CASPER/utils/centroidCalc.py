
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from typing import Optional
import logging



def compute_centroids_leiden(
    adata_sc,
    logger,
    n_top_hvgs=None,
    n_pcs=30,
    neighbors_k=15,
    leiden_resolution=1.0,
    use_hvg_subset=True,
):

    ad = adata_sc.copy()

    # -----------------------------
    # HVG
    # -----------------------------
    if use_hvg_subset and n_top_hvgs is not None:

        X = ad.X.toarray() if sp.issparse(ad.X) else ad.X

        if np.max(X) < 50 or not np.allclose(X, np.round(X)):
            hv_flavor = "cell_ranger"
        else:
            hv_flavor = "seurat_v3"

        sc.pp.highly_variable_genes(
            ad,
            n_top_genes=n_top_hvgs,
            flavor=hv_flavor
        )

        ad = ad[:, ad.var["highly_variable"]].copy()

    # -----------------------------
    # Dense matrix
    # -----------------------------
    if sp.issparse(ad.X):
        ad.X = ad.X.toarray()

    ad.X = np.nan_to_num(ad.X)

    # -----------------------------
    # PCA + Leiden
    # -----------------------------
    sc.pp.pca(
        ad,
        n_comps=n_pcs,
        svd_solver="arpack"
    )

    sc.pp.neighbors(
        ad,
        n_neighbors=neighbors_k,
        n_pcs=n_pcs
    )

    sc.tl.leiden(
        ad,
        resolution=leiden_resolution,
        key_added="leiden",
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )

    # -----------------------------
    # Centroids
    # -----------------------------
    expr_df = pd.DataFrame(
        ad.X,
        index=ad.obs_names,
        columns=ad.var_names
    )

    centroids_df = expr_df.groupby(
        ad.obs["leiden"].astype(str)
    ).mean()

    # -----------------------------
    # Cell -> Cluster
    # -----------------------------
    cell_cluster_df = pd.DataFrame({
        "cell_id": ad.obs_names,
        "cluster": ad.obs["leiden"].astype(str).values
    })

    # -----------------------------
    # Cluster -> Cells
    # -----------------------------
    cluster_to_cells = {}

    for c in sorted(
        ad.obs["leiden"].unique(),
        key=lambda x: int(x)
    ):
        cluster_to_cells[str(c)] = (
            cell_cluster_df[
                cell_cluster_df.cluster == str(c)
            ]["cell_id"].tolist()
        )

    # -----------------------------
    # Cell ID -> Index
    # -----------------------------
    cell_to_idx = {
        cell: idx
        for idx, cell in enumerate(ad.obs_names)
    }

    # -----------------------------
    # Cluster -> Cell Indices
    # -----------------------------
    cluster_to_indices = {}

    for cluster, cells in cluster_to_cells.items():

        cluster_to_indices[cluster] = [
            cell_to_idx[x]
            for x in cells
        ]

    return (
        centroids_df,
        ad,
        cell_cluster_df,
        cluster_to_cells,
        cluster_to_indices
    )