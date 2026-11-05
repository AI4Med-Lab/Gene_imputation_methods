#!/usr/bin/env python3
"""
python3 hvgTest.py \
    --filename bc_data \
    --sp_data spatial.h5ad \
    --pred_dir /path/to/SpaGene/fold_results \
    --hvg_dir /path/to/HVG_results \
    --method SpaGENE

"""

import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc

from compute_metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Five-fold HVG imputation evaluation")
    parser.add_argument("--sp_data", type=str, default="spatial.h5ad")
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--hvg_dir", type=str, required=True)
    parser.add_argument("--sp_path", type=str, default=None,
                        help="Optional full path to the spatial .h5ad file")
    parser.add_argument("--n_splits", type=int, default=5)
    return parser.parse_args()


def load_spatial(args):
    if args.sp_path:
        data_path = args.sp_path
    else:
        parent = "updatedDataset" if args.filename == "merFISH" else "processedData"
        data_path = os.path.join(
            "YOUR PATH...", parent, args.filename, args.sp_data
        )
    print("Spatial file:", data_path)
    adata = sc.read_h5ad(data_path)
    adata.var_names = pd.Index(adata.var_names.astype(str)).str.strip().str.upper()
    if not adata.var_names.is_unique:
        raise ValueError("Duplicate spatial gene names after uppercasing.")

    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print("Filtered spatial shape:", adata.shape)
    return adata


def merge_data(args, adata_spatial):
    fold_dfs = []
    n_cells = adata_spatial.n_obs

    for fold in range(1, args.n_splits + 1):
        path = os.path.join(args.pred_dir, f"fold_{fold}_pred.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing fold prediction: {path}")

        df = pd.read_csv(path)
        df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed:")].copy()
        df.columns = pd.Index(df.columns.astype(str)).str.strip().str.upper()

        if len(df) != n_cells:
            raise ValueError(
                f"Fold {fold}: {len(df)} prediction rows, but filtered ST has "
                f"{n_cells} cells. This index-free script requires the same "
                "cells in the same order; do not use fold-filtered outputs "
                "without first aligning them."
            )
        if df.columns.duplicated().any():
            dup = df.columns[df.columns.duplicated()].tolist()
            raise ValueError(f"Fold {fold}: duplicate predicted genes: {dup[:10]}")

        print(f"Fold {fold} predictions: {df.shape}")
        fold_dfs.append(df.reset_index(drop=True))

    y_pred = pd.concat(fold_dfs, axis=1)
    if y_pred.columns.duplicated().any():
        dup = y_pred.columns[y_pred.columns.duplicated()].tolist()
        raise ValueError(f"Genes appear in multiple folds: {dup[:10]}")

    selected_genes = [g for g in adata_spatial.var_names if g in y_pred.columns]
    if not selected_genes:
        raise ValueError("No genes match between predictions and spatial data.")

    y_pred = y_pred.loc[:, selected_genes]
    print("Merged predictions:", y_pred.shape)
    print("Matched genes:", len(selected_genes))
    return y_pred


def get_ground_truth(adata_spatial, pred_genes):
    X = adata_spatial[:, pred_genes].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return pd.DataFrame(np.asarray(X), columns=pred_genes)


def evaluate_hvg(args, y_true, y_pred):
    print("\n" + "=" * 65)
    print("Dataset:", args.filename)
    print("Method:", args.method)
    print("=" * 65)

    results = []
    for pct in (1, 5, 10, 25):
        hvg_path = os.path.join(
            args.hvg_dir, args.filename, f"top_{pct}percent_hvg.npy"
        )
        hvg_genes = np.load(hvg_path, allow_pickle=False)
        hvg_genes = [str(g).strip().upper() for g in hvg_genes]
        selected = list(dict.fromkeys(
            g for g in hvg_genes if g in y_pred.columns and g in y_true.columns
        ))

        print(f"\nTop {pct}% HVGs: {len(selected)} matched genes")
        if not selected:
            print("No HVGs available.")
            continue

        metrics = compute_metrics(
            y_true[selected].to_numpy(), y_pred[selected].to_numpy(),
            genes=selected,
        )
        # The supplied compute_metrics() returns aggregate correlation keys;
        # it does NOT return 'spearman_corrs' or 'pearson_corrs'.
        pcc = float(metrics["pearson_mean"])
        srcc = float(metrics["spearman_mean_genewise"])
        print(f"PCC: {pcc:.5f} | SRCC: {srcc:.5f}")
        results.append({"Top %": f"{pct}%", "Genes": len(selected),
                        "PCC": pcc, "SRCC": srcc})

    if not results:
        print("No valid HVG results available.")
        return

    result_df = pd.DataFrame(results)
    print("\nFINAL HVG RESULTS")
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nCopyable SRCC:")
    for val in result_df["SRCC"]:
        print(f"{val:.5f}")
    print("\nCopyable PCC:")
    for val in result_df["PCC"]:
        print(f"{val:.5f}")


def main():
    args = parse_args()
    adata = load_spatial(args)
    pred = merge_data(args, adata)
    truth = get_ground_truth(adata, pred.columns.tolist())
    if pred.shape != truth.shape:
        raise ValueError(f"Prediction/truth mismatch: {pred.shape} vs {truth.shape}")
    evaluate_hvg(args, truth, pred)


if __name__ == "__main__":
    main()
