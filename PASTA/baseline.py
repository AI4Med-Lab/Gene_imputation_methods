"""Modular PASTA gene-holdout benchmarking implementation."""

'''
python3 -u baseline.py \
    --sc_data /path/to/scRNA.h5ad \
    --sp_data /path/to/spatial.h5ad \
    --document Example \
    --filename Example \
    --rand 0 \
    --n_splits 5 \
    --epochs 200 \
    --learning_rate 0.05 \
    --m_chunk_rows 64 \
    --optimizer adam \
    --device cuda \
    --normalize log1p



'''



from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Dict, List
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.model_selection import KFold
import torch
from data_utils import (seed_everything, dense_float32, make_uppercase_unique,
                        maybe_normalize, intersect_and_filter_genes,
                        load_pathway_genes, deterministic_subsample_sc)
from spatial import extract_coords, extract_celltypes, build_same_type_neighbors
from model import train_pasta, project_test_genes, mapping_memory_report
from metrics import compute_metrics

LOGGER = logging.getLogger("PASTA-baseline")


def make_output_dir(args) -> Path:
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        document = args.document or "PASTABaseline"
        filename = args.filename or "dataset"
        out = Path("results") / document / filename / "PASTA"
    out.mkdir(parents=True, exist_ok=True)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--sc_data", required=True, help="scRNA-seq .h5ad")
    p.add_argument("--sp_data", required=True, help="Spatial .h5ad")
    p.add_argument("--document", default=None, help="Compatibility output label")
    p.add_argument("--filename", default=None, help="Dataset label")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--rand", type=int, default=0)
    p.add_argument("--n_splits", type=int, default=5)

    p.add_argument("--coord_key", default="auto")
    p.add_argument("--celltype_key", default="auto")
    p.add_argument("--ncell_thres", type=int, default=10)

    p.add_argument("--pathway_file", default=None, help="Optional pathway gene list or CSV/TSV with a gene column")
    p.add_argument("--pathway_name", default=None, help="Optional pathway value when pathway_file also has a pathway column")

    p.add_argument("--normalize", choices=["none", "log1p"], default="none")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--lambda_1", type=float, default=1.0, help="Reconstruction loss weight")
    p.add_argument("--lambda_2", type=float, default=2.0, help="Global similarity weight")
    p.add_argument("--lambda_3", type=float, default=1.0, help="Pathway similarity weight")
    p.add_argument("--lambda_4", type=float, default=10.0, help="Neighborhood similarity weight")
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam", help="Adam matches official repository; SGD uses less GPU memory")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--print_every", type=int, default=25)
    p.add_argument("--m_chunk_rows", type=int, default=64, help="Rows per CPU-backed mapping chunk; smaller uses less GPU memory but is slower")
    p.add_argument("--map_chunk_rows", type=int, default=None, help="Deprecated alias for --m_chunk_rows")
    p.add_argument("--max_sc_cells", type=int, default=0, help="0 = all scRNA cells; positive value deterministically subsamples reference")
    p.add_argument("--mape_threshold", type=float, default=0.1)
    p.add_argument("--save_predictions", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.map_chunk_rows is not None:
        LOGGER.warning("--map_chunk_rows is deprecated; using it as --m_chunk_rows=%d.", args.map_chunk_rows)
        args.m_chunk_rows = args.map_chunk_rows
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    seed_everything(args.rand)

    out_dir = make_output_dir(args)
    LOGGER.info("Output directory: %s", out_dir.resolve())

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    if device.type == "cuda":
        LOGGER.info("GPU: %s", torch.cuda.get_device_name(device))

    # =========================================================================
    # Load data using the same path convention as your other baselines.
    #
    # If an explicit/absolute path exists, use it directly.
    # Otherwise:
    #   merFISH ST  -> updatedDataset/merFISH/
    #   other ST    -> processedData/<dataset>/
    #   scRNA       -> processedData/<dataset>/
    # =========================================================================
    sc_arg = Path(args.sc_data)
    sp_arg = Path(args.sp_data)

    if sc_arg.exists():
        sc_path = sc_arg
    else:
        sc_path = (
            Path("YOUR DATASET PATH...")
            / str(args.filename)
            / args.sc_data
        )

    if sp_arg.exists():
        sp_path = sp_arg
    elif args.filename == "merFISH":
        sp_path = (
            Path("YOUR DATASET PATH...")
            / str(args.filename)
            / args.sp_data
        )
    else:
        sp_path = (
            Path("YOUR DATASET PATH...")
            / str(args.filename)
            / args.sp_data
        )

    LOGGER.info("Reading scRNA: %s", sc_path)
    LOGGER.info("Reading ST: %s", sp_path)

    if not sc_path.exists():
        raise FileNotFoundError(
            f"scRNA file not found: {sc_path}"
        )
    if not sp_path.exists():
        raise FileNotFoundError(
            f"Spatial file not found: {sp_path}"
        )

    adata_sc = ad.read_h5ad(sc_path)
    adata_sp = ad.read_h5ad(sp_path)

    # Avoid AnnData implicit index-conversion warnings where possible.
    adata_sc.obs_names = adata_sc.obs_names.astype(str)
    adata_sc.var_names = adata_sc.var_names.astype(str)
    adata_sp.obs_names = adata_sp.obs_names.astype(str)
    adata_sp.var_names = adata_sp.var_names.astype(str)

    print(f"sp shape before filtering: {adata_sp.shape}")
    print(f"sc shape before filtering: {adata_sc.shape}")

    # =========================================================================
    # Same basic filtering used in your existing baseline pipeline.
    # Do this BEFORE extracting coordinates/cell types so metadata remains
    # aligned with the filtered ST observations.
    # =========================================================================
    sc.pp.filter_cells(adata_sc, min_counts=1)
    sc.pp.filter_genes(adata_sc, min_counts=1)

    sc.pp.filter_cells(adata_sp, min_counts=1)
    sc.pp.filter_genes(adata_sp, min_counts=1)

    LOGGER.info(
        "After nonzero filtering | scRNA=%s | ST=%s",
        adata_sc.shape,
        adata_sp.shape,
    )

    # Uppercase symbols before intersection to avoid case mismatches.
    adata_sc = make_uppercase_unique(adata_sc, "scRNA")
    adata_sp = make_uppercase_unique(adata_sp, "ST")

    if args.normalize == "none":
        LOGGER.info(
            "Normalization: NONE (input expression values are used as provided)."
        )
    else:
        LOGGER.info(
            "Normalization: normalize_total(1e4) + log1p on each modality."
        )
        adata_sc = maybe_normalize(adata_sc, args.normalize)
        adata_sp = maybe_normalize(adata_sp, args.normalize)

    # Common genes + remove genes that are all-zero in either modality.
    adata_sc, adata_sp, common_genes = intersect_and_filter_genes(
        adata_sc,
        adata_sp,
    )

    print(f"The number of common genes: {len(common_genes)}")
    print(f"Spatial after intersection: {adata_sp.shape}")
    print(f"scRNA after intersection: {adata_sc.shape}")

    LOGGER.info(
        "Common nonzero genes: %d",
        len(common_genes),
    )

    # Coordinates/cell types after cell filtering, but before optional scRNA
    # subsampling. These arrays therefore remain aligned with adata_sp.
    coords = extract_coords(
        adata_sp,
        args.coord_key,
    )

    celltypes, celltype_key_used = extract_celltypes(
        adata_sp,
        args.celltype_key,
    )

    # Optional reproducible scRNA reference subsampling.
    adata_sc = deterministic_subsample_sc(
        adata_sc,
        args.max_sc_cells,
        args.rand,
    )

    LOGGER.info(
        "Final matrices | scRNA=%s | ST=%s",
        adata_sc.shape,
        adata_sp.shape,
    )

    pathway_set = load_pathway_genes(args.pathway_file, args.pathway_name)
    if pathway_set is None:
        LOGGER.info("Pathway context: all visible/train genes in each fold (generic gene-imputation adaptation).")
    else:
        LOGGER.info("Loaded pathway definition with %d genes.", len(pathway_set))

    LOGGER.info("Building exact same-cell-type spatial %d-NN once for all folds...", args.ncell_thres)
    neighbor_groups = build_same_type_neighbors(coords, celltypes, args.ncell_thres)
    n_neighbor_groups = sum(1 for _, _, nbr in neighbor_groups if nbr.shape[1] > 0)
    LOGGER.info("Neighbor groups: %d total, %d usable.", len(neighbor_groups), n_neighbor_groups)

    mem = mapping_memory_report(
        adata_sc.n_obs,
        adata_sp.n_obs,
        args.optimizer,
        device,
    )

    cpu_mapping_gib = mem["mapping_matrix_gib"]
    cpu_adam_gib = 2.0 * cpu_mapping_gib
    cpu_total_gib = 3.0 * cpu_mapping_gib

    LOGGER.info(
        "Logical mapping size: %d x %d = %.3f B entries | "
        "M=%.2f GiB CPU | Adam states=%.2f GiB CPU | "
        "total persistent mapping+Adam ~= %.2f GiB CPU",
        adata_sc.n_obs,
        adata_sp.n_obs,
        mem["mapping_entries"] / 1e9,
        cpu_mapping_gib,
        cpu_adam_gib,
        cpu_total_gib,
    )

    if device.type == "cuda":
        LOGGER.info(
            "GPU total memory: %.2f GiB; only one M chunk is moved "
            "to GPU at a time.",
            mem["gpu_total_gib"],
        )

    config = vars(args).copy()
    config.update(
        {
            "sc_shape_after_preprocessing": list(adata_sc.shape),
            "sp_shape_after_preprocessing": list(adata_sp.shape),
            "n_common_genes": len(common_genes),
            "celltype_key_used": celltype_key_used,
            "n_celltypes": int(len(np.unique(celltypes))),
            "memory_estimate": mem,
            "mapping_storage": "CPU-backed row chunks",
            "m_chunk_rows_effective": args.m_chunk_rows,
        }
    )
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.rand)
    genes_np = np.asarray(common_genes)

    fold_metrics: List[Dict[str, float]] = []
    all_true = []
    all_pred = []
    all_gene_names = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(genes_np), start=1):

        # if fold != 5:
        #     continue
        fold_seed = args.rand + fold - 1
        seed_everything(fold_seed)
        train_genes = genes_np[train_idx].tolist()
        test_genes = genes_np[test_idx].tolist()

        print("test_genes:=>", test_genes)
        LOGGER.info("=" * 70)
        LOGGER.info("Fold %d/%d | train genes=%d | test genes=%d", fold, args.n_splits, len(train_genes), len(test_genes))

        S_train = dense_float32(adata_sc[:, train_genes].X)
        G_train = dense_float32(adata_sp[:, train_genes].X)
        S_test = dense_float32(adata_sc[:, test_genes].X)
        Y_true = dense_float32(adata_sp[:, test_genes].X)

        if pathway_set is None:
            pathway_mask = np.ones(len(train_genes), dtype=bool)
        else:
            pathway_mask = np.asarray([g in pathway_set for g in train_genes], dtype=bool)
            LOGGER.info("Fold %d visible pathway genes: %d/%d", fold, int(pathway_mask.sum()), len(train_genes))
            if pathway_mask.sum() == 0:
                raise ValueError(f"Fold {fold}: no pathway genes are visible among training genes.")

        model, history = train_pasta(
            S_train,
            G_train,
            pathway_mask,
            neighbor_groups,
            device=device,
            seed=fold_seed,
            epochs=args.epochs,
            lr=args.learning_rate,
            lambdas=(args.lambda_1, args.lambda_2, args.lambda_3, args.lambda_4),
            print_every=args.print_every,
            m_chunk_rows=args.m_chunk_rows,
            optimizer_name=args.optimizer,
        )

        Y_pred = project_test_genes(model, S_test, device=device, m_chunk_rows=args.m_chunk_rows)
        LOGGER.info("Fold %d predicted shape: %s", fold, Y_pred.shape)

        metrics, gene_df = compute_metrics(Y_true, Y_pred, test_genes, args.mape_threshold)
        metrics.update({"fold": fold, "phase": "test", "n_train_genes": len(train_genes), "n_test_genes": len(test_genes)})
        fold_metrics.append(metrics)

        LOGGER.info(
            "Fold %d | Pearson=%.5f | Spearman=%.5f | L1=%.5f | L2=%.5f | R2=%.5f | masked MAPE=%.3f%%",
            fold,
            metrics["pearson_mean"],
            metrics["spearman_mean_genewise"],
            metrics["l1_error_mean"],
            metrics["l2_errors_mean"],
            metrics["r2_scores_mean"],
            metrics["masked_mape_pct"],
        )

        pd.DataFrame(history).to_csv(out_dir / f"fold_{fold}_training_history.csv", index=False)
        gene_df.to_csv(out_dir / f"fold_{fold}_gene_metrics.csv", index=False)
        (out_dir / f"fold_{fold}_train_genes.txt").write_text("\n".join(train_genes) + "\n")
        (out_dir / f"fold_{fold}_test_genes.txt").write_text("\n".join(test_genes) + "\n")

        if args.save_predictions:
 
            true_df = pd.DataFrame(
                Y_true,
                columns=test_genes,
                index=adata_sp.obs_names
            )

            pred_df = pd.DataFrame(
                Y_pred,
                columns=test_genes,
                index=adata_sp.obs_names
            )

            true_df.to_csv(
                out_dir / f"fold_{fold}_y_true.csv"
            )

            pred_df.to_csv(
                out_dir / f"fold_{fold}_y_pred.csv"
            )

        all_true.append(Y_true)
        all_pred.append(Y_pred)
        all_gene_names.extend(test_genes)

        del model, S_train, G_train, S_test, Y_true, Y_pred
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)

    numeric_cols = [
        "pearson_mean",
        "spearman_mean_genewise",
        "l1_error_mean",
        "l2_errors_mean",
        "r2_scores_mean",
        "masked_mape_pct",
    ]
    summary_rows = []
    for col in numeric_cols:
        vals = fold_df[col].to_numpy(dtype=float)
        summary_rows.append(
            {
                "metric": col,
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals, ddof=0)),
                "mean_pm_std": f"{np.nanmean(vals):.5f} ± {np.nanstd(vals, ddof=0):.5f}",
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_metrics.csv", index=False)

    # Genes are disjoint across CV folds, so concatenating columns reconstructs the
    # full gene set (in fold order) for a combined evaluation.
    Y_true_all = np.concatenate(all_true, axis=1)
    Y_pred_all = np.concatenate(all_pred, axis=1)
    combined_metrics, combined_gene_df = compute_metrics(
        Y_true_all, Y_pred_all, all_gene_names, args.mape_threshold
    )
    combined_gene_df.to_csv(out_dir / "combined_gene_metrics.csv", index=False)
    pd.DataFrame([combined_metrics]).to_csv(out_dir / "combined_metrics.csv", index=False)

    LOGGER.info("=" * 70)
    LOGGER.info("5-fold summary:")
    for row in summary_rows:
        LOGGER.info("  %-26s %s", row["metric"], row["mean_pm_std"])
    LOGGER.info(
        "Combined | Pearson=%.5f | Spearman=%.5f | L1=%.5f | L2=%.5f | R2=%.5f | masked MAPE=%.3f%%",
        combined_metrics["pearson_mean"],
        combined_metrics["spearman_mean_genewise"],
        combined_metrics["l1_error_mean"],
        combined_metrics["l2_errors_mean"],
        combined_metrics["r2_scores_mean"],
        combined_metrics["masked_mape_pct"],
    )
    LOGGER.info("Saved results to: %s", out_dir.resolve())

if __name__ == "__main__":
    main()
