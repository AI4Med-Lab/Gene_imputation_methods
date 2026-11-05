#!/usr/bin/env python3
"""

(vista) 2023csz0012@small:~/AmitCode/code/baselines/VISTA-main$ nohup python3 baseline.py \
    --sc_data 'scRNAseq_full_compatible.h5ad' \
    --sp_data 'spatial.h5ad' \
    --document 'lungsCAN' \
    --filename 'lungsCAN' \
    --rand 0 \
    --n_splits 5 \
    --epochs 200 \
    --batch_size 128 \
    --n_latent 32 \
    --neighbor_size 20 \
    --lr 1e-4 \
    --gradient_clip_val 10.0 \
    > lungsCAN_VISTA.out 2>&1 &
    

nohup python3 baseline.py     --sc_data 'scRNAseq_full.h5ad'     --sp_data 'spatial.h5ad'     --document 'lungsCAN'     --filename 'lungsCAN'
    --rand 0     --n_splits 5     --epochs 200     --batch_size 128     --n_latent 32     --neighbor_size 20     --lr 1e-4     --gradient_clip_val 10.0     > lungsCAN_VISTA.out 2>&1 &

VISTA 5-fold gene-imputation baseline.

Protocol matches the user's other missing-gene baselines:
  * intersect scRNA-seq and ST genes
  * 5-fold KFold over genes only
  * scRNA keeps ALL common genes
  * ST exposes ONLY the train genes in each fold
  * the SAME ST cells are retained in every fold
  * held-out ST genes are used only after training for evaluation
  * save per-fold predictions/truth/genes/cells/metrics
  * save fold_metrics.csv, metrics_summary.csv and a complete CV prediction matrix

VISTA itself is count-based. Training therefore uses count/count-like matrices.
For comparable metrics, normalized VISTA output is transformed as log1p(pred * 1e4),
while the aligned ST truth is independently normalize_total(1e4) + log1p transformed.
This script is intended for the stabilized VISTA module where zero-library values and
spatial graph posterior updates are handled safely.
"""

import argparse
import gc
import os
import random
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import torch
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr
from sklearn.model_selection import KFold

from vista import GIMVI_GCN


def dense_float32(x):
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)



# -----------------------------------------------------------------------------
# Exact benchmark metric implementation used by the other baselines.
# Metrics are computed gene-wise and then averaged.
# NOTE: l2_errors_mean is gene-wise mean squared error despite the old name.
# -----------------------------------------------------------------------------
def compute_metrics(y_test, preds_all, genes=None):
    y_test = np.asarray(y_test, dtype=np.float64)
    preds_all = np.asarray(preds_all, dtype=np.float64)

    if y_test.shape != preds_all.shape:
        raise ValueError(
            f"Metric shape mismatch: {y_test.shape} vs {preds_all.shape}"
        )

    errors = []
    l1_errors = []
    r2_scores = []
    pearson_corrs = []
    spearman_corrs = []

    for i in range(y_test.shape[1]):
        preds = preds_all[:, i]
        target_vals = y_test[:, i]

        l1_error_value = float(np.mean(np.abs(target_vals - preds)))
        l2_error = float(np.mean((preds - target_vals) ** 2))

        total_variance = np.sum((target_vals - np.mean(target_vals)) ** 2)
        if total_variance == 0:
            r2_score = 0.0
        else:
            r2_score = float(
                1 - np.sum((target_vals - preds) ** 2) / total_variance
            )

        if len(target_vals) < 2 or len(preds) < 2:
            pearson_corr = 0.0
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConstantInputWarning)
                try:
                    pearson_corr, _ = pearsonr(
                        target_vals.flatten(), preds.flatten()
                    )
                    if np.isnan(pearson_corr):
                        pearson_corr = 0.0
                except (ConstantInputWarning, ValueError, FloatingPointError):
                    pearson_corr = 0.0

        if len(target_vals) < 2 or len(preds) < 2:
            spearman_corr = 0.0
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConstantInputWarning)
                try:
                    spearman_corr, _ = spearmanr(
                        target_vals.flatten(), preds.flatten()
                    )
                    if np.isnan(spearman_corr):
                        spearman_corr = 0.0
                except (ConstantInputWarning, ValueError, FloatingPointError):
                    spearman_corr = 0.0

        errors.append(l2_error)
        l1_errors.append(l1_error_value)
        r2_scores.append(r2_score)
        pearson_corrs.append(float(pearson_corr))
        spearman_corrs.append(float(spearman_corr))

    return {
        "pearson_mean": float(f"{np.mean(pearson_corrs):.4f}"),
        "spearman_mean_genewise": float(f"{np.mean(spearman_corrs):.4f}"),
        "l1_error_mean": float(f"{np.mean(l1_errors):.4f}"),
        "l2_errors_mean": float(f"{np.mean(errors):.4f}"),
        "r2_scores_mean": float(f"{np.mean(r2_scores):.4f}"),
        "pearson_std": float(f"{np.std(pearson_corrs):.4f}"),
        "l2_error_q1": float(f"{np.percentile(errors, 25):.4f}"),
        "l2_error_q2": float(f"{np.median(errors):.4f}"),
        "l2_error_q3": float(f"{np.percentile(errors, 75):.4f}"),
        "r2_score_q1": float(f"{np.percentile(r2_scores, 25):.4f}"),
        "r2_score_q2": float(f"{np.median(r2_scores):.4f}"),
        "r2_score_q3": float(f"{np.percentile(r2_scores, 75):.4f}"),
    }

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import scvi
        scvi.settings.seed = seed
    except Exception:
        pass


def apply_layer_as_x(adata, layer, label):
    if layer is None:
        return adata
    if layer not in adata.layers:
        raise KeyError(
            f"{label}: requested layer '{layer}' not found. "
            f"Available layers: {list(adata.layers.keys())}"
        )
    adata = adata.copy()
    adata.X = adata.layers[layer].copy()
    return adata


def check_count_like(adata, label, sample_size=200000):
    """Warn rather than fail; VISTA/scVI will also validate count data."""
    x = adata.X
    if sp.issparse(x):
        vals = x.data
    else:
        vals = np.asarray(x).reshape(-1)

    if vals.size == 0:
        raise ValueError(f"{label}: expression matrix is empty")

    if vals.size > sample_size:
        rng = np.random.default_rng(0)
        vals = vals[rng.choice(vals.size, size=sample_size, replace=False)]

    vals = np.asarray(vals, dtype=np.float64)
    if not np.isfinite(vals).all():
        raise ValueError(f"{label}: X contains NaN/Inf")
    if np.min(vals) < -1e-8:
        raise ValueError(
            f"{label}: negative values found. VISTA requires non-negative count/count-like input."
        )

    nonneg = vals[vals >= 0]
    if nonneg.size:
        frac_integer = np.mean(np.isclose(nonneg, np.rint(nonneg), atol=1e-4))
        if frac_integer < 0.98:
            warnings.warn(
                f"{label}: only {frac_integer:.1%} of sampled nonnegative values "
                "look integer-like. VISTA expects raw/count-like expression. "
                "Use --sc_layer/--sp_layer if counts are stored in a layer."
            )


def ensure_spatial_coordinates(adata):
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    elif "coords" in adata.obsm:
        coords = np.asarray(adata.obsm["coords"], dtype=np.float32)
        adata.obsm["spatial"] = coords
    elif {"center_x", "center_y"}.issubset(adata.obs.columns):
        coords = adata.obs[["center_x", "center_y"]].to_numpy(dtype=np.float32)
        adata.obsm["spatial"] = coords
    elif {"Centroid_X", "Centroid_Y"}.issubset(adata.obs.columns):
        coords = adata.obs[["Centroid_X", "Centroid_Y"]].to_numpy(dtype=np.float32)
        adata.obsm["spatial"] = coords
    elif {"x", "y"}.issubset(adata.obs.columns):
        coords = adata.obs[["x", "y"]].to_numpy(dtype=np.float32)
        adata.obsm["spatial"] = coords
    else:
        raise KeyError(
            "VISTA requires spatial coordinates. Expected adata.obsm['spatial'], "
            "adata.obsm['coords'], obs[['center_x','center_y']], "
            "obs[['Centroid_X','Centroid_Y']], or obs[['x','y']]."
        )

    if coords.ndim != 2 or coords.shape[0] != adata.n_obs or coords.shape[1] < 2:
        raise ValueError(
            f"Invalid spatial coordinates shape {coords.shape}; expected ({adata.n_obs}, >=2)."
        )
    if not np.isfinite(coords).all():
        raise ValueError("Spatial coordinates contain NaN/Inf")

    # VISTA expects 2-D locations.
    adata.obsm["spatial"] = coords[:, :2].astype(np.float32, copy=False)


def normalize_log_copy(adata, target_sum=1e4):
    out = adata.copy()
    sc.pp.normalize_total(out, target_sum=target_sum)
    sc.pp.log1p(out)
    return out

def uppercase_var_names(adata):
    adata = adata.copy()
    adata.var_names = pd.Index(
        [str(g).strip().upper() for g in adata.var_names]
    )
    if not adata.var_names.is_unique:
        warnings.warn(
            "Gene names became duplicated after uppercasing; making them unique."
        )
        adata.var_names_make_unique()
    return adata


def get_output_cell_ids(adata):
    if "cell_id" in adata.obs.columns:
        ids = adata.obs["cell_id"].astype(str).to_numpy()
        if len(np.unique(ids)) == len(ids):
            return ids
        warnings.warn(
            "obs['cell_id'] is not unique; using AnnData obs_names for output."
        )
    return adata.obs_names.astype(str).to_numpy()


def parse_args():
    p = argparse.ArgumentParser(
        description="VISTA baseline with the same 5-fold gene split used by the other baselines."
    )
    p.add_argument("--sc_data", type=str, required=True)
    p.add_argument("--sp_data", type=str, required=True)
    p.add_argument("--document", type=str, required=True)
    p.add_argument("--filename", type=str, required=True)
    p.add_argument("--rand", type=int, default=0)

    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--n_latent", type=int, default=32)
    p.add_argument("--neighbor_size", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gradient_clip_val", type=float, default=10.0)
    p.add_argument("--start_fold", type=int, default=1, help="Start training from this fold (1 to 5)")

    # Optional count layers. Leave unset when X already contains counts.
    p.add_argument("--sc_layer", type=str, default=None)
    p.add_argument("--sp_layer", type=str, default=None)

    p.add_argument(
        "--uppercase_genes",
        action="store_true",
        help="Strip and uppercase gene names in both datasets before intersection.",
    )

    p.add_argument(
        "--result_root",
        type=str,
        default="Result",
        help="Outputs go to RESULT_ROOT/document/VISTA/",
    )
    p.add_argument("--save_model", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.rand)

    print("=" * 70)
    print("VISTA 5-fold gene-imputation baseline")
    print("=" * 70)
    print("scRNA:", args.sc_data)
    print("Spatial:", args.sp_data)
    print("Seed:", args.rand)


    if args.filename != "merFISH":
        adata_spatial = sc.read_h5ad(
            f"/home/2023csz0012/AmitCode/data/processedData/"
            f"{args.filename}/{args.sp_data}"
        )
    else:
        adata_spatial = sc.read_h5ad(
            f"/home/2023csz0012/AmitCode/data/updatedDataset/"
            f"{args.filename}/{args.sp_data}"
        )

    adata_seq = sc.read_h5ad(
        f"/home/2023csz0012/AmitCode/data/processedData/"
        f"{args.filename}/{args.sc_data}"
    )


    adata_seq = apply_layer_as_x(adata_seq, args.sc_layer, "scRNA")
    adata_spatial = apply_layer_as_x(adata_spatial, args.sp_layer, "ST")

    if args.uppercase_genes:
        adata_seq = uppercase_var_names(adata_seq)
        adata_spatial = uppercase_var_names(adata_spatial)

    if not adata_seq.var_names.is_unique:
        print("[warning] scRNA var_names are not unique; making unique.")
        adata_seq.var_names_make_unique()
    if not adata_spatial.var_names.is_unique:
        print("[warning] ST var_names are not unique; making unique.")
        adata_spatial.var_names_make_unique()
    if not adata_seq.obs_names.is_unique:
        print("[warning] scRNA obs_names are not unique; making unique.")
        adata_seq.obs_names_make_unique()
    if not adata_spatial.obs_names.is_unique:
        print("[warning] ST obs_names are not unique; making unique.")
        adata_spatial.obs_names_make_unique()

    ensure_spatial_coordinates(adata_spatial)

    print("Before filtering:")
    print("  ST:", adata_spatial.shape)
    print("  scRNA:", adata_seq.shape)

    check_count_like(adata_seq, "scRNA")
    check_count_like(adata_spatial, "ST")

    # Same initial filtering convention as the other baselines.
    sc.pp.filter_cells(adata_seq, min_counts=1)
    sc.pp.filter_genes(adata_seq, min_counts=1)
    sc.pp.filter_cells(adata_spatial, min_counts=1)
    sc.pp.filter_genes(adata_spatial, min_counts=1)

    # Preserve ST gene order, matching the existing baseline scaffold.
    seq_gene_set = set(adata_seq.var_names)
    # common_genes = np.array(
    #     [g for g in adata_spatial.var_names if g in seq_gene_set], dtype=object
    # )

    common_genes = np.intersect1d(
    np.asarray(adata_spatial.var_names, dtype=str),
    np.asarray(adata_seq.var_names, dtype=str)
    )
    if len(common_genes) < args.n_splits:
        raise ValueError(
            f"Only {len(common_genes)} common genes; cannot run {args.n_splits}-fold CV."
        )

    print("Number of common genes:", len(common_genes))

    adata_seq = adata_seq[:, common_genes].copy()
    adata_spatial = adata_spatial[:, common_genes].copy()

    # ============================================================
    # Evaluation copy.
    # VISTA TRAINING remains on count/count-like data.
    # Metrics use normalize_total(1e4) + log1p, matching the benchmark.
    # ============================================================
    adata_spatial_eval = normalize_log_copy(adata_spatial, target_sum=1e4)

    print("scRNA shape:", adata_seq.shape)
    print("Spatial shape:", adata_spatial.shape)

    print("After intersection:")
    print("  ST:", adata_spatial.shape)
    print("  scRNA:", adata_seq.shape)

    outdir = os.path.join(args.result_root, args.document, "VISTA")
    csv_dir = os.path.join(outdir, "fold_results")
    model_dir = os.path.join(outdir, "models")
    os.makedirs(csv_dir, exist_ok=True)
    if args.save_model:
        os.makedirs(model_dir, exist_ok=True)

    # Complete cross-validated prediction matrix. Same ST rows in every fold.
    output_cell_ids = get_output_cell_ids(adata_spatial)
    original_obs_names = adata_spatial.obs_names.copy()

    all_pred_res = np.full(
        (adata_spatial.n_obs, len(common_genes)), np.nan, dtype=np.float32
    )
    fold_metrics = {}

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.rand)

    for fold, (train_ind, test_ind) in enumerate(kf.split(common_genes), start=1):

        if fold < args.start_fold:
            print(f"Skipping Fold {fold}")
            continue

        train_gene = common_genes[train_ind]
        test_gene = common_genes[test_ind]

        print("\n" + "=" * 70)
        print(f"Fold {fold}/{args.n_splits}")
        print(f"Train genes: {len(train_gene)} | Test genes: {len(test_gene)}")
        print("=" * 70)

        # IMPORTANT:
        #   scRNA retains ALL common genes -> decoder can learn held-out genes.
        #   ST contains ONLY train genes -> held-out ST values are never seen.
        seq_fold = adata_seq.copy()
        sp_fold = adata_spatial[:, train_gene].copy()

        # Keep EXACTLY the same ST cells in every gene fold.
        # The stabilized VISTA module safely clamps zero visible-gene library sizes,
        # so cells must not be removed fold-by-fold.
        if sp_fold.n_obs != adata_spatial.n_obs:
            raise RuntimeError(
                f"Fold {fold}: ST cell count changed unexpectedly "
                f"({sp_fold.n_obs} vs {adata_spatial.n_obs})."
            )
        if sp_fold.n_obs < 2:
            raise ValueError(f"Fold {fold}: fewer than 2 ST cells.")

        visible_sum = np.asarray(sp_fold.X.sum(axis=1)).reshape(-1)
        zero_visible = int(np.sum(visible_sum <= 0))
        print(
            f"ST cells in fold {fold}: {sp_fold.n_obs}/{adata_spatial.n_obs} | "
            f"zero-visible-count cells kept: {zero_visible}"
        )
        del visible_sum

        # Fold-local row index for spatial graph construction.
        sp_fold.obs["_vista_row"] = np.arange(sp_fold.n_obs, dtype=np.int64)
        ensure_spatial_coordinates(sp_fold)

        if not np.array_equal(
            sp_fold.obs_names.to_numpy(), original_obs_names.to_numpy()
        ):
            raise RuntimeError(f"Fold {fold}: ST row order changed.")

        # Register fresh fold-specific AnnData objects with scVI.
        GIMVI_GCN.setup_anndata(seq_fold)
        GIMVI_GCN.setup_anndata(sp_fold, obs_names="_vista_row")

        model = GIMVI_GCN(
            seq_fold,
            sp_fold,
            n_latent=args.n_latent,
            neighbor_size=args.neighbor_size,
        )

        print(f"Training VISTA fold {fold} for {args.epochs} epochs...")
        model.train(
            max_epochs=args.epochs,
            use_gpu=True if torch.cuda.is_available() else False,
            train_size=1.0,          # gene split only: no extra cell holdout
            batch_size=args.batch_size,
            plan_kwargs={"lr": args.lr},
            gradient_clip_val=args.gradient_clip_val,
        )

        print(f"Imputing VISTA fold {fold}...")
        # VISTA's implementation returns a one-element list containing the
        # spatial prediction, unlike stock gimVI which commonly returns both.
        imputation = model.get_imputed_values(
            normalized=True,
            deterministic=True,
            batch_size=args.batch_size,
        )[0]
        imputation = np.asarray(imputation, dtype=np.float32)

        expected = (sp_fold.n_obs, len(common_genes))
        if imputation.shape != expected:
            raise ValueError(
                f"Fold {fold}: unexpected VISTA output {imputation.shape}; expected {expected}."
            )

        pred_normalized = imputation[:, test_ind]
        my_test = np.log1p(
            np.clip(pred_normalized, a_min=0.0, a_max=None) * 1e4
        ).astype(np.float32)

        spatial_obs_names = sp_fold.obs_names.copy()
        true_data = dense_float32(
            adata_spatial_eval[:, test_gene].X
        )

        if my_test.shape != true_data.shape:
            raise ValueError(
                f"Fold {fold}: prediction/true shape mismatch "
                f"{my_test.shape} vs {true_data.shape}"
            )
        if not np.isfinite(my_test).all():
            raise ValueError(f"Fold {fold}: VISTA predictions contain NaN/Inf")

        results_val = compute_metrics(true_data, my_test, genes=test_gene)
        results_val = {
            "fold": fold,
            "phase": "val",
            "loss": float(results_val.get("l2_errors_mean", 0.0)),
            **results_val,
        }
        fold_metrics[fold] = results_val

        print("Fold metrics:")
        print(results_val)

        # Same cells and same row order in every fold.
        all_pred_res[:, test_ind] = my_test

        pd.DataFrame(
            my_test, index=output_cell_ids, columns=test_gene
        ).to_csv(os.path.join(csv_dir, f"fold_{fold}_pred.csv"), index=True)

        pd.DataFrame(
            true_data, index=output_cell_ids, columns=test_gene
        ).to_csv(os.path.join(csv_dir, f"fold_{fold}_true.csv"), index=True)

        pd.DataFrame({"train_gene": train_gene}).to_csv(
            os.path.join(csv_dir, f"fold_{fold}_train_genes.csv"), index=False
        )
        pd.DataFrame({"test_gene": test_gene}).to_csv(
            os.path.join(csv_dir, f"fold_{fold}_test_genes.csv"), index=False
        )
        # Compatibility with earlier baseline filename.
        pd.DataFrame({"test_gene": test_gene}).to_csv(
            os.path.join(csv_dir, f"fold_{fold}_genes.csv"), index=False
        )
        pd.DataFrame({"cell_id": output_cell_ids}).to_csv(
            os.path.join(csv_dir, f"fold_{fold}_cells.csv"), index=False
        )
        pd.DataFrame([results_val]).to_csv(
            os.path.join(csv_dir, f"fold_{fold}_metrics.csv"), index=False
        )

        if args.save_model:
            model.save(
                os.path.join(model_dir, f"fold_{fold}"),
                overwrite=True,
                save_anndata=False,
            )

        del model, imputation, pred_normalized, my_test, true_data, seq_fold, sp_fold
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Aggregate metrics
    # -------------------------------------------------------------------------
    df_metrics = pd.DataFrame.from_dict(fold_metrics, orient="index")
    metrics_path = os.path.join(csv_dir, "fold_metrics.csv")
    df_metrics.to_csv(metrics_path, index=False)

    numeric_cols = df_metrics.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "fold"]
    df_mean = df_metrics[numeric_cols].mean().to_frame().T
    df_mean.index = ["mean"]
    df_std = df_metrics[numeric_cols].std().to_frame().T
    df_std.index = ["std"]
    df_summary = pd.concat([df_mean, df_std])
    summary_path = os.path.join(csv_dir, "metrics_summary.csv")
    df_summary.to_csv(summary_path, index=True)

    if not np.isfinite(all_pred_res).all():
        n_bad = int(np.sum(~np.isfinite(all_pred_res)))
        raise ValueError(
            f"Complete CV prediction matrix contains {n_bad} NaN/Inf values."
        )

    all_pred_df = pd.DataFrame(
        all_pred_res,
        index=output_cell_ids,
        columns=common_genes,
    )
    all_pred_path = os.path.join(outdir, "VISTA_all_predictions.csv")
    all_pred_df.to_csv(all_pred_path, index=True)

    # Also save the aligned metric-scale truth for convenient combined analysis.
    true_all_path = os.path.join(outdir, "ST_truth_log1p_1e4.csv")
    pd.DataFrame(
        dense_float32(adata_spatial_eval.X),
        index=output_cell_ids,
        columns=adata_spatial_eval.var_names,
    ).to_csv(true_all_path, index=True)

    print("\n" + "=" * 70)
    print("VISTA 5-fold evaluation completed")
    print("=" * 70)
    print(df_summary)
    print("Fold metrics:", metrics_path)
    print("Summary:", summary_path)
    print("Complete predictions:", all_pred_path)


if __name__ == "__main__":
    main()
