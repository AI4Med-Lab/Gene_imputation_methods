#!/usr/bin/env python3
"""Run SpaGene five-fold held-out-gene benchmarking."""

'''

nohup python3 -u baseline.py \
  --sc_data scRNA.h5ad \
  --sp_data spatial.h5ad \
  --document starMAP \
  --filename starMAP \
  --rand 0 \
  --n_splits 5 \
  --transform log1p \
  --uppercase_genes \
  > SpaGene_starMAP.log 2>&1 &

'''


import argparse
import gc
import json
import os
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.model_selection import KFold
from model import SpaGeneNet
from training import train_autoencoder, train_translation, predict_full
from compute_metrics import compute_metrics
from utils import (set_seed, dense_float32, apply_transform,
                   filter_nonempty, align_common_genes)

def parse_args():
    p = argparse.ArgumentParser(
        description="SpaGene 5-fold gene-imputation baseline for AnnData inputs."
    )
    p.add_argument("--sc_data", type=str, required=True)
    p.add_argument("--sp_data", type=str, required=True)
    p.add_argument("--document", type=str, required=True)
    p.add_argument("--filename", type=str, required=True)
    p.add_argument("--rand", type=int, default=0)

    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--eval_batch_size", type=int, default=1024)
    p.add_argument("--epochs_ae", type=int, default=100)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr_ae", type=float, default=1e-3)
    p.add_argument("--lr_trans", type=float, default=1e-3)
    p.add_argument("--lr_disc", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)

    p.add_argument("--lambda_cyc", type=float, default=1.0)
    p.add_argument("--lambda_id", type=float, default=1.0)
    p.add_argument("--lambda_adv", type=float, default=0.1)
    p.add_argument("--lambda_disc", type=float, default=0.05)
    p.add_argument("--lambda_gp", type=float, default=3.0)
    p.add_argument("--gradient_clip", type=float, default=10.0)

    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-4)

    p.add_argument(
        "--transform",
        choices=["none", "sqrt", "log1p"],
        default="none",
        help=(
            "Expression transform applied to BOTH domains and to evaluation truth. "
            "'none' uses .X exactly as stored. The published SpaGene default is sqrt."
        ),
    )
    p.add_argument(
        "--uppercase_genes",
        action="store_true",
        help="Uppercase/strip gene names before intersection.",
    )
    p.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--result_root", type=str, default="Result")

    p.add_argument(
        "--sc_path",
        type=str,
        default=None,
        help="Optional full path to scRNA h5ad; overrides filename-based path.",
    )
    p.add_argument(
        "--sp_path",
        type=str,
        default=None,
        help="Optional full path to ST h5ad; overrides filename-based path.",
    )

    return p.parse_args()


def resolve_paths(args):
    if args.sp_path is not None:
        sp_path = args.sp_path
    else:
        sp_path = (
            f"YOUR PATH..."
            f"{args.filename}/{args.sp_data}"
        )

    if args.sc_path is not None:
        sc_path = args.sc_path
    else:
        sc_path = (
            f"YOUR PATH..."
            f"{args.filename}/{args.sc_data}"
        )
    return sc_path, sp_path


def save_model_state(model, path, metadata):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def main():
    args = parse_args()
    set_seed(args.rand)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sc_path, sp_path = resolve_paths(args)

    print("=" * 76)
    print("SpaGene 5-fold gene-imputation baseline")
    print("=" * 76)
    print("scRNA:", sc_path)
    print("Spatial:", sp_path)
    print("Device:", device)
    print("Seed:", args.rand)
    print("Transform:", args.transform)

    adata_spatial = sc.read_h5ad(sp_path)
    adata_seq = sc.read_h5ad(sc_path)

    print("Before filtering:")
    print("  ST:", adata_spatial.shape)
    print("  scRNA:", adata_seq.shape)

    adata_spatial = filter_nonempty(adata_spatial, "ST")
    adata_seq = filter_nonempty(adata_seq, "scRNA")

    adata_spatial, adata_seq, common_genes = align_common_genes(adata_spatial, adata_seq, uppercase=args.uppercase_genes)

    if len(common_genes) < args.n_splits:
        raise ValueError(
            f"Only {len(common_genes)} common genes; cannot run {args.n_splits}-fold CV."
        )

    # Transform after alignment so training and truth are on exactly the same scale.
    adata_spatial = apply_transform(adata_spatial, args.transform)
    adata_seq = apply_transform(adata_seq, args.transform)

    x_st_full = dense_float32(adata_spatial.X)
    x_sc_full = dense_float32(adata_seq.X)

    if not np.isfinite(x_st_full).all() or not np.isfinite(x_sc_full).all():
        raise ValueError("Expression matrices contain NaN/Inf after preprocessing.")

    print("After alignment:")
    print("  ST:", x_st_full.shape)
    print("  scRNA:", x_sc_full.shape)
    print("  Common genes:", len(common_genes))
    print(
        "  ST min/max/mean:",
        float(np.min(x_st_full)),
        float(np.max(x_st_full)),
        float(np.mean(x_st_full)),
    )
    print(
        "  scRNA min/max/mean:",
        float(np.min(x_sc_full)),
        float(np.max(x_sc_full)),
        float(np.mean(x_sc_full)),
    )

    outdir = os.path.join(args.result_root, args.document, "SpaGene")
    csv_dir = os.path.join(outdir, "fold_results")
    model_dir = os.path.join(outdir, "models")
    os.makedirs(csv_dir, exist_ok=True)
    if args.save_model:
        os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(outdir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    all_pred_res = np.full(
        (adata_spatial.n_obs, len(common_genes)), np.nan, dtype=np.float32
    )
    fold_metrics = {}

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.rand)

    for fold, (train_ind, test_ind) in enumerate(kf.split(common_genes), start=1):
        # Fold-specific seed keeps runs reproducible while avoiding identical dropout
        # streams across folds.
        set_seed(args.rand + fold - 1)

        train_gene = common_genes[train_ind]
        test_gene = common_genes[test_ind]

        print("\n" + "=" * 76)
        print(f"Fold {fold}/{args.n_splits}")
        print(f"Train genes: {len(train_gene)} | Test genes: {len(test_gene)}")
        print("=" * 76)

        # Critical protocol:
        # ST contains only train genes. scRNA retains all common genes.
        x_st = np.ascontiguousarray(x_st_full[:, train_ind], dtype=np.float32)
        x_sc = np.ascontiguousarray(x_sc_full, dtype=np.float32)

        model = SpaGeneNet(
            n_st_genes=len(train_ind),
            n_sc_genes=len(common_genes),
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")

        print("Stage 1/2: pretraining ST autoencoder")
        st_ae_loss = train_autoencoder(
            model.enc_st,
            model.dec_st,
            x_st,
            epochs=args.epochs_ae,
            batch_size=args.batch_size,
            lr=args.lr_ae,
            device=device,
            use_amp=args.amp,
            label="ST",
        )

        print("Stage 1/2: pretraining scRNA autoencoder")
        sc_ae_loss = train_autoencoder(
            model.enc_sc,
            model.dec_sc,
            x_sc,
            epochs=args.epochs_ae,
            batch_size=args.batch_size,
            lr=args.lr_ae,
            device=device,
            use_amp=args.amp,
            label="scRNA",
        )

        print("Stage 2/2: training SpaGene translators/discriminators")
        best_visible_corr = train_translation(
            model,
            x_st,
            x_sc,
            train_indices_in_sc=np.asarray(train_ind, dtype=np.int64),
            args=args,
            device=device,
        )

        print("Imputing held-out genes...")
        pred_full = predict_full(model, x_st, args.eval_batch_size, device)
        y_pred = np.asarray(pred_full[:, test_ind], dtype=np.float32)

        # Held-out ST truth is touched only now, after training/model selection.
        y_true = np.asarray(x_st_full[:, test_ind], dtype=np.float32)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Fold {fold}: prediction/true shape mismatch "
                f"{y_pred.shape} vs {y_true.shape}"
            )
        if not np.isfinite(y_pred).all():
            raise ValueError(f"Fold {fold}: predictions contain NaN/Inf")

        all_pred_res[:, test_ind] = y_pred

        results_val = compute_metrics(y_true, y_pred, genes=test_gene)
        results_val = {
            "fold": fold,
            "phase": "val",
            "loss": float(results_val.get("l2_errors_mean", 0.0)),
            "st_ae_mse": st_ae_loss,
            "sc_ae_mse": sc_ae_loss,
            "visible_checkpoint_pearson": best_visible_corr,
            **results_val,
        }
        fold_metrics[fold] = results_val

        print("Fold metrics:")
        print(results_val)

        obs_names = adata_spatial.obs_names
        pd.DataFrame(y_pred, index=obs_names, columns=test_gene).to_csv(os.path.join(csv_dir, f"fold_{fold}_pred.csv"), index=True)
        pd.DataFrame(y_true, index=obs_names, columns=test_gene).to_csv(os.path.join(csv_dir, f"fold_{fold}_true.csv"), index=True)
        pd.DataFrame({"train_gene": train_gene}).to_csv(os.path.join(csv_dir, f"fold_{fold}_train_genes.csv"), index=False)
        pd.DataFrame({"test_gene": test_gene}).to_csv(os.path.join(csv_dir, f"fold_{fold}_test_genes.csv"), index=False)
        pd.DataFrame({"cell_id": obs_names}).to_csv(os.path.join(csv_dir, f"fold_{fold}_cells.csv"), index=False)
        pd.DataFrame([results_val]).to_csv(os.path.join(csv_dir, f"fold_{fold}_metrics.csv"), index=False)

        if args.save_model:
            save_model_state(
                model,
                os.path.join(model_dir, f"fold_{fold}.pt"),
                {
                    "fold": fold,
                    "train_genes": train_gene.tolist(),
                    "test_genes": test_gene.tolist(),
                    "common_genes": common_genes.tolist(),
                    "transform": args.transform,
                },
            )

        del model, pred_full, y_pred, y_true, x_st
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Aggregate outputs
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

    pred_path = os.path.join(outdir, "SpaGene_all_predictions.csv")
    pd.DataFrame(
        all_pred_res,
        index=adata_spatial.obs_names,
        columns=common_genes,
    ).to_csv(pred_path, index=True)

    truth_path = os.path.join(outdir, "ST_truth.csv")
    pd.DataFrame(
        x_st_full,
        index=adata_spatial.obs_names,
        columns=common_genes,
    ).to_csv(truth_path, index=True)

    print("\n" + "=" * 76)
    print("SpaGene 5-fold evaluation completed")
    print("=" * 76)
    print(df_summary)
    print("Fold metrics:", metrics_path)
    print("Summary:", summary_path)
    print("Complete CV predictions:", pred_path)
    print("Truth:", truth_path)


if __name__ == "__main__":
    main()
