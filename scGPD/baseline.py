# JSG
'''

nohup python3 -u baseline.py \
  --sc_data scRNA.h5ad \
  --sp_data spatial.h5ad \
  --document lungsCAN \
  --filename lungsCAN \
  --rand 0 \
  > scGPD_lungsCAN.log 2>&1 &
'''



import os
import sys
import random
import warnings
import argparse

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F

from scipy import sparse
from sklearn.model_selection import KFold

from compute_metrics import compute_metrics, spearmanrr

warnings.filterwarnings("ignore")


# =============================================================================
# scGPD import
# =============================================================================
# This file is intended to live in the scGPD-main repository root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from scGPD import (
    scGPD,
    ExpressionDataset,
    MSELoss,
)
from scGPD import functions as scgpd_functions


# =============================================================================
# GPU-safe patch for the repository's correlated Bernoulli-Concrete sampler
# =============================================================================
# The uploaded functions.py creates z on CPU:
#     z = torch.rand(uniform_shape)
# while L/model may be on CUDA.  Patch the function at runtime so no repository
# source file has to be edited separately.
def _bernoulli_concrete_sample_device_safe(
    logits,
    temperature,
    L,
    shape=torch.Size([]),
):
    uniform_shape = torch.Size(shape) + logits.shape

    z = torch.rand(
        uniform_shape,
        dtype=logits.dtype,
        device=logits.device,
    )

    # L must be on the same device/dtype as logits.
    L = L.to(device=logits.device, dtype=logits.dtype)

    u = z @ L.t()

    normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=u.dtype, device=u.device),
        torch.tensor(1.0, dtype=u.dtype, device=u.device),
    )
    u = normal.cdf(u)

    eps = torch.finfo(u.dtype).eps
    u = torch.clamp(u, min=eps, max=1.0 - eps)

    return torch.sigmoid(
        (
            F.logsigmoid(logits)
            - F.logsigmoid(-logits)
            + torch.log(u)
            - torch.log1p(-u)
        )
        / temperature
    )


scgpd_functions.bernoulli_concrete_sample = (
    _bernoulli_concrete_sample_device_safe
)


# =============================================================================
# Arguments
# =============================================================================
parser = argparse.ArgumentParser(
    description="Run adapted scGPD with the same 5-fold gene masking protocol"
)

parser.add_argument("--sc_data", type=str, default="dataset5_seq_915.h5ad")
parser.add_argument("--sp_data", type=str, default="dataset5_spatial_915.h5ad")
parser.add_argument("--document", type=str, default="dataset5")
parser.add_argument("--filename", type=str, default="Dataset")
parser.add_argument("--rand", type=int, default=0)

# scGPD options
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--scgpd_epochs", type=int, default=200)
parser.add_argument("--scgpd_batchsize", type=int, default=64)
parser.add_argument("--scgpd_lr", type=float, default=1e-3)
parser.add_argument("--scgpd_lam", type=float, default=1e-4)
parser.add_argument("--scgpd_lookback", type=int, default=10)
parser.add_argument("--scgpd_inference_batchsize", type=int, default=4096)

args = parser.parse_args()


# =============================================================================
# Reproducibility
# =============================================================================
random.seed(args.rand)
np.random.seed(args.rand)
torch.manual_seed(args.rand)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.rand)


# =============================================================================
# General settings
# =============================================================================
n_splits = 5

print("adata_spatial:", args.sp_data)
print("adata_seq:", args.sc_data)


# =============================================================================
# Load data -- same path logic as your existing baseline
# =============================================================================

adata_spatial = sc.read_h5ad(
    f"YOUR PATH..."
    f"{args.filename}/{args.sp_data}"
)

adata_seq = sc.read_h5ad(
    f"YOUR PATH..."
    f"{args.filename}/{args.sc_data}"
)

print(f"sp shape before filtering: {adata_spatial.shape}")
print(f"sc shape before filtering: {adata_seq.shape}")


# =============================================================================
# Preprocessing -- same as your existing baseline pipeline
# =============================================================================
sc.pp.filter_cells(adata_seq, min_counts=1)
sc.pp.filter_genes(adata_seq, min_counts=1)

sc.pp.filter_cells(adata_spatial, min_counts=1)
sc.pp.filter_genes(adata_spatial, min_counts=1)

common_genes = np.intersect1d(
    adata_spatial.var_names,
    adata_seq.var_names,
)

adata_spatial = adata_spatial[:, common_genes].copy()
adata_seq = adata_seq[:, common_genes].copy()

print(f"The number of common genes: {len(common_genes)}")
print(f"Spatial after intersection: {adata_spatial.shape}")
print(f"scRNA after intersection: {adata_seq.shape}")


# =============================================================================
# PRECOMPUTED MATRICES USED BY ALL BASELINES
# =============================================================================
# scRNA: normalize_total(1e4) + log1p
adata_seq2 = adata_seq.copy()
sc.pp.normalize_total(adata_seq2, target_sum=1e4)
sc.pp.log1p(adata_seq2)
data_seq_array = adata_seq2.X

# Spatial: preserve your merFISH rule
adata_spatial2 = adata_spatial.copy()

sc.pp.normalize_total(adata_spatial2, target_sum=1e4)
sc.pp.log1p(adata_spatial2)

data_spatial_array = adata_spatial2.X

sp_genes = np.array(adata_spatial.var_names)


def to_dataframe(x, columns):
    """Convert sparse or dense matrix to a pandas DataFrame."""
    if sparse.issparse(x):
        x = x.toarray()
    else:
        x = np.asarray(x)

    return pd.DataFrame(x, columns=columns)


sp_data = to_dataframe(
    data_spatial_array,
    sp_genes,
)

sc_data = to_dataframe(
    data_seq_array,
    sp_genes,
)

print(
    f"sp_data: {sp_data.shape}, "
    f"sc_data: {sc_data.shape}"
)


# =============================================================================
# Utilities
# =============================================================================
def build_correlation_factor(X):
    """
    Build the L factor required by scGPD from OUTER TRAIN genes only.

    X shape:
        [n_scRNA_cells, n_outer_train_genes]

    Returns:
        L where corr_psd ~= L @ L.T
    """
    print(
        "Building scGPD correlation matrix from "
        f"{X.shape[1]} OUTER train genes..."
    )

    # Correlation across genes.
    corr = np.corrcoef(
        X,
        rowvar=False,
    )

    corr = np.asarray(
        corr,
        dtype=np.float64,
    )

    corr = np.nan_to_num(
        corr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Force exact symmetry.
    corr = (corr + corr.T) / 2.0

    # Ensure diagonal is valid.
    np.fill_diagonal(
        corr,
        1.0,
    )

    # Numerical PSD repair.
    eigvals, eigvecs = np.linalg.eigh(corr)

    eigvals = np.clip(
        eigvals,
        1e-5,
        None,
    )

    corr_psd = (
        eigvecs
        @ np.diag(eigvals)
        @ eigvecs.T
    )

    corr_psd = (
        corr_psd
        + corr_psd.T
    ) / 2.0

    corr_psd += (
        np.eye(
            corr_psd.shape[0],
            dtype=np.float64,
        )
        * 1e-5
    )

    L = np.linalg.cholesky(
        corr_psd
    ).astype(np.float32)

    print(
        "scGPD L shape:",
        L.shape,
    )

    return L


@torch.no_grad()
def predict_scgpd_reconstruction(
    model,
    X,
    device,
    batch_size=4096,
):
    """
    Deterministic reconstruction with the trained Stage-I gate probabilities.

    scGPD's SelectorMLP.forward() samples a stochastic gate every call.
    For evaluation we use a deterministic hard gate:
        probability > 0.5

    This is also the threshold used by scGPD eliminate() to report retained genes.
    """
    model.eval()

    gate_probs = (
        model.input_layer.probs
        .detach()
        .to(device)
    )

    hard_gate = (
        gate_probs > 0.5
    ).to(torch.float32)

    n_selected = int(
        hard_gate.sum().item()
    )

    # Extremely defensive fallback in case no gate crosses 0.5.
    if n_selected == 0:
        print(
            "WARNING: no gate probability > 0.5; "
            "using continuous gate probabilities for inference."
        )
        gate = gate_probs
    else:
        gate = hard_gate

    print(
        f"Deterministic scGPD inference gates active: "
        f"{n_selected}/{gate_probs.numel()}"
    )

    pred_batches = []

    for start in range(
        0,
        X.shape[0],
        batch_size,
    ):
        end = min(
            start + batch_size,
            X.shape[0],
        )

        x_batch = torch.as_tensor(
            X[start:end],
            dtype=torch.float32,
            device=device,
        )

        # SelectorMLP Stage-I uses:
        #   gated input -> model.mlp -> full reconstruction.
        pred_batch = model.mlp(
            x_batch * gate
        )

        pred_batches.append(
            pred_batch.detach().cpu().numpy()
        )

    return np.concatenate(
        pred_batches,
        axis=0,
    )


# =============================================================================
# scGPD baseline
# =============================================================================
def scGPD_impute():

    print("\n============================================================")
    print("We run adapted scGPD for this data")
    print("Same OUTER 5-fold GENE split as the other baselines")
    print("NO extra outer cell split")
    print("============================================================\n")

    global sc_data, sp_data, adata_seq, adata_spatial

    outdir = os.path.join(
        "Result",
        args.document,
        "scGPD",
    )

    csv_dir = os.path.join(
        outdir,
        "fold_results",
    )

    os.makedirs(
        csv_dir,
        exist_ok=True,
    )

    raw_shared_gene = np.array(
        adata_spatial.var_names
    )

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=args.rand,
    )

    all_pred_res = np.zeros(
        (
            adata_spatial.n_obs,
            len(raw_shared_gene),
        ),
        dtype=np.float32,
    )

    fold_metrics = {}

    # Full preprocessed matrices.
    X_sc_full = sc_data.values.astype(
        np.float32,
        copy=False,
    )

    X_st_full = sp_data.values.astype(
        np.float32,
        copy=False,
    )

    if torch.cuda.is_available():
        device = torch.device(
            f"cuda:{args.gpu}"
        )
    else:
        device = torch.device("cpu")

    print("scGPD device:", device)

    # idx = 1

    for fold, (train_ind, test_ind) in enumerate(
        kf.split(raw_shared_gene), start=1
    ):

        train_gene = raw_shared_gene[
            train_ind
        ]

        test_gene = raw_shared_gene[
            test_ind
        ]

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"scGPD Fold {fold}"
        )

        print(
            f"Train genes: {len(train_gene)}"
        )

        print(
            f"Test genes:  {len(test_gene)}"
        )

        print(
            "=" * 70
        )

        # ---------------------------------------------------------
        # SAME gene protocol as other baselines
        #
        # scRNA input  = only OUTER train genes
        # scRNA target = all common genes
        # ST input     = only OUTER train genes
        # Evaluation   = only OUTER test genes
        # ---------------------------------------------------------
        X_sc_input = X_sc_full[
            :,
            train_ind,
        ]

        X_sc_target = X_sc_full

        X_st_input = X_st_full[
            :,
            train_ind,
        ]

        y_true = X_st_full[
            :,
            test_ind,
        ]

        print(
            "scRNA input:",
            X_sc_input.shape,
        )

        print(
            "scRNA target:",
            X_sc_target.shape,
        )

        print(
            "ST input:",
            X_st_input.shape,
        )

        print(
            "Ground-truth held-out ST:",
            y_true.shape,
        )

        # ---------------------------------------------------------
        # scGPD datasets
        # ---------------------------------------------------------
        # You requested gene split only. Therefore there is no separate
        # benchmark cell split here. scGPD still requires train and val
        # dataset objects, so both reference the same scRNA data.
        train_dataset = ExpressionDataset(
            X_sc_input,
            X_sc_target,
        )

        val_dataset = ExpressionDataset(
            X_sc_input,
            X_sc_target,
        )

        # ---------------------------------------------------------
        # Correlation factor L from TRAIN GENES ONLY
        # ---------------------------------------------------------
        L_np = build_correlation_factor(
            X_sc_input
        )

        L = torch.as_tensor(
            L_np,
            dtype=torch.float32,
            device=device,
        )

        # ---------------------------------------------------------
        # Construct actual scGPD wrapper
        # ---------------------------------------------------------
        loss_fn = MSELoss()

        selector = scGPD(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            loss_fn=loss_fn,
            device=device,
            L=L,
            hidden=[128, 128],
            seed=args.rand,
        )

        # ---------------------------------------------------------
        # Stage-I correlated binary-gate reconstruction
        # ---------------------------------------------------------
        # eliminate() requires target < input_size in this repository.
        # The current uploaded implementation does NOT create another
        # train/test gene split from this target.  It trains one Stage-I
        # binary-gate reconstruction model on all OUTER train genes.
        eliminate_target = max(
            1,
            len(train_ind) - 1,
        )

        print(
            "Training scGPD Stage-I reconstruction..."
        )

        selected_relative_inds, model = selector.eliminate(
            target=eliminate_target,
            lam_init=args.scgpd_lam,
            mbsize=args.scgpd_batchsize,
            max_nepochs=args.scgpd_epochs,
            lr=args.scgpd_lr,
            start_temperature=10.0,
            end_temperature=0.01,
            lookback=args.scgpd_lookback,
            bar=True,
            verbose=True,
        )

        # selected_relative_inds indexes X_sc_input, i.e. the OUTER
        # train-gene matrix, not all common genes.
        selected_relative_inds = np.asarray(
            selected_relative_inds,
            dtype=int,
        )

        selected_global_inds = train_ind[
            selected_relative_inds
        ]

        selected_gene_names = raw_shared_gene[
            selected_global_inds
        ]

        print(
            "scGPD retained genes:",
            len(selected_gene_names),
            "/",
            len(train_gene),
        )

        pd.DataFrame(
            {
                "selected_gene": selected_gene_names
            }
        ).to_csv(
            os.path.join(
                csv_dir,
                f"fold_{fold}_selected_genes.csv",
            ),
            index=False,
        )

        # ---------------------------------------------------------
        # Apply trained Stage-I reconstruction model to ST
        # ---------------------------------------------------------
        pred_full = predict_scgpd_reconstruction(
            model=model,
            X=X_st_input,
            device=device,
            batch_size=args.scgpd_inference_batchsize,
        )

        # Model output is full common transcriptome.
        y_pred = pred_full[
            :,
            test_ind,
        ].astype(
            np.float32,
            copy=False,
        )

        print(
            "Full reconstruction:",
            pred_full.shape,
        )

        print(
            "Prediction:",
            y_pred.shape,
        )

        print(
            "Ground truth:",
            y_true.shape,
        )

        # ---------------------------------------------------------
        # Same metrics as your other baselines
        # ---------------------------------------------------------
        results_val = compute_metrics(
            y_true,
            y_pred,
            genes=test_gene,
        )

        results_val.update(
            {
                "fold": fold,
                "phase": "val",
                "loss": float(
                    results_val.get(
                        "l2_errors_mean",
                        0.0,
                    )
                ),
            }
        )

        fold_metrics[
            fold
        ] = results_val

        print(
            "\nFold metrics:"
        )

        print(
            results_val
        )

        all_pred_res[
            :,
            test_ind,
        ] = y_pred

        # ---------------------------------------------------------
        # Save fold output
        # ---------------------------------------------------------
        pd.DataFrame(
            y_pred,
            columns=test_gene,
        ).to_csv(
            os.path.join(
                csv_dir,
                f"fold_{fold}_pred.csv",
            ),
            index=False,
        )

        pd.DataFrame(
            y_true,
            columns=test_gene,
        ).to_csv(
            os.path.join(
                csv_dir,
                f"fold_{fold}_true.csv",
            ),
            index=False,
        )

        # Free fold GPU memory.
        del (
            selector,
            model,
            L,
            L_np,
            pred_full,
            y_pred,
            train_dataset,
            val_dataset,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # idx += 1

    # =============================================================================
    # Metrics summary
    # =============================================================================
    df_metrics = pd.DataFrame.from_dict(
        fold_metrics,
        orient="index",
    )

    df_metrics.to_csv(
        os.path.join(
            csv_dir,
            "fold_metrics.csv",
        ),
        index=False,
    )

    numeric_cols = df_metrics.select_dtypes(
        include=[np.number]
    ).columns

    df_mean = (
        df_metrics[
            numeric_cols
        ]
        .mean()
        .to_frame(
            "mean"
        )
        .T
    )

    df_std = (
        df_metrics[
            numeric_cols
        ]
        .std()
        .to_frame(
            "std"
        )
        .T
    )

    df_summary = pd.concat(
        [
            df_mean,
            df_std,
        ]
    )

    df_summary.to_csv(
        os.path.join(
            csv_dir,
            "metrics_summary.csv",
        ),
        index=True,
    )

    # Save the complete 5-fold reconstructed test-gene matrix.
    pd.DataFrame(
        all_pred_res,
        columns=raw_shared_gene,
    ).to_csv(
        os.path.join(
            outdir,
            "all_pred_res.csv",
        ),
        index=False,
    )

    print(
        "\n============================================================"
    )
    print(
        "scGPD finished"
    )
    print(
        "all_pred_res:",
        all_pred_res.shape,
    )
    print(
        "============================================================"
    )

    return all_pred_res


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":

    scGPD_result = scGPD_impute()

    print(
        "scGPD final prediction shape:",
        scGPD_result.shape,
    )
