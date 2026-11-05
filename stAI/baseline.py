'''
nohup python3 -u baseline.py \
  --sc_data scRNAseq_full.h5ad \
  --sp_data spatial.h5ad \
  --document merFISHBaselineFullset_new \
  --filename merFISH \
  --rand 0 \
  --gpu 0 \
  > stAI_merFISH.log 2>&1 &


'''



import os
import sys
import copy
import random
import warnings
import argparse

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse
from sklearn.model_selection import KFold

from compute_metrics import compute_metrics, spearmanrr
# from process.result_analysis import clustering_metrics
# from process.result_analysis import *
# from process.data import *

warnings.filterwarnings("ignore")

# IMPORTANT:
# Do NOT set the global default tensor type to CUDA.
# PyTorch DataLoader/RandomSampler uses a CPU torch.Generator internally.
# Making CUDA the global default causes:
# RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'
#
# stAI explicitly moves its tensors/model to cuda:<device_id>, so no global
# CUDA default is needed.
torch.set_default_dtype(torch.float32)

# -----------------------------------------------------------------------------
# stAI import
# Expected repository structure:
# baseline/stAI-main/stAI/main.py
# -----------------------------------------------------------------------------
STAI_ROOT = os.path.join("baseline", "stAI-main")
if STAI_ROOT not in sys.path:
    sys.path.append(STAI_ROOT)

from stAI.main import train_and_impute
from stAI.utils import process_celltype


# =============================================================================
# Arguments
# =============================================================================
parser = argparse.ArgumentParser(description="Run stAI with 5-fold gene masking")
parser.add_argument("--sc_data", type=str, default="dataset5_seq_915.h5ad")
parser.add_argument("--sp_data", type=str, default="dataset5_spatial_915.h5ad")
parser.add_argument("--document", type=str, default="dataset5")
parser.add_argument("--filename", type=str, default="Dataset")
parser.add_argument("--rand", type=int, default=0)

# stAI-specific options
parser.add_argument("--stai_epochs", type=int, default=200)
parser.add_argument("--stai_inner_folds", type=int, default=5)
parser.add_argument("--stai_lr", type=float, default=0.002)
parser.add_argument("--stai_spatial_knn", type=int, default=10)
parser.add_argument("--stai_rna_batchsize", type=int, default=-1)
parser.add_argument("--stai_spatial_batchsize", type=int, default=-1)
parser.add_argument("--stai_inference_batchsize", type=int, default=2048)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--celltype_col", type=str, default=None)

args = parser.parse_args()


# =============================================================================
# General settings
# =============================================================================
n_splits = 5

print("adata_spatial:", args.sp_data)
print("adata_seq:", args.sc_data)


# =============================================================================
# Load data -- same path logic as your current baseline file
# =============================================================================
if args.filename != "merFISH":
    adata_spatial = sc.read_h5ad(
        f"/home/2023csz0012/AmitCode/data/processedData/{args.filename}/{args.sp_data}"
    )
else:
    adata_spatial = sc.read_h5ad(
        f"/home/2023csz0012/AmitCode/data/updatedDataset/{args.filename}/{args.sp_data}"
    )

adata_seq = sc.read_h5ad(
    f"/home/2023csz0012/AmitCode/data/processedData/{args.filename}/{args.sc_data}"
)

print(f"sp shape before filtering: {adata_spatial.shape}")
print(f"sc shape before filtering: {adata_seq.shape}")


# =============================================================================
# Amit preprocessing -- kept consistent with your existing baseline pipeline
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
# scRNA matrix: normalize_total(1e4) + log1p
adata_seq2 = adata_seq.copy()
sc.pp.normalize_total(adata_seq2, target_sum=1e4)
sc.pp.log1p(adata_seq2)
data_seq_array = adata_seq2.X

# Spatial matrix: preserve your existing merFISH rule
adata_spatial2 = adata_spatial.copy()
# if args.filename != "merFISH":
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


sp_data = to_dataframe(data_spatial_array, sp_genes)
sc_data = to_dataframe(data_seq_array, sp_genes)

print(f"sp_data: {sp_data.shape}, sc_data: {sc_data.shape}")


# =============================================================================
# Utility
# =============================================================================
def dataSparse(x):
    """Convert sparse or dense matrix to a NumPy array."""
    if sparse.issparse(x):
        x = x.toarray()
    else:
        x = np.asarray(x)
    return x


def make_dense_float32(x):
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def ensure_celltype_column(adata):
    """
    stAI expects adata_rna.obs['celltype'] because the original model contains
    an auxiliary scRNA cell-type classifier.

    If real annotations are available, use them.

    If the scRNA AnnData has no annotation column (for example only n_counts),
    create one dummy class for all cells. In stAI_impute() we then set
    lam_clf = 0.0, so this dummy label does NOT contribute to training.
    """
    adata = adata.copy()
    original_obs_columns = list(adata.obs.columns)
    using_dummy_celltype = False

    if args.celltype_col is not None:
        if args.celltype_col not in adata.obs.columns:
            raise ValueError(
                f"--celltype_col '{args.celltype_col}' was not found. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        source_col = args.celltype_col
        adata.obs["celltype"] = adata.obs[source_col].astype(str)

    elif "celltype" in adata.obs.columns:
        source_col = "celltype"
        adata.obs["celltype"] = adata.obs[source_col].astype(str)

    else:
        candidates = [
            "cell_type",
            "subclass",
            "class_label",
            "label",
            "leiden",
        ]
        source_col = next((c for c in candidates if c in adata.obs.columns), None)

        if source_col is not None:
            adata.obs["celltype"] = adata.obs[source_col].astype(str)

        else:
            # No biological cell-type labels are available.
            # Keep stAI's API satisfied with a single dummy class.
            # The classification loss is disabled in stAI_impute().
            using_dummy_celltype = True
            source_col = "__dummy_celltype__"
            print(
                "\n[stAI warning] No real scRNA cell-type annotation was found."
            )
            print(
                f"Original scRNA obs columns: {original_obs_columns}"
            )

            adata.obs["celltype"] = "unknown"
            print(
                "[stAI warning] Using one dummy cell type ('unknown') and "
                "disabling the auxiliary classification loss (lam_clf=0.0)."
            )

    print(f"stAI celltype source: {source_col}")
    print(f"Number of cell types: {adata.obs['celltype'].nunique()}")

    return adata, using_dummy_celltype


# =============================================================================
# stAI baseline
# =============================================================================
def stAI_impute():
    """
    Run stAI under the SAME outer 5-fold gene masking protocol as Tangram/SpaGE.

    For each outer fold:
        - all outer train genes are used by stAI (e.g. 180 genes)
        - there is NO inner 144/36 gene split
        - ST test genes are completely absent from training
        - scRNA test-gene expression is supplied only as SC_ref
        - one stAI model is trained per outer fold
        - final prediction is evaluated on the held-out ST genes

    To preserve stAI's cross-modal imputation objective without hiding another
    subset of genes, the same outer training genes are used as both:
        gene_fit = train_gene
        gene_calibration = train_gene

    Thus the model learns the ST<->scRNA relationship using all measured genes,
    analogous to Tangram learning its mapping from all outer training genes.
    """

    print("\n============================================================")
    print("Running stAI")
    print("============================================================\n")

    global sc_data, sp_data, adata_seq2, adata_spatial2

    # -------------------------------------------------------------------------
    # Parameters from the repository's demo/osmFISH_impute.yaml
    # -------------------------------------------------------------------------
    model_parameters = {
        "d_hidden": 64,
        "d_latent": 16,
        "lam_clf": 0.1,
        "lam_cos": 1,
        "lam_genegraph": 0,
        "lam_impute": 1,
        "lam_mmd": 0.1,
        "lam_recon": 1,
    }

    # -------------------------------------------------------------------------
    # Memory-safe minibatches for large datasets.
    # Original stAI uses -1 (= full batch), which creates an O((N_ST+N_SC)^2)
    # MMD matrix and is impossible for large merFISH.  When the user leaves the
    # batch-size arguments at -1, choose batches that give approximately the
    # same number of ST and scRNA batches because stAI trains with zip(...).
    # -------------------------------------------------------------------------
    if args.stai_spatial_batchsize == -1:
        spatial_batchsize = min(4096, adata_spatial2.n_obs)
    else:
        spatial_batchsize = args.stai_spatial_batchsize

    spatial_steps = max(1, adata_spatial2.n_obs // spatial_batchsize)

    if args.stai_rna_batchsize == -1:
        rna_batchsize = max(2, adata_seq2.n_obs // spatial_steps)
        rna_batchsize = min(rna_batchsize, adata_seq2.n_obs)
    else:
        rna_batchsize = args.stai_rna_batchsize

    training_parameters = {
        "device_id": args.gpu,
        "eval_step": 50,
        "k_folds": args.stai_inner_folds,
        "lr": args.stai_lr,
        "num_epoch": args.stai_epochs,
        "rna_batchsize": rna_batchsize,
        "spatial_batchsize": spatial_batchsize,
        "spatial_knn": args.stai_spatial_knn,
        "inference_batchsize": args.stai_inference_batchsize,
    }

    st_steps = adata_spatial2.n_obs // spatial_batchsize
    sc_steps = adata_seq2.n_obs // rna_batchsize
    print(
        f"stAI batches -> spatial_batchsize={spatial_batchsize} "
        f"({st_steps} full batches), rna_batchsize={rna_batchsize} "
        f"({sc_steps} full batches), inference_batchsize={args.stai_inference_batchsize}"
    )

    # -------------------------------------------------------------------------
    # Sanity checks
    # -------------------------------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The uploaded stAI code constructs torch.device('cuda:<id>') inside "
            "train_and_impute, so CUDA is required unless stAI/main.py is modified."
        )

    if "spatial" not in adata_spatial2.obsm:
        raise ValueError(
            "stAI requires adata_spatial.obsm['spatial']. "
            "Please create it before running this baseline."
        )

    print("Spatial coordinate shape:", adata_spatial2.obsm["spatial"].shape)

    # stAI requires celltype labels in RNA data.
    adata_seq_stai, using_dummy_celltype = ensure_celltype_column(adata_seq2)
    if using_dummy_celltype:
        model_parameters["lam_clf"] = 0.0
        print("stAI lam_clf set to 0.0 because real cell-type labels are unavailable.")

    # -------------------------------------------------------------------------
    # Output folders
    # -------------------------------------------------------------------------
    outdir = os.path.join("Result", args.document, "stAI")
    csv_dir = os.path.join(outdir, "fold_results")
    os.makedirs(csv_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # SAME outer gene split as your SpaGE function
    # -------------------------------------------------------------------------
    raw_shared_gene = np.array(adata_spatial2.var_names)

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=args.rand,
    )

    all_pred_res = np.zeros(
        (adata_spatial2.n_obs, adata_spatial2.n_vars),
        dtype=np.float32,
    )

    print("all_pred_res:", all_pred_res.shape)

    fold_metrics = {}
    gene_seq = []

    # -------------------------------------------------------------------------
    # OUTER 5-fold evaluation
    # -------------------------------------------------------------------------
    for idx, (train_ind, test_ind) in enumerate(kf.split(raw_shared_gene), start=1):

        print("\n============================================================")
        print(f"===== stAI OUTER Fold {idx}/{n_splits} =====")
        print(
            f"Number of train genes: {len(train_ind)}, "
            f"Number of test genes: {len(test_ind)}"
        )
        print("============================================================")

        train_gene = raw_shared_gene[train_ind]
        test_gene = raw_shared_gene[test_ind]

        gene_seq.append(test_gene.tolist())

        # ---------------------------------------------------------------------
        # IMPORTANT: ST AnnData contains ONLY OUTER TRAIN GENES.
        # This prevents held-out ST genes from entering training.
        # ---------------------------------------------------------------------
        adata_st_fold = adata_spatial2[:, train_gene].copy()

        # scRNA training AnnData also uses train genes only.
        adata_sc_fold = adata_seq_stai[:, train_gene].copy()

        # Keep test-gene scRNA expression separately as reference for imputation.
        # This is equivalent to original stAI run_impute(): SC_ref is extracted
        # BEFORE scaling the training-gene matrices.
        SC_ref = make_dense_float32(sc_data[test_gene].values)

        print("ST outer-train matrix:", adata_st_fold.shape)
        print("scRNA outer-train matrix:", adata_sc_fold.shape)
        print("SC_ref test-gene matrix:", SC_ref.shape)
        print("Expected final fold prediction:", (adata_st_fold.n_obs, len(test_gene)))

        # ---------------------------------------------------------------------
        # stAI label preprocessing
        # process_celltype creates adata_sc_fold.obs['intlabel'].
        # ST celltype is not required for imputation.
        # ---------------------------------------------------------------------
        process_celltype(adata_st_fold, adata_sc_fold)

        # ---------------------------------------------------------------------
        # Original stAI scales TRAINING-GENE matrices before train_and_impute.
        # This is NOT normalize_total/log1p.
        # We preserve this original stAI behavior.
        # ---------------------------------------------------------------------
        sc.pp.scale(adata_sc_fold, max_value=10)
        sc.pp.scale(adata_st_fold, max_value=10)

        adata_sc_fold.X = make_dense_float32(adata_sc_fold.X)
        adata_st_fold.X = make_dense_float32(adata_st_fold.X)

        # ---------------------------------------------------------------------
        # SAME SPLIT AS TANGRAM:
        #
        # Use ALL outer training genes. There is NO second 144/36 split.
        #
        # We use the same 180 measured genes for:
        #   gene_fit         -> model input
        #   gene_calibration -> cross-modal supervision on those measured genes
        #
        # The 45 outer test genes remain completely absent from ST training.
        # ---------------------------------------------------------------------
        gene_fit = train_gene
        gene_calibration = train_gene

        fold_dir = os.path.join(outdir, f"fold_{idx}")
        os.makedirs(fold_dir, exist_ok=True)

        # Save the outer gene split for reproducibility.
        pd.DataFrame({"train_gene": train_gene}).to_csv(
            os.path.join(csv_dir, f"fold_{idx}_train_genes.csv"),
            index=False,
        )
        pd.DataFrame({"test_gene": test_gene}).to_csv(
            os.path.join(csv_dir, f"fold_{idx}_test_genes.csv"),
            index=False,
        )

        print("\n------------------------------------------------------------")
        print(f"Outer fold {idx} | stAI single model")
        print(f"training_genes={len(gene_fit)} | test_genes={len(test_gene)}")
        print("NO inner gene split")
        print("------------------------------------------------------------")

        # ST_gt=None is deliberate: held-out ST expression is never supplied
        # during training. SC_ref contains only scRNA expression for the
        # held-out genes and is used for final spatial prediction.
        train_and_impute(
            adata_spatial=adata_st_fold,
            adata_rna=adata_sc_fold,
            ST_gt=None,
            SC_ref=SC_ref,
            gene_fit=gene_fit,
            gene_calibration=gene_calibration,
            model_parameters=model_parameters,
            training_parameters=training_parameters,
            random_seed=args.rand,
            save_dir=fold_dir,
        )

        prediction_file = os.path.join(
            fold_dir,
            "impute_res",
            f"epoch_{training_parameters['num_epoch']}.npy",
        )

        if not os.path.exists(prediction_file):
            raise FileNotFoundError(
                f"stAI prediction was not produced: {prediction_file}"
            )

        Imp_Genes = np.load(prediction_file).astype(np.float32)

        expected_shape = (adata_st_fold.n_obs, len(test_gene))
        if Imp_Genes.shape != expected_shape:
            raise ValueError(
                "Unexpected stAI prediction shape. "
                f"Got {Imp_Genes.shape}, expected {expected_shape}."
            )

        print("\nFinal stAI fold prediction shape:", Imp_Genes.shape)

        # Store predictions in their original outer-test gene positions.
        all_pred_res[:, test_ind] = Imp_Genes

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---------------------------------------------------------------------
        # Evaluation -- true ST test genes are accessed ONLY here.
        # ---------------------------------------------------------------------
        y_true = sp_data[test_gene].values
        y_pred = np.asarray(Imp_Genes)

        results_val = compute_metrics(
            y_true,
            y_pred,
            genes=test_gene,
        )

        results_val.update(
            {
                "fold": idx,
                "phase": "val",
                "loss": float(results_val.get("l2_errors_mean", 0.0)),
            }
        )

        fold_metrics[idx] = results_val

        print("\nFold metrics:")
        print(results_val)
        print(
            f"[Fold {idx}] stAI val Spearman: "
            f"{results_val['spearman_mean_genewise']:.4f}, "
            f"Pearson: {results_val['pearson_mean']:.4f}"
        )

        # ---------------------------------------------------------------------
        # Save predictions and ground truth exactly like your SpaGE function.
        # ---------------------------------------------------------------------
        df_pred = pd.DataFrame(y_pred, columns=test_gene)
        df_true = pd.DataFrame(y_true, columns=test_gene)

        df_pred.to_csv(
            os.path.join(csv_dir, f"fold_{idx}_pred.csv"),
            index=False,
        )
        df_true.to_csv(
            os.path.join(csv_dir, f"fold_{idx}_true.csv"),
            index=False,
        )

        np.save(
            os.path.join(csv_dir, f"fold_{idx}_pred.npy"),
            y_pred,
        )
        np.save(
            os.path.join(csv_dir, f"fold_{idx}_true.npy"),
            y_true,
        )

    # -------------------------------------------------------------------------
    # Save metrics
    # -------------------------------------------------------------------------
    df_metrics = pd.DataFrame.from_dict(fold_metrics, orient="index")
    df_metrics.to_csv(
        os.path.join(csv_dir, "fold_metrics.csv"),
        index=False,
    )

    print(f"\nSaved all fold metrics to {csv_dir}/fold_metrics.csv")

    numeric_cols = df_metrics.select_dtypes(include=[np.number]).columns

    df_mean = df_metrics[numeric_cols].mean().to_frame(name="mean").T
    df_std = df_metrics[numeric_cols].std().to_frame(name="std").T

    df_summary = pd.concat([df_mean, df_std])
    df_summary.to_csv(
        os.path.join(csv_dir, "metrics_summary.csv"),
        index=True,
    )

    print(f"Saved mean/std metrics to {csv_dir}/metrics_summary.csv")

    # Save full reconstructed cross-validation prediction matrix.
    np.save(
        os.path.join(csv_dir, "all_pred_res.npy"),
        all_pred_res,
    )

    print("\n============================================================")
    print("stAI 5-fold finished")
    print("all_pred_res:", all_pred_res.shape)
    print("============================================================")

    if "pearson_mean" in df_metrics.columns:
        print(
            "Pearson mean +/- std: "
            f"{df_metrics['pearson_mean'].mean():.6f} +/- "
            f"{df_metrics['pearson_mean'].std():.6f}"
        )

    if "spearman_mean_genewise" in df_metrics.columns:
        print(
            "Spearman mean +/- std: "
            f"{df_metrics['spearman_mean_genewise'].mean():.6f} +/- "
            f"{df_metrics['spearman_mean_genewise'].std():.6f}"
        )

    return all_pred_res, gene_seq


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    stAI_result, stAI_gene_seq = stAI_impute()

    print("\nstAI result shape:", stAI_result.shape)
    print("Number of outer folds:", len(stAI_gene_seq))
