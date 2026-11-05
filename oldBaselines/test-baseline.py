import numpy as np
import pandas as pd
import random
import sys
import os
import scipy.stats as st
import copy
from sklearn.model_selection import KFold
import pandas as pd
import scanpy as sc
import warnings
from os.path import join
from scipy import sparse

from sklearn.model_selection import train_test_split
import torch
from utils.compute_metrics import compute_metrics, spearmanrr
warnings.filterwarnings('ignore')
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
from process.result_analysis import clustering_metrics
torch.set_float32_matmul_precision('high')
from process.result_analysis import *
# from baseline.scvi.model import GIMVI
import uniport as up
import scvi
from scvi.external import GIMVI
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--sc_data", type=str, default="dataset5_seq_915.h5ad")
parser.add_argument("--sp_data", type=str, default='dataset5_spatial_915.h5ad')
parser.add_argument("--document", type=str, default='dataset5')
parser.add_argument("--filename", type=str, default='Dataset')
parser.add_argument("--rand", type=int, default=0)
args = parser.parse_args()


n_splits = 5 
print("adata_spatial: ", args.sp_data)
print("adata_seq: ", args.sc_data)

adata_spatial = sc.read_h5ad(f"/paste/your/path/{args.filename}/{args.sp_data}")
adata_seq = sc.read_h5ad(f"/paste/your/path/{args.filename}/{args.sc_data}")


# =============================================================
# Align gene names before finding common genes
# =============================================================

print("\nBefore gene-name alignment")
print("ST shape:", adata_spatial.shape)
print("scRNA shape:", adata_seq.shape)

print("First ST genes:")
print(list(adata_spatial.var_names[:20]))

print("First scRNA genes:")
print(list(adata_seq.var_names[:20]))

# Convert gene names to strings, strip spaces, uppercase
adata_spatial.var_names = (
    adata_spatial.var_names
    .astype(str)
    .str.strip()
    .str.upper()
)

adata_seq.var_names = (
    adata_seq.var_names
    .astype(str)
    .str.strip()
    .str.upper()
)

if not adata_spatial.var_names.is_unique:
    adata_spatial.var_names_make_unique()

if not adata_seq.var_names.is_unique:
    adata_seq.var_names_make_unique()

# -------------------------------------------------------------
# Filter empty cells/genes
# -------------------------------------------------------------
sc.pp.filter_cells(
    adata_seq,
    min_counts=1
)

sc.pp.filter_genes(
    adata_seq,
    min_counts=1
)

sc.pp.filter_cells(
    adata_spatial,
    min_counts=1
)

sc.pp.filter_genes(
    adata_spatial,
    min_counts=1
)

# -------------------------------------------------------------
# Find common genes
# Preserve ST gene order
# -------------------------------------------------------------
seq_gene_set = set(
    adata_seq.var_names
)


#only for bc_data
# common_genes = np.asarray(
#     [
#         gene
#         for gene in adata_spatial.var_names
#         if gene in seq_gene_set
#     ],
#     dtype=object
# )



common_genes = np.intersect1d(adata_spatial.var_names, adata_seq.var_names)



print("\nNumber of common genes:", len(common_genes))

if len(common_genes) == 0:
    print("\nERROR: no common genes.")
    print("ST example:")
    print(list(adata_spatial.var_names[:30]))

    print("\nscRNA example:")
    print(list(adata_seq.var_names[:30]))

    raise ValueError(
        "No common genes between spatial and scRNA data."
    )

# -------------------------------------------------------------
# Put both datasets in exactly the same gene order
# -------------------------------------------------------------
adata_spatial = adata_spatial[
    :,
    common_genes
].copy()

adata_seq = adata_seq[
    :,
    common_genes
].copy()

print("\nAfter intersection")
print("ST:", adata_spatial.shape)
print("scRNA:", adata_seq.shape)

assert adata_spatial.n_vars > 0
assert adata_seq.n_vars > 0

assert np.array_equal(
    adata_spatial.var_names,
    adata_seq.var_names
)

print("Gene alignment: OK")

#=============================================================
print(f"The number of common genes:: {len(common_genes)}")

adata_seq2 = adata_seq.copy()
# tangram
adata_seq3 =  adata_seq2.copy()
sc.pp.normalize_total(adata_seq2, target_sum=1e4)
sc.pp.log1p(adata_seq2)
data_seq_array = adata_seq2.X

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




def dataSparse(x):
    """Convert sparse or dense matrix to a pandas DataFrame."""
    if sparse.issparse(x):
        x = x.toarray()
    else:
        x = np.asarray(x)
    return x        


def SpaGE_impute():

    print ('We run SpaGE for this data\n')
    sys.path.append("baseline/SpaGE-master/")
    from SpaGE.main import SpaGE
    global sc_data, sp_data, adata_seq, adata_spatial, adata_seq_test


    outdir1 = os.path.join("Result", args.document, "spage")
    csv_dir = os.path.join(outdir1, "fold_results")
    os.makedirs(csv_dir, exist_ok=True)


    raw_shared_gene = np.array(adata_spatial.var_names)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.rand)
    kf.get_n_splits(raw_shared_gene)
    torch.manual_seed(10)
    idx = 1
    # if args.filename == "bc_data":
    #     all_pred_res = np.zeros_like(adata_spatial.X.toarray())
    # else:
    all_pred_res = np.zeros_like(dataSparse(adata_spatial.X))

    print(f"all_pred_res: {all_pred_res.shape}")

    fold_metrics = {}
    gene_seq = []
    for train_ind, test_ind in kf.split(raw_shared_gene):
        print("\n===== Fold %d =====\nNumber of train genes: %d, Number of test genes: %d" % (
            idx, len(train_ind), len(test_ind)))
        train_gene = raw_shared_gene[train_ind]
        test_gene = raw_shared_gene[test_ind]
        pv = len(train_gene) // 2
        sp_data_partial = sp_data[train_gene]
        gene_seq.append(test_gene.tolist())

        print(f"gene_seq:: {gene_seq}")

        Imp_Genes = SpaGE(sp_data_partial, sc_data, n_pv=50 if args.filename != "osmFISH" else 13,
                          genes_to_predict=test_gene)

        print(f"Imp_Genes: {Imp_Genes.shape}")

        print(f"all_pred_res: {all_pred_res.shape}")

        all_pred_res[:, test_ind] = Imp_Genes
        

        y_true = sp_data[test_gene].values
        y_pred = np.asarray(Imp_Genes)

        results_val = compute_metrics(y_true, y_pred, genes=test_gene)
        results_val.update({
            'fold': idx,
            'phase': 'val',
            'loss': float(results_val.get('l2_errors_mean', 0.0))
        })

        fold_metrics[idx] = results_val

        print(results_val)
        print(f"[Fold {idx}] SpaGE val Spearman: {results_val['spearman_mean_genewise']:.4f}, "
                        f"Pearson: {results_val['pearson_mean']:.4f}")


        # Save per-fold predictions and ground truth
        df_pred = pd.DataFrame(Imp_Genes.to_numpy(), columns=test_gene)
        df_true = pd.DataFrame(y_true, columns=test_gene)
        df_pred.to_csv(os.path.join(csv_dir, f"fold_{idx}_pred.csv"), index=False)
        df_true.to_csv(os.path.join(csv_dir, f"fold_{idx}_true.csv"), index=False)

        idx += 1

    df_metrics = pd.DataFrame.from_dict(fold_metrics, orient="index")
    df_metrics.to_csv(os.path.join(csv_dir, "fold_metrics.csv"), index=False)
    print(f"Saved all fold metrics to {csv_dir}/fold_metrics.csv")

    # Compute mean and std
    numeric_cols = df_metrics.select_dtypes(include=[np.number]).columns
    df_mean = df_metrics[numeric_cols].mean().to_frame(name="mean").T
    df_std = df_metrics[numeric_cols].std().to_frame(name="std").T

    # Combine into one summary table
    df_summary = pd.concat([df_mean, df_std])
    df_summary.to_csv(os.path.join(csv_dir, "metrics_summary.csv"), index=False)
    print(f"Saved mean and std metrics to {csv_dir}/metrics_summary.csv")

    print(f"all_pred_res :::: {all_pred_res.shape}")

    return all_pred_res, gene_seq








def Tangram_impute(annotate=None, modes='clusters', density='rna_count_based'):

    import torch
    import baseline.tangram as tg
    from sklearn.model_selection import KFold
    import os
    import numpy as np
    import pandas as pd
    import scanpy as sc

    print('We run Tangram for this data\n')
    global adata_seq, adata_seq2, adata_seq3, adata_spatial, adata_spatial2

    fold_metrics = {}
    outdir1 = os.path.join("Result", args.document, "tangram")
    csv_dir = os.path.join(outdir1, "fold_results")
    os.makedirs(csv_dir, exist_ok=True)
    
    raw_shared_gene = np.array(adata_spatial.var_names)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.rand)
    torch.manual_seed(10)
    idx = 1

    all_pred_res = np.zeros((adata_spatial.n_obs, len(raw_shared_gene)), dtype=np.float32)

    for train_ind, test_ind in kf.split(raw_shared_gene):
        print("\n===== Fold %d =====\nNumber of train genes: %d, Number of test genes: %d" % 
              (idx, len(train_ind), len(test_ind)))
        
        train_gene = np.array(raw_shared_gene[train_ind])
        test_gene = np.array(raw_shared_gene[test_ind])

        adata_seq_tmp = adata_seq2.copy()
        adata_spatial_tmp = adata_spatial2.copy()

        # Subset spatial dataset for training
        adata_spatial_partial = adata_spatial_tmp[:, train_gene].copy()

        # Handle cell type annotations / clustering
        if annotate is None:
            RNA_data_adata_label = adata_seq3.copy()
            sc.pp.normalize_total(RNA_data_adata_label)
            sc.pp.log1p(RNA_data_adata_label)
            sc.pp.highly_variable_genes(RNA_data_adata_label)
            RNA_data_adata_label = RNA_data_adata_label[:, RNA_data_adata_label.var.highly_variable]
            sc.pp.scale(RNA_data_adata_label, max_value=10)
            sc.tl.pca(RNA_data_adata_label)
            sc.pp.neighbors(RNA_data_adata_label)
            sc.tl.leiden(RNA_data_adata_label, resolution=0.5)
            adata_seq_tmp.obs['leiden'] = RNA_data_adata_label.obs.leiden
        else:
            global CellTypeAnnotate
            adata_seq_tmp.obs['leiden'] = CellTypeAnnotate

        # Preprocess AnnData pair for Tangram mapping
        tg.pp_adatas(adata_seq_tmp, adata_spatial_partial, genes=train_gene) 

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Run Tangram mapping and gene projection
        if modes == 'clusters':
            ad_map = tg.map_cells_to_space(
                adata_seq_tmp, 
                adata_spatial_partial, 
                device=device, 
                mode=modes, 
                cluster_label='leiden', 
                density_prior=density
            )
            ad_ge = tg.project_genes(ad_map, adata_seq_tmp, cluster_label='leiden')
        else:
            ad_map = tg.map_cells_to_space(
                adata_seq_tmp, 
                adata_spatial_partial, 
                device=device
            )
            ad_ge = tg.project_genes(ad_map, adata_seq_tmp)

        df_ge_full = pd.DataFrame(
            ad_ge.X, 
            index=adata_spatial_tmp.obs_names, 
            columns=[str(g).upper() for g in ad_ge.var_names]
        )
        
        # Ensure test genes exist in projected matrix and match exact fold test order
        df_pred_raw = df_ge_full.reindex(columns=test_gene)
        assert not df_pred_raw.isna().any().any(), "Some test genes missing from Tangram projection"

        y_pred = df_pred_raw.to_numpy(dtype=np.float32)  # already log1p scale, no extra transform


        X_true = adata_spatial2[:, test_gene].X
        if sparse.issparse(X_true):
            y_true = X_true.toarray().astype(np.float32)
        else:
            y_true = np.asarray(X_true, dtype=np.float32)

        if "_original_row" in adata_spatial.obs.columns:
            spatial_row_idx = adata_spatial.obs["_original_row"].to_numpy().astype(int)
        else:
            spatial_row_idx = np.arange(adata_spatial.n_obs, dtype=int)
            
        np.save(os.path.join(csv_dir, f"fold_{idx}_row_indices.npy"), spatial_row_idx)

        # Record metrics for validation check
        results_val = compute_metrics(y_true, y_pred, genes=test_gene)
        results_val = {
            'fold': idx,
            'phase': 'val',
            'loss': float(results_val.get('l2_errors_mean', 0.0)),
            **results_val
        }
        fold_metrics[idx] = results_val

        print(f"[Fold {idx}] Tangram val Spearman: {results_val.get('spearman_mean_genewise', 0.0):.4f}, "
              f"Pearson: {results_val.get('pearson_mean', 0.0):.4f}")

        # Save per-fold predictions and ground truth CSVs
        df_pred = pd.DataFrame(y_pred, columns=test_gene)
        df_true = pd.DataFrame(y_true, columns=test_gene)

        df_pred.to_csv(os.path.join(csv_dir, f"fold_{idx}_pred.csv"), index=False)
        df_true.to_csv(os.path.join(csv_dir, f"fold_{idx}_true.csv"), index=False)


        all_pred_res[:, test_ind] = y_pred
        idx += 1

    # Save aggregated fold metrics
    df_metrics = pd.DataFrame.from_dict(fold_metrics, orient="index")
    df_metrics.to_csv(os.path.join(csv_dir, "fold_metrics.csv"), index=False)
    print(f"Saved all fold metrics to {csv_dir}/fold_metrics.csv")

    numeric_cols = df_metrics.select_dtypes(include=[np.number]).columns
    df_mean = df_metrics[numeric_cols].mean().to_frame(name="mean").T
    df_std = df_metrics[numeric_cols].std().to_frame(name="std").T

    df_summary = pd.concat([df_mean, df_std])
    df_summary.to_csv(os.path.join(csv_dir, "metrics_summary.csv"), index=False)
    print(f"Saved mean and std metrics to {csv_dir}/metrics_summary.csv")

    return all_pred_res


def gimVI_impute():

    print("We run gimVI for this data\n")

    import os
    import gc
    import copy
    import numpy as np
    import pandas as pd
    import torch
    import scanpy as sc
    import scipy.sparse as sp
    import anndata

    global adata_seq, adata_spatial, adata_spatial2

    if not hasattr(anndata, "read"):
        anndata.read = anndata.read_h5ad

    import anndata._core.sparse_dataset as sparse_dataset

    if not hasattr(sparse_dataset, "SparseDataset"):
        sparse_dataset.SparseDataset = (
            sparse_dataset.BaseCompressedSparseDataset
        )



    print("SCVI VERSION:", scvi.__version__)
    print("SCVI PATH:", scvi.__file__)

    # ==========================================================
    # Output
    # ==========================================================

    outdir1 = os.path.join(
        "Result",
        args.document,
        "gimVI"
    )

    csv_dir = os.path.join(
        outdir1,
        "fold_results"
    )

    os.makedirs(
        csv_dir,
        exist_ok=True
    )

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("gimVI output:", outdir1)
    print("Fold output:", csv_dir)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # ==========================================================
    # IMPORTANT:
    #
    # Gene ordering is taken from the SAME processed ST matrix
    # used by the other baselines.
    #
    # But GIMVI TRAINING uses raw adata_spatial / adata_seq.
    # ==========================================================

    raw_shared_gene = np.asarray(
        adata_spatial2.var_names
    )

    Genes = list(raw_shared_gene)

    print(
        "Total spatial genes:",
        len(raw_shared_gene)
    )

    # ==========================================================
    # Check genes exist in raw matrices
    # ==========================================================

    missing_sc = np.setdiff1d(
        raw_shared_gene,
        adata_seq.var_names
    )

    missing_st = np.setdiff1d(
        raw_shared_gene,
        adata_spatial.var_names
    )

    if len(missing_sc) > 0:
        raise ValueError(
            f"{len(missing_sc)} genes missing from raw scRNA: "
            f"{missing_sc[:20]}"
        )

    if len(missing_st) > 0:
        raise ValueError(
            f"{len(missing_st)} genes missing from raw ST: "
            f"{missing_st[:20]}"
        )

    # ==========================================================
    # Utility: inspect matrix
    # ==========================================================

    def check_matrix(name, X):

        if sp.issparse(X):
            values = X.data
        else:
            values = np.asarray(X).ravel()

        if values.size == 0:
            print(f"{name}: EMPTY")
            return

        n_nan = np.isnan(values).sum()
        n_inf = np.isinf(values).sum()

        integer_fraction = np.mean(
            np.isclose(
                values,
                np.round(values),
                atol=1e-5
            )
        )

        print(
            f"{name}: "
            f"min={values.min():.6f}, "
            f"max={values.max():.6f}, "
            f"NaN={n_nan}, "
            f"Inf={n_inf}, "
            f"integer_like={integer_fraction:.4f}"
        )

        if n_nan > 0:
            raise ValueError(
                f"{name} contains NaN before GIMVI training."
            )

        if n_inf > 0:
            raise ValueError(
                f"{name} contains Inf before GIMVI training."
            )

        if values.min() < 0:
            raise ValueError(
                f"{name} contains negative expression values."
            )

        if integer_fraction < 0.95:
            print(
                f"WARNING: {name} does not look like raw counts. "
                f"GIMVI should preferably receive raw count data."
            )


    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=args.rand
    )

    torch.manual_seed(10)
    np.random.seed(10)

    try:
        scvi.settings.seed = 10
    except Exception:
        pass


    all_pred_res = np.full(
        (
            adata_spatial2.n_obs,
            adata_spatial2.n_vars
        ),
        np.nan,
        dtype=np.float32
    )

    print(
        "all_pred_res:",
        all_pred_res.shape
    )

    fold_metrics = {}

    # ==========================================================
    # Folds
    # ==========================================================

    for idx, (train_ind, test_ind) in enumerate(kf.split(raw_shared_gene), start=1):

        print("\n")
        print("=" * 70)
        print(f"Fold {idx}")
        print(f"Train genes: {len(train_ind)}")
        print(f"Test genes:  {len(test_ind)}")
        print(f"Test genes:  {raw_shared_gene[test_ind]}")
        print("=" * 70)


        # if idx <= 5:
        #     print(f"Skipping Fold {idx}")
        #     # idx += 1
        #     continue

        train_gene = raw_shared_gene[
            train_ind
        ]

        test_gene = raw_shared_gene[
            test_ind
        ]



        seq_data = (
            adata_seq[
                :,
                Genes
            ]
            .copy()
        )

        adata_spatial_partial = (adata_spatial[:,train_gene].copy())

        adata_spatial_partial.obs["_original_row"] = np.arange(adata_spatial_partial.n_obs)
        print("Before filtering:")

        print(
            "seq_data:",
            seq_data.shape
        )

        print("spatial partial:",adata_spatial_partial.shape)

        # ======================================================
        # Inspect BEFORE training
        # ======================================================

        check_matrix(
            "scRNA RAW",
            seq_data.X
        )

        check_matrix(
            "ST RAW",
            adata_spatial_partial.X
        )

        # ======================================================
        # Remove zero-library cells.
        #
        # Official GIMVI tutorial uses min_counts=1.
        # ======================================================

        sc.pp.filter_cells(
            seq_data,
            min_counts=1
        )

        sc.pp.filter_cells(
            adata_spatial_partial,
            min_counts=1
        )


        spatial_row_idx = (
            adata_spatial_partial
            .obs["_original_row"]
            .to_numpy()
            .astype(int)
        )

        print(
            "Retained ST rows:",
            len(spatial_row_idx)
        )

        row_idx_path = os.path.join(
            csv_dir,
            f"fold_{idx}_row_indices.npy"
        )

        np.save(
            row_idx_path,
            spatial_row_idx
        )

        print(
            f"Saved row indices: {row_idx_path}"
        )


        print(
            "\nAfter zero-count filtering:"
        )

        print(
            "seq_data:",
            seq_data.shape
        )

        print(
            "spatial partial:",
            adata_spatial_partial.shape
        )

        print(
            "Removed scRNA cells:",
            adata_seq.n_obs - seq_data.n_obs
        )

        print(
            "Removed ST cells:",
            adata_spatial.n_obs
            - adata_spatial_partial.n_obs
        )

        # ======================================================
        # Remember which ST cells survived.
        #
        # This is important because GIMVI output rows now
        # correspond to these cells only.
        # ======================================================

        spatial_obs_names = (
            adata_spatial_partial
            .obs_names
            .copy()
        )

        # ======================================================
        # Setup GIMVI
        # ======================================================


        import numpy as np
        from scipy import sparse

        def check_gimvi_data(adata, name):
            X = adata.X

            if sparse.issparse(X):
                vals = X.data
                row_sum = np.asarray(X.sum(axis=1)).ravel()
                col_sum = np.asarray(X.sum(axis=0)).ravel()
            else:
                X = np.asarray(X)
                vals = X.ravel()
                row_sum = X.sum(axis=1)
                col_sum = X.sum(axis=0)

            print(f"\n===== {name} =====")
            print("shape:", adata.shape)

            print("min:", vals.min())
            print("max:", vals.max())

            print("NaN:", np.isnan(vals).sum())
            print("Inf:", np.isinf(vals).sum())
            print("negative:", np.sum(vals < 0))

            print("zero-count cells:", np.sum(row_sum == 0))
            print("zero-count genes:", np.sum(col_sum == 0))

            # raw counts can be stored as float, but values should be count-like
            print(
                "non-integer-like values:",
                np.sum(np.abs(vals - np.round(vals)) > 1e-5)
            )


        check_gimvi_data(seq_data, "scRNA")
        check_gimvi_data(adata_spatial_partial, "ST")

        GIMVI.setup_anndata(
            seq_data
        )

        GIMVI.setup_anndata(
            adata_spatial_partial
        )



        print(
            f"\nTraining GIMVI fold {idx}..."
        )

        model = GIMVI(
            seq_data,
            adata_spatial_partial
        )


        model.train(
            max_epochs=200,

            train_size=0.99,
            validation_size=0.01,

            use_gpu=True,

            batch_size=128,

            plan_kwargs={
                "lr": 1e-4
            },

            gradient_clip_val=10.0
        )


        print(
            f"Running imputation Fold {idx}..."
        )

        _, imputation = (
            model.get_imputed_values(
                normalized=True,
                deterministic=True,
                batch_size=128
            )
        )

        imputation = np.asarray(
            imputation,
            dtype=np.float32
        )

        print(
            "Full imputation:",
            imputation.shape
        )

        # Expected:
        #
        # [retained ST cells, 456 genes]

        if imputation.shape[1] != len(Genes):
            raise ValueError(
                "Unexpected GIMVI output gene dimension: "
                f"{imputation.shape[1]} vs {len(Genes)}"
            )


        pred_normalized = (
            imputation[
                :,
                test_ind
            ]
        )


        my_test = np.log1p(
            pred_normalized * 1e4
        ).astype(
            np.float32
        )

        print(
            "Predicted test:",
            my_test.shape
        )

        # ======================================================
        # Ground truth
        #
        # Use ONLY the same retained spatial cells.
        # ======================================================

        true_data = (
            adata_spatial2[
                spatial_row_idx,
                test_ind
            ].X
        )

        if sp.issparse(true_data):
            true_data = true_data.toarray()

        true_data = np.asarray(
            true_data,
            dtype=np.float32
        )

        true_data = np.asarray(
            true_data,
            dtype=np.float32
        )

        print(
            "True test:",
            true_data.shape
        )

        if my_test.shape != true_data.shape:
            raise ValueError(
                f"Prediction / ground-truth mismatch: "
                f"{my_test.shape} vs {true_data.shape}"
            )

        # ======================================================
        # Check output before metrics
        # ======================================================

        if not np.isfinite(my_test).all():
            raise ValueError(
                f"Fold {idx}: prediction contains NaN/Inf."
            )

        # ======================================================
        # SAME METRIC AS stPlus
        # ======================================================

        results_val = compute_metrics(
            true_data,
            my_test,
            genes=test_gene
        )

        results_val = {
            "fold": idx,
            "phase": "val",
            "loss": float(
                results_val.get(
                    "l2_errors_mean",
                    0.0
                )
            ),
            **results_val
        }

        fold_metrics[
            idx
        ] = results_val

        print("\nFold metrics:")
        print(results_val)

        # ======================================================
        # Put predictions into COMPLETE matrix
        # ======================================================
        if np.any(
            spatial_row_idx < 0
        ):
            raise ValueError(
                "Some filtered spatial cells cannot be "
                "mapped back to adata_spatial2."
            )

        all_pred_res[
            np.ix_(
                spatial_row_idx,
                test_ind
            )
        ] = my_test

        # ======================================================
        # Save fold results
        # ======================================================

        df_pred = pd.DataFrame(
            my_test,
            columns=test_gene
        )

        df_true = pd.DataFrame(
            true_data,
            columns=test_gene
        )

        pred_path = os.path.join(
            csv_dir,
            f"fold_{idx}_pred.csv"
        )

        true_path = os.path.join(
            csv_dir,
            f"fold_{idx}_true.csv"
        )

        df_pred.to_csv(
            pred_path,
            index=False
        )

        df_true.to_csv(
            true_path,
            index=False
        )

        # Test gene list
        pd.DataFrame({
            "test_gene": test_gene
        }).to_csv(
            os.path.join(
                csv_dir,
                f"fold_{idx}_genes.csv"
            ),
            index=False
        )

        # Save cells actually evaluated
        pd.DataFrame({
            "cell_id": spatial_obs_names
        }).to_csv(
            os.path.join(
                csv_dir,
                f"fold_{idx}_cells.csv"
            ),
            index=False
        )

        # Save fold metric immediately
        pd.DataFrame(
            [results_val]
        ).to_csv(
            os.path.join(
                csv_dir,
                f"fold_{idx}_metrics.csv"
            ),
            index=False
        )

        print(
            f"Saved predictions: {pred_path}"
        )

        print(
            f"Saved ground truth: {true_path}"
        )

        # ======================================================
        # Free GPU RAM before next fold
        # ======================================================

        del model
        del imputation
        del pred_normalized

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ==========================================================
    # All fold metrics
    # ==========================================================

    df_metrics = pd.DataFrame.from_dict(
        fold_metrics,
        orient="index"
    )

    metrics_path = os.path.join(
        csv_dir,
        "fold_metrics.csv"
    )

    df_metrics.to_csv(
        metrics_path,
        index=False
    )

    print(
        "\nSaved all fold metrics to:",
        metrics_path
    )

    # ==========================================================
    # Mean ± std
    # ==========================================================

    numeric_cols = (
        df_metrics
        .select_dtypes(
            include=[np.number]
        )
        .columns
        .tolist()
    )

    numeric_cols = [
        col
        for col in numeric_cols
        if col != "fold"
    ]

    df_mean = (
        df_metrics[
            numeric_cols
        ]
        .mean()
        .to_frame()
        .T
    )

    df_mean.index = [
        "mean"
    ]

    df_std = (
        df_metrics[
            numeric_cols
        ]
        .std()
        .to_frame()
        .T
    )

    df_std.index = [
        "std"
    ]

    df_summary = pd.concat(
        [
            df_mean,
            df_std
        ]
    )

    summary_path = os.path.join(
        csv_dir,
        "metrics_summary.csv"
    )

    df_summary.to_csv(
        summary_path,
        index=True
    )

    # ==========================================================
    # Complete CV prediction
    # ==========================================================

    all_pred_df = pd.DataFrame(
        all_pred_res,
        index=adata_spatial2.obs_names,
        columns=raw_shared_gene
    )

    all_pred_path = os.path.join(
        outdir1,
        "gimVI_all_predictions.csv"
    )

    all_pred_df.to_csv(
        all_pred_path,
        index=True
    )

    print("\n========================================")
    print("gimVI 5-fold evaluation completed")
    print("========================================")

    print("\nMetric summary:")
    print(df_summary)

    print(
        "\nComplete predictions:",
        all_pred_path
    )

    return all_pred_res


def uniport_impute():
    """
    UniPort imputation baseline, split by gene count (like Tangram).
    Saves per-fold predictions, ground truth, and summary metrics.
    """
    import scipy
    global sc_data, sp_data, adata_seq, adata_spatial, adata_seq2, adata_spatial2

    print("We run UniPort for this data\n")

    torch.set_default_tensor_type('torch.FloatTensor')

    outdir1 = os.path.join("Result", args.document, "uniport")
    csv_dir = os.path.join(outdir1, "fold_results")
    os.makedirs(csv_dir, exist_ok=True)

    # Get all shared genes between ST and scRNA-seq
    raw_shared_gene = np.array(adata_spatial.var_names)
    print(f"Total shared genes for KFold split: {len(raw_shared_gene)}")

    # Split based on gene count
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.rand)
    kf.get_n_splits(raw_shared_gene)
    torch.manual_seed(10)
    idx = 1

    # Store all results
    all_pred_res = np.zeros((adata_spatial.n_obs, adata_spatial.n_vars), dtype=np.float32)
    fold_metrics = {}
    # --- Loop over folds ---
    for train_ind, test_ind in kf.split(raw_shared_gene):
        print(f"\n===== Fold {idx} =====")
        print(f"Number of train genes: {len(train_ind)}, Number of test genes: {len(test_ind)}")

        train_gene = raw_shared_gene[train_ind]
        test_gene = raw_shared_gene[test_ind]

        # --- Prepare adatas ---
        adata_seq_tmp = adata_seq2.copy()
        adata_spatial_tmp = adata_spatial2.copy()

        adata_spatial_tmp.obs['domain_id'] = 0
        adata_spatial_tmp.obs['domain_id'] = adata_spatial_tmp.obs['domain_id'].astype('category')
        adata_spatial_tmp.obs['source'] = 'ST'

        adata_seq_tmp.obs['domain_id'] = 1
        adata_seq_tmp.obs['domain_id'] = adata_seq_tmp.obs['domain_id'].astype('category')
        adata_seq_tmp.obs['source'] = 'RNA'

        adata_cm = ad.concat(
            [adata_spatial_tmp, adata_seq_tmp],
            axis=0,
            join="inner",
            label="domain_id",
            keys=["0", "1"],
            index_unique=None,
        )
        print(f"[Fold {idx}] Combined AnnData shape: {adata_cm.shape}")

        # Split spatial and RNA subsets
        spatial_data = adata_cm[adata_cm.obs['source'] == 'ST'].copy()
        seq_data = adata_cm[adata_cm.obs['source'] == 'RNA'].copy()

        # Train on a subset of genes
        spatial_data_partial = spatial_data[:, train_gene].copy()

        # Combine for UniPort input
        # adata_cm = spatial_data_partial.concatenate(seq_data, join='inner', batch_key='domain_id')

        adata_cm = ad.concat(
            [spatial_data_partial, seq_data],
            axis=0,
            join="inner",
            label="domain_id",
            keys=["0", "1"],
            index_unique=None,
        )

        print(f"[Fold {idx}] Combined shape after train gene selection: {adata_cm.shape}")

        # --- Preprocessing ---
        up.batch_scale(adata_cm)
        up.batch_scale(spatial_data_partial)
        up.batch_scale(seq_data)

        # seq_data.X = scipy.sparse.coo_matrix(seq_data.X)
        seq_data.X = scipy.sparse.csr_matrix(seq_data.X)
        # spatial_data_partial.X = scipy.sparse.coo_matrix(spatial_data_partial.X)
        spatial_data_partial.X = scipy.sparse.csr_matrix(spatial_data_partial.X)

        adatas = [spatial_data_partial, seq_data]


        print("\n===== UniPort DEBUG =====", flush=True)
        print("Number of adatas:", len(adatas), flush=True)

        for i, a in enumerate(adatas):
            print(f"\nDataset {i}", flush=True)
            print("shape:", a.shape, flush=True)
            print("obs columns:", list(a.obs.columns), flush=True)

            if "domain_id" in a.obs.columns:
                print(
                    "domain_id unique:",
                    a.obs["domain_id"].unique(),
                    flush=True
                )
            else:
                print("domain_id missing", flush=True)

        print("\nadata_cm shape:", adata_cm.shape, flush=True)
        print("adata_cm obs columns:", list(adata_cm.obs.columns), flush=True)

        if "domain_id" in adata_cm.obs.columns:
            print(
                "adata_cm domain_id unique:",
                adata_cm.obs["domain_id"].unique(),
                flush=True
            )

        print("==========================\n", flush=True)
        n_domains = len(adatas)

        # Each separate AnnData
        for i, a in enumerate(adatas):
            a.obs["domain_id"] = pd.Categorical(
                np.full(a.n_obs, i, dtype=np.int64),
                categories=list(range(n_domains))
            )

        # Combined/common-gene AnnData
        domain_ids = np.concatenate([
            np.full(a.n_obs, i, dtype=np.int64)
            for i, a in enumerate(adatas)
        ])

        adata_cm.obs["domain_id"] = pd.Categorical(
            domain_ids,
            categories=list(range(n_domains))
        )

        print("\n===== UniPort domain check =====", flush=True)

        for i, a in enumerate(adatas):
            print(
                f"Dataset {i}:",
                "unique =", a.obs["domain_id"].unique().tolist(),
                "dtype =", a.obs["domain_id"].dtype,
                "categories =", a.obs["domain_id"].cat.categories.tolist(),
                flush=True
            )

        print(
            "adata_cm:",
            "unique =", adata_cm.obs["domain_id"].unique().tolist(),
            "dtype =", adata_cm.obs["domain_id"].dtype,
            "categories =", adata_cm.obs["domain_id"].cat.categories.tolist(),
            flush=True
        )

        print("================================\n", flush=True)
        # --- Train UniPort ---
        adata = up.Run(adatas=adatas, adata_cm=adata_cm, lambda_kl=5.0, loss_type='MSE', model_info=False)

        if sparse.issparse(spatial_data_partial.X):
            spatial_data_partial.X = spatial_data_partial.X.toarray()

        # --- Predict unseen genes ---
        adata_predict = up.Run(adata_cm=spatial_data_partial, out='predict', pred_id=1)
        model_res = pd.DataFrame(
            adata_predict.obsm["predict"],
            index=spatial_data_partial.obs_names,
            columns=raw_shared_gene
        )

        # --------------------------------------------------
        # Predictions
        # --------------------------------------------------
        y_pred = model_res.loc[:, test_gene].to_numpy(
            dtype=np.float32
        )

        # --------------------------------------------------
        # Ground truth
        # AnnData slicing may return SparseCSRMatrixView
        # --------------------------------------------------
        X_true = adata_spatial_tmp[:, test_gene].X

        if scipy.sparse.issparse(X_true):
            X_true = X_true.toarray()

        y_true = np.asarray(
            X_true,
            dtype=np.float32
        )

        y_true = np.nan_to_num(
            y_true,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        y_pred = np.nan_to_num(
            y_pred,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        print("y_true shape:", y_true.shape)
        print("y_pred shape:", y_pred.shape)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Fold {idx}: shape mismatch "
                f"y_true={y_true.shape}, y_pred={y_pred.shape}"
            )

        all_pred_res[:, test_ind] = y_pred



        # --- Compute metrics ---
        results_val = compute_metrics(y_true, y_pred, genes=test_gene)
        results_val.update({
            'fold': idx,
            'phase': 'val',
            'loss': float(results_val.get('l2_errors_mean', 0.0))
        })
        fold_metrics[idx] = results_val

        print(results_val)
        print(f"[Fold {idx}] Spearman: {results_val['spearman_mean_genewise']:.4f}, "
              f"Pearson: {results_val['pearson_mean']:.4f}, "
              f"L2 loss: {results_val['l2_errors_mean']:.4f}")

        # --- Save per-fold results ---
        df_pred = pd.DataFrame(y_pred, columns=test_gene)
        df_true = pd.DataFrame(y_true, columns=test_gene)
        df_pred.to_csv(os.path.join(csv_dir, f"fold_{idx}_pred.csv"), index=False)
        df_true.to_csv(os.path.join(csv_dir, f"fold_{idx}_true.csv"), index=False)

        idx += 1

    # --- Save all fold metrics ---
    df_metrics = pd.DataFrame.from_dict(fold_metrics, orient="index")
    df_metrics.to_csv(os.path.join(csv_dir, "fold_metrics.csv"), index=False)
    print(f"Saved all fold metrics to {csv_dir}/fold_metrics.csv")

    # --- Compute mean and std ---
    numeric_cols = df_metrics.select_dtypes(include=[np.number]).columns
    df_mean = df_metrics[numeric_cols].mean().to_frame(name="mean").T
    df_std = df_metrics[numeric_cols].std().to_frame(name="std").T

    # Combine and save
    df_summary = pd.concat([df_mean, df_std])
    df_summary.to_csv(os.path.join(csv_dir, "metrics_summary.csv"), index=False)
    print(f"Saved mean and std metrics to {csv_dir}/metrics_summary.csv")

    print("\n UniPort imputation complete.")
    return all_pred_res



Data = args.document
outdir = 'Result/' + Data + '/'
if not os.path.exists(outdir):
    # os.mkdir(outdir)
    os.makedirs(outdir, exist_ok=True)



SpaGE_result, gene_seq_test = SpaGE_impute() 
SpaGE_result_pd = pd.DataFrame(SpaGE_result, columns=sp_genes)
os.makedirs(f"{outdir}/spage",  exist_ok=True)
SpaGE_result_pd.to_csv(outdir +  'spage/SpaGE_impute.csv',header = 1, index = 1)
print(f"{outdir}/SpaGE_impute.csv")

Tangram_result = Tangram_impute() 
Tangram_result_pd = pd.DataFrame(Tangram_result, columns=sp_genes)
os.makedirs(f"{outdir}/tangram",  exist_ok=True)
Tangram_result_pd.to_csv(outdir +  'tangram/Tangram_impute.csv',header = 1, index = 1)

gimVI_result = gimVI_impute() 
gimVI_result_pd = pd.DataFrame(gimVI_result, columns=sp_genes)
os.makedirs(f"{outdir}/gimVI",  exist_ok=True)
gimVI_result_pd.to_csv(outdir +  'gimVI/gimVI_impute.csv',header = 1, index = 1)

uniport_result = uniport_impute()
uniport_result_pd = pd.DataFrame(uniport_result, columns=sp_genes)
os.makedirs(f"{outdir}/uniport",  exist_ok=True)
uniport_result_pd.to_csv(outdir +  'uniport/uniport_impute.csv',header = 1, index = 1)

