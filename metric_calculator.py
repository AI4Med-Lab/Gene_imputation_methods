#!/usr/bin/env python3
"""

Command to RUN:

python3 metric_calculator.py \
    --filename bc_data \
    --method SpaGENE \
    --sp_path /path/to/spatial.h5ad \
    --sc_path /path/to/scRNA_common.h5ad \
    --pred_dir /path/to/SpaGene/fold_results

"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as st
from scipy import sparse
from scipy.stats import pearsonr, spearmanr, ConstantInputWarning

from compute_metrics import compute_metrics
from process.result_analysis import clustering_metrics

warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser(description="Five-fold spatial gene-imputation metric calculator")
parser.add_argument("--sc_data", default="scRNA_common.h5ad")
parser.add_argument("--sp_data", default="spatial.h5ad")
parser.add_argument("--document", default="dataset5")
parser.add_argument("--filename", required=True)
parser.add_argument("--method", required=True)
parser.add_argument("--pred_dir", default=None,
                    help="Directory containing fold_results or the fold CSV files")
parser.add_argument("--sp_path", default=None, help="Full path to spatial .h5ad")
parser.add_argument("--sc_path", default=None, help="Full path to scRNA .h5ad")
parser.add_argument("--rand", type=int, default=0)
args = parser.parse_args()


def kl_divergence_genewise(y_true, y_pred, eps=1e-10):
    # Original KL definition: normalize across cells for each gene.
    y_true = np.clip(np.asarray(y_true, dtype=float), eps, None)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), eps, None)
    y_true = y_true / y_true.sum(axis=0, keepdims=True)
    y_pred = y_pred / y_pred.sum(axis=0, keepdims=True)
    kl_per_gene = np.sum(y_true * np.log(y_true / y_pred), axis=0)
    return kl_per_gene.mean(), kl_per_gene.std(), kl_per_gene


def data_path(filename, name, is_spatial):
    root = "/home/2023csz0012/AmitCode/data"
    branch = "updatedDataset" if is_spatial and filename == "merFISH" else "processedData"
    return os.path.join(root, branch, filename, name)


spatial_path = args.sp_path or data_path(args.filename, args.sp_data, True)
sc_path = args.sc_path or data_path(args.filename, args.sc_data, False)
print("Spatial:", spatial_path)
print("scRNA:", sc_path)
adata_spatial = sc.read_h5ad(spatial_path)
adata_seq = sc.read_h5ad(sc_path)

for adata in (adata_spatial, adata_seq):
    adata.var_names = pd.Index(adata.var_names.astype(str)).str.strip().str.upper()
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()

sc.pp.filter_cells(adata_seq, min_counts=1)
sc.pp.filter_genes(adata_seq, min_counts=1)
sc.pp.filter_cells(adata_spatial, min_counts=1)
sc.pp.filter_genes(adata_spatial, min_counts=1)

# Preserve the original spatial gene order when aligning to scRNA-seq.
seq_genes = set(adata_seq.var_names)
common_genes = [g for g in adata_spatial.var_names if g in seq_genes]
if not common_genes:
    raise ValueError("No common ST/scRNA-seq genes after preprocessing")
adata_spatial = adata_spatial[:, common_genes].copy()
adata_seq = adata_seq[:, common_genes].copy()
adata_spatial2 = adata_spatial.copy()
sc.pp.normalize_total(adata_spatial2, target_sum=1e4)
sc.pp.log1p(adata_spatial2)
sp_genes = np.asarray(adata_spatial2.var_names)
data_spatial_array = adata_spatial2.X
if sparse.issparse(data_spatial_array):
    data_spatial_metric = data_spatial_array.toarray()
else:
    data_spatial_metric = np.asarray(data_spatial_array)
print("Filtered spatial truth:", data_spatial_metric.shape)


def cal_ssim(im1, im2, M):
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, M
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim = l12 * c12 * s12

    return ssim


def scale_max(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = content / content.max()
        result = pd.concat([result, content], axis=1)
    return result


def scale_z_score(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = st.zscore(content)
        content = pd.DataFrame(content, columns=[label])
        result = pd.concat([result, content], axis=1)
    return result


def scale_plus(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = content / content.sum()
        result = pd.concat([result, content], axis=1)
    return result


def logNorm(df):
    df = np.log1p(df)
    df = st.zscore(df)
    return df


class CalculateMeteics:
    def __init__(self, raw_data, genes_name,impute_count_file, prefix, metric):
        self.impute_count_file = impute_count_file
        print(f"gene_list length +++++++++++++ {len(genes_name)}")

        if sparse.issparse(raw_data):
            print("Raw matrix is sparse -> converting to dense")
            raw_data = raw_data.toarray()
        elif isinstance(raw_data, pd.DataFrame):
            raw_data = raw_data.to_numpy()
        else:
            raw_data = np.asarray(raw_data)
        self.raw_count = pd.DataFrame(raw_data, columns=genes_name)
        self.raw_count.columns = [x.upper() for x in self.raw_count.columns]
        self.raw_count = self.raw_count.T
        self.raw_count = self.raw_count.loc[~self.raw_count.index.duplicated(keep='first')].T
        self.raw_count = self.raw_count.fillna(1e-20)

        self.impute_count = pd.read_csv(impute_count_file, header=0)
        # self.impute_count = self.impute_count.loc[:, (self.impute_count != 0).any(axis=0)]
        print(f"impute_count :::::::::::::::::::::::::::::::: {self.impute_count.shape}")
        self.impute_count = self.impute_count.loc[:, self.impute_count.columns.isin([g.upper() for g in genes_name])]
        print(f"impute_count ++++: {self.impute_count.shape}")
        self.impute_count.columns = [x.upper() for x in self.impute_count.columns]
        self.impute_count = self.impute_count.T
        self.impute_count = self.impute_count.loc[~self.impute_count.index.duplicated(keep='first')].T
        self.impute_count = self.impute_count.fillna(1e-20)
        self.prefix = prefix
        self.metric = metric
        self.impute_count.index = self.raw_count.index
    def SSIM(self, raw, impute, scale='scale_max'):
        print(f"raw {raw.shape} and impute {impute.shape}")
        if scale == 'scale_max':
            raw = scale_max(raw)
            impute = scale_max(impute)
        else:
            print('Please note you do not scale data by scale max')
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    ssim = 0
                else:
                    raw_col = raw.loc[:, label]
                    impute_col = impute.loc[:, label]
                    impute_col = impute_col.fillna(1e-20)
                    raw_col = raw_col.fillna(1e-20)
                    M = max(raw_col.max(), impute_col.max())
                    raw_col_2 = np.array(raw_col)
                    raw_col_2 = raw_col_2.reshape(raw_col_2.shape[0], 1)
                    impute_col_2 = np.array(impute_col)
                    impute_col_2 = impute_col_2.reshape(impute_col_2.shape[0], 1)
                    ssim = cal_ssim(raw_col_2, impute_col_2, M)

                ssim_df = pd.DataFrame(ssim, index=["SSIM"], columns=[label])
                result = pd.concat([result, ssim_df], axis=1)
        else:
            print("columns error")
        return result

    def SPCC(self, raw, impute, scale=None):
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    spearmanr = 0
                else:
                    raw_col = raw.loc[:, label]
                    impute_col = impute.loc[:, label]
                    impute_col = impute_col.fillna(1e-20)
                    raw_col = raw_col.fillna(1e-20)
                    spearmanr, _ = st.spearmanr(raw_col, impute_col)
                spearman_df = pd.DataFrame(spearmanr, index=["SPCC"], columns=[label])
                result = pd.concat([result, spearman_df], axis=1)
        else:
            print("columns error")
        return result

    def JS(self, raw, impute, scale='scale_plus'):
        if scale == 'scale_plus':
            raw = scale_plus(raw)
            impute = scale_plus(impute)
        else:
            print('Please note you do not scale data by plus')
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                print(label)
                if label not in impute.columns:
                    JS = 1
                else:
                    raw_col = raw.loc[:, label]
                    impute_col = impute.loc[:, label]
                    raw_col = raw_col.fillna(1e-20)
                    impute_col = impute_col.fillna(1e-20)
                    # print(f"raw_col {raw_col.index} ")
                    # print(f"impute_col {impute_col.index} ")
                    M = (raw_col + impute_col) / 2
                    print(f"raw_col: {raw_col.shape}, impute_col: {impute_col.shape} M: {M.shape}")
                    JS = 0.5 * st.entropy(raw_col, M) + 0.5 * st.entropy(impute_col, M)
                JS_df = pd.DataFrame(JS, index=["JS"], columns=[label])
                result = pd.concat([result, JS_df], axis=1)
        else:
            print("columns error")
        return result

    def RMSE(self, raw, impute, scale='zscore'):
        if scale == 'zscore':
            raw = scale_z_score(raw)
            impute = scale_z_score(impute)
        else:
            print('Please note you do not scale data by zscore')
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    RMSE = 1.5
                else:
                    raw_col = raw.loc[:, label]
                    impute_col = impute.loc[:, label]
                    impute_col = impute_col.fillna(1e-20)
                    raw_col = raw_col.fillna(1e-20)
                    RMSE = np.sqrt(((raw_col - impute_col) ** 2).mean())

                RMSE_df = pd.DataFrame(RMSE, index=["RMSE"], columns=[label])
                result = pd.concat([result, RMSE_df], axis=1)
        else:
            print("columns error")
        return result


    def cluster(self, raw, impu, scale=None):

        ad_sp = adata_spatial_metric.copy()
        ad_sp.X = np.asarray(raw)

        cpy_x = adata_spatial_metric.copy()
        cpy_x.X = np.asarray(impu)

        sc.tl.pca(ad_sp)
        sc.pp.neighbors(ad_sp, n_pcs=30, n_neighbors=30)
        sc.tl.leiden(ad_sp)
        tmp_adata1 = ad_sp

        sc.tl.pca(cpy_x)
        sc.pp.neighbors(cpy_x, n_pcs=30, n_neighbors=30)
        sc.tl.leiden(cpy_x)
        tmp_adata2 = cpy_x

        tmp_adata2.obs['class'] = tmp_adata1.obs['leiden']
        ari = clustering_metrics(tmp_adata2, 'class', 'leiden', "ARI")
        ami = clustering_metrics(tmp_adata2, 'class', 'leiden', "AMI")
        homo = clustering_metrics(tmp_adata2, 'class', 'leiden', "Homo")
        nmi = clustering_metrics(tmp_adata2, 'class', 'leiden', "NMI")
        
        
        # tmp_adata2 = get_N_clusters(cpy_x, 23, 'leiden')
        
        # tmp_adata2.obs['class'] = ad_sp.obs['subclass_label']
        # ari = clustering_metrics(tmp_adata2, 'class', 'leiden', "ARI")
        # ami = clustering_metrics(tmp_adata2, 'class', 'leiden', "AMI")
        # homo = clustering_metrics(tmp_adata2, 'class', 'leiden', "Homo")
        # nmi = clustering_metrics(tmp_adata2, 'class', 'leiden', "NMI")
        
        result = pd.DataFrame([[ari, ami, homo, nmi]], columns=["ARI", "AMI", "Homo", "NMI"])
        return result

    def compute_all(self):
        raw = self.raw_count
        print(f"raw : {raw.shape}")
        impute = self.impute_count
        print(f"impute : {impute.shape}")
        prefix = self.prefix

        # --- Compute metrics ---
        SSIM_gene = self.SSIM(raw, impute)
        Spearman_gene = self.SPCC(raw, impute)
        JS_gene = self.JS(raw, impute)
        RMSE_gene = self.RMSE(raw, impute)

        # --- Combine gene-wise results ---
        result_gene = pd.concat([Spearman_gene, SSIM_gene, RMSE_gene, JS_gene], axis=0)
        result_gene_T = result_gene.T  # genes as rows

        # --- Compute mean and std for each metric ---
        summary = {}
        for metric_name in result_gene.index:
            values = result_gene.loc[metric_name, :].values
            summary[metric_name + "_mean"] = np.mean(values)
            summary[metric_name + "_std"] = np.std(values, ddof=0)

        summary_df = pd.DataFrame(summary, index=["Summary"]).T
        summary_df = summary_df.rename_axis("Metric")

        # --- Save outputs ---
        # 1️⃣ Gene-wise metrics
        result_gene_T.to_csv(prefix + "_gene_Metrics.txt", sep='\t', header=True, index=True)

        # 2️⃣ Mean & std summary
        summary_df.to_csv(prefix + "_summary_Metrics.csv", sep=',', header=True, index=True)

        # --- Cluster metrics ---
        cluster_result = self.cluster(raw, impute)
        cluster_result.to_csv(prefix + "_cluster_Metrics.txt", sep='\t', header=True, index=True)

        print(f"Saved results:\n - Gene metrics: {prefix}_gene_Metrics.txt\n - Summary metrics: {prefix}_summary_Metrics.csv\n - Cluster metrics: {prefix}_cluster_Metrics.txt")

        return result_gene




def find_or_merge_imputes(impute_count_dir, expected_rows, expected_genes,
                          ground_truth, mape_threshold=0.1):
    """Merge five gene folds and compute metrics without any row-index files.

    Requires all fold CSVs to use the same filtered spatial cell order.
    """
    fold_dir = args.pred_dir or os.path.join(impute_count_dir, "fold_results")
    if os.path.isdir(os.path.join(fold_dir, "fold_results")):
        fold_dir = os.path.join(fold_dir, "fold_results")
    if not os.path.isdir(fold_dir):
        raise FileNotFoundError(f"Prediction directory not found: {fold_dir}")

    method_name = args.method
    expected_genes = [str(g).strip().upper() for g in expected_genes]
    gene_to_index = {g: i for i, g in enumerate(expected_genes)}
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    if ground_truth.shape != (expected_rows, len(expected_genes)):
        raise ValueError("Spatial ground-truth matrix shape mismatch")

    pred_files = sorted(
        [f for f in os.listdir(fold_dir)
         if f.startswith("fold_") and f.endswith("_pred.csv")],
        key=lambda f: int(f.split("_")[1])
    )
    fold_numbers = [int(f.split("_")[1]) for f in pred_files]
    if fold_numbers != [1, 2, 3, 4, 5]:
        raise ValueError(f"Expected folds 1..5, found: {fold_numbers}")

    metric_dir = os.path.join(impute_count_dir, "fold_metrics_common_cells")
    os.makedirs(metric_dir, exist_ok=True)
    fold_dfs, all_gene_metric_dfs, all_fold_summaries = [], [], []

    for fold, pred_file in zip(fold_numbers, pred_files):
        df = pd.read_csv(os.path.join(fold_dir, pred_file))
        df.columns = [str(c).strip().upper() for c in df.columns]
        # Keep only actual gene columns; excludes CSV index/cell identifiers.
        df = df[[c for c in df.columns if c in gene_to_index]].copy()
        if df.columns.duplicated().any():
            raise ValueError(f"Fold {fold}: duplicate predicted gene columns")
        if len(df) != expected_rows:
            raise ValueError(
                f"Fold {fold}: {len(df)} rows vs {expected_rows} filtered ST cells. "
                "Without row-index files, filtered-fold predictions cannot be aligned safely."
            )
        if df.shape[1] == 0:
            raise ValueError(f"Fold {fold}: no matching genes")
        if not np.isfinite(df.to_numpy(dtype=float)).all():
            raise ValueError(f"Fold {fold}: predictions contain NaN/Inf")

        fold_genes = df.columns.tolist()
        true_fold = ground_truth[:, [gene_to_index[g] for g in fold_genes]]
        pred_fold = df.to_numpy(dtype=np.float64)
        metric_result = compute_metrics(true_fold, pred_fold, genes=fold_genes)
        kl_mean, kl_std, kl_per_gene = kl_divergence_genewise(true_fold, pred_fold)

        gene_rows = []
        for j, gene in enumerate(fold_genes):
            yt, yp = true_fold[:, j], pred_fold[:, j]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConstantInputWarning)
                pearson = float(pearsonr(yt, yp).statistic) if len(yt) >= 2 else 0.0
                spearman = float(spearmanr(yt, yp).statistic) if len(yt) >= 2 else 0.0
            pearson = pearson if np.isfinite(pearson) else 0.0
            spearman = spearman if np.isfinite(spearman) else 0.0
            mask = np.isfinite(yt) & np.isfinite(yp) & (yt > mape_threshold)
            gene_mape = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask]) * 100)) if mask.any() else np.nan
            gene_rows.append({
                "method": method_name, "fold": fold, "gene": gene,
                "pearson": pearson, "spearman": spearman,
                "mape": gene_mape, "kl_divergence": float(kl_per_gene[j]),
                "n_cells": expected_rows, "mape_cells": int(mask.sum())
            })
        gene_metric_df = pd.DataFrame(gene_rows)
        gene_metric_df.to_csv(os.path.join(metric_dir, f"fold_{fold}_gene_metrics.csv"), index=False)
        all_gene_metric_dfs.append(gene_metric_df)
        mask = np.isfinite(true_fold) & np.isfinite(pred_fold) & (true_fold > mape_threshold)
        fold_mape = float(np.mean(np.abs((true_fold[mask] - pred_fold[mask]) / true_fold[mask]) * 100)) if mask.any() else np.nan
        all_fold_summaries.append({
            "method": method_name, "fold": fold,
            "n_cells": expected_rows, "n_genes": len(fold_genes),
            "pearson_mean": metric_result["pearson_mean"],
            "pearson_std": float(np.nanstd(gene_metric_df["pearson"])),
            "spearman_mean": metric_result["spearman_mean_genewise"],
            "spearman_std": float(np.nanstd(gene_metric_df["spearman"])),
            "mape_gene_mean": float(np.nanmean(gene_metric_df["mape"])),
            "mape_gene_std": float(np.nanstd(gene_metric_df["mape"])),
            "mape_fold": fold_mape,
            "kl_mean": float(kl_mean), "kl_std": float(kl_std)
        })
        fold_dfs.append(df.reset_index(drop=True))
        print(f"Fold {fold}: predictions={df.shape}, truth={true_fold.shape}")

    all_gene_metrics = pd.concat(all_gene_metric_dfs, ignore_index=True)
    all_gene_metrics.to_csv(os.path.join(metric_dir, f"{method_name}_all_fold_gene_metrics.csv"), index=False)
    fold_summary_df = pd.DataFrame(all_fold_summaries)
    fold_summary_df.to_csv(os.path.join(metric_dir, f"{method_name}_fold_summary.csv"), index=False)

    final_summary = pd.DataFrame({
        "metric": ["Pearson", "Spearman", "MAPE_Gene", "MAPE_Fold", "KL"],
        "mean": [float(np.nanmean(fold_summary_df[k])) for k in
                 ["pearson_mean", "spearman_mean", "mape_gene_mean", "mape_fold", "kl_mean"]],
        "std": [float(np.nanstd(fold_summary_df[k], ddof=0)) for k in
                ["pearson_mean", "spearman_mean", "mape_gene_mean", "mape_fold", "kl_mean"]]
    })
    final_summary.to_csv(os.path.join(metric_dir, f"{method_name}_final_summary.csv"), index=False)
    print("\nFINAL FIVE-FOLD SUMMARY\n", final_summary.to_string(index=False))

    merged_df = pd.concat(fold_dfs, axis=1)
    if merged_df.columns.duplicated().any():
        raise ValueError("A gene appears in more than one held-out fold")
    missing = [g for g in expected_genes if g not in merged_df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} predicted genes: {missing[:20]}")
    merged_df = merged_df[expected_genes]
    merged_filename = f"{method_name}_imputes.csv"
    merged_df.to_csv(os.path.join(impute_count_dir, merged_filename), index=False)
    return [merged_filename]


def CalDataMetric(data_dir):
    print("Evaluating:", data_dir)
    os.makedirs(data_dir, exist_ok=True)
    merged_files = find_or_merge_imputes(
        impute_count_dir=data_dir,
        expected_rows=data_spatial_metric.shape[0],
        expected_genes=sp_genes,
        ground_truth=data_spatial_metric,
        mape_threshold=0.1,
    )
    metrics = ["SPCC(gene)", "SSIM(gene)", "RMSE(gene)", "JS(gene)"]
    metric = ["SPCC", "SSIM", "RMSE", "JS"]
    medians, methods = [], []
    for file_name in merged_files:
        prefix = os.path.join(data_dir, args.method)
        impute_path = os.path.join(data_dir, file_name)
        cm = CalculateMeteics(data_spatial_metric, sp_genes, impute_path, prefix, metric)
        cm.compute_all()
        per_gene = pd.read_csv(prefix + "_gene_Metrics.txt", sep="\t", index_col=0)
        median = [np.median(per_gene[col]) for col in metric]
        cluster = pd.read_csv(prefix + "_cluster_Metrics.txt", sep="\t", index_col=0)
        row = pd.concat([pd.DataFrame([median], columns=metrics), cluster.reset_index(drop=True)], axis=1)
        medians.append(row)
        methods.append(args.method)
    if medians:
        summary = pd.concat(medians, ignore_index=True)
        summary.index = methods
        summary.to_csv(os.path.join(data_dir, "final_result.csv"), index=True)


if __name__ == "__main__":
    Data = args.pred_dir if args.pred_dir is not None else os.path.join(args.document, args.method)
    if os.path.basename(os.path.normpath(Data)) == "fold_results":
        Data = os.path.dirname(os.path.normpath(Data))
    CalDataMetric(Data)
