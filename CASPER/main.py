#JSG

'''
Command to run:=

python3 main.py --config AttentionParams.json

'''


import random
import warnings
import time
import h5py
import csv
import anndata as ad
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import ConcatDataset
from datetime import datetime
import logging
import argparse
import random
import json
import pandas as pd
import scipy.sparse as sp
from scipy import sparse
from typing import Tuple, List
from scipy.stats import pearsonr
from torch.nn import DataParallel
from sklearn.model_selection import train_test_split, KFold
from utils.compute_metrics import compute_metrics, spearmanrr
from utils.setup_logger import setup_logging
from models.Casper import Casper
from utils.centroidCalc import compute_centroids_leiden
from models.noCentroid import Casper_NoCentroid, FNN_Mimic
import inspect
from torch_geometric.utils import add_self_loops

torch.manual_seed(10)
random.seed(10)

print("Jai shri Ganesh!!!")
parser = argparse.ArgumentParser(description="Script that uses a JSON configuration file")
parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration file')

args = parser.parse_args()

try:
    with open(args.config, 'r') as f:
        params = json.load(f)
except FileNotFoundError:
    print(f"Error: File not found at {args.config}")
except json.JSONDecodeError:
    print(f"Error: Failed to parse JSON file at {args.config}")

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_save_dir = os.path.join(params['result_path'], f"result_{timestamp}")
os.makedirs(model_save_dir,  exist_ok=True)

json_file = os.path.join(model_save_dir,f"experiment_info.json")

with open(json_file, "w") as exp:
    json.dump(params, exp, indent=4)


log_file = os.path.join(model_save_dir, "results_log.txt")
logger = setup_logging(log_file)
logger.info("Logging setup complete.")
logger.info(f"Experiment information saved to the path: {json_file}")


metrics_csv_path = os.path.join(model_save_dir, "metrics_result.csv")

csv_dir = os.path.join(model_save_dir, "fold_results")
os.makedirs(csv_dir,  exist_ok=True)

n_splits = 5 
adata_spatial = sc.read_h5ad(f"{params['data_path']}/{params['dataset_name']}/spatial.h5ad")
adata_seq = sc.read_h5ad(f"{params['data_path']}/{params['dataset_name']}/scRNA_common.h5ad")


sc.pp.filter_cells(adata_seq, min_counts=1)
sc.pp.filter_genes(adata_seq, min_counts=1)
sc.pp.filter_cells(adata_spatial, min_counts=1)
sc.pp.filter_genes(adata_spatial, min_counts=1)

sc.pp.normalize_total(adata_seq, target_sum=1e4)
sc.pp.log1p(adata_seq)

sc.pp.normalize_total(adata_spatial, target_sum=1e4)
sc.pp.log1p(adata_spatial)



# Write CSV headers (only once before the loop starts)
with open(metrics_csv_path, mode='w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=[
        'fold','epoch','phase','loss','pearson_mean', 'spearman_mean_genewise',
        'l1_error_mean','l2_errors_mean','r2_scores_mean',
        'pearson_std','l2_error_q1','l2_error_q2','l2_error_q3',
        'r2_score_q1','r2_score_q2','r2_score_q3', 'MAPE', 'RMSE'
    ])
    writer.writeheader()



def run_5fold_training(st_adata, sc_adata, common_genes, params):
    """
    st_adata: AnnData with only the common genes (spots x genes)
    sc_adata: AnnData with only the common genes (cells x genes)
    """
    def get_device(): return torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


    st_adata_test = st_adata.copy()


    logger.info(f"[info] Spatial data split: train={st_adata.n_obs}, test={st_adata_test.n_obs}")

    device = get_device()

    logger.info(f"[info] Using device: {device}")
    raw_shared_gene = np.array(common_genes)
    # Optional HVG selection parameters
    n_top_hvgs = params.get("n_top_hvg", None)
    use_hvg = params.get("use_hvg", False)
    logger.info(f"[info] use_hvg={use_hvg}, n_top_hvgs={n_top_hvgs}")
    n_pcs = params.get("n_pcs", 50)
    print(f"sc_adata {sc_adata.shape}")
    centroids_df, ad_proc, cell_cluster_df, cluster_to_cells, cluster_to_indices = compute_centroids_leiden(sc_adata,
                                           logger,
                                           n_top_hvgs=n_top_hvgs if use_hvg else None,
                                           n_pcs = max(1,min(sc_adata.n_obs - 1, sc_adata.n_vars - 1, sc_adata.n_vars // 2)),
                                           neighbors_k=params.get("neighbors_k", 15),
                                           leiden_resolution=params.get("leiden_resolution", 1.0),
                                           use_hvg_subset=use_hvg)
    centroids = centroids_df.values.astype(np.float32)  # shape: (n_centroids, n_genes)
    n_centroids = centroids.shape[0]
    cluster_labels = cell_cluster_df["cluster"].tolist()
    centroids_2 = centroids.copy()

    st_X = st_adata.X.copy()
    if sp.issparse(st_X):
        st_X = st_X.toarray()
    sc_X = sc_adata.X.copy()
    if sp.issparse(sc_X):
        sc_X = sc_X.toarray()

    st_X_test = st_adata_test.X.toarray() if sp.issparse(st_adata_test.X) else st_adata_test.X

    n_spots, n_genes = st_X.shape
    # logger.info(f"[info] After intersection: n_spots={n_spots}, n_genes={n_genes}, n_centroids={n_centroids}")

    # 2) 5-fold gene-wise CV
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=params.get("rand_seed", 0))
    all_pred = np.zeros_like(st_X_test, dtype=np.float32)
    fold_metrics = {}

    class SpotDataset(Dataset):

        def __init__(self, X_in, Y_target):

            self.X = self._to_tensor(X_in)
            self.Y = self._to_tensor(Y_target)

            if len(self.X) != len(self.Y):
                raise ValueError(
                    "X_in and Y_target must have the same number of samples."
                )

        @staticmethod
        def _to_tensor(data):

            # AnnData -> matrix
            if isinstance(data, ad.AnnData):
                data = data.X

            # Sparse -> dense
            if sp.issparse(data):
                data = data.toarray()

            # NumPy -> Tensor
            if isinstance(data, np.ndarray):
                return torch.from_numpy(
                    data.astype(np.float32, copy=False)
                )

            # Tensor -> float32
            if isinstance(data, torch.Tensor):
                return data.to(dtype=torch.float32)

            raise TypeError(
                f"Unsupported data type: {type(data)}"
            )

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]



    for fold, (train_idx, test_idx) in enumerate(kf.split(raw_shared_gene), start=1):
        
        best_spearman = -np.inf
        best_state = None
        wait = 0

        train_gene = list(raw_shared_gene[train_idx])
        test_gene = list(raw_shared_gene[test_idx])
        print(f"text_genes: {test_gene}")
        logger.info(f"\n[fold {fold}] Train genes: {len(train_idx)} | Test genes: {len(test_idx)}")


        test_gene_names = raw_shared_gene[test_idx]
        available_genes = [g for g in test_gene_names if g in centroids_df.columns]
        sc_centroids_genes = torch.from_numpy(centroids_df[available_genes].values.astype(np.float32)).to(device)
        temp = sc_X.copy()
        Y_target = temp[:, test_idx].copy()
        X_in = temp[:, train_idx]


        temp = st_adata.copy()
        Y_target_test = temp[:, test_idx].copy()
        X_in_test = temp[:, train_idx] 

        temp = centroids_2.copy()
        centroid = temp[:, train_idx] 
        # centroid = temp.copy()

        print("X_in_test========= ",type(X_in_test))
        print("Y_target_test==========", type(Y_target_test))

        sc_centroid_tensor = torch.from_numpy(centroid).to(device)  # (n_centroids, n_genes)
        print("sc_centroids shape:", sc_centroid_tensor.shape)

        print("X_in shape:", X_in.shape)
        print("sc_X shape:", sc_X.shape)
        # Build model
        st_in_dim = X_in.shape[1]
        sc_gene_dim = sc_centroid_tensor.shape[1]
        emb_dim = params['emb_dim']
        out_dim = len(test_idx)

        if params['model'] == "CASPER":
            model = Casper(
                st_dim=st_in_dim,
                sc_dim=sc_gene_dim,
                out_dim=out_dim,
                emb_dim=emb_dim,
                n_heads=params.get("n_heads", 4),
                dropout=params.get("dropout", 0.1)
            ).to(device)

        else:
            logger.info(f" No model found!!!")


        if fold == 1:
            logger.info(f"Model architecture:\n{model}")

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=params.get("lr", 1e-3),
                                     weight_decay=params.get("weight_decay", 1e-5))
        loss_fn = nn.MSELoss()

        # Dataset + DataLoader
        dataset_train = SpotDataset(X_in, Y_target)

        dataset_test = SpotDataset(X_in_test, Y_target_test)

        train_loader = DataLoader(dataset_train, batch_size=params.get("batch_size", 128),
                            shuffle=True)
        test_loader =  DataLoader(dataset_test, batch_size=params.get("batch_size", 128),
                            shuffle=False)

        # Early stopping setup
        best_val = np.inf
        best_state = None
        patience = params.get("early_stop_patience", 5)
        wait = 0
        epochs = params.get("epochs", 100)

        # =========================
        #   EPOCH LOOP
        # =========================
        for ep in range(1, epochs + 1):
            t0 = time.time()
            model.train()

            epoch_train_loss = 0.0
            all_y_true, all_y_pred = [], []
            pearson_train_corr, spearman_train_corr = [], []

            for batch_idx, (xb, yb) in enumerate(train_loader):

                xb = xb.to(device)
                yb = yb.to(device)
                # positive_x = positive_x.to(device)
                # negative_x = negative_x.to(device)
                # preds, emb_anchor, emb_positive, emb_negative = output[0], output[4], output[5], output[6]
                if params['model'] == "CASPER":
                    preds, _ = model(st_input = xb, sc_centroids = sc_centroid_tensor, per_head=True)
                else:
                    print("No Loss Found!!!")


                if params["loss"] == "MSE":
                    print(f"===== {preds.shape} ======= {yb.shape}")
                    loss = loss_fn(preds, yb)
                else:
                    print("No Loss Found!!!")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # ---- Metrics ----
                try:
                    batch_pearson, _ = pearsonr(
                        preds.detach().cpu().numpy().flatten(),
                        yb.detach().cpu().numpy().flatten()
                    )
                except Exception:
                    batch_pearson = np.nan

                pearson_train_corr.append(batch_pearson)

                batch_spearman = spearmanrr(
                    preds.detach().cpu(), yb.detach().cpu()
                )
                spearman_train_corr.append(batch_spearman)

                epoch_train_loss += loss.item()
                all_y_true.append(yb.detach().cpu().numpy())
                all_y_pred.append(preds.detach().cpu().numpy())

            # ====== End of training for this epoch ======
            avg_train_loss = epoch_train_loss / len(train_loader)
            all_y_true = np.vstack(all_y_true)
            all_y_pred = np.vstack(all_y_pred)
            results_train = compute_metrics(all_y_true, all_y_pred, genes=None)
            results_train.update({
                'fold': fold,
                'epoch': ep,
                'phase': 'train',
                'loss': round(avg_train_loss, 4)
            })

            logger.info(f"[Fold {fold}] Epoch {ep} Train Loss={avg_train_loss:.4f} | "
                        f"Pearson={np.nanmean(pearson_train_corr):.4f} | "
                        f"Spearman={np.nanmean(spearman_train_corr):.4f} | "
                        f"Time={time.time()-t0:.1f}s")

            # ========== VALIDATION ==========
            model.eval()

            preds_all = []
            Y_val_all = []

            val_loss_sum = 0.0
            val_sample_count = 0

            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb.to(device, dtype=torch.float32)
                    yb = yb.to(device, dtype=torch.float32)

                    print(
                        f"Input shape: {xb.shape} | "
                        f"Target shape: {yb.shape}"
                    )

                    # if isinstance(output, tuple):
                    #     preds, emb_anchor, emb_positive, emb_negative = output[0], output[4], output[5], output[6]
                    # elif isinstance(output, dict):
                    #     preds = output.get(
                    #         "predictions",
                    #         output.get("pred"),
                    #     )
                    #     if preds is None:
                    #         raise KeyError(
                    #             "Model output dictionary contains neither "
                    #             "'predictions' nor 'pred'."
                    #         )
                    # else:
                    #     preds = output

                    if params["model"] == "CASPER":
                        preds, _ = model(
                            st_input=xb,
                            sc_centroids=sc_centroid_tensor,
                            positive_x=None,
                            negative_x=None,
                            per_head=True,
                        )


                    else:
                        raise ValueError(
                            f"Unsupported model: {params['model']}"
                        )

                    if not torch.is_tensor(preds):
                        raise TypeError(
                            f"Predictions must be a tensor, but received "
                            f"{type(preds)}"
                        )

                    if preds.shape != yb.shape:
                        raise ValueError(
                            f"Prediction-target shape mismatch: "
                            f"preds={preds.shape}, yb={yb.shape}"
                        )

                    print(f"Prediction shape: {preds.shape}")

                    # Calculate batch MSE for monitoring.
                    batch_loss = F.mse_loss(
                        preds,
                        yb,
                        reduction="mean",
                    )

                    batch_size = yb.shape[0]

                    val_loss_sum += batch_loss.item() * batch_size
                    val_sample_count += batch_size

                    preds_all.append(
                        preds.detach().cpu().numpy()
                    )
                    Y_val_all.append(
                        yb.detach().cpu().numpy()
                    )

                    print(
                        f"Collected batches: {len(preds_all)} | "
                        f"Latest predictions: {preds_all[-1].shape} | "
                        f"Latest targets: {Y_val_all[-1].shape}"
                    )

            if not preds_all:
                raise RuntimeError(
                    "The test_loader produced no validation batches."
                )

            preds_all = np.concatenate(
                preds_all,
                axis=0,
            )

            Y_val_all = np.concatenate(
                Y_val_all,
                axis=0,
            )

            print(
                f"Final prediction shape: {preds_all.shape} | "
                f"Final target shape: {Y_val_all.shape}"
            )

            if preds_all.shape != Y_val_all.shape:
                raise ValueError(
                    f"Final shape mismatch: "
                    f"predictions={preds_all.shape}, "
                    f"targets={Y_val_all.shape}"
                )

            if params["loss"] == "MSE":
                val_loss = mean_squared_error(
                    Y_val_all.reshape(-1),
                    preds_all.reshape(-1),
                )
            
            else:
                val_loss = val_loss_sum / val_sample_count

            results_val = compute_metrics(
                Y_val_all,
                preds_all,
            )

            results_val.update({
                "fold": fold,
                "epoch": ep,
                "phase": "val",
                "loss": round(float(val_loss), 4),
            })

            print(f"results_val: {results_val}")

            logger.info(f"[Fold {fold}] Epoch {ep} Val Loss={loss.item():.4f} | "
                        f"Pearson={results_val['pearson_mean']:.4f} | "
                        f"Spearman={results_val['spearman_mean_genewise']:.4f} | "
                        f"Time={time.time()-t0:.1f}s")

            # ========== WRITE BOTH TRAIN AND VAL ROWS ==========
            with open(metrics_csv_path, mode='a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    'fold','epoch','phase','loss','pearson_mean', 'spearman_mean_genewise',
                    'l1_error_mean','l2_errors_mean','r2_scores_mean',
                    'pearson_std','l2_error_q1','l2_error_q2','l2_error_q3',
                    'r2_score_q1','r2_score_q2','r2_score_q3'
                ])
                writer.writerow({key: results_train.get(key, "") for key in writer.fieldnames})
                writer.writerow({key: results_val.get(key, "") for key in writer.fieldnames})

            # ========== EARLY STOPPING ==========
            if float(results_val['spearman_mean_genewise']) > float(best_spearman):

                #results save ======= 
                all_pred[:, test_idx] = preds_all
                print(f"preds_all: {preds_all.shape}")
                df_pred = pd.DataFrame(preds_all, columns=test_gene)
                df_true = pd.DataFrame(Y_val_all, columns=test_gene)
                df_pred.to_csv(os.path.join(csv_dir, f"fold_{fold}_pred.csv"), header = 1, index = 1)
                df_true.to_csv(os.path.join(csv_dir, f"fold_{fold}_true.csv"), header = 1, index = 1)


                best_spearman = results_val['spearman_mean_genewise']
                best_state = {'model': {k: v.cpu().clone() for k, v in model.state_dict().items()}}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info(f"[Fold {fold}] Early stopping at epoch {ep}. Best Spearman={best_spearman:.4f}")
                    # break

        if best_state is not None:
            logger.info(f"[Fold {fold} best model (Spearman={best_spearman:.4f}) before saving.")
        else:
            logger.warning(f"[Fold {fold}] No improvement recorded — saving final model instead.")
            best_state = {'model': {k: v.cpu().clone() for k, v in model.state_dict().items()}}

        # After training, save the best model for this fold
        ckpt_path = os.path.join(model_save_dir, f"model_fold{fold}.pt")
        torch.save(best_state['model'], ckpt_path)
        logger.info(f"[Fold {fold}] Saved best model checkpoint: {ckpt_path}")
    return all_pred




'''
run_5fold_training function ends here,


'''


adata_spatial.var_names = (
    adata_spatial.var_names.astype(str).str.strip().str.upper()
)

adata_seq.var_names = (
    adata_seq.var_names.astype(str).str.strip().str.upper()
)

# if params["dataset_name"] == "bc_data":
#     common_genes = [g for g in adata_spatial.var_names if g in adata_seq.var_names]
# else:
common_genes = np.intersect1d(
adata_spatial.var_names,
adata_seq.var_names
)

logger.info(f"[info] Found {len(common_genes)} common genes between scRNA and ST data.")

if len(common_genes) == 0:
    logger.error("❌ No overlapping genes found between scRNA and spatial datasets.")
    sys.exit(1)

adata_seq = adata_seq[:, common_genes].copy()
adata_spatial = adata_spatial[:, common_genes].copy()

logger.info(f"[info] Subsetting done: scRNA ({adata_seq.shape}) | ST ({adata_spatial.shape})")

# Run training and imputation
logger.info("[info] Starting 5-fold gene-wise cross-validation training...")

all_pred_matrix = run_5fold_training(adata_spatial, adata_seq, common_genes, params)
pd.DataFrame(all_pred_matrix, columns=common_genes).to_csv(
    os.path.join(csv_dir, "attention_imputes.csv"), header = 1, index = 1
)