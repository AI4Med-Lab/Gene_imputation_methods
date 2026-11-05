import numpy as np
import pandas as pd
import os
import scipy.stats as st
from sklearn.model_selection import KFold
import pandas as pd
import scanpy as sc
import warnings
import torch
from utils.compute_metrics import compute_metrics, spearmanrr
from sklearn.model_selection import train_test_split

from model.stDiff_model import DiT_stDiff
from model.stDiff_scheduler import NoiseScheduler
from model.stDiff_train import normal_train_stDiff
from model.sample import sample_stDiff
from process.result_analysis import clustering_metrics

from scipy import sparse

warnings.filterwarnings('ignore')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

from process.data import *

import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--sc_data", type=str, default="dataset5_seq_915.h5ad")
parser.add_argument("--sp_data", type=str, default='dataset5_spatial_915.h5ad')
parser.add_argument("--document", type=str, default='dataset5')
parser.add_argument("--batch_size", type=int, default=512)  # 2048
parser.add_argument("--hidden_size", type=int, default=1024) # 512
parser.add_argument("--filename", type=str, default='')
parser.add_argument("--noise_std", type=float, default=10)
parser.add_argument("--head", type=int, default=16)
parser.add_argument("--step", type=int, default=1500) #1500
parser.add_argument("--epoch", type=int, default=900) #900
parser.add_argument("--rand", type=int, default=0)
args = parser.parse_args()

# ******** preprocess ********

# raw data 
n_splits = 5 # cross-validation sets number


adata_spatial = sc.read_h5ad(f"/paste/your/path/here/{args.filename}/{args.sp_data}")

adata_seq = sc.read_h5ad(f"/paste/your/path/here/{args.filename}/{args.sc_data}")

adata_seq.var_names = pd.Index(
    [str(g).strip().upper() for g in adata_seq.var_names]
)

adata_spatial.var_names = pd.Index(
    [str(g).strip().upper() for g in adata_spatial.var_names]
)

adata_seq.var_names_make_unique()
adata_spatial.var_names_make_unique()

print("\n========== AFTER GENE NAME FIX ==========")
print("scRNA shape:", adata_seq.shape)
print("Spatial shape:", adata_spatial.shape)

common_genes = [
    g for g in adata_seq.var_names
    if g in set(adata_spatial.var_names)
]

print("Common genes:", len(common_genes))
print("First common genes:", common_genes[:20])

if len(common_genes) == 0:
    raise ValueError("No common genes after gene-name normalization!")

print("=========================================\n")




sc.pp.filter_cells(adata_seq, min_counts=1)
sc.pp.filter_genes(adata_seq, min_counts=1)
sc.pp.filter_cells(adata_spatial, min_counts=1)
sc.pp.filter_genes(adata_spatial, min_counts=1)




common_genes = np.intersect1d(adata_spatial.var_names, adata_seq.var_names)


adata_spatial = adata_spatial[:, common_genes].copy()
adata_seq = adata_seq[:, common_genes].copy()

# print(adata_spatial.shape)
# print(adata_seq.shape)

adata_seq2 = data_augment(adata_seq.copy(), True, noise_std=args.noise_std)
adata_spatial2 = adata_spatial.copy()

sc.pp.normalize_total(adata_seq2, target_sum=1e4)
sc.pp.log1p(adata_seq2)

print("\n========== DEBUG BEFORE SCALE ==========")
print("adata_seq2 shape:", adata_seq2.shape)
print("adata_seq2 genes:", adata_seq2.var_names[:20].tolist())
print("Number of genes:", adata_seq2.n_vars)

print("adata_spatial shape:", adata_spatial.shape)
print("Spatial genes:", adata_spatial.var_names[:20].tolist())
print("========================================\n")  


adata_seq2 = scale(adata_seq2)
data_seq_array = adata_seq2.X



adata_spatial2 = adata_spatial.copy()
# if args.filename != "merFISH":
sc.pp.normalize_total(adata_spatial2, target_sum=1e4)
sc.pp.log1p(adata_spatial2)
data_spatial_array = adata_spatial2.X

outdir1 = os.path.join("Result", args.document, "stDiff")
csv_dir = os.path.join(outdir1, "fold_results")
os.makedirs(csv_dir, exist_ok=True)


adata_seq_train = adata_spatial2.copy()
adata_seq_test = adata_spatial2.copy()


print(f"adata_spatial  {adata_spatial}")

sp_genes = np.array(adata_spatial.var_names)

print(f"data_spatial_array TYPE : {type(data_spatial_array)}")
print(f"data_spatial_array SHAPE : {data_spatial_array.shape}")
print(f"data_seq_array TYPE : {type(data_seq_array)}")
print(f"data_seq_array SHAPE : {data_seq_array.shape}")
print("len(sp_genes):", len(sp_genes))

if sparse.issparse(data_spatial_array):
    data_spatial_array = data_spatial_array.toarray()

if sparse.issparse(data_seq_array):
    data_seq_array = data_seq_array.toarray()

sp_data = pd.DataFrame(
    data=data_spatial_array,
    columns=sp_genes
)

sc_data = pd.DataFrame(
    data=data_seq_array,
    columns=sp_genes
)

print("sp_data:", sp_data.shape)
print("sc_data:", sc_data.shape)

print(f"adata_seq_test::: {adata_seq_test}")


test_array = adata_seq_test.X

if sparse.issparse(test_array):
    test_array = test_array.toarray()

sc_data_test = pd.DataFrame(
    data=test_array,
    columns=sp_genes
)



print(adata_spatial.shape)
print(adata_seq.shape)

def diffusion_impute():
  
    lr = 0.00016046744893538737 
    depth = 6 
    num_epoch = args.epoch
    diffusion_step =  args.step # 1500 
    batch_size = args.batch_size 
    hidden_size = args.hidden_size 
    head = args.head

    raw_shared_gene = adata_spatial.var_names
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.rand)   # 0
    kf.get_n_splits(raw_shared_gene)
    torch.manual_seed(10)  # 10
    idx = 1
    all_pred_res = np.zeros_like(data_spatial_array)
    all_pred_res_1 = np.zeros_like(sc_data_test)
    fold_metrics = {}
    for train_ind, test_ind in kf.split(raw_shared_gene):
        print("\n===== Fold %d =====\nNumber of train genes: %d, Number of test genes: %d" % (
            idx, len(train_ind), len(test_ind)))

        train_gene = np.array(raw_shared_gene[train_ind])
        test_gene = np.array(raw_shared_gene[test_ind])
        

        doc =args.document # 'Dataset11_std+scale_new'
        save_path = 'stDiff-ckpt/' + doc + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        save_path_prefix = save_path + 'stDiff%d.pt' % (idx)
        
        
        # mask
        cell_num = data_spatial_array.shape[0]
        gene_num = data_spatial_array.shape[1]
        mask = np.ones((gene_num,), dtype='float32')
        gene_ids_test = test_ind
        mask[gene_ids_test] = 0

        if args.filename == "bc_data":
            seq = data_seq_array.toarray()
            st = data_spatial_array.toarray()
        elif args.filename == "Crunch":
            seq = data_seq_array.toarray()
            st = data_spatial_array
        else:
            seq = data_seq_array
            st = data_spatial_array
        data_seq_masked = seq * mask
        data_spatial_masked = st * mask

        seq = seq * 2 - 1
        data_seq_masked = data_seq_masked * 2 - 1

        data_ary = data_spatial_array
        st = st * 2 - 1
        data_spatial_masked = data_spatial_masked * 2 - 1

        dataloader = get_data_loader(
            seq,
            data_seq_masked,
            batch_size=batch_size,  # 原2048  ssp 1024  d3 1024  994 512
            is_shuffle=True)

        seed = 1202
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        model = DiT_stDiff(
            input_size=gene_num,  # 158  33   994       169
            hidden_size=hidden_size,  # 512      1024       512
            depth=depth,
            num_heads=head,
            classes=6,  # 16   6
            mlp_ratio=4.0,
            dit_type='dit'
        )

        device = torch.device('cuda')
        model.to(device)

        diffusion_step = diffusion_step

        model.train()

        if not os.path.isfile(save_path_prefix):

            normal_train_stDiff(model,
                                 dataloader=dataloader,
                                 lr=lr,
                                 num_epoch=num_epoch,
                                 diffusion_step=diffusion_step,
                                 device=device,
                                 pred_type='noise',
                                 mask=mask)

            # torch.save(model.state_dict(), save_path_prefix)
        else:
            model.load_state_dict(torch.load(save_path_prefix))

        gt = data_spatial_masked
        noise_scheduler = NoiseScheduler(
            num_timesteps=diffusion_step,
            beta_schedule='cosine'
        )

        dataloader = get_data_loader(
            data_spatial_masked,
            data_spatial_masked,
            batch_size=batch_size,  # 33 1024    994 512
            is_shuffle=False)

        # sample
        model.eval()
        imputation = sample_stDiff(model,
                                    device=device,
                                    dataloader=dataloader,
                                    noise_scheduler=noise_scheduler,
                                    mask=mask,
                                    gt=gt,
                                    num_step=diffusion_step,
                                    sample_shape=(cell_num, gene_num),
                                    is_condi=True,
                                    sample_intermediate=diffusion_step,
                                    model_pred_type='noise',
                                    is_classifier_guidance=False,
                                    omega=0.2)

        gene_ids_test = np.array(gene_ids_test).flatten() 


        all_pred_res_1[:, gene_ids_test] = imputation[:,  gene_ids_test]
        y_pred = imputation[:,  gene_ids_test]


        y_true = sc_data_test.iloc[:, gene_ids_test]
        results_val = compute_metrics(np.array(y_true), np.array(y_pred), genes=test_gene)
        results_val = {
            'fold': idx,
            'phase': 'val',
            'loss': float(results_val.get('l2_errors_mean', 0.0)),
            **results_val
        }


        print(f"[Fold {idx}] Spearman: {results_val['spearman_mean_genewise']:.4f}, "
            f"Pearson: {results_val['pearson_mean']:.4f}, "
            f"L2 loss: {results_val['l2_errors_mean']:.4f}")

        df_pred = pd.DataFrame(y_pred, columns=test_gene)
        df_true = pd.DataFrame(y_true, columns=test_gene)

        df_pred.to_csv(os.path.join(csv_dir, f"fold_{idx}_pred.csv"), index=False)
        df_true.to_csv(os.path.join(csv_dir, f"fold_{idx}_true.csv"), index=False)
        fold_metrics[idx] = results_val
        idx += 1

    df_metrics = pd.DataFrame.from_dict(fold_metrics, orient="index")
    df_metrics.to_csv(os.path.join(csv_dir, "fold_metrics.csv"), index=False)
    print(f"Saved all fold metrics to {csv_dir}/fold_metrics.csv")

    # Compute mean and std
    numeric_cols = df_metrics.select_dtypes(include=[np.number]).columns
    df_mean = df_metrics[numeric_cols].mean().to_frame(name="mean").T
    df_std = df_metrics[numeric_cols].std().to_frame(name="std").T

    # Combine into one summary table
    # stDiff_result_pd = pd.DataFrame(SpaGE_result, columns=sp_genes)

    df_summary = pd.concat([df_mean, df_std])
    df_summary.to_csv(os.path.join(csv_dir, "metrics_summary.csv"), index=False)
    print(f"Saved mean and std metrics to {csv_dir}/metrics_summary.csv")

    impu = (all_pred_res_1 + 1) / 2
    # cpy_x = adata_spatial.copy()
    # cpy_x.X = impu
    return impu



Data =args.document # 
outdir = 'Result/' + Data + '/'
if not os.path.exists(outdir):
    # os.mkdir(outdir)
    os.makedirs(outdir, exist_ok=True)



diffusion_result = diffusion_impute() # ok
diffusion_result_pd = pd.DataFrame(diffusion_result, columns=sp_genes)
print("diffusion_result_pd: ", diffusion_result_pd.shape)
diffusion_result_pd.to_csv(outdir +  '/diffusion_impute.csv',header = 1, index = 1)

