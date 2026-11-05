import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from options import Options
from dataset import ImputationDataset
from model import ImputeModule

from utils import seed_everything
from compute_metrics import compute_metrics, spearmanrr


os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# ============================================================
# HELPER: CONVERT compute_metrics OUTPUT TO ONE ROW
# ============================================================
def metrics_to_dataframe(metrics, fold_no):
    """
    Convert output of compute_metrics() into one DataFrame row.

    Expected/preferred output:
        dict
        pandas Series
        one-row pandas DataFrame
    """

    if isinstance(metrics, dict):
        metrics_df = pd.DataFrame([metrics])

    elif isinstance(metrics, pd.Series):
        metrics_df = metrics.to_frame().T

    elif isinstance(metrics, pd.DataFrame):
        metrics_df = metrics.copy()

        # Usually compute_metrics should return one row.
        # If index is meaningful, reset it.
        if len(metrics_df) != 1:
            print(
                "[Warning] compute_metrics returned",
                len(metrics_df),
                "rows."
            )

    elif isinstance(metrics, (list, tuple, np.ndarray)):
        # Fallback only.
        metrics = np.asarray(metrics).reshape(-1)

        metrics_df = pd.DataFrame(
            [{
                f"metric_{i}": value
                for i, value in enumerate(metrics)
            }]
        )

    else:
        metrics_df = pd.DataFrame(
            [{"result": metrics}]
        )

    # Remove fold if compute_metrics itself returned one
    if "fold" in metrics_df.columns:
        metrics_df = metrics_df.drop(
            columns=["fold"]
        )

    metrics_df.insert(
        0,
        "fold",
        fold_no
    )

    return metrics_df


# ============================================================
# SAVE / UPDATE ALL-FOLD METRICS
# ============================================================
def save_all_fold_metrics(opt, metrics_df, fold_no):

    all_metrics_path = os.path.join(
        opt.result_dir,
        "SpaIM_all_fold_metrics.csv"
    )

    # --------------------------------------------------------
    # Load previous folds if file exists
    # --------------------------------------------------------
    if os.path.exists(all_metrics_path):

        old_df = pd.read_csv(
            all_metrics_path
        )

        # Remove this fold if we are rerunning it
        if "fold" in old_df.columns:
            old_df = old_df[
                old_df["fold"] != fold_no
            ]

        all_metrics_df = pd.concat(
            [
                old_df,
                metrics_df
            ],
            ignore_index=True
        )

    else:

        all_metrics_df = metrics_df.copy()

    # --------------------------------------------------------
    # Sort Fold 1 -> Fold 5
    # --------------------------------------------------------
    all_metrics_df = (
        all_metrics_df
        .sort_values("fold")
        .reset_index(drop=True)
    )

    all_metrics_df.to_csv(
        all_metrics_path,
        index=False
    )

    print("\n" + "=" * 80)
    print("ALL FOLD METRICS")
    print("=" * 80)
    print(all_metrics_df)

    print(
        "\nSaved all-fold metrics:",
        all_metrics_path
    )

    # ========================================================
    # CALCULATE MEAN +/- STD
    # ========================================================

    numeric_df = all_metrics_df.drop(
        columns=["fold"],
        errors="ignore"
    )

    numeric_df = numeric_df.select_dtypes(
        include=[np.number]
    )

    summary_rows = []

    for metric_name in numeric_df.columns:

        values = (
            numeric_df[metric_name]
            .dropna()
            .to_numpy(dtype=float)
        )

        if len(values) == 0:
            continue

        metric_mean = np.mean(values)

        # NumPy population std, same as np.std(values)
        metric_std = np.std(
            values,
            ddof=0
        )

        summary_rows.append({
            "metric": metric_name,
            "n_folds": len(values),
            "mean": metric_mean,
            "std": metric_std,
            "mean+-std":
                f"{metric_mean:.5f} +/- {metric_std:.5f}"
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = os.path.join(
        opt.result_dir,
        "SpaIM_mean_std.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\n" + "=" * 80)
    print("MEAN +/- STD")
    print("=" * 80)

    print(summary_df)

    print(
        "\nSaved mean/std:",
        summary_path
    )


# ============================================================
# VALIDATION
# ============================================================
def val(opt):

    # ========================================================
    # LOAD VALIDATION DATA
    # ========================================================
    valdataset = ImputationDataset(
        opt,
        istrain='val'
    )

    # Because spa_val is transposed:
    #
    # obs_names = test genes
    # var_names = ST cells/spots
    #
    gene_names, cell_names = (
        valdataset.get_eval_names()
    )

    gene_names = list(gene_names)
    cell_names = list(cell_names)

    fold_no = opt.kfold + 1

    print("\n" + "=" * 80)
    print(f"VALIDATION FOLD {fold_no}")
    print("=" * 80)

    print(
        "Number of test genes :",
        len(gene_names)
    )

    print(
        "Number of ST cells   :",
        len(cell_names)
    )

    opt.sc_dim = (
        valdataset.get_cluster_dim()
    )

    valdataloader = (
        torch.utils.data.DataLoader(
            valdataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=0
        )
    )

    # ========================================================
    # LOAD TRAINED MODEL
    # ========================================================

    opt.sc_dim = valdataset.seq_val.shape[1]
    opt.st_dim = valdataset.spa_val.shape[1]

    print("SC dimension:", opt.sc_dim)
    print("ST dimension:", opt.st_dim)
    model = ImputeModule(opt)

    if opt.parallel:

        model = (
            torch.nn
            .DataParallel(model)
            .cuda()
            .module
        )

    else:

        model = model.cuda()

    checkpoint_path = os.path.join(
        opt.save_path,
        'last_%d.pth' % opt.kfold
    )

    print(
        "Loading checkpoint:",
        checkpoint_path
    )

    model.load(
        checkpoint_path
    )

    # ========================================================
    # INFERENCE
    # ========================================================
    pred_batches = []
    true_batches = []

    with torch.no_grad():

        for i, (
            seq,
            st_style,
            spa,
            seq_cls
        ) in enumerate(valdataloader):

            inputs = {
                'scx': seq,
                'st_style': st_style,
                'sc_cls': seq_cls
            }

            # IMPORTANT:
            # Inference must happen INSIDE the loop.
            model.set_input(
                inputs,
                istrain=0
            )

            out = model.inference()

            pred = (
                out['st_fake']
                .detach()
                .cpu()
                .numpy()
            )

            true = (
                spa
                .detach()
                .cpu()
                .numpy()
            )

            pred_batches.append(
                pred
            )

            true_batches.append(
                true
            )

    # ========================================================
    # CONCATENATE ALL TEST GENE BATCHES
    # ========================================================
    pred_result = np.concatenate(
        pred_batches,
        axis=0
    )

    true_result = np.concatenate(
        true_batches,
        axis=0
    )

    print(
        "\nBefore transpose:"
    )

    print(
        "Prediction:",
        pred_result.shape
    )

    print(
        "Ground truth:",
        true_result.shape
    )

    # --------------------------------------------------------
    # SpaIM:
    #
    # test genes × ST cells
    #
    # Change to our baseline convention:
    #
    # ST cells × test genes
    # --------------------------------------------------------
    pred_result = pred_result.T
    true_result = true_result.T

    # Original SpaIM behavior
    pred_result[
        pred_result < 0
    ] = 0

    print(
        "\nAfter transpose:"
    )

    print(
        "Prediction:",
        pred_result.shape
    )

    print(
        "Ground truth:",
        true_result.shape
    )

    # ========================================================
    # SAFETY CHECKS
    # ========================================================
    assert (
        pred_result.shape ==
        true_result.shape
    ), (
        f"Prediction shape "
        f"{pred_result.shape} != "
        f"truth shape "
        f"{true_result.shape}"
    )

    assert (
        pred_result.shape[0] ==
        len(cell_names)
    ), (
        f"ST-cell mismatch: "
        f"{pred_result.shape[0]} "
        f"vs {len(cell_names)}"
    )

    assert (
        pred_result.shape[1] ==
        len(gene_names)
    ), (
        f"Gene mismatch: "
        f"{pred_result.shape[1]} "
        f"vs {len(gene_names)}"
    )

    # ========================================================
    # DATAFRAMES
    # ========================================================
    pred_df = pd.DataFrame(
        pred_result,
        index=cell_names,
        columns=gene_names
    )

    true_df = pd.DataFrame(
        true_result,
        index=cell_names,
        columns=gene_names
    )

    # ========================================================
    # SAVE PREDICTION CSV
    # ========================================================
    pred_csv = os.path.join(
        opt.result_dir,
        f"fold_{fold_no}_pred.csv"
    )

    pred_df.to_csv(
        pred_csv
    )

    print(
        "\nPrediction saved:",
        pred_csv
    )

    # ========================================================
    # SAVE TRUE MATRIX CSV
    # ========================================================
    truth_csv = os.path.join(
        opt.result_dir,
        f"fold_{fold_no}_truth.csv"
    )

    true_df.to_csv(
        truth_csv
    )

    print(
        "Ground truth saved:",
        truth_csv
    )

    # ========================================================
    # KEEP ORIGINAL SpaIM PICKLE OUTPUT
    # ========================================================
    pred_df.to_pickle(
        os.path.join(
            opt.save_path,
            'impute_result_%d.pkl'
            % opt.kfold
        )
    )

    true_df.to_pickle(
        os.path.join(
            opt.save_path,
            'input_result_%d.pkl'
            % opt.kfold
        )
    )

    # ========================================================
    # CALCULATE YOUR BASELINE METRICS
    # ========================================================
    print("\n" + "=" * 80)
    print(
        f"CALCULATING METRICS - "
        f"FOLD {fold_no}"
    )
    print("=" * 80)

    metrics = compute_metrics(
        true_result,
        pred_result
    )

    print(
        "\nRaw metric result:"
    )

    print(
        metrics
    )

    # ========================================================
    # CONVERT TO ONE FOLD ROW
    # ========================================================
    metrics_df = metrics_to_dataframe(
        metrics,
        fold_no
    )

    print(
        "\nFold metric row:"
    )

    print(
        metrics_df
    )

    # ========================================================
    # SAVE INDIVIDUAL FOLD METRICS
    # ========================================================
    fold_metric_path = os.path.join(
        opt.result_dir,
        f"fold_{fold_no}_metrics.csv"
    )

    metrics_df.to_csv(
        fold_metric_path,
        index=False
    )

    print(
        "\nIndividual fold metrics saved:",
        fold_metric_path
    )

    # ========================================================
    # UPDATE COMBINED 5-FOLD FILE
    # ========================================================
    save_all_fold_metrics(
        opt,
        metrics_df,
        fold_no
    )


# ============================================================
# DATA AUGMENTATION
# ============================================================
def Data_augmentation(
    data1,
    data2,
    data3,
    times=2,
    zero_fraction=0.1
):

    all_data = [
        data1
    ]

    for _ in range(
        times - 1
    ):

        device = data1.device

        mask = (
            torch.rand(
                data1.shape,
                device=device
            )
            <
            zero_fraction
        )

        augmented_data = torch.where(
            mask,
            torch.zeros_like(
                data1
            ),
            data1
        )

        all_data.append(
            augmented_data
        )

    concatenated_data1 = torch.cat(
        all_data,
        dim=0
    )

    concatenated_data2 = torch.cat(
        [data2] * times,
        dim=0
    )

    concatenated_data3 = torch.cat(
        [data3] * times,
        dim=0
    )

    return (
        concatenated_data1,
        concatenated_data2,
        concatenated_data3
    )


# ============================================================
# TRAIN
# ============================================================
def train(opt):

    dataset = ImputationDataset(
        opt,
        istrain='train'
    )
    opt.st_dim = dataset.spa_train.shape[1]

    print("ST dimension:", opt.st_dim)


    dataloader = (
        torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=0
        )
    )

    opt.sc_dim = (
        dataset.get_cluster_dim()
    )

    print("\n" + "=" * 80)
    print(
        f"TRAINING FOLD "
        f"{opt.kfold + 1}"
    )
    print("=" * 80)

    print(
        "Training genes:",
        len(dataset)
    )

    print(
        "SC cluster dimension:",
        opt.sc_dim
    )

    model = ImputeModule(
        opt
    )

    if opt.parallel:

        model = (
            torch.nn
            .DataParallel(model)
            .cuda()
            .module
        )

    else:

        model = model.cuda()

    # ========================================================
    # EPOCH LOOP
    # ========================================================
    tqdm_bar = tqdm(
        range(opt.epochs)
    )

    for epoch in tqdm_bar:

        for i, (
            seq,
            spa,
            sc_style,
            st_style,
            seq_cls
        ) in enumerate(dataloader):

            # =================================================
            # ORIGINAL SpaIM DATA AUGMENTATION
            # =================================================
            seq, spa, st_style = Data_augmentation(
                seq,
                spa,
                st_style,
                times=4,
                zero_fraction=0.5
            )

            inputs = {
                'scx': seq,
                'stx': spa,
                'sc_style': sc_style,
                'st_style': st_style,
                'sc_cls': seq_cls
            }

            model.set_input(
                inputs,
                istrain=1
            )

            model.update_parameters()

            loss_stat = (
                model.get_current_loss()
            )

            msg = (
                'Epoch: {}/{}, '
                'Iter: {}/{}, '
                .format(
                    epoch + 1,
                    opt.epochs,
                    i + 1,
                    len(dataloader)
                )
            )

            for k, v in (
                loss_stat.items()
            ):

                msg += (
                    ' %s: %.4f'
                    % (k, v)
                )

            tqdm_bar.set_description(
                msg
            )

    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================
    checkpoint_path = os.path.join(
        opt.save_path,
        'last_%d.pth'
        % opt.kfold
    )

    model.save(
        checkpoint_path
    )

    print(
        "\nSaved checkpoint:",
        checkpoint_path
    )


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':

    opt = Options().parse()

    seed_everything(
        opt.seed
    )

    # ========================================================
    # RESULT DIRECTORY
    # ========================================================
    # opt.result_dir = os.path.join(
    #     "results",
    #     "SpaIM",
    #     opt.dataset_name
    # )

    opt.result_dir = os.path.join(
        opt.result_dir,
        opt.dataset_name
    )

    os.makedirs(opt.result_dir, exist_ok=True)

    os.makedirs(
        opt.result_dir,
        exist_ok=True
    )

    os.makedirs(
        opt.save_path,
        exist_ok=True
    )

    print(
        "Results directory:",
        opt.result_dir
    )

    print(
        "Checkpoint directory:",
        opt.save_path
    )

    print(
        "Dataset:",
        opt.dataset_name
    )

    print(
        "Fold:",
        opt.kfold + 1
    )

    # ========================================================
    # TRAIN + TEST
    # ========================================================
    if opt.val_only == 0:

        train(
            opt
        )

        val(
            opt
        )

    else:

        val(
            opt
        )

    torch.cuda.empty_cache()








# python3 main.py     --dataset_name seqFISH     --seq_data /home/2023csz0012/AmitCode/data/processedData/seqFISH/scRNA_common.h5ad     --spa_data /home/2023csz0012/AmitCode/data/processedData/seqFISH/spatial.h5ad     --kfold 3     --val_only 0     --cluster leiden     --save_path checkpoints/SpaIM



# merFISH
# python3 main.py     --dataset_name merFISH     --seq_data /home/2023csz0012/AmitCode/data/processedData/merFISH/scRNAseq_full.h5ad     --spa_data /home/2023csz0012/AmitCode/data/updatedDataset/merFISH/spatial_2.h5ad     --kfold 3     --val_only 0     --cluster leiden     --save_path checkpoints/SpaIM