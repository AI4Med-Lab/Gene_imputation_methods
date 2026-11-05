# scGPD Baseline for Spatial Transcriptomics Gene Imputation

This repository contains a **benchmarking adaptation** of [scGPD (single-cell informed Gene Panel Design)](https://github.com/TinaGuo/scGPD) for missing-gene prediction in spatial transcriptomics (ST) using a single-cell RNA-seq (scRNA-seq) reference.

**Scope:** This repository provides the gene-holdout evaluation pipeline. The original scGPD model is an **external dependency**, not a model newly developed here. See the original [scGPD repository](https://github.com/TinaGuo/scGPD) for its implementation and citation information.

## Repository structure

```text
.
├── baseline.py          # Adapted scGPD training and gene-wise CV pipeline
├── compute_metrics.py   # Gene-wise evaluation metrics
└── README.md
```

## Requirements and installation

Use a Python environment with PyTorch, Scanpy, NumPy, pandas, SciPy, scikit-learn, and the other dependencies required by scGPD.

Install the original scGPD package (unless it is already installed in your environment):

```bash
git clone https://github.com/TinaGuo/scGPD.git
cd scGPD
python3 -m pip install -e .

OR 

pip install scGPD
```


Verify the package is available in the **same Python environment** used to run the baseline:

```bash
python3 -c "from scGPD import scGPD, ExpressionDataset, MSELoss; print('scGPD OK')"
```

Expected output:

```text
scGPD OK
```

The pipeline imports `compute_metrics` from the local `compute_metrics.py`. Keep that file beside `baseline.py`. If your metrics file imports `torchsort`, install it separately in the active environment as needed.

## Input data

The pipeline expects two AnnData (`.h5ad`) files:

| Input | Description |
| --- | --- |
| scRNA-seq reference | Cells × genes expression matrix; e.g., `scRNA.h5ad` |
| Spatial transcriptomics | Spatial cells/spots × measured genes expression matrix; e.g., `spatial.h5ad` |

The current script uses dataset paths constructed inside `baseline.py`. **Before running, replace the `YOUR PATH...` placeholders with actual paths.** The `--sc_data` and `--sp_data` options currently specify filenames, not arbitrary absolute paths.

## Method: gene-wise cross-validation

The baseline adapts scGPD to evaluate missing ST genes:

1. Split the shared gene set into five shuffled folds using `KFold` and the specified random seed.
2. In each fold, use the **training genes** as model inputs for both the scRNA-seq reference and the ST observations.
3. Use the **full common gene set** as the scRNA-seq reconstruction target.
4. Construct the scGPD gene-correlation factor from **training genes only**.
5. Train scGPD's Stage-I reconstruction model on the scRNA-seq reference.
6. Reconstruct full gene expression from ST training-gene inputs using deterministic gate selection (threshold 0.5; continuous-probability fallback if none is selected).
7. Evaluate **only the held-out ST genes** against their measured expression.

The held-out ST gene expression is not used as a model input or training target. The scRNA-seq reference, however, retains expression for all common genes as intended by this adaptation.

> **Current-script note:** The uploaded `baseline.py` includes `if fold != 5: continue`. **As written, it runs only fold 5.** Remove those two lines to run all five folds. Do not interpret the exported `all_pred_res.csv` as a complete five-fold prediction matrix when only one fold was run: other columns remain zero-initialized.

## Running the baseline

Once the dataset-path placeholders have been updated, run from the directory containing `baseline.py` and `compute_metrics.py`:

```bash
python3 -u baseline.py \
  --sc_data scRNA.h5ad \
  --sp_data spatial.h5ad \
  --document ExampleDataset \
  --filename ExampleDataset \
  --rand 0 \
  --gpu 0 \
  --scgpd_epochs 200 \
  --scgpd_batchsize 64 \
  --scgpd_lr 0.001
```

To run it in the background on Linux:

```bash
nohup python3 -u baseline.py \
  --sc_data scRNA.h5ad \
  --sp_data spatial.h5ad \
  --document ExampleDataset \
  --filename ExampleDataset \
  --rand 0 \
  > scGPD_run.log 2>&1 &
```

Additional options in the supplied script include `--scgpd_lam`, `--scgpd_lookback`, and `--scgpd_inference_batchsize`. Run `python3 baseline.py --help` to see the arguments.

## Evaluation metrics

The local `compute_metrics.py` evaluates **each held-out gene across spatial cells**, then averages per-gene scores. The supplied implementation reports:

| Key | Metric |
| --- | --- |
| `pearson_mean` | Mean gene-wise Pearson correlation (PCC) |
| `spearman_mean_genewise` | Mean gene-wise Spearman rank correlation (SRCC) |
| `l1_error_mean` | Mean gene-wise absolute error (MAE) |
| `l2_errors_mean` | Mean gene-wise squared error (MSE) |
| `r2_scores_mean` | Mean gene-wise coefficient of determination (R²) |
| `pearson_std` | Standard deviation of gene-wise Pearson correlations |

It also reports error and R² quartiles. The supplied metrics function rounds reported values to four decimal places and assigns zero correlation to constant inputs. The function `spearmanrr` (if included) is a separate differentiable helper and is **not used** in the displayed evaluation call.

## Outputs

With `--document ExampleDataset`, the script creates:

```text
Result/
└── ExampleDataset/
    └── scGPD/
        ├── all_pred_res.csv
        └── fold_results/
            ├── fold_1_pred.csv
            ├── fold_1_true.csv
            ├── fold_1_selected_genes.csv
            ├── ...
            ├── fold_metrics.csv
            └── metrics_summary.csv
```

Only files for **executed** folds are generated. `metrics_summary.csv` summarizes the folds that ran; its standard deviation is a **sample standard deviation** via pandas `.std()`. In the uploaded script, the summary also includes numeric metadata such as `fold`, so select the relevant metric columns for reporting.

## Attribution

This project **uses and adapts** the scGPD software developed by its original authors:

- Original source: https://github.com/TinaGuo/scGPD

Please consult the original repository or associated publication for the authoritative citation, license, and methodological details. This benchmark does not claim to introduce the original scGPD architecture.
