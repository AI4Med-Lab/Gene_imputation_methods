# Spatial Transcriptomics Missing-Gene Imputation Baselines

This folder contains baseline experiments for **missing-gene prediction in spatial transcriptomics (ST)** using paired ST and single-cell RNA-sequencing (scRNA-seq) data. The supplied benchmarking script includes **SpaGE, Tangram, gimVI, and uniPort**. The same working folder also contains **stDiff**, which is used as a separate diffusion-based baseline following the method described in its original publication.

> **Scope:** The uploaded combined script executes SpaGE, Tangram, gimVI, and uniPort. It does **not** call stDiff. Run stDiff through its own implementation and instructions. The stDiff code was not included in the supplied script, so its exact command and settings are not documented here.

## Methods

| Method | General approach | Implementation in this project |
| --- | --- | --- |
| **SpaGE** | Principal-vector alignment and nearest-neighbor transfer from scRNA-seq | Gene-wise five-fold held-out evaluation |
| **Tangram** | Mapping scRNA-seq cells/clusters to spatial locations and projecting gene expression | Gene-wise five-fold held-out evaluation |
| **gimVI** | Variational integration of scRNA-seq and spatial gene-expression data | Gene-wise five-fold held-out evaluation; uses raw expression counts for training |
| **uniPort** | Integration of data modalities with a variational autoencoder | Gene-wise five-fold held-out evaluation |
| **stDiff** | Conditional diffusion-based gene-expression imputation | Separate implementation, following the original stDiff paper |

## Inputs

Two AnnData `.h5ad` files are required for the combined baseline script:

- `spatial.h5ad`: spatial transcriptomics measurements, cells/spots by genes.
- `scRNA.h5ad`: reference single-cell RNA-seq expression, cells by genes.

Both must have valid gene names in `var_names`. The script standardizes gene names by stripping whitespace and converting to uppercase, filters empty cells and genes, and intersects the genes present in both modalities. It normalizes and log-transforms copies of the aligned data (`normalize_total(target_sum=1e4)` followed by `log1p`) for appropriate baselines. **gimVI training uses count data**, whereas its held-out evaluation is aligned with the processed ST truth.

## Five-fold gene holdout

The combined script uses `KFold(n_splits=5, shuffle=True, random_state=--rand)` over shared genes:

1. Approximately 80% of shared genes are available as ST input features in each fold.
2. Approximately 20% of shared ST genes are held out for evaluation.
3. The scRNA-seq reference supplies the corresponding reference expression.
4. Predictions are evaluated against held-out ST measurements.
5. Fold predictions and metrics are saved to method-specific output folders.

There is no separate *outer* cell cross-validation split in the supplied script. Some methods may apply their own filtering or internal model-training split. For gimVI, filtered spatial-cell row indices are saved to support alignment.

## Installation

Use a Python environment with the dependencies required by the baselines you plan to execute, including:

```bash
pip install numpy pandas scipy scanpy anndata scikit-learn torch
```

The script additionally imports `scvi-tools` (for gimVI), `uniport`, and local implementations of SpaGE and Tangram. Install those packages and ensure the local paths match the directory layout of your repository. The project also imports `utils.compute_metrics` and `process.result_analysis`; those modules must be accessible. stDiff has **its own installation requirements**; follow its original repository.

## Running the combined baselines

Before execution, replace `/paste/your/path/` in the combined script with the correct data directory. The script's command-line arguments are `--sc_data`, `--sp_data`, `--document`, `--filename`, and `--rand`.

```bash
python3 -u baseline.py \
  --sc_data scRNA.h5ad \
  --sp_data spatial.h5ad \
  --document ExampleDataset \
  --filename ExampleDataset \
  --rand 0
```

**Note:** The uploaded script calls SpaGE, Tangram, gimVI, and uniPort sequentially at its end. Running this command runs all four methods, rather than selecting one. Rename `baseline.py` in the example if your actual combined script has a different filename.

## Running stDiff

stDiff is maintained separately in this working folder and follows its original paper: **“stDiff: a diffusion model for imputing spatial transcriptomics through single-cell transcriptomics.”** It uses a conditional diffusion process learned from scRNA-seq to impute unobserved ST gene expression.

For stDiff installation, supported options, and model-specific settings, see the [official stDiff repository](https://github.com/fdu-wangfeilab/stDiff). Use the actual stDiff entry-point script in your local folder; its invocation cannot be established from the combined baseline source supplied here.

When reporting stDiff results alongside the other methods, confirm that the **held-out gene splits, cell alignment, expression scale, and evaluation definitions** match those of the benchmark.

## Evaluation and outputs

The combined script imports `compute_metrics` from `utils.compute_metrics` to evaluate held-out predictions. Its exact formulas and constant-gene handling are determined by that local file. The script logs `pearson_mean`, `spearman_mean_genewise`, and `l2_errors_mean` where provided.

An illustrative output layout for the combined script is:

```text
Result/
└── ExampleDataset/
    ├── spage/
    │   ├── fold_results/
    │   │   ├── fold_1_pred.csv
    │   │   ├── fold_1_true.csv
    │   │   ├── ...
    │   │   ├── fold_metrics.csv
    │   │   └── metrics_summary.csv
    │   └── SpaGE_impute.csv
    ├── tangram/
    │   ├── fold_results/
    │   └── Tangram_impute.csv
    ├── gimVI/
    │   ├── fold_results/
    │   └── gimVI_impute.csv
    └── uniport/
        ├── fold_results/
        └── uniport_impute.csv
```

The **gimVI** fold directory additionally saves `fold_N_row_indices.npy` and other fold-specific metadata. stDiff outputs depend on its separate implementation and are not assumed to have this exact structure.

## Reproducibility notes

- Use the same data files, preprocessing policy, and held-out gene splits across methods when interpreting comparisons.
- Be aware that `np.intersect1d` sorts common gene names, which determines the gene order used by `KFold`.
- Keep row-index files when a baseline filters cells after the initial preprocessing.
- Record each method's package version, hyperparameters, GPU/CUDA configuration, and random seed.
- The combined script uses method-specific defaults and hard-coded settings. Verify them before claiming a strictly matched hyperparameter comparison.

## Attribution and original code

This repository collects **benchmark adaptations of existing published methods**. Model authorship remains with the respective original researchers. Cite the original publication and repository for each method when using the results.

- **stDiff paper:** [stDiff: a diffusion model for imputing spatial transcriptomics through single-cell transcriptomics](https://pmc.ncbi.nlm.nih.gov/articles/PMC11021815/)
- **stDiff source:** https://github.com/fdu-wangfeilab/stDiff

The other baseline methods (SpaGE, Tangram, gimVI, and uniPort) should likewise be credited to their respective original papers and implementations.
