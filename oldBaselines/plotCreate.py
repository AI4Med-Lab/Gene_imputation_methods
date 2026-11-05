#!/usr/bin/env python3
"""Compare observed and cross-validation-imputed spatial gene expression via joint UMAP.

Prediction files must be named fold_<N>_pred.csv. Each CSV contains cell rows
and predicted gene columns. Optional fold_<N>_row_indices.npy files map each
CSV row to its original (zero-based) spatial cell row. Without index files,
all fold CSVs MUST have the same row order, indicated by --row-space.


python3 -u plot_umap.py \
    --spatial /data/Dataset/merFISH/spatial.h5ad \
    --pred-dir /your/results/CSV/path \
    --dataset merFISH \
    --model scGPD \
    --output-dir ./Plots \
    --n-pcs 50 \
    --n-neighbors 15 \
    --seed 0


"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--spatial', type=Path, required=True, help='Observed spatial .h5ad')
    p.add_argument('--pred-dir', type=Path, required=True, help='Directory with fold_<N>_pred.csv')
    p.add_argument('--output-dir', type=Path, default=Path('plots'))
    p.add_argument('--dataset', default='dataset', help='Output filename label')
    p.add_argument('--model', default='model', help='Output filename label')
    p.add_argument('--indices', type=Path, help='Optional .npy of original ST row IDs to include')
    p.add_argument('--row-space', choices=('auto', 'original', 'filtered', 'external'), default='auto',
                   help='Meaning of CSV rows when fold row-index files are absent')
    p.add_argument('--input-normalized', action='store_true',
                   help='Spatial .X is already library-normalized and log1p transformed')
    p.add_argument('--target-sum', type=float, default=1e4)
    p.add_argument('--min-counts', type=int, default=1)
    p.add_argument('--n-pcs', type=int, default=50)
    p.add_argument('--n-neighbors', type=int, default=15)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-cells', type=int, default=None,
                   help='Optionally sample this many aligned cells to limit memory (same cells in both groups)')
    p.add_argument('--point-size', type=float, default=None)
    p.add_argument('--dpi', type=int, default=300)
    return p.parse_args()


def normalize_gene_names(names):
    return pd.Index([str(x).strip().upper() for x in names])


def require_unique(arr, label):
    if len(np.unique(arr)) != len(arr):
        raise ValueError(f'{label} contains duplicate values; alignment would be ambiguous')


def positions_for_indices(source, needed):
    """Find rows of source for needed IDs, preserving needed order."""
    source = np.asarray(source, dtype=np.int64)
    needed = np.asarray(needed, dtype=np.int64)
    require_unique(source, 'Source row indices')
    order = np.argsort(source)
    sorted_source = source[order]
    pos = np.searchsorted(sorted_source, needed)
    valid = pos < len(sorted_source)
    matches = np.zeros(len(pos), dtype=bool)
    matches[valid] = sorted_source[pos[valid]] == needed[valid]
    if not np.all(matches):
        raise ValueError(f'{(~matches).sum()} requested cell IDs absent from the source')
    return order[pos]


def read_predictions(path):
    df = pd.read_csv(path)
    # Remove only the common pandas index export, not genuine gene columns.
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df.columns = normalize_gene_names(df.columns)
    if not df.columns.is_unique:
        raise ValueError(f'Duplicate gene columns (after name normalization) in {path}')
    if not all(pd.api.types.is_numeric_dtype(t) for t in df.dtypes):
        raise ValueError(f'Non-numeric prediction column in {path}')
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError(f'NaN/Inf prediction values in {path}')
    return df


def merge_folds(args, original_n, retained_original):
    files = sorted(args.pred_dir.glob('fold_*_pred.csv'),
                   key=lambda f: int(re.fullmatch(r'fold_(\d+)_pred\.csv', f.name).group(1))
                   if re.fullmatch(r'fold_(\d+)_pred\.csv', f.name) else -1)
    if not files:
        raise FileNotFoundError(f'No fold_*_pred.csv files found in {args.pred_dir}')
    files = [f for f in files if re.fullmatch(r'fold_(\d+)_pred\.csv', f.name)]
    fold_ids = [int(re.fullmatch(r'fold_(\d+)_pred\.csv', f.name).group(1)) for f in files]
    index_paths = [args.pred_dir / f'fold_{i}_row_indices.npy' for i in fold_ids]
    has_indices = [p.is_file() for p in index_paths]
    if any(has_indices) and not all(has_indices):
        raise ValueError('Index files found for only some folds. Supply all, or remove them and specify --row-space.')
    external = None
    if args.indices is not None:
        external = np.asarray(np.load(args.indices), dtype=np.int64).ravel()
        require_unique(external, 'External indices')
        if np.any((external < 0) | (external >= original_n)):
            raise ValueError('External indices must refer to original spatial rows')

    if all(has_indices):
        fold_indices = [np.asarray(np.load(p), dtype=np.int64).ravel() for p in index_paths]
        for idx, path in zip(fold_indices, index_paths):
            require_unique(idx, str(path))
            if np.any((idx < 0) | (idx >= original_n)):
                raise ValueError(f'Row IDs outside original spatial range: {path}')
        used = fold_indices[0]
        for idx in fold_indices[1:]:
            used = np.intersect1d(used, idx)
        used = np.intersect1d(used, retained_original)
        if external is not None:
            used = np.intersect1d(used, external)
        used = np.sort(used)
    else:
        fold_indices = None
        first_count = len(pd.read_csv(files[0], usecols=[0]))
        candidates = {
            'original': np.arange(original_n, dtype=np.int64),
            'filtered': retained_original,
        }
        if external is not None:
            candidates['external'] = external
        if args.row_space == 'auto':
            matching = [name for name, idx in candidates.items() if len(idx) == first_count]
            if len(matching) != 1:
                raise ValueError(f'Cannot infer CSV row space uniquely ({matching}). Specify --row-space explicitly.')
            mode = matching[0]
        else:
            mode = args.row_space
        if mode not in candidates or len(candidates[mode]) != first_count:
            raise ValueError(f'--row-space {mode} is incompatible with prediction rows or missing --indices')
        prediction_ids = candidates[mode]
        used = np.intersect1d(prediction_ids, retained_original)
        if external is not None:
            used = np.intersect1d(used, external)
        used = np.sort(used)

    if used.size == 0:
        raise ValueError('No shared cells survive alignment and filtering')
    if args.max_cells is not None:
        if args.max_cells < 3:
            raise ValueError('--max-cells must be at least 3')
        if used.size > args.max_cells:
            used = np.sort(np.random.default_rng(args.seed).choice(used, args.max_cells, replace=False))

    frames = []
    for i, file in enumerate(files):
        print(f'Loading {file}')
        df = read_predictions(file)
        ids = fold_indices[i] if fold_indices is not None else prediction_ids
        if len(df) != len(ids):
            raise ValueError(f'Rows in {file} ({len(df)}) != corresponding index length ({len(ids)})')
        aligned = df.iloc[positions_for_indices(ids, used)].reset_index(drop=True)
        frames.append(aligned)
    merged = pd.concat(frames, axis=1)
    if merged.columns.duplicated().any():
        duplicated = merged.columns[merged.columns.duplicated()].unique().tolist()
        raise ValueError(f'Genes predicted in multiple folds: {duplicated[:20]}. '
                         'Expected nonoverlapping held-out genes; resolve duplicates before comparison.')
    return used, merged


def main():
    args = parse_args()
    if args.n_neighbors < 2 or args.n_pcs < 2:
        raise ValueError('--n-neighbors and --n-pcs must be >= 2')
    observed = sc.read_h5ad(args.spatial)
    original_n = observed.n_obs
    observed.obs['_original_row'] = np.arange(original_n, dtype=np.int64)
    observed.var_names = normalize_gene_names(observed.var_names)
    if not observed.var_names.is_unique:
        raise ValueError('Spatial gene names are duplicated after uppercase/strip normalization')
    sc.pp.filter_cells(observed, min_counts=args.min_counts)
    sc.pp.filter_genes(observed, min_counts=args.min_counts)
    retained = observed.obs['_original_row'].to_numpy(dtype=np.int64)
    if not args.input_normalized:
        sc.pp.normalize_total(observed, target_sum=args.target_sum)
        sc.pp.log1p(observed)

    used, pred = merge_folds(args, original_n, retained)
    observed_positions = positions_for_indices(retained, used)
    shared = observed.var_names[observed.var_names.isin(pred.columns)]
    if len(shared) < 3:
        raise ValueError(f'Only {len(shared)} common genes; need at least 3')
    real = observed[observed_positions, shared].copy()
    imputed = ad.AnnData(X=pred.loc[:, shared].to_numpy(dtype=np.float32),
                         obs=pd.DataFrame(index=real.obs_names.copy()),
                         var=pd.DataFrame(index=shared.copy()))
    # Use distinct names for stacked rows, but retain pairing via the used indices file.
    real.obs_names = pd.Index(['observed_' + str(i) for i in used])
    imputed.obs_names = pd.Index(['imputed_' + str(i) for i in used])
    combined = ad.concat([real, imputed], label='source', keys=['Observed', 'Imputed'], join='inner')

    # Joint PCA/UMAP: both sources must share the same feature space and embedding.
    # Caution: dense scaling on very large data can use substantial memory.
    if sparse.issparse(combined.X):
        combined.X = combined.X.toarray().astype(np.float32)
    else:
        combined.X = np.asarray(combined.X, dtype=np.float32)
    if not np.isfinite(combined.X).all():
        raise ValueError('Joint expression contains NaN or Inf')
    sc.pp.scale(combined, max_value=10)
    n_pcs = min(args.n_pcs, combined.n_vars - 1, combined.n_obs - 1)
    sc.tl.pca(combined, n_comps=n_pcs, random_state=args.seed)
    sc.pp.neighbors(combined, n_neighbors=min(args.n_neighbors, combined.n_obs - 1),
                    n_pcs=n_pcs, random_state=args.seed)
    sc.tl.umap(combined, random_state=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{args.dataset}_{args.model}'
    png_path = args.output_dir / f'{stem}_umap.png'
    np.save(args.output_dir / f'{stem}_used_row_indices.npy', used)
    size = args.point_size if args.point_size is not None else (1 if combined.n_obs > 1_000_000 else 3 if combined.n_obs > 200_000 else 20)
    ax = sc.pl.umap(combined, color='source', size=size, alpha=0.5,
                    frameon=True, title='', show=False)
    ax.figure.savefig(png_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(ax.figure)
    print(f'Completed: {len(used):,} paired cells, {len(shared)} genes')
    print(f'Plot: {png_path}')


if __name__ == '__main__':
    main()
