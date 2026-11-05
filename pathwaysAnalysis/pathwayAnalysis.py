#!/usr/bin/env python3
"""
python3 pathwayAnalysis.py \
    --fname merFISH \
    --model SpaGENE \
    --pred_path /SpaGene/fold_results \
    --pathway_path /path/to/pathways.csv \
    --spatial_path /path/to/spatial.h5ad \
    --base /path/to/project


"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


def dense(x):
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def read_predictions(path):
    """Read a prediction CSV or merge fold_*_pred.csv by gene columns."""
    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        fold_dir = path
        files = glob.glob(os.path.join(fold_dir, 'fold_*_pred.csv'))
        if not files:
            fold_dir = os.path.join(path, 'fold_results')
            files = glob.glob(os.path.join(fold_dir, 'fold_*_pred.csv'))
        if files:
            files.sort(key=lambda p: int(os.path.basename(p).split('_')[1]))
        else:
            csv_files = glob.glob(os.path.join(path, '*.csv'))
            if len(csv_files) != 1:
                raise FileNotFoundError(
                    f'Expected fold_*_pred.csv or exactly one CSV in {path}'
                )
            files = csv_files
    else:
        raise FileNotFoundError(f'Prediction path not found: {path}')

    frames = []
    n_rows = None
    for path_i in files:
        df = pd.read_csv(path_i)
        # Remove index column introduced by pandas .to_csv(index=True).
        df = df.loc[:, ~df.columns.astype(str).str.match(r'^Unnamed:')]
        df.columns = pd.Index([str(g).strip().upper() for g in df.columns])
        if df.columns.duplicated().any():
            raise ValueError(f'Duplicate gene names in {path_i}')
        if n_rows is not None and len(df) != n_rows:
            raise ValueError('Fold predictions have different cell counts; cannot concatenate without cell IDs.')
        n_rows = len(df)
        print('Loaded:', path_i, df.shape)
        frames.append(df.reset_index(drop=True))

    merged = pd.concat(frames, axis=1)
    if merged.columns.duplicated().any():
        raise ValueError('Duplicate prediction genes across folds.')
    return merged


def main():
    parser = argparse.ArgumentParser(description='Pathway analysis without row-index files')
    parser.add_argument('--fname', required=True, help='Dataset name')
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--pred_path', required=True, help='Prediction CSV or fold directory')
    parser.add_argument('--pathway_path', required=True, help='Pathway coverage CSV')
    parser.add_argument('--spatial_path', default=None, help='ST .h5ad path; otherwise uses --base')
    parser.add_argument('--base', default=None, help='Dataset root containing data/processedData/')
    parser.add_argument('--min_pathway_genes', type=int, default=10)
    parser.add_argument('--max_pathways', type=int, default=20)
    parser.add_argument('--output_root', default='PathwayResults')
    args = parser.parse_args()

    # Local benchmark metrics module (same implementation as the existing project).
    if args.base:
        sys.path.insert(0, os.path.join(args.base, 'code'))
    try:
        from utils.compute_metrics import compute_metrics
    except ImportError as exc:
        raise ImportError(
            'Cannot import utils.compute_metrics. Run from the project root or set --base '
            'to your project directory containing code/utils/compute_metrics.py.'
        ) from exc

    spatial_path = args.spatial_path
    if spatial_path is None:
        if args.base is None:
            parser.error('Specify --spatial_path or --base')
        spatial_path = os.path.join(args.base, 'data', 'processedData', args.fname, 'spatial.h5ad')

    spatial = sc.read_h5ad(spatial_path)
    spatial.var_names = pd.Index([str(g).strip().upper() for g in spatial.var_names])
    if not spatial.var_names.is_unique:
        spatial.var_names_make_unique()
    sc.pp.filter_cells(spatial, min_counts=1)
    sc.pp.filter_genes(spatial, min_counts=1)
    sc.pp.normalize_total(spatial, target_sum=1e4)
    sc.pp.log1p(spatial)
    y_true = pd.DataFrame(dense(spatial.X), columns=spatial.var_names)

    y_pred = read_predictions(args.pred_path)
    # No index-based reordering: the CSV must already be in filtered ST cell order.
    if len(y_pred) != len(y_true):
        raise ValueError(
            f'Prediction has {len(y_pred)} cells, filtered ST truth has {len(y_true)}. '
            'Without row indices, automatic alignment is unsafe. Supply predictions '
            'already aligned to the filtered spatial data.'
        )
    if len(y_pred) == 0:
        raise ValueError('No spatial cells available for evaluation.')
    print('Assuming prediction and filtered ST truth rows have identical cell order.')

    genes = [g for g in y_pred.columns if g in y_true.columns]
    if not genes:
        raise ValueError('No genes shared between predictions and spatial truth.')
    y_pred = y_pred[genes]
    y_true = y_true[genes]
    print('Aligned prediction/truth:', y_pred.shape, y_true.shape)

    pathway = pd.read_csv(args.pathway_path)
    required = {'source', 'pathway', 'p-value', 'q-value', 'shared_pathway_genes'}
    missing = required.difference(pathway.columns)
    if missing:
        raise ValueError(f'Pathway CSV missing columns: {sorted(missing)}')
    for col in ['p-value', 'q-value']:
        pathway[col] = pd.to_numeric(pathway[col], errors='coerce')
    pathway = pathway.loc[(pathway['p-value'] < 0.05) & (pathway['q-value'] < 0.05)].copy()
    if 'n_ST_sc_shared' in pathway.columns:
        pathway['n_ST_sc_shared'] = pd.to_numeric(pathway['n_ST_sc_shared'], errors='coerce')
        pathway = pathway.loc[pathway['n_ST_sc_shared'] >= args.min_pathway_genes]
    pathway = pathway.sort_values(['q-value', 'p-value']).drop_duplicates(['source', 'pathway'])
    if 'n_ST_sc_shared' in pathway.columns:
        pathway = pathway.sort_values(['n_ST_sc_shared', 'q-value'], ascending=[False, True])

    rows = []
    for _, pathway_row in pathway.iterrows():
        if pd.isna(pathway_row['shared_pathway_genes']):
            continue
        pathway_genes = {
            x.strip().upper() for x in str(pathway_row['shared_pathway_genes']).split(';') if x.strip()
        }
        selected = [g for g in genes if g in pathway_genes]
        if len(selected) < args.min_pathway_genes:
            continue
        results = compute_metrics(y_true[selected].values, y_pred[selected].values)
        rows.append({
            'p-value': pathway_row['p-value'],
            'q-value': pathway_row['q-value'],
            'pathway': pathway_row['pathway'],
            'source': pathway_row['source'],
            'n_ST_sc_shared': pathway_row.get('n_ST_sc_shared', np.nan),
            'cluster': pathway_row.get('cluster', ''),
            'n_genes': len(selected),
            'genes': ';'.join(selected),
            'n_cells': len(y_true),
            **results,
        })
        if args.max_pathways > 0 and len(rows) >= args.max_pathways:
            break

    output_dir = os.path.join(args.output_root, args.fname)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{args.fname}_{args.model}_pathway_metrics.csv')
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f'Pathways evaluated: {len(rows)}; results saved: {output_path}')


if __name__ == '__main__':
    main()
