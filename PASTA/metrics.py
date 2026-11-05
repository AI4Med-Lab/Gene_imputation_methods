
"""PASTA gene-holdout benchmark metrics using shared compute_metrics."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from compute_metrics import compute_metrics as shared_compute_metrics


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    genes: Sequence[str],
    mape_threshold: float = 0.1,
) -> Tuple[Dict[str, float], pd.DataFrame]:

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: truth={y_true.shape}, pred={y_pred.shape}"
        )

    if y_true.ndim != 2:
        raise ValueError("Expected 2-D arrays.")

    if len(genes) != y_true.shape[1]:
        raise ValueError("Gene count does not match matrix columns.")

    # ==========================================
    # YOUR EXISTING COMPUTE METRICS
    # ==========================================

    metrics = shared_compute_metrics(
        y_true,
        y_pred,
        genes=genes
    )

    # ==========================================
    # PER-GENE METRICS
    # ==========================================

    rows = []

    for j, gene in enumerate(genes):

        yt = y_true[:, j]
        yp = y_pred[:, j]

        # Use your existing metric implementation
        # to preserve its correlation and R2 behavior.
        gene_metrics = shared_compute_metrics(
            yt.reshape(-1, 1),
            yp.reshape(-1, 1),
            genes=[gene]
        )

        # Masked MAPE: original PASTA definition
        mask = (
            np.isfinite(yt)
            & np.isfinite(yp)
            & (np.abs(yt) > mape_threshold)
        )

        if np.any(mask):
            mape = float(
                np.mean(
                    np.abs((yt[mask] - yp[mask]) / yt[mask])
                ) * 100.0
            )
        else:
            mape = np.nan

        rows.append({
            "gene": gene,
            "pearson": gene_metrics["pearson_mean"],
            "spearman": gene_metrics["spearman_mean_genewise"],
            "l1_mae": gene_metrics["l1_error_mean"],
            "l2_mse": gene_metrics["l2_errors_mean"],
            "r2": gene_metrics["r2_scores_mean"],
            "masked_mape_pct": mape,
        })

    gene_df = pd.DataFrame(rows)

    # ==========================================
    # FOLD-WISE MASKED MAPE
    # ==========================================

    flat_mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
        & (np.abs(y_true) > mape_threshold)
    )

    if np.any(flat_mask):
        overall_mape = float(
            np.mean(
                np.abs(
                    (y_true[flat_mask] - y_pred[flat_mask])
                    / y_true[flat_mask]
                )
            ) * 100.0
        )
    else:
        overall_mape = np.nan

    # ==========================================
    # ADD PASTA-SPECIFIC METRICS
    # ==========================================

    metrics["masked_mape_pct"] = overall_mape
    metrics["spearman_std"] = float(
        gene_df["spearman"].std(ddof=0)
    )

    return metrics, gene_df
