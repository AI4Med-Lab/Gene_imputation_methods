"""Gene-wise evaluation metrics for SpaGene and related baselines."""
import warnings
import numpy as np
from scipy.stats import pearsonr, spearmanr, ConstantInputWarning

def compute_metrics(y_test, preds_all, genes=None):
    """
    Computes metrics L2 errors R2 scores and Pearson correlations for each target/gene.

    Parameters
    ----------
    y_test : np.ndarray
        Ground truth values with shape [n_samples, n_targets].
    preds_all : np.ndarray
        Predictions with shape [n_samples, n_targets].
    genes : sequence, optional
        Gene names corresponding to targets.

    Returns
    -------
    dict
        Metrics using the same implementation as the user's other baselines.
    """

    y_test = np.asarray(y_test)
    preds_all = np.asarray(preds_all)

    if y_test.shape != preds_all.shape:
        raise ValueError(
            f"Metric shape mismatch: y_test={y_test.shape}, preds={preds_all.shape}"
        )
    if y_test.ndim != 2:
        raise ValueError(f"Expected 2-D metric arrays, got {y_test.ndim}-D")
    if genes is not None and len(genes) != y_test.shape[1]:
        raise ValueError(
            f"genes has length {len(genes)}, expected {y_test.shape[1]}"
        )

    errors = []
    l1_errors = []
    r2_scores = []
    pearson_corrs = []
    pearson_genes = []
    spearman_corrs = []
    spearman_genes = []

    for i, target in enumerate(range(y_test.shape[1])):
        preds = preds_all[:, target]
        target_vals = y_test[:, target]

        l1_error_value = float(np.mean(np.abs(target_vals - preds)))

        # Mean squared error for this gene (named L2 in the existing baseline metric).
        l2_error = float(np.mean((preds - target_vals) ** 2))

        # R2 score for this gene.
        total_variance = np.sum((target_vals - np.mean(target_vals)) ** 2)
        if total_variance == 0:
            r2_score = 0.0
        else:
            r2_score = float(
                1 - np.sum((target_vals - preds) ** 2) / total_variance
            )

        # Pearson correlation.
        if len(target_vals) < 2 or len(preds) < 2:
            pearson_corr = 0.0
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConstantInputWarning)
                try:
                    pearson_corr, _ = pearsonr(
                        target_vals.flatten(), preds.flatten()
                    )
                    pearson_corr = (
                        pearson_corr if not np.isnan(pearson_corr) else 0.0
                    )
                except ConstantInputWarning:
                    pearson_corr = 0.0

        # Spearman correlation.
        if len(target_vals) < 2 or len(preds) < 2:
            spearman_corr = 0.0
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConstantInputWarning)
                try:
                    spearman_corr, _ = spearmanr(
                        target_vals.flatten(), preds.flatten()
                    )
                    spearman_corr = (
                        spearman_corr if not np.isnan(spearman_corr) else 0.0
                    )
                except ConstantInputWarning:
                    spearman_corr = 0.0

        errors.append(l2_error)
        l1_errors.append(l1_error_value)
        r2_scores.append(r2_score)
        pearson_corrs.append(pearson_corr)
        spearman_corrs.append(spearman_corr)

        if genes is not None:
            pearson_genes.append(
                {"name": genes[i], "pearson_corr": pearson_corr}
            )
            spearman_genes.append(
                {"name": genes[i], "spearman_corr": spearman_corr}
            )

    results = {
        "pearson_mean": float(f"{np.mean(pearson_corrs): .4f}"),
        "spearman_mean_genewise": float(
            f"{np.mean(spearman_corrs): .4f}"
        ),
        "l1_error_mean": float(f"{np.mean(l1_errors): .4f}"),
        "l2_errors_mean": float(f"{np.mean(errors):.4f}"),
        "r2_scores_mean": float(f"{np.mean(r2_scores):.4f}"),
        "pearson_std": float(f"{np.std(pearson_corrs):.4f}"),
        "l2_error_q1": float(f"{np.percentile(errors, 25): .4f}"),
        "l2_error_q2": float(f"{np.median(errors): .4f}"),
        "l2_error_q3": float(f"{np.percentile(errors, 75): .4f}"),
        "r2_score_q1": float(f"{np.percentile(r2_scores, 25): .4f}"),
        "r2_score_q2": float(f"{np.median(r2_scores): .4f}"),
        "r2_score_q3": float(f"{np.percentile(r2_scores, 75): .4f}"),
    }

    return results



def spearmanrr(pred, target, **kw):
    """Differentiable rank correlation (requires torchsort when invoked)."""
    import torchsort
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return (pred * target).sum()
