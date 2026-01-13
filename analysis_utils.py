"""
Utility functions for analyzing embedding datasets across tasks.

Includes:
- Bootstrapped confidence intervals for metrics
- Optional PCA dimensionality reduction
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.metrics import precision_score, recall_score, mean_absolute_error


def bootstrap_ci(
    metric_func,
    y_true,
    y_pred,
    sample_weight=None,
    n_bootstrap=1000,
    alpha=0.05,
    seed=0,
    average = 'binary',
    zero_division = np.nan,
):
    """
    Compute bootstrap confidence interval for a metric function.

    Args:
        metric_func: callable, e.g. sklearn.metrics.precision_score
        y_true, y_pred: arrays
        sample_weight: optional, array-like
        n_bootstrap: number of bootstrap samples
        alpha: significance level
        seed: random seed

    Returns:
        mean, (lower, upper)
    """
    np.random.seed(seed)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if sample_weight is not None:
        sample_weight = np.array(sample_weight)

    scores = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        weights = sample_weight[indices] if sample_weight is not None else None
        score = metric_func(
            y_true[indices], y_pred[indices], sample_weight=weights, average = average,
            zero_division = zero_division
        )
        scores.append(score)

    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    return np.mean(scores), (lower, upper)


def compute_bootstrap_metrics(df, label_col="label", pred_col="predicted_label", sample_weight=None, n_bootstrap=1000,
                              average = 'binary'):
    """
    Compute precision and recall with bootstrap confidence intervals.

    Args:
        df: DataFrame containing true and predicted labels.
        label_col: Name of true label column.
        pred_col: Name of predicted label column.
        sample_weight: Optional weights array.

    Returns:
        Dict with mean and (lower, upper) for both precision and recall.
    """
    precision = bootstrap_ci(precision_score, df[label_col], df[pred_col],sample_weight=sample_weight, n_bootstrap=n_bootstrap, 
                             average = average)
    recall = bootstrap_ci(recall_score, df[label_col], df[pred_col], sample_weight=sample_weight, n_bootstrap=n_bootstrap,
                          average = average)
    return {"precision": precision, "recall": recall}


def summarize_all_metrics(datasets,
                          label_col="label",
                          pred_col="predicted_label",
                          n_bootstrap=1000,
                          average = 'binary'):
    """
    Compute precision and recall with bootstrap confidence intervals across multiple datasets.

    Args:
        datasets: Dict of dataset name → DataFrame.
        label_col: Name of true label column.
        pred_col: Name of predicted label column.
        n_bootstrap: Number of bootstrap samples.

    Returns:
        DataFrame with precision and recall metrics, including mean and confidence bounds.
    """
    summary = []

    for name, df in datasets.items():
        weight_col = "w" if "w" in df.columns else None

        metrics = compute_bootstrap_metrics(
            df,
            label_col=label_col,
            pred_col=pred_col,
            sample_weight=df[weight_col] if weight_col else None,
            n_bootstrap=n_bootstrap,
            average = average,
        )

        for metric_name, (mean, ci) in metrics.items():
            summary.append({
                "set": name,
                "metric": metric_name,
                "mean": mean,
                "lower": ci[0],
                "upper": ci[1],
            })

    return pd.DataFrame(summary)


def bootstrap_delta_metrics(df, group_col, group_a, group_b,
                            label_col="label", pred_col="predicted_label",
                            sample_weight=None, n_bootstrap=1000, alpha=0.05, seed=0,
                            average = 'binary'):
    """
    Compute bootstrapped Δprecision and Δrecall between two groups.

    Args:
        df: DataFrame containing true and predicted labels.
        group_col: Column defining groups to compare (e.g. gender, language).
        group_a, group_b: Group values to compare (Δ = A − B).
        label_col, pred_col: Column names for labels and predictions.
        sample_weight: Optional weight column.
        n_bootstrap: Number of bootstrap samples.
        alpha: Significance level (e.g. 0.05 for 95% CI).
        seed: Random seed.

    Returns:
        Dict with Δprecision and Δrecall and their confidence intervals.
    """
    np.random.seed(seed)
    a = df[df[group_col] == group_a]
    b = df[df[group_col] == group_b]

    deltas = []
    for _ in range(n_bootstrap):
        aa = a.sample(frac=1, replace=True)
        bb = b.sample(frac=1, replace=True)
        w_a = aa[sample_weight] if sample_weight is not None and sample_weight in aa else None
        w_b = bb[sample_weight] if sample_weight is not None and sample_weight in bb else None

        pa = precision_score(aa[label_col], aa[pred_col], average= average, sample_weight=w_a, zero_division=np.nan)
        ra = recall_score(aa[label_col], aa[pred_col], average=average, sample_weight=w_a, zero_division=np.nan)
        pb = precision_score(bb[label_col], bb[pred_col], average=average, sample_weight=w_b, zero_division=np.nan)
        rb = recall_score(bb[label_col], bb[pred_col], average=average, sample_weight=w_b, zero_division=np.nan)

        deltas.append((pa - pb, ra - rb))

    arr = np.array(deltas)
    mean = arr.mean(axis=0)
    low, high = np.percentile(arr, [100*alpha/2, 100*(1-alpha/2)], axis=0)

    return {
        "Δprecision": mean[0], "Δprecision_low": low[0], "Δprecision_high": high[0],
        "Δrecall": mean[1], "Δrecall_low": low[1], "Δrecall_high": high[1],
    }


def summarize_all_deltas(datasets, group_col, group_a, group_b,
                         label_col="label", pred_col="predicted_label",
                         n_bootstrap=1000, average = 'binary'):
    """Compute Δprecision and Δrecall across multiple datasets (dict or list)."""
    summary = []
    for name, df in datasets.items():
        weight_col = "w" if "w" in df.columns else None

        delta = bootstrap_delta_metrics(
            df, group_col, group_a, group_b,
            label_col, pred_col,
            sample_weight=weight_col,
            n_bootstrap=n_bootstrap,
            average = average
        )

        for metric_name in ["Δprecision", "Δrecall"]:
            summary.append({
                "set": name,
                "metric": metric_name,
                "mean": delta[metric_name],
                "lower": delta[f"{metric_name}_low"],
                "upper": delta[f"{metric_name}_high"],
            })

    return pd.DataFrame(summary)


def safe_mean_absolute_error(y_true, y_pred, sample_weight=None, average = None, zero_division = None):
    """
    Wrapper around sklearn's mean_absolute_error.
    Returns 0 if inputs are empty, otherwise computes MAE normally.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    return mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)


def compute_bootstrap_mae(
    y_true,
    y_pred,
    sample_weight=None,
    n_bootstrap=1000,
    alpha=0.05,
):
    """
    Compute Mean Absolute Error (MAE) with bootstrap confidence interval.

    Args:
        y_true: Array of true target values.
        y_pred: Array of predicted values.
        sample_weight: Optional array of sample weights.
        n_bootstrap: Number of bootstrap samples.
        alpha: Significance level (e.g. 0.05 for 95% CI).

    Returns:
        mean, (lower, upper)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if sample_weight is not None:
        sample_weight = np.array(sample_weight)

    mae_mean, mae_ci = bootstrap_ci(
        safe_mean_absolute_error,
        y_true,
        y_pred,
        sample_weight=sample_weight,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        average = None, 
        zero_division= None
    )
    return mae_mean, mae_ci


def summarize_all_mae(datasets,
                      label_col="y_true",
                      pred_col="y_pred",
                      n_bootstrap=1000):
    """
    Compute MAE with bootstrap confidence intervals across multiple datasets.

    Args:
        datasets: Dict of dataset name → DataFrame.
        label_col: Column name for true target values.
        pred_col: Column name for predicted values.
        n_bootstrap: Number of bootstrap samples.

    Returns:
        DataFrame with MAE for each dataset, including mean and confidence bounds.
    """
    summary = []
    for name, df in datasets.items():
        weight_col = "w" if "w" in df.columns else None

        mae_mean, mae_ci = compute_bootstrap_mae(
            df[label_col],
            df[pred_col],
            sample_weight=df[weight_col] if weight_col else None,
            n_bootstrap=n_bootstrap,
        )

        summary.append({
            "set": name,
            "metric": "MAE",
            "mean": mae_mean,
            "lower": mae_ci[0],
            "upper": mae_ci[1],
        })

    return pd.DataFrame(summary)


def bootstrap_delta_mae(df, group_col, group_a, group_b,
                        label_col="y_true", pred_col="y_pred",
                        sample_weight=None, n_bootstrap=1000, alpha=0.05, seed=0):
    """
    Compute bootstrapped ΔMAE (difference in Mean Absolute Error) between two groups.

    Args:
        df: DataFrame containing true and predicted values.
        group_col: Column defining groups to compare (e.g. gender, dataset split).
        group_a, group_b: Group values to compare (Δ = A − B).
        label_col, pred_col: Column names for true and predicted values.
        sample_weight: Optional name of weight column.
        n_bootstrap: Number of bootstrap samples.
        alpha: Significance level (e.g. 0.05 for 95% CI).
        seed: Random seed.

    Returns:
        Dict with ΔMAE and its confidence interval bounds.
    """
    np.random.seed(seed)
    a = df[df[group_col] == group_a]
    b = df[df[group_col] == group_b]

    deltas = []
    for _ in range(n_bootstrap):
        aa = a.sample(frac=1, replace=True)
        bb = b.sample(frac=1, replace=True)

        w_a = aa[sample_weight] if sample_weight is not None and sample_weight in aa else None
        w_b = bb[sample_weight] if sample_weight is not None and sample_weight in bb else None

        mae_a = safe_mean_absolute_error(aa[label_col], aa[pred_col], sample_weight=w_a)
        mae_b = safe_mean_absolute_error(bb[label_col], bb[pred_col], sample_weight=w_b)
        deltas.append(mae_a - mae_b)

    arr = np.array(deltas)
    mean = arr.mean()
    low, high = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "ΔMAE": mean,
        "ΔMAE_low": low,
        "ΔMAE_high": high,
    }


def summarize_all_delta_mae(datasets, group_col, group_a, group_b,
                            label_col="y_true", pred_col="y_pred",
                            n_bootstrap=1000):
    """Compute ΔMAE across multiple datasets (dict or list)."""
    summary = []
    for name, df in datasets.items():
        weight_col = "w" if "w" in df.columns else None

        delta = bootstrap_delta_mae(
            df, group_col, group_a, group_b,
            label_col, pred_col,
            sample_weight=weight_col,
            n_bootstrap=n_bootstrap,
        )

        summary.append({
            "set": name,
            "metric": "ΔMAE",
            "mean": delta["ΔMAE"],
            "lower": delta["ΔMAE_low"],
            "upper": delta["ΔMAE_high"],
        })

    return pd.DataFrame(summary)


def prepare_embeddings(df, embedding_col="cls_embedding", use_pca=True, n_components=5):
    """
    Convert embedding column to separate columns, with optional PCA reduction.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the embedding column.
    embedding_col : str
        Column name with embedding vectors (e.g. 'cls_embedding' or 'mean_embedding').
    use_pca : bool
        Whether to apply PCA to reduce dimensionality.
    n_components : int
        Number of PCA components to keep if use_pca=True.

    Returns
    -------
    df : pd.DataFrame
        Updated DataFrame with embedding columns (either reduced or full).
    """

    # Filter out rows with empty embeddings
    df = df[df[embedding_col].apply(lambda x: isinstance(x, np.ndarray) and x.size > 0)].reset_index(drop=True)

    # Stack into matrix
    emb_matrix = np.vstack(df[embedding_col].values)

    if use_pca:
        from sklearn.decomposition import PCA

        print(f"Applying PCA to reduce from {emb_matrix.shape[1]} to {n_components} dimensions...")
        pca = PCA(n_components=n_components, random_state=42)
        emb_matrix = pca.fit_transform(emb_matrix)
        col_prefix = "pca_"
        print(f"Total variance explained by {n_components} components: {pca.explained_variance_ratio_.sum():.2%}")
    else:
        print(f"Using full embedding of size {emb_matrix.shape[1]}...")
        col_prefix = "embedding_"

    # Create DataFrame for embeddings
    embedding_df = pd.DataFrame(emb_matrix, columns=[f"{col_prefix}{i}" for i in range(emb_matrix.shape[1])])

    # Merge with other columns
    df = pd.concat([df, embedding_df], axis=1)
    return df

