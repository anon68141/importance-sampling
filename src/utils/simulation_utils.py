from scipy.special import expit
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score


def generate_data(
    n=1000,
    p=3,
    beta_1=None,
    beta_2=4.0,
    intercept=0.0,
    prop_A=0.5,
    mu_A0=None,
    mu_A1=None,
    sigma=1.0,
    seed=0,
    target_p_y=None  
):
    """
    Generate synthetic data with covariate shift and optional label shift.

    Args:
        n: Number of samples to generate.
        p: Number of covariates (features).
        beta_1: Coefficients for covariates in the outcome model (defaults to ones).
        beta_2: Coefficient for the treatment effect in the outcome model.
        intercept: Intercept term in the logistic model for Y.
        prop_A: Probability of receiving treatment A = 1.
        mu_A0: Mean vector of covariates when A = 0 (defaults to zeros).
        mu_A1: Mean vector of covariates when A = 1 (defaults to ones).
        sigma: Standard deviation for covariate generation.
        seed: Random seed for reproducibility.
        target_p_y: Optional target proportion of Y = 1 to induce label shift.

    Returns:
        DataFrame with columns:
            - x1, x2, ..., xp: generated covariates
            - A: binary treatment indicator
            - Y: binary outcome indicator

    Raises:
        ValueError: If resampling fails because one of the outcome classes has zero samples.
    """

    np.random.seed(seed)

    if beta_1 is None:
        beta_1 = np.ones(p)
    beta_1 = np.array(beta_1)

    if mu_A0 is None:
        mu_A0 = np.zeros(p)
    if mu_A1 is None:
        mu_A1 = np.ones(p)

    # Sample A ~ Bernoulli(prop_A)
    A = np.random.binomial(1, prop_A, size=n)

    # Generate X | A (covariate shift)
    X = np.zeros((n, p))
    for i in range(n):
        mu = mu_A1 if A[i] == 1 else mu_A0
        X[i, :] = np.random.normal(loc=mu, scale=sigma)

    # Compute logits and probabilities for Y
    logits_Y = intercept + X @ beta_1 + A * beta_2 
    P_Y = expit(logits_Y)
    Y = np.random.binomial(1, P_Y)

    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'x{i+1}' for i in range(p)])
    df['A'] = A
    df['Y'] = Y

    # Resample to induce true label shift (optional)
    if target_p_y is not None:
        # Separate positive and negative cases
        df_1 = df[df['Y'] == 1]
        df_0 = df[df['Y'] == 0]

        # Edge case handling
        if len(df_1) == 0 or len(df_0) == 0:
            raise ValueError("Resampling failed: original Y=0 or Y=1 class has zero samples.")

        # Compute how many samples to draw from each class
        n1 = int(round(target_p_y * n))
        n0 = n - n1

        df = pd.concat([
            df_1.sample(n=n1, replace=True),
            df_0.sample(n=n0, replace=True)
        ]).sample(frac=1).reset_index(drop=True)  # Shuffle

    return df


def metric_diff_bootstrap(df_target, df_other, metric_fn, n_bootstrap=1000, weighted=False):
    diffs = []

    for _ in range(n_bootstrap):
        target_sample = resample(df_target)
        other_sample = resample(df_other)

        y_true_target = target_sample['Y']
        y_pred_target = target_sample['yhat']
        metric_target = metric_fn(y_true_target, y_pred_target)

        y_true_other = other_sample['Y']
        y_pred_other = other_sample['yhat']

        if weighted and 'w' in df_other.columns:
            weights = other_sample['w']
            metric_other = metric_fn(y_true_other, y_pred_other, sample_weight=weights)
        else:
            metric_other = metric_fn(y_true_other, y_pred_other)

        diffs.append(abs(metric_target - metric_other))

    diffs = np.array(diffs)
    return {
        'mean_abs_diff': diffs.mean(),
        'std': diffs.std(),
        'ci_2.5%': np.percentile(diffs, 2.5),
        'ci_97.5%': np.percentile(diffs, 97.5),
    }


def summarize_metric_differences(datasets, label_col="Y", pred_col="yhat", n_bootstrap=1000):
    """
    Compute metric differences (precision & recall) vs Target dataset.
    Returns DataFrame like: [set, metric, mean_abs_diff, ci_2.5%, ci_97.5%]
    """
    df_target = datasets["Target"]
    summary = []

    for name, df in datasets.items():
        if name == "Target":
            continue  # skip target vs itself

        weighted = ("w" in df.columns)

        for metric_name, metric_fn in zip(["precision", "recall"], [precision_score, recall_score]):
            stats = metric_diff_bootstrap(
                df_target=df_target,
                df_other=df,
                metric_fn=metric_fn,
                n_bootstrap=n_bootstrap,
                weighted=weighted
            )

            summary.append({
                "set": name,
                "metric": metric_name,
                **stats  # expands mean_abs_diff, std, ci_2.5%, ci_97.5%
            })

    return pd.DataFrame(summary)