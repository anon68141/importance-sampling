# ratio density estimation 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import ceil
from sklearn.metrics import roc_auc_score

class ImportanceSampler:
    """Importance sampling via random forest-based density ratio estimation."""

    def __init__(self, source, target, ignore_cols=None, model = 'rf'):
        self.source = source.copy()
        self.target = target.copy()
        self.ignore_cols = ignore_cols if ignore_cols is not None else []
        self.model = model
        self.weights_ = None
        self._clf = None

    def fit(self):
        # Drop ignored columns and missing values
        X_source = self.source.drop(columns=self.ignore_cols, errors='ignore').dropna()
        X_target = self.target.drop(columns=self.ignore_cols, errors='ignore').dropna()

        X_source["__label__"] = 0  # Source domain label
        X_target["__label__"] = 1  # Target domain label

        # Combine and shuffle
        df_combined = pd.concat([X_source, X_target], axis=0)
        df_combined = shuffle(df_combined, random_state=42)

        X = df_combined.drop(columns="__label__")
        y = df_combined["__label__"]

        # Fallback: if no features remain, assign uniform weights
        if X.shape[1] == 0:
            print("Warning: No features left after ignoring columns. Using uniform weights.")
            n_source = len(self.source)
            self.weights_ = np.ones(n_source) / n_source
            self._clf = None
            return

        if self.model == "rf":
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        elif self.model == "logreg":
            clf = LogisticRegression(
                class_weight="balanced",
                solver="liblinear"
            )
        else:
            raise ValueError("model must be 'rf' or 'logreg'")
        
        clf.fit(X, y)

        # Predict D(x) for source samples
        X_source_features = X_source.drop(columns="__label__")
        D = clf.predict_proba(X_source_features)[:, 1]

        # Compute importance weights
        eps = 1e-6
        weights = D / (1.0 - D + eps)
        self.weights_ = np.nan_to_num(weights / np.sum(weights), nan=0.0)

        # Store the classifier
        self._clf = clf

    def sample(self, n, attempts=10, replace=False, stratify=None, frac_y_1 = None, label_col = None):
        source = self.source.copy()
        source["w"] = self.weights_

        if frac_y_1 is not None:
            # If user doesn't specify label column, fall back to "y"
            col = label_col if label_col is not None else "y"

            if col not in source.columns:
                raise ValueError(
                    f"To adjust weights for target p(y=1), the source dataset must contain "
                    f"a label column '{col}'. Provide label_col='your_column_name'."
                )

            # Compute current weighted proportion
            current = (source.loc[source[col] == 1, "w"].sum()) / (source["w"].sum() + 1e-12)

            # Compute correction multipliers
            mult_pos = frac_y_1 / (current + 1e-12)
            mult_neg = (1 - frac_y_1) / (1 - current + 1e-12)

            # Apply corrections
            source.loc[source[col] == 1, "w"] *= mult_pos
            source.loc[source[col] == 0, "w"] *= mult_neg

            # Renormalize
            source["w"] /= source["w"].sum()

        if stratify is not None:
            n_levels = source[stratify].nunique()
            if n > n_levels:
                raise ValueError(f"Trying to sample {n} observations from {n_levels} categories!")

            source = source.groupby(stratify)

        best_subset = None
        best_score = -np.inf

        for i in range(attempts):
            if stratify is not None:
                subset = source.sample(n=1, weights='w', replace=replace, random_state=1).sample(n=n, weights='w', replace=replace, random_state=1)
            else:
                subset = source.sample(n=n, weights='w', replace=replace, random_state=1)

            # Use average predicted probability from the logistic model as score proxy
            if self._clf is not None:
                X_sub = subset.drop(columns=self.ignore_cols + ['w'], errors='ignore')
                score = self._clf.predict_proba(X_sub)[:, 1].mean()
            else:
                # No model available (no features). Use the mean weight as a score.
                score = subset["w"].mean()
            if score > best_score:
                best_score = score
                best_subset = subset
                print(f"Current best logistic target prob avg at iter {i+1}: {score:.3f}")

        return best_subset.drop(columns="w"), best_score, source

    def plot(self, subset, maxcols=4):
        subset = subset.drop(columns=self.ignore_cols, errors='ignore')
        nrows = int(ceil(subset.shape[1] / maxcols))
        ncols = min(subset.shape[1], maxcols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * ncols, 2 * nrows), constrained_layout=True)
        # Ensure axes is always a 1D array
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = np.array([axes])

        colors = {'source': 'tab:blue', 'target': 'tab:orange', 'subset': 'tab:blue'}
        ls = {'source': '-', 'target': '-', 'subset': '--'}

        for i, col in enumerate(subset.columns):
            ax = axes[i]
            for name, df in zip(['source', 'target', 'subset'], [self.source, self.target, subset]):
                if pd.api.types.is_numeric_dtype(df[col]):
                    sns.kdeplot(df[col], ax=ax, color=colors[name], linestyle=ls[name], fill=False, label=name)
                else:
                    value_counts = df[col].value_counts(normalize=True)
                    ax.bar(value_counts.index.astype(str), value_counts.values, alpha=0.5, label=name, color=colors[name])
            ax.set_title(col)

        for j in range(i + 1, nrows * ncols):
            axes[j].axis('off')

        ax.legend(frameon=False)
        sns.despine()
        return ax
