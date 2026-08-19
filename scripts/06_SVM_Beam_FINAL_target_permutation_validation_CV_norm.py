import random
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parent
SUMMARY_FILE = (
    ROOT / "CSV" / "Input" / "Normalization"
    / "Final_Revised_Beam_RawFoldNorm_AllFeatures_All1000"
    / "Evaluation" / "Beam"
    / "Beam_final_summary.csv"
)
summary_df = pd.read_csv(SUMMARY_FILE)
summary_dict = dict(zip(summary_df["Metric"], summary_df["Value"]))
OBSERVED_MSE = float(summary_dict["MSE"])
OBSERVED_MAE = float(summary_dict["MAE"])
OBSERVED_R2 = float(summary_dict["R2"])
OBSERVED_R = float(summary_dict["Pearson r"])
OBSERVED_P = float(summary_dict["Pearson p"])

print("\nLoaded observed metrics from final Beam analysis:")
print(f"MSE:       {OBSERVED_MSE:.6f}")
print(f"MAE:       {OBSERVED_MAE:.6f}")
print(f"R²:        {OBSERVED_R2:.6f}")
print(f"Pearson r: {OBSERVED_R:.6f}")

TEST = "Beam"
BEHAVIOR_FILE = "Beamwalk24h.csv"
TARGET_COL = "Beam"
UNIT = "AU"
MSE_UNIT = "AU²"

N_PERMUTATIONS = 1000
N_REPEATS_PER_PERMUTATION = 2

N_OUTER_SPLITS = 5
N_INNER_SPLITS = 5
RANDOM_SEED = 123

total_cores = multiprocessing.cpu_count()
N_JOBS = max(1, int(total_cores * 0.90))

feature_path = (
    ROOT / "CSV" / "Input" / "Normalization"
    / "Median_raw_foldnorm_Beam_Grip_matched324.csv"
)

behavior_path = ROOT / "CSV" / "Output" / BEHAVIOR_FILE

OUT = (
    ROOT / "CSV" / "Input" / "Normalization"
    / "Final_Revised_Beam_RawFoldNorm_AllFeatures_All1000"
    / "Evaluation" / TEST
    / "Permutation_Test"
)
OUT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# PLOT STYLE
# =============================================================================

matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42

TITLE_FONT_SIZE = 22
AXIS_LABEL_FONT_SIZE = 22
TICK_LABEL_FONT_SIZE = 18
LEGEND_FONT_SIZE = 16
PANEL_FONT_SIZE = 28
AXIS_LINEWIDTH = 3.5
TICK_WIDTH = 2.5
TICK_LENGTH = 7


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(AXIS_LINEWIDTH)
    ax.spines["left"].set_linewidth(AXIS_LINEWIDTH)
    ax.tick_params(
        axis="both",
        which="major",
        width=TICK_WIDTH,
        length=TICK_LENGTH,
        labelsize=TICK_LABEL_FONT_SIZE,
    )
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(False)


# =============================================================================
# MODALITY-WISE FOLD NORMALIZATION
# =============================================================================

class ModalityWiseScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.modality_stats_ = {}

    def _get_modality(self, col):
        col = str(col)

        if col.endswith("_ADC"):
            return "ADC"
        if col.endswith("_T2") or col.endswith("_T2w"):
            return "T2"
        if col.endswith("_NormFA") or col.endswith("_FA"):
            return "FA"
        if col.endswith("_NormAD") or col.endswith("_AD"):
            return "AD"
        if col.endswith("_NormRD") or col.endswith("_RD"):
            return "RD"
        if col.endswith("_NormTrace") or col.endswith("_Trace") or col.endswith("_trace"):
            return "Trace"

        return "Other"

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()

        if hasattr(X, "columns"):
            X_df.columns = X.columns

        self.columns_ = X_df.columns.tolist()
        self.modality_stats_ = {}

        for modality in ["ADC", "T2", "FA", "AD", "RD", "Trace"]:
            cols = [c for c in self.columns_ if self._get_modality(c) == modality]

            if len(cols) == 0:
                continue

            values = X_df[cols].values.astype(float).ravel()

            mean = np.nanmean(values)
            std = np.nanstd(values)

            if std == 0 or np.isnan(std):
                std = 1.0

            self.modality_stats_[modality] = {
                "columns": cols,
                "mean": mean,
                "std": std,
            }

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()

        if hasattr(X, "columns"):
            X_df.columns = X.columns
        else:
            X_df.columns = self.columns_

        X_scaled = X_df.copy()

        for modality, stats in self.modality_stats_.items():
            cols = stats["columns"]
            mean = stats["mean"]
            std = stats["std"]

            X_scaled[cols] = (X_scaled[cols] - mean) / std

        return X_scaled


# =============================================================================
# MODEL SETUP
# =============================================================================

pipeline = Pipeline([
    ("scaler", ModalityWiseScaler()),
    ("svr", SVR())
])

param_grid = {
    "svr__kernel": ["linear", "rbf"],
    "svr__C": [0.1, 1, 10],
    "svr__gamma": ["scale", "auto"],
    "svr__epsilon": [0.01, 0.1],
}


def safe_pearson(y_true, y_pred):
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        r, p = pearsonr(y_true, y_pred)
        return float(r), float(p)
    return np.nan, np.nan


# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 80)
print("FINAL BEAM TARGET-PERMUTATION VALIDATION")
print("=" * 80)
print("Feature file:", feature_path)
print("Behavior file:", behavior_path)
print("N_PERMUTATIONS:", N_PERMUTATIONS)
print("N_REPEATS_PER_PERMUTATION:", N_REPEATS_PER_PERMUTATION)
print("n_jobs:", N_JOBS)

df_features = pd.read_csv(feature_path)
df_behavior = pd.read_csv(behavior_path)

feature_ids = (
    df_features["AnimalID"]
    .astype(str)
    .str.strip()
    .str.replace("FRat", "Rat", regex=False)
    .str.replace("Frat", "Rat", regex=False)
    .str.replace("frat", "Rat", regex=False)
)

behavior_ids = (
    df_behavior.iloc[:, 0]
    .astype(str)
    .str.strip()
    .str.replace("FRat", "Rat", regex=False)
    .str.replace("Frat", "Rat", regex=False)
    .str.replace("frat", "Rat", regex=False)
)

y_values = df_behavior.iloc[:, 1].astype(float).values

df_features["AnimalID"] = feature_ids

behavior_order = pd.DataFrame({
    "AnimalID": behavior_ids,
    TARGET_COL: y_values
})

merged = behavior_order.merge(
    df_features,
    on="AnimalID",
    how="left"
)

if merged.isna().any().any():
    missing_animals = merged.loc[
        merged.isna().any(axis=1),
        "AnimalID"
    ].tolist()
    print("Missing animals after merge:")
    print(missing_animals)
    raise RuntimeError("Feature/behavior alignment failed.")

sample_ids = merged["AnimalID"].values
y = merged[TARGET_COL].values.astype(float)
X_df = merged.drop(columns=["AnimalID", TARGET_COL])

print("Samples:", X_df.shape[0])
print("Features:", X_df.shape[1])
print(f"{TEST} range: {float(np.min(y))} to {float(np.max(y))}")


# =============================================================================
# REPEATED NESTED CV FUNCTION
# =============================================================================

def run_repeated_nested_cv(permute_training_labels=False, base_seed=RANDOM_SEED):
    """
    Observed model:
        permute_training_labels = False

    Permutation model:
        permute_training_labels = True

    In permutation mode:
        labels are shuffled only within the outer-training fold.
        Held-out test labels remain real.
    """

    n_samples = len(y)
    repeat_predictions = np.full((N_REPEATS_PER_PERMUTATION, n_samples), np.nan)

    for rep in range(N_REPEATS_PER_PERMUTATION):

        current_seed = int(base_seed + rep)

        outer_cv = KFold(
            n_splits=N_OUTER_SPLITS,
            shuffle=True,
            random_state=current_seed
        )

        y_pred_rep = np.full(n_samples, np.nan)

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_df), start=1):

            X_train = X_df.iloc[train_idx]
            X_test = X_df.iloc[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            if permute_training_labels:
                rng = np.random.default_rng(base_seed + 10000 * rep + fold_idx)
                y_train_used = rng.permutation(y_train)
            else:
                y_train_used = y_train

            inner_cv = KFold(
                n_splits=N_INNER_SPLITS,
                shuffle=True,
                random_state=current_seed + fold_idx
            )

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring="neg_mean_squared_error",
                cv=inner_cv,
                n_jobs=N_JOBS,
                refit=True,
                pre_dispatch="2*n_jobs"
            )

            grid.fit(X_train, y_train_used)

            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)

            y_pred_rep[test_idx] = y_pred

        if np.isnan(y_pred_rep).any():
            raise RuntimeError(f"Missing predictions in repeat {rep + 1}")

        repeat_predictions[rep, :] = y_pred_rep

        if (rep + 1) == 1 or (rep + 1) % 25 == 0:
            print(
                f"  repeat {rep + 1}/{N_REPEATS_PER_PERMUTATION} "
                f"({'permuted' if permute_training_labels else 'observed'})",
                flush=True
            )

    final_pred = np.median(repeat_predictions, axis=0)

    mse = mean_squared_error(y, final_pred)
    mae = mean_absolute_error(y, final_pred)
    r2 = r2_score(y, final_pred)
    r, p = safe_pearson(y, final_pred)

    return {
        "true": y.copy(),
        "pred": final_pred,
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "pearson_r": r,
        "pearson_parametric_p": p
    }

# =============================================================================
# TARGET PERMUTATIONS
# =============================================================================

print("\nRunning target permutations...")
rng_global = np.random.default_rng(RANDOM_SEED)
perm_rows = []

for perm_idx in range(1, N_PERMUTATIONS + 1):

    perm_seed = int(rng_global.integers(0, 2**31 - 1))

    result = run_repeated_nested_cv(
        permute_training_labels=True,
        base_seed=perm_seed
    )

    perm_rows.append({
        "permutation": perm_idx,
        "seed": perm_seed,
        "mse": result["mse"],
        "mae": result["mae"],
        "r2": result["r2"],
        "pearson_r": result["pearson_r"],
        "pearson_parametric_p": result["pearson_parametric_p"],
    })

    if perm_idx % 10 == 0:
        pd.DataFrame(perm_rows).to_csv(
            OUT / f"{TEST}_target_permutation_results_partial.csv",
            index=False
        )

    if perm_idx == 1 or perm_idx % 25 == 0:
        print(
            f"Permutation {perm_idx}/{N_PERMUTATIONS}: "
            f"MSE={result['mse']:.4f}, "
            f"R²={result['r2']:.4f}, "
            f"r={result['pearson_r']:.4f}",
            flush=True
        )

perm_df = pd.DataFrame(perm_rows)
perm_df.to_csv(OUT / f"{TEST}_target_permutation_results.csv", index=False)


# =============================================================================
# EMPIRICAL P-VALUES
# =============================================================================

p_mse = (1 + np.sum(perm_df["mse"].values <= OBSERVED_MSE)) / (N_PERMUTATIONS + 1)
p_r2 = (1 + np.sum(perm_df["r2"].values >= OBSERVED_R2)) / (N_PERMUTATIONS + 1)
p_r = (1 + np.sum(perm_df["pearson_r"].values >= OBSERVED_R)) / (N_PERMUTATIONS + 1)

summary = pd.DataFrame([{
    "observed_mse": OBSERVED_MSE,
    "observed_mae": OBSERVED_MAE,
    "observed_r2": OBSERVED_R2,
    "observed_pearson_r": OBSERVED_R,
    "observed_parametric_pearson_p": OBSERVED_P,
    "n_permutations": N_PERMUTATIONS,
    "n_repeats_per_permutation": N_REPEATS_PER_PERMUTATION,
    "n_outer_splits": N_OUTER_SPLITS,
    "n_inner_splits": N_INNER_SPLITS,
    "empirical_p_mse": p_mse,
    "empirical_p_r2": p_r2,
    "empirical_p_pearson_r": p_r,
    "median_permuted_mse": float(np.median(perm_df["mse"])),
    "median_permuted_r2": float(np.median(perm_df["r2"])),
    "median_permuted_pearson_r": float(np.median(perm_df["pearson_r"])),
    "min_permuted_mse": float(np.min(perm_df["mse"])),
    "max_permuted_r2": float(np.max(perm_df["r2"])),
    "max_permuted_pearson_r": float(np.max(perm_df["pearson_r"])),
}])

summary.to_csv(OUT / f"{TEST}_target_permutation_summary.csv", index=False)

print("\n" + "=" * 80)
print("TARGET PERMUTATION SUMMARY")
print("=" * 80)
print(f"Observed MSE:       {OBSERVED_MSE:.6f}")
print(f"Observed MAE:       {OBSERVED_MAE:.6f}")
print(f"Observed R²:        {OBSERVED_R2:.6f}")
print(f"Observed Pearson r: {OBSERVED_R:.6f}")
print(f"Observed Pearson p: {OBSERVED_P:.6f}")
print(f"Empirical p(MSE):   {p_mse:.6g}")
print(f"Empirical p(R²):    {p_r2:.6g}")
print(f"Empirical p(r):     {p_r:.6g}")


# =============================================================================
# PLOTS
# =============================================================================

fig, axes = plt.subplots(2, 1, figsize=(10, 8.5))

# Panel A: MSE
ax = axes[0]
ax.hist(perm_df["mse"].values, bins=60, edgecolor="black", linewidth=1.0)

ax.axvline(
    OBSERVED_MSE,
    linestyle="--",
    linewidth=3,
    label=f"Observed MSE = {OBSERVED_MSE:.2f}"
)

clean_axes(ax)

ax.set_xlabel(f"MSE [{MSE_UNIT}]", fontweight="bold", fontsize=AXIS_LABEL_FONT_SIZE)
ax.set_ylabel("Frequency", fontweight="bold", fontsize=AXIS_LABEL_FONT_SIZE)
ax.set_title(
    f"{TEST} Walk Permutation Test: MSE null distribution",
    fontweight="bold",
    fontsize=TITLE_FONT_SIZE,
    pad=10
)
ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)

ax.text(
    -0.06, 1.10, "A",
    transform=ax.transAxes,
    fontsize=PANEL_FONT_SIZE,
    fontweight="bold",
    va="top",
    ha="left"
)

# Panel B: Pearson r
ax = axes[1]
ax.hist(perm_df["pearson_r"].values, bins=60, edgecolor="black", linewidth=1.0)

ax.axvline(
    OBSERVED_R,
    linestyle="--",
    linewidth=3,
    label=f"Observed r = {OBSERVED_R:.2f}"
)

clean_axes(ax)

ax.set_xlabel("Pearson r", fontweight="bold", fontsize=AXIS_LABEL_FONT_SIZE)
ax.set_ylabel("Frequency", fontweight="bold", fontsize=AXIS_LABEL_FONT_SIZE)
ax.set_title(
    f"{TEST} Walk Permutation Test: Pearson r null distribution",
    fontweight="bold",
    fontsize=TITLE_FONT_SIZE,
    pad=10
)
ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)

ax.text(
    -0.06, 1.10, "B",
    transform=ax.transAxes,
    fontsize=PANEL_FONT_SIZE,
    fontweight="bold",
    va="top",
    ha="left"
)

plt.tight_layout(pad=1.8)

fig.savefig(
    OUT / f"{TEST}_target_permutation_combined_null_distributions.pdf",
    bbox_inches="tight",
    pad_inches=0.2
)

fig.savefig(
    OUT / f"{TEST}_target_permutation_combined_null_distributions.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.2
)

plt.show()

print("\nDone. Outputs saved to:")
print(OUT)