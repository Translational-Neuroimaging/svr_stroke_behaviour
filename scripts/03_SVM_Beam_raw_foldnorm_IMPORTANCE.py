import ast
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
from sklearn.linear_model import LinearRegression


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parent

TEST = "Beam"  # Beam only for now

N_REPEATS = 100
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 5
RANDOM_SEED = 123

COMPUTE_FEATURE_IMPORTANCE = True
FEATURE_IMPORTANCE_MAX_MODELS = 100
N_IMPORTANCE_REPEATS = 3

OUT_DIR = (
    ROOT / "CSV" / "Input" / "Normalization"
    / "Final_Revised_Beam_RawFoldNorm_FeatureImportance"
    / "Evaluation" / TEST
)

# Use 90% of available CPU cores
total_cores = multiprocessing.cpu_count()
N_JOBS = max(1, int(total_cores * 0.9))


OUT_DIR.mkdir(parents=True, exist_ok=True)


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
AXIS_LINEWIDTH = 3.5
TICK_WIDTH = 2.5
TICK_LENGTH = 7


def clean_axes(ax):
    """Remove top/right box, thicken remaining axes."""
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
# HELPER FUNCTIONS
# =============================================================================

def find_file(filename: str) -> Path:
    """Find file recursively from ROOT."""
    direct = ROOT / filename
    if direct.exists():
        return direct

    matches = list(ROOT.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} inside {ROOT}")
    return matches[0]


def read_behavior_file(path: Path):
    """Read behavior CSV and return IDs if present plus target vector."""
    df_y = pd.read_csv(path)

    # Assume first column is animal ID and second column is behavior value
    id_col = df_y.columns[0]
    val_col = df_y.columns[1]

    ids = df_y[id_col].astype(str).values
    y = df_y[val_col].values.astype(float)

    return ids, y, df_y


def align_features_to_behavior(feature_df, animal_ids, y_ids, y):
    """Align feature rows to behavior file if possible."""
    animal_ids = pd.Series(animal_ids).astype(str)
    y_ids = pd.Series(y_ids).astype(str)

    if set(y_ids).issubset(set(animal_ids)):
        tmp = feature_df.copy()
        tmp["__animal_id__"] = animal_ids.values

        y_df = pd.DataFrame({
            "__animal_id__": y_ids.values,
            "__target__": y
        })

        merged = y_df.merge(tmp, on="__animal_id__", how="left")

        if merged.isna().any().any():
            raise ValueError("Alignment by animal ID produced missing values.")

        y_aligned = merged["__target__"].values.astype(float)
        X_aligned = merged.drop(columns=["__animal_id__", "__target__"])

        print("Feature rows aligned to behavior file using animal IDs.")
        return X_aligned, y_aligned, y_ids.values

    else:
        print("WARNING: Could not align by animal IDs. Using current row order.")
        if len(feature_df) != len(y):
            raise ValueError("Feature table and behavior file have different lengths.")
        return feature_df.copy(), y.astype(float), animal_ids.values


# =============================================================================
# LOAD MATCHED RAW FEATURE TABLE
# =============================================================================

feature_path = (
    ROOT / "CSV" / "Input" / "Normalization"
    / "Median_raw_foldnorm_Beam_Grip_matched324.csv"
)

behavior_path = ROOT / "CSV" / "Output" / "Beamwalk24h.csv"

print("Feature file:", feature_path)
print("Behavior file:", behavior_path)

df_features = pd.read_csv(feature_path)
df_behavior = pd.read_csv(behavior_path)

# Animal IDs
feature_ids = df_features["AnimalID"].astype(str).str.strip()
behavior_ids = df_behavior.iloc[:, 0].astype(str).str.strip()

# Behavioral target = second column
y_values = df_behavior.iloc[:, 1].astype(float).values

# Align feature table to behavior file order
df_features["AnimalID"] = feature_ids

behavior_order = pd.DataFrame({
    "AnimalID": behavior_ids,
    "Beam": y_values
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

aligned_ids = merged["AnimalID"].values
y = merged["Beam"].values.astype(float)

# Drop metadata columns
X_df = merged.drop(columns=["AnimalID", "Beam"])

print("Final aligned X shape:", X_df.shape)
print("Final y shape:", y.shape)
print(f"Beam range: {np.min(y)} to {np.max(y)}")

feature_names = X_df.columns.tolist()

##

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
                "std": std
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
    "svr__C": [0.1, 1, 10],#, 100],
    "svr__gamma": ["scale", "auto"],#, 0.01, 0.1, 1],
    "svr__epsilon": [0.01, 0.1],#, 0.5],
}

rng_global = np.random.RandomState(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =============================================================================
# STORAGE
# =============================================================================

n_samples = len(y)

repeat_predictions = np.full((N_REPEATS, n_samples), np.nan)
repeat_mse = []
repeat_mae = []
repeat_r2 = []
repeat_r = []

model_rows = []
all_outer_mse = []
all_outer_mae = []
all_outer_r2 = []

importance_sum = np.zeros(len(feature_names), dtype=float)
importance_count = np.zeros(len(feature_names), dtype=float)
importance_model_counter = 0


# =============================================================================
# REPEATED NESTED CROSS-VALIDATION
# =============================================================================

print("\n" + "=" * 80)
print("RUNNING REPEATED NESTED CV WITH FOLD-WISE NORMALIZATION")
print("=" * 80)
print(f"N_REPEATS: {N_REPEATS}")
print(f"Outer folds: {N_OUTER_SPLITS}")
print(f"Inner folds: {N_INNER_SPLITS}")
print(f"Features: {len(feature_names)}")
print(f"n_jobs: {N_JOBS}")
print("=" * 80 + "\n")

for rep in range(N_REPEATS):

    current_seed = RANDOM_SEED + rep

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

        print(f"Repeat {rep + 1}/{N_REPEATS}, fold {fold_idx}/{N_OUTER_SPLITS}", flush=True)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)
        y_pred_rep[test_idx] = y_pred

        mse_test = mean_squared_error(y_test, y_pred)
        mae_test = mean_absolute_error(y_test, y_pred)
        r2_test = r2_score(y_test, y_pred)

        all_outer_mse.append(mse_test)
        all_outer_mae.append(mae_test)
        all_outer_r2.append(r2_test)

        model_rows.append({
            "repeat": rep + 1,
            "seed": current_seed,
            "fold": fold_idx,
            "mse_outer_test": float(mse_test),
            "mae_outer_test": float(mae_test),
            "r2_outer_test": float(r2_test),
            "best_params": str(grid.best_params_),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx))
        })

        # ---------------------------------------------------------------------
        # Permutation feature importance on held-out fold
        # ---------------------------------------------------------------------
        do_importance = COMPUTE_FEATURE_IMPORTANCE and (
            FEATURE_IMPORTANCE_MAX_MODELS is None
            or importance_model_counter < FEATURE_IMPORTANCE_MAX_MODELS
        )

        if do_importance:
            baseline_mse = mse_test
            rng_imp = np.random.RandomState(current_seed + fold_idx + 999)

            X_test_np = X_test.copy()

            for j, feature in enumerate(feature_names):
                inc_values = []

                for _ in range(N_IMPORTANCE_REPEATS):
                    X_perm = X_test_np.copy()
                    permuted_values = X_perm.iloc[:, j].values.copy()
                    rng_imp.shuffle(permuted_values)
                    X_perm.iloc[:, j] = permuted_values

                    y_perm_pred = best_model.predict(X_perm)
                    perm_mse = mean_squared_error(y_test, y_perm_pred)
                    inc_values.append(perm_mse - baseline_mse)

                importance_sum[j] += np.mean(inc_values)
                importance_count[j] += 1

            importance_model_counter += 1

    # Repeat-level performance
    if np.isnan(y_pred_rep).any():
        raise RuntimeError(f"Missing predictions in repeat {rep + 1}")

    repeat_predictions[rep, :] = y_pred_rep

    mse_rep = mean_squared_error(y, y_pred_rep)
    mae_rep = mean_absolute_error(y, y_pred_rep)
    r2_rep = r2_score(y, y_pred_rep)
    r_rep, _ = pearsonr(y, y_pred_rep)

    repeat_mse.append(mse_rep)
    repeat_mae.append(mae_rep)
    repeat_r2.append(r2_rep)
    repeat_r.append(r_rep)

    if (rep + 1) == 1 or (rep + 1) % 25 == 0:
        print(
            f"Repeat {rep + 1}/{N_REPEATS}: "
            f"MSE={mse_rep:.4f}, MAE={mae_rep:.4f}, "
            f"R²={r2_rep:.4f}, r={r_rep:.4f}"
        )


# =============================================================================
# FINAL ENSEMBLE PREDICTION
# =============================================================================

final_pred = np.median(repeat_predictions, axis=0)

overall_mse = mean_squared_error(y, final_pred)
overall_mae = mean_absolute_error(y, final_pred)
overall_r2 = r2_score(y, final_pred)
overall_r, overall_p = pearsonr(y, final_pred)

print("\n" + "=" * 80)
print("FINAL RAW-FEATURE FOLD-NORMALIZED BEAM SUMMARY")
print("=" * 80)
print(f"Overall MSE:       {overall_mse:.6f}")
print(f"Overall MAE:       {overall_mae:.6f}")
print(f"Overall R²:        {overall_r2:.6f}")
print(f"Overall Pearson r: {overall_r:.6f}")
print(f"Pearson p-value:   {overall_p:.6g}")
print("=" * 80)


# =============================================================================
# SAVE TABLES
# =============================================================================

summary_df = pd.DataFrame({
    "Metric": [
        "MSE",
        "MAE",
        "R2",
        "Pearson r",
        "Pearson p",
        "N repeats",
        "N outer folds",
        "N inner folds",
        "N samples",
        "N features",
        "Feature importance models used"
    ],
    "Value": [
        overall_mse,
        overall_mae,
        overall_r2,
        overall_r,
        overall_p,
        N_REPEATS,
        N_OUTER_SPLITS,
        N_INNER_SPLITS,
        n_samples,
        len(feature_names),
        importance_model_counter
    ]
})

summary_df.to_csv(OUT_DIR / "Beam_final_summary.csv", index=False)

pred_df = pd.DataFrame({
    "AnimalID": aligned_ids,
    "True_Beam": y,
    "Predicted_Beam": final_pred
})
pred_df.to_csv(OUT_DIR / "Beam_final_true_vs_predicted.csv", index=False)

model_metrics_df = pd.DataFrame(model_rows)
model_metrics_df.to_csv(OUT_DIR / "Beam_final_all_model_metrics.csv", index=False)

repeat_metrics_df = pd.DataFrame({
    "repeat": np.arange(1, N_REPEATS + 1),
    "mse": repeat_mse,
    "mae": repeat_mae,
    "r2": repeat_r2,
    "pearson_r": repeat_r
})
repeat_metrics_df.to_csv(OUT_DIR / "Beam_final_repeat_metrics.csv", index=False)


# =============================================================================
# FEATURE IMPORTANCE OUTPUT
# =============================================================================

if COMPUTE_FEATURE_IMPORTANCE and importance_model_counter > 0:
    mean_mse_increase = importance_sum / np.maximum(importance_count, 1)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_mse_increase": mean_mse_increase,
        "models_used": importance_count.astype(int)
    })

    importance_df = importance_df.sort_values(
        "mean_mse_increase",
        ascending=False
    ).reset_index(drop=True)

    importance_df["rank_by_mean_mse_increase"] = np.arange(1, len(importance_df) + 1)

    importance_df.head(20).to_csv(
        OUT_DIR / "Beam_final_top20_features.csv",
        index=False
    )

    importance_df.to_csv(
        OUT_DIR / "Beam_all_feature_importance.csv",
        index=False
    )
    # =============================================================================
    # FIGURE: TOP FEATURE IMPORTANCE
    # =============================================================================

    if COMPUTE_FEATURE_IMPORTANCE and importance_model_counter > 0:
        top_n = 20
        top_df = importance_df.head(top_n).copy()
        top_df = top_df.iloc[::-1]  # reverse so most important is at top

        plt.figure(figsize=(12, 10))
        ax = plt.gca()

        plt.barh(
            top_df["feature"],
            top_df["mean_mse_increase"],
            edgecolor="black",
            linewidth=1.2
        )

        clean_axes(ax)
        ax.tick_params(axis="y", labelsize=14)

        plt.xlabel(
            "Increase in MSE after permutation [AU²]",
            fontweight="bold",
            fontsize=AXIS_LABEL_FONT_SIZE
        )

        plt.ylabel(
            "MRI feature",
            fontweight="bold",
            fontsize=AXIS_LABEL_FONT_SIZE
        )

        plt.title(
            "Top 20 Beam Walk SVR feature importance",
            fontweight="bold",
            fontsize=TITLE_FONT_SIZE
        )

        plt.tight_layout(pad=1.5)

        plt.savefig(
            OUT_DIR / "Beam_top20_feature_importance.pdf",
            bbox_inches="tight",
            pad_inches=0.2
        )

        plt.savefig(
            OUT_DIR / "Beam_top20_feature_importance.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.2
        )

        plt.show()

    print("\nTop 20 features by held-out permutation importance:")
    print(importance_df.head(20).to_string(index=False))

else:
    print("\nFeature importance was not computed.")


# =============================================================================
# FIGURE 1: MSE HISTOGRAM
# =============================================================================

plt.figure(figsize=(8, 6))
ax = plt.gca()

plt.hist(repeat_mse, bins=60, edgecolor="black", linewidth=1.2)

ax.axvline(
    np.median(repeat_mse),
    linestyle="--",
    linewidth=3,
    label=f"Median MSE = {np.median(repeat_mse):.2f}"
)

clean_axes(ax)

plt.xlabel("MSE [AU²]", fontweight="bold", fontsize=AXIS_LABEL_FONT_SIZE)
plt.ylabel("Frequency", fontweight="bold", fontsize=AXIS_LABEL_FONT_SIZE)
plt.title("Distribution of MSE across repeated nested CV", fontweight="bold", fontsize=TITLE_FONT_SIZE)
plt.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
plt.tight_layout(pad=2.0)

plt.savefig(OUT_DIR / "Beam_final_MSE_distribution.pdf", bbox_inches="tight")
plt.savefig(OUT_DIR / "Beam_final_MSE_distribution.png", dpi=300, bbox_inches="tight")
plt.show()


# =============================================================================
# FIGURE 2: ALL MODEL LINES + FINAL ENSEMBLE POINTS
# =============================================================================

plt.figure(figsize=(8, 6))
ax = plt.gca()

x_min, x_max = float(np.min(y)), float(np.max(y))
x_range = np.linspace(x_min, x_max, 100).reshape(-1, 1)

for rep in range(N_REPEATS):
    y_pred_rep = repeat_predictions[rep, :]

    lr = LinearRegression()
    lr.fit(y.reshape(-1, 1), y_pred_rep.reshape(-1, 1))
    y_fit = lr.predict(x_range)

    plt.plot(
        x_range,
        y_fit,
        color="gray",
        alpha=0.04,
        linewidth=1
    )

plt.scatter(
    y,
    final_pred,
    s=80,
    edgecolor="black",
    linewidth=1.2,
    label="Median ensemble prediction"
)

plt.plot(
    [x_min, x_max],
    [x_min, x_max],
    linestyle="--",
    linewidth=3,
    label="Ideal fit"
)

clean_axes(ax)

plt.xlabel("True Value [AU]", fontweight="bold", fontsize=AXIS_LABEL_FONT_SIZE)
plt.ylabel("Predicted Value [AU]", fontweight="bold", fontsize=AXIS_LABEL_FONT_SIZE)
plt.title("Held-out prediction lines from all repeated models", fontweight="bold", fontsize=TITLE_FONT_SIZE)

# Add padding so markers are not clipped
x_pad = 0.2
y_pad = 0.3

plt.xlim(x_min - x_pad, x_max + x_pad)

plt.ylim(
    np.min(final_pred) - y_pad,
    np.max(final_pred) + y_pad
)

plt.plot([], [], " ", label=f"Overall R² = {overall_r2:.2f}, r = {overall_r:.2f}")
plt.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
plt.tight_layout()

plt.savefig(OUT_DIR / "Beam_final_all_model_lines.pdf", bbox_inches="tight",pad_inches=0.2)
plt.savefig(OUT_DIR / "Beam_final_all_model_lines.png", dpi=300, bbox_inches="tight",pad_inches=0.2)
plt.show()


# =============================================================================
# FIGURE 3: FINAL TRUE VS PREDICTED
# =============================================================================

plt.figure(figsize=(8, 6))
ax = plt.gca()

plt.scatter(
    y,
    final_pred,
    s=90,
    edgecolor="black",
    linewidth=1.2
)

plt.plot(
    [x_min, x_max],
    [x_min, x_max],
    linestyle="--",
    linewidth=3,
    label="Ideal fit"
)

lr_final = LinearRegression()
lr_final.fit(y.reshape(-1, 1), final_pred.reshape(-1, 1))
y_fit_final = lr_final.predict(x_range)

plt.plot(
    x_range,
    y_fit_final,
    linewidth=3,
    label=f"Linear fit: y={lr_final.coef_[0][0]:.2f}x+{lr_final.intercept_[0]:.2f}"
)

clean_axes(ax)

plt.xlabel("True Value [AU]", fontweight="bold", fontsize=AXIS_LABEL_FONT_SIZE)
plt.ylabel("Predicted Value [AU]", fontweight="bold", fontsize=AXIS_LABEL_FONT_SIZE)
plt.title("Beam: repeated nested CV ensemble prediction", fontweight="bold", fontsize=TITLE_FONT_SIZE)

# Add padding so markers are not clipped
x_pad = 0.2
y_pad = 0.3

plt.xlim(x_min - x_pad, x_max + x_pad)

plt.ylim(
    np.min(final_pred) - y_pad,
    np.max(final_pred) + y_pad
)

plt.plot([], [], " ", label=f"R² = {overall_r2:.2f}, r = {overall_r:.2f}")
plt.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
plt.tight_layout()

plt.savefig(OUT_DIR / "Beam_final_true_vs_predicted.pdf", bbox_inches="tight",pad_inches=0.2)
plt.savefig(OUT_DIR / "Beam_final_true_vs_predicted.png", dpi=300, bbox_inches="tight",pad_inches=0.2)
plt.show()


print("\nDone. Outputs saved to:")
print(OUT_DIR)