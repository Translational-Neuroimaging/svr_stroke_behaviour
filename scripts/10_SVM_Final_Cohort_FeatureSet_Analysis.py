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

TEST = input("Which Test? Beam, Grip, or Sticky: ").strip()
if TEST == "Beam":
    BEHAVIOR_FILE = "Beamwalk24h.csv"
    TARGET_COL = "Beam"
    UNIT = "AU"
    MSE_UNIT = "AU²"
elif TEST == "Grip":
    BEHAVIOR_FILE = "Grip24h.csv"
    TARGET_COL = "Grip"
    UNIT = "AU"
    MSE_UNIT = "AU²"
elif TEST == "Sticky":
    BEHAVIOR_FILE = "Sticky24h.csv"
    TARGET_COL = "Sticky"
    UNIT = "sec"
    MSE_UNIT = "sec²"
else:
    raise ValueError("TEST must be Beam, Grip, or Sticky")


COHORT = input("Which Cohort? All, Control, or Ambroxol: ").strip()
if COHORT not in ["All", "Control", "Ambroxol"]:
    raise ValueError("COHORT must be All, Control, or Ambroxol")


FEATURE_SET = input("Which Feature Set? MRI, MRI_Voxels, or RegionalVoxels: ").strip()
if FEATURE_SET not in ["MRI", "MRI_Voxels", "RegionalVoxels"]:
    raise ValueError("FEATURE_SET must be MRI, MRI_Voxels, or RegionalVoxels")

N_REPEATS = 1000
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 5
RANDOM_SEED = 123
FEATURE_IMPORTANCE_INPUT = input("Compute feature importance? yes or no: ").strip().lower()

if FEATURE_IMPORTANCE_INPUT in ["yes", "y"]:
    COMPUTE_FEATURE_IMPORTANCE = True
elif FEATURE_IMPORTANCE_INPUT in ["no", "n"]:
    COMPUTE_FEATURE_IMPORTANCE = False
else:
    raise ValueError("Please enter yes or no for feature importance.")

FEATURE_IMPORTANCE_MAX_MODELS = 100
N_IMPORTANCE_REPEATS = 3

total_cores = multiprocessing.cpu_count()
N_JOBS = max(1, int(total_cores * 0.9))


# =============================================================================
# INPUT FILES
# =============================================================================

if FEATURE_SET == "MRI":

    feature_path = (
        ROOT
        / "CSV" / "Input" / "Normalization"
        / "Median_raw_foldnorm_Beam_Grip_matched324.csv"
    )

elif FEATURE_SET == "MRI_Voxels":

    feature_path = (
        ROOT
        / "CSV" / "Input" / "Normalization"
        / "Median_raw_foldnorm_Beam_Grip_matched324_plusVoxels.csv"
    )

elif FEATURE_SET == "RegionalVoxels":

    feature_path = (
        ROOT
        / "CSV" / "Input"
        / "Stroke_VOI_per_area_(GMM)Beam_Grip.csv"
    )

else:
    raise ValueError("Unknown FEATURE_SET")

behavior_path = ROOT / "CSV" / "Output" / BEHAVIOR_FILE
treatment_path = ROOT / "Treatment_2.xlsx"


OUT_DIR = (
    ROOT / "CSV" / "Input" / "Normalization"
    / f"Final_Revised_{TEST}_RawFoldNorm_{FEATURE_SET}_{COHORT}_All1000"
    / "Evaluation" / TEST
)
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


def clean_animal_id(series):
    return (
        series.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("FRat", "Rat", regex=False)
        .str.replace("Frat", "Rat", regex=False)
        .str.replace("frat", "Rat", regex=False)
        .str.strip()
    )


def get_cohort_ids(treatment_path, cohort):

    treat = pd.read_excel(treatment_path)

    if "AnimalID" not in treat.columns:
        raise RuntimeError("AnimalID column not found in Treatment_2.xlsx.")

    if "Ambroxol" not in treat.columns:
        raise RuntimeError("Ambroxol column not found in Treatment_2.xlsx.")

    treat["AnimalID"] = clean_animal_id(treat["AnimalID"])
    treat["Ambroxol"] = pd.to_numeric(treat["Ambroxol"], errors="coerce")

    print("Treatment values:")
    print(treat["Ambroxol"].value_counts(dropna=False))

    if cohort == "All":
        selected = treat["AnimalID"]

    elif cohort == "Control":
        selected = treat.loc[treat["Ambroxol"] == 0, "AnimalID"]

    elif cohort == "Ambroxol":
        selected = treat.loc[treat["Ambroxol"] == 1, "AnimalID"]

    else:
        raise ValueError(f"Unknown cohort: {cohort}")

    selected = selected.drop_duplicates()

    print(f"Selected {cohort} animals:", len(selected))
    print(sorted(selected.tolist()))

    if len(selected) == 0:
        raise RuntimeError(f"No animals found for cohort: {cohort}")

    return selected.tolist()
# =============================================================================
# MODALITY-WISE / FEATURE-SET-WISE SCALER
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

        # StrokeVolume_Voxels and regional voxel-load columns go here.
        return "Other"

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()

        if hasattr(X, "columns"):
            X_df.columns = X.columns

        self.columns_ = X_df.columns.tolist()
        self.modality_stats_ = {}

        for modality in ["ADC", "T2", "FA", "AD", "RD", "Trace", "Other"]:
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
# LOAD DATA
# =============================================================================

print("=" * 80)
print("FINAL REPEATED NESTED CV ANALYSIS")
print("=" * 80)
print("TEST:", TEST)
print("COHORT:", COHORT)
print("FEATURE_SET:", FEATURE_SET)
print("Feature file:", feature_path)
print("Behavior file:", behavior_path)
print("Treatment file:", treatment_path)
print("Output:", OUT_DIR)

# Load feature table
if feature_path.suffix.lower() in [".xlsx", ".xls"]:
    df_features = pd.read_excel(feature_path)
else:
    df_features = pd.read_csv(feature_path, sep=None, engine="python", encoding="utf-8-sig")

# Load behavior and treatment tables
df_behavior = pd.read_csv(behavior_path)
treat = pd.read_excel(treatment_path)
if "AnimalID" not in df_features.columns:
    df_features = df_features.rename(columns={df_features.columns[0]: "AnimalID"})
# Clean IDs
df_features["AnimalID"] = clean_animal_id(df_features["AnimalID"])
df_behavior.iloc[:, 0] = clean_animal_id(df_behavior.iloc[:, 0])

# Select cohort using Treatment_2.xlsx
cohort_ids = get_cohort_ids(treatment_path, COHORT)
cohort_ids = set(cohort_ids)

print("Selected cohort animals:", len(cohort_ids))

# Filter feature and behavior tables
df_features = df_features[df_features["AnimalID"].isin(cohort_ids)].copy()
df_behavior = df_behavior[df_behavior.iloc[:, 0].isin(cohort_ids)].copy()

print("Feature animals after cohort filter:", df_features.shape[0])
print("Behavior animals after cohort filter:", df_behavior.shape[0])

if df_features.shape[0] == 0:
    raise RuntimeError("No feature rows left after cohort filtering.")

if df_behavior.shape[0] == 0:
    raise RuntimeError("No behavior rows left after cohort filtering.")

# Keep only numeric feature columns
feature_cols = [c for c in df_features.columns if c != "AnimalID"]

metadata_like = []
for c in feature_cols:
    if str(c).lower() in [
        "brainregion", "group", "treatment", "therapy", "condition",
        "animal", "rat", "id"
    ]:
        metadata_like.append(c)

if metadata_like:
    print("Dropping metadata-like columns:", metadata_like)
    df_features = df_features.drop(columns=metadata_like)

feature_cols = [c for c in df_features.columns if c != "AnimalID"]

for c in feature_cols:
    df_features[c] = pd.to_numeric(df_features[c], errors="coerce")

all_nan_cols = [c for c in feature_cols if df_features[c].isna().all()]
if all_nan_cols:
    print("Dropping all-NaN feature columns:", all_nan_cols)
    df_features = df_features.drop(columns=all_nan_cols)

feature_cols = [c for c in df_features.columns if c != "AnimalID"]

# Align behavior to features
behavior_order = pd.DataFrame({
    "AnimalID": df_behavior.iloc[:, 0].values,
    TARGET_COL: df_behavior.iloc[:, 1].astype(float).values,
})

merged = behavior_order.merge(df_features, on="AnimalID", how="left")

feature_cols = [c for c in merged.columns if c not in ["AnimalID", TARGET_COL]]

if merged[feature_cols].isna().any().any():
    missing_animals = merged.loc[
        merged[feature_cols].isna().any(axis=1),
        "AnimalID"
    ].tolist()

    print("Missing or incomplete animals after merge:")
    print(missing_animals)
    raise RuntimeError("Feature/behavior alignment failed or feature table contains NaNs.")

aligned_ids = merged["AnimalID"].values
y = merged[TARGET_COL].values.astype(float)
X_df = merged[feature_cols].copy()

feature_names = X_df.columns.tolist()

print("Final aligned X shape:", X_df.shape)
print("Final y shape:", y.shape)
print(f"{TEST} range: {float(np.min(y))} to {float(np.max(y))}")
print("Number of feature names:", len(feature_names))
print("First feature:", feature_names[0])
print("Last feature:", feature_names[-1])
print("Animals used:")
print(aligned_ids)
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
print(f"Samples: {n_samples}")
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

        print(
            f"{TEST} | {COHORT} | {FEATURE_SET} | "
            f"Repeat {rep + 1}/{N_REPEATS}, fold {fold_idx}/{N_OUTER_SPLITS}",
            flush=True
        )

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
            "test": TEST,
            "cohort": COHORT,
            "feature_set": FEATURE_SET,
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

    if np.isnan(y_pred_rep).any():
        raise RuntimeError(f"Missing predictions in repeat {rep + 1}")

    repeat_predictions[rep, :] = y_pred_rep

    mse_rep = mean_squared_error(y, y_pred_rep)
    mae_rep = mean_absolute_error(y, y_pred_rep)
    r2_rep = r2_score(y, y_pred_rep)

    if len(np.unique(y)) > 1 and len(np.unique(y_pred_rep)) > 1:
        r_rep, _ = pearsonr(y, y_pred_rep)
    else:
        r_rep = np.nan

    repeat_mse.append(mse_rep)
    repeat_mae.append(mae_rep)
    repeat_r2.append(r2_rep)
    repeat_r.append(r_rep)

    if (rep + 1) == 1 or (rep + 1) % 25 == 0:
        print(
            f"Repeat {rep + 1}/{N_REPEATS}: "
            f"MSE={mse_rep:.4f}, MAE={mae_rep:.4f}, "
            f"R²={r2_rep:.4f}, r={r_rep:.4f}",
            flush=True
        )


# =============================================================================
# FINAL ENSEMBLE PREDICTION
# =============================================================================

final_pred = np.median(repeat_predictions, axis=0)

overall_mse = mean_squared_error(y, final_pred)
overall_mae = mean_absolute_error(y, final_pred)
overall_r2 = r2_score(y, final_pred)

if len(np.unique(y)) > 1 and len(np.unique(final_pred)) > 1:
    overall_r, overall_p = pearsonr(y, final_pred)
else:
    overall_r, overall_p = np.nan, np.nan

print("\n" + "=" * 80)
print(f"FINAL SUMMARY | {TEST} | {COHORT} | {FEATURE_SET}")
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
        "Test", "Cohort", "Feature set",
        "MSE", "MAE", "R2", "Pearson r", "Pearson p",
        "N repeats", "N outer folds", "N inner folds",
        "N samples", "N features", "Feature importance models used"
    ],
    "Value": [
        TEST, COHORT, FEATURE_SET,
        overall_mse, overall_mae, overall_r2, overall_r, overall_p,
        N_REPEATS, N_OUTER_SPLITS, N_INNER_SPLITS,
        n_samples, len(feature_names), importance_model_counter
    ]
})

summary_df.to_csv(OUT_DIR / f"{TEST}_{COHORT}_{FEATURE_SET}_final_summary.csv", index=False)

pred_df = pd.DataFrame({
    "AnimalID": aligned_ids,
    f"True_{TEST}": y,
    f"Predicted_{TEST}": final_pred
})
pred_df.to_csv(OUT_DIR / f"{TEST}_{COHORT}_{FEATURE_SET}_final_true_vs_predicted.csv", index=False)

model_metrics_df = pd.DataFrame(model_rows)
model_metrics_df.to_csv(OUT_DIR / f"{TEST}_{COHORT}_{FEATURE_SET}_final_all_model_metrics.csv", index=False)

repeat_metrics_df = pd.DataFrame({
    "repeat": np.arange(1, N_REPEATS + 1),
    "mse": repeat_mse,
    "mae": repeat_mae,
    "r2": repeat_r2,
    "pearson_r": repeat_r
})
repeat_metrics_df.to_csv(OUT_DIR / f"{TEST}_{COHORT}_{FEATURE_SET}_final_repeat_metrics.csv", index=False)


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

    importance_df.to_csv(OUT_DIR / f"{TEST}_{COHORT}_{FEATURE_SET}_final_all_feature_importance.csv", index=False)
    importance_df.head(20).to_csv(OUT_DIR / f"{TEST}_{COHORT}_{FEATURE_SET}_final_top20_features.csv", index=False)

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

plt.xlabel(f"MSE [{MSE_UNIT}]", fontweight="bold", fontsize=AXIS_LABEL_FONT_SIZE)
plt.ylabel("Frequency", fontweight="bold", fontsize=AXIS_LABEL_FONT_SIZE)
plt.title(
    f"{TEST}: MSE distribution\n{COHORT}, {FEATURE_SET}",
    fontweight="bold",
    fontsize=TITLE_FONT_SIZE
)
plt.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
plt.tight_layout(pad=2.0)

plt.savefig(OUT_DIR / f"{TEST}_{COHORT}_{FEATURE_SET}_final_MSE_distribution.pdf", bbox_inches="tight", pad_inches=0.2)
plt.savefig(OUT_DIR / f"{TEST}_{COHORT}_{FEATURE_SET}_final_MSE_distribution.png", dpi=300, bbox_inches="tight", pad_inches=0.2)
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

plt.xlabel(f"True Value [{UNIT}]", fontweight="bold", fontsize=AXIS_LABEL_FONT_SIZE)
plt.ylabel(f"Predicted Value [{UNIT}]", fontweight="bold", fontsize=AXIS_LABEL_FONT_SIZE)
plt.title(
    f"{TEST}: held-out prediction lines\n{COHORT}, {FEATURE_SET}",
    fontweight="bold",
    fontsize=TITLE_FONT_SIZE
)

x_pad = 0.05 * (x_max - x_min) if x_max > x_min else 0.2
y_pad = 0.05 * (np.max(final_pred) - np.min(final_pred)) if np.max(final_pred) > np.min(final_pred) else 0.3

plt.xlim(x_min - x_pad, x_max + x_pad)
plt.ylim(np.min(final_pred) - y_pad, np.max(final_pred) + y_pad)

plt.plot([], [], " ", label=f"R² = {overall_r2:.2f}, r = {overall_r:.2f}")
plt.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
plt.tight_layout()

plt.savefig(OUT_DIR / f"{TEST}_{COHORT}_{FEATURE_SET}_final_all_model_lines.pdf", bbox_inches="tight", pad_inches=0.2)
plt.savefig(OUT_DIR / f"{TEST}_{COHORT}_{FEATURE_SET}_final_all_model_lines.png", dpi=300, bbox_inches="tight", pad_inches=0.2)
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

plt.xlabel(f"True Value [{UNIT}]", fontweight="bold", fontsize=AXIS_LABEL_FONT_SIZE)
plt.ylabel(f"Predicted Value [{UNIT}]", fontweight="bold", fontsize=AXIS_LABEL_FONT_SIZE)
plt.title(
    f"{TEST}: repeated nested CV ensemble prediction\n{COHORT}, {FEATURE_SET}",
    fontweight="bold",
    fontsize=TITLE_FONT_SIZE
)

plt.xlim(x_min - x_pad, x_max + x_pad)
plt.ylim(np.min(final_pred) - y_pad, np.max(final_pred) + y_pad)

plt.plot([], [], " ", label=f"R² = {overall_r2:.2f}, r = {overall_r:.2f}")
plt.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
plt.tight_layout()

plt.savefig(OUT_DIR / f"{TEST}_{COHORT}_{FEATURE_SET}_final_true_vs_predicted.pdf", bbox_inches="tight", pad_inches=0.2)
plt.savefig(OUT_DIR / f"{TEST}_{COHORT}_{FEATURE_SET}_final_true_vs_predicted.png", dpi=300, bbox_inches="tight", pad_inches=0.2)
plt.show()


print("\nDone. Outputs saved to:")
print(OUT_DIR)