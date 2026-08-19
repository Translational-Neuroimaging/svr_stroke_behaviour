from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parent

MRI_FEATURE_FILE = (
    ROOT / "CSV" / "Input" / "Normalization"
    / "Median_raw_foldnorm_Beam_Grip_matched324.csv"
)

LESION_FEATURE_FILE = (
    ROOT / "CSV" / "Input"
    / "Stroke_VOI_per_area_(GMM)Beam_Grip.csv"
)

BEAM_FILE = ROOT / "CSV" / "Output" / "Beamwalk24h.csv"

OUT_DIR = (
    ROOT / "CSV" / "Input" / "Normalization"
    / "Beam_HighLow_PolarMaps"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Beam Walk: higher score = worse deficit.
LOW_LABEL = "Low Beam Walk score"
HIGH_LABEL = "High Beam Walk score"

# Top 10 MRI features from permutation importance
MRI_TOP_FEATURES = [
    "Insular cortex ipsilesional_T2",
    "Amygdala ipsilesional_T2",
    "Somatosensory cortex ipsilesional_T2",
    "Frontal cortex contralesional_ADC",
    "Somatosensory cortex ipsilesional_ADC",
    "Insular cortex ipsilesional_ADC",
    "Internal capsule ipsilesional_T2",
    "Amygdala ipsilesional_ADC",
    "Frontal cortex ipsilesional_ADC",
    "Frontal cortex ipsilesional_T2",
]

# Top 10 regional lesion burden features from permutation importance
LESION_TOP_FEATURES = [
    "Somatosensory cortex ipsilesional",
    "Amygdala ipsilesional",
    "Insular cortex ipsilesional",
    "Olfactory ipsilesional",
    "Visual cortex ipsilesional",
    "Parietal cortex ipsilesional",
    "Putamen ipsilesional",
    "Internal capsule ipsilesional",
    "Nucleus accumbens ipsilesional",
    "Frontal cortex ipsilesional",
]

# =============================================================================
# STYLE
# =============================================================================

matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42

COLOR_LOW = "#2f5aa8"
COLOR_HIGH = "#b2202b"
AXIS_LINEWIDTH = 2.5


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


def load_feature_table(path):
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")

    if "AnimalID" not in df.columns:
        df = df.rename(columns={df.columns[0]: "AnimalID"})

    df["AnimalID"] = clean_animal_id(df["AnimalID"])

    for c in df.columns:
        if c != "AnimalID":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def load_beam(path):
    df = pd.read_csv(path)
    df.iloc[:, 0] = clean_animal_id(df.iloc[:, 0])

    return pd.DataFrame({
        "AnimalID": df.iloc[:, 0].values,
        "Beam": pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
    }).dropna()


def resolve_column(df, requested):
    """
    Tries to match feature names even if suffix is T2 vs T2w etc.
    """
    if requested in df.columns:
        return requested

    candidates = [
        requested,
        requested.replace("_T2w", "_T2"),
        requested.replace("_T2", "_T2w"),
        requested.replace("_Trace", "_NormTrace"),
        requested.replace("_FA", "_NormFA"),
        requested.replace("_AD", "_NormAD"),
        requested.replace("_RD", "_NormRD"),
    ]

    for c in candidates:
        if c in df.columns:
            return c

    # Loose matching
    simple_req = requested.lower().replace(" ", "").replace("_", "")
    for c in df.columns:
        simple_c = str(c).lower().replace(" ", "").replace("_", "")
        if simple_req == simple_c:
            return c

    raise RuntimeError(f"Could not find column: {requested}")


def zscore_columns(df, cols):
    out = df.copy()

    for c in cols:
        vals = out[c].astype(float).values
        mean = np.nanmean(vals)
        std = np.nanstd(vals)

        if std == 0 or np.isnan(std):
            std = 1.0

        out[c] = (vals - mean) / std

    return out


def sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return np.std(x, ddof=1) / np.sqrt(len(x))


def make_group_summary(df, cols, group_col, representation):
    rows = []

    for c in cols:
        for group in [LOW_LABEL, HIGH_LABEL]:
            values = df.loc[df[group_col] == group, c].astype(float).values

            rows.append({
                "feature_representation": representation,
                "feature": c,
                "group": group,
                "n": np.sum(np.isfinite(values)),
                "mean_z": np.nanmean(values),
                "sem_z": sem(values),
                "median_z": np.nanmedian(values),
            })

    return pd.DataFrame(rows)


def short_label(label):
    replacements = {
        "ipsilesional": "ipsi",
        "contralesional": "contra",
        "Somatosensory cortex": "Somatosensory",
        "Insular cortex": "Insula",
        "Frontal cortex": "Frontal",
        "Internal capsule": "Int. capsule",
        "Nucleus accumbens": "N. accumbens",
        "Visual cortex": "Visual",
        "Parietal cortex": "Parietal",
        "Amygdala": "Amygdala",
        "Olfactory": "Olfactory",
        "Putamen": "Putamen",
    }

    out = label
    for k, v in replacements.items():
        out = out.replace(k, v)

    out = out.replace("_T2", "\nT2w")
    out = out.replace("_ADC", "\nADC")

    return out


def polar_plot(ax, labels, low_values, high_values, title):
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    angles_closed = np.r_[angles, angles[0]]
    low_closed = np.r_[low_values, low_values[0]]
    high_closed = np.r_[high_values, high_values[0]]

    ax.plot(
        angles_closed,
        low_closed,
        color=COLOR_LOW,
        linewidth=3,
        marker="o",
        label=LOW_LABEL,
    )
    ax.fill(
        angles_closed,
        low_closed,
        color=COLOR_LOW,
        alpha=0.12,
    )

    ax.plot(
        angles_closed,
        high_closed,
        color=COLOR_HIGH,
        linewidth=3,
        marker="o",
        label=HIGH_LABEL,
    )
    ax.fill(
        angles_closed,
        high_closed,
        color=COLOR_HIGH,
        alpha=0.12,
    )

    ax.set_xticks(angles)
    ax.set_xticklabels([short_label(x) for x in labels], fontsize=10, fontweight="bold")

    ax.set_ylim(-1.5, 1.5)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(["-1", "-0.5", "0", "0.5", "1"], fontsize=9)

    ax.spines["polar"].set_linewidth(AXIS_LINEWIDTH)
    ax.grid(True, linewidth=1.0, alpha=0.45)

    ax.set_title(title, fontsize=18, fontweight="bold", pad=30)


# =============================================================================
# LOAD DATA
# =============================================================================

df_mri = load_feature_table(MRI_FEATURE_FILE)
df_lesion = load_feature_table(LESION_FEATURE_FILE)
df_beam = load_beam(BEAM_FILE)

mri_cols = [resolve_column(df_mri, c) for c in MRI_TOP_FEATURES]
lesion_cols = [resolve_column(df_lesion, c) for c in LESION_TOP_FEATURES]

df_mri = df_mri[["AnimalID"] + mri_cols].copy()
df_lesion = df_lesion[["AnimalID"] + lesion_cols].copy()

# Merge all data
merged = (
    df_beam
    .merge(df_mri, on="AnimalID", how="inner")
    .merge(df_lesion, on="AnimalID", how="inner", suffixes=("_MRI", "_Lesion"))
)

# Split animals by Beam median
beam_median = merged["Beam"].median()

merged["Beam_group"] = np.where(
    merged["Beam"] <= beam_median,
    LOW_LABEL,
    HIGH_LABEL
)

print("Beam median:", beam_median)
print(merged["Beam_group"].value_counts())

# Normalize MRI and lesion features independently across all animals.
# This is for visualization only.
all_feature_cols = mri_cols + lesion_cols
merged_z = zscore_columns(merged, all_feature_cols)

# Save normalized animal-level data
merged_z.to_csv(
    OUT_DIR / "Beam_high_low_normalized_feature_values.csv",
    index=False
)

# Group summaries
summary_mri = make_group_summary(
    merged_z,
    mri_cols,
    "Beam_group",
    "Regional MRI features"
)

summary_lesion = make_group_summary(
    merged_z,
    lesion_cols,
    "Beam_group",
    "Regional lesion burden"
)

summary = pd.concat([summary_mri, summary_lesion], ignore_index=True)
summary.to_csv(
    OUT_DIR / "Beam_high_low_group_summary.csv",
    index=False
)

# Extract group means for polar maps
def group_means(summary_df, cols, representation, group):
    return np.array([
        summary_df.loc[
            (summary_df["feature_representation"] == representation)
            & (summary_df["feature"] == c)
            & (summary_df["group"] == group),
            "mean_z"
        ].values[0]
        for c in cols
    ])

mri_low = group_means(summary, mri_cols, "Regional MRI features", LOW_LABEL)
mri_high = group_means(summary, mri_cols, "Regional MRI features", HIGH_LABEL)

lesion_low = group_means(summary, lesion_cols, "Regional lesion burden", LOW_LABEL)
lesion_high = group_means(summary, lesion_cols, "Regional lesion burden", HIGH_LABEL)

# =============================================================================
# FIGURE 1: TWO POLAR MAPS, LOW VS HIGH OVERLAID
# =============================================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(18, 9),
    subplot_kw=dict(polar=True)
)

polar_plot(
    axes[0],
    mri_cols,
    mri_low,
    mri_high,
    "Regional MRI features\nTop Beam Walk predictors"
)

polar_plot(
    axes[1],
    lesion_cols,
    lesion_low,
    lesion_high,
    "Regional lesion burden\nTop Beam Walk predictors"
)

fig.suptitle(
    "Imaging characteristics of low vs. high Beam Walk score animals",
    fontsize=24,
    fontweight="bold",
    y=0.98
)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=2,
    frameon=False,
    fontsize=14
)

plt.tight_layout(rect=[0, 0.08, 1, 0.94])

fig.savefig(
    OUT_DIR / "Beam_HighLow_PolarMaps_MRI_and_LesionBurden.pdf",
    bbox_inches="tight",
    pad_inches=0.2
)

fig.savefig(
    OUT_DIR / "Beam_HighLow_PolarMaps_MRI_and_LesionBurden.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.2
)

plt.show()

# =============================================================================
# FIGURE 2: FOUR POLAR MAPS, LOW AND HIGH SEPARATED
# =============================================================================

fig2, axes2 = plt.subplots(
    2,
    2,
    figsize=(18, 14),
    subplot_kw=dict(polar=True)
)

def single_group_polar(ax, labels, values, title, color):
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles_closed = np.r_[angles, angles[0]]
    values_closed = np.r_[values, values[0]]

    ax.plot(angles_closed, values_closed, color=color, linewidth=3, marker="o")
    ax.fill(angles_closed, values_closed, color=color, alpha=0.18)

    ax.set_xticks(angles)
    ax.set_xticklabels([short_label(x) for x in labels], fontsize=10, fontweight="bold")

    ax.set_ylim(-1.5, 1.5)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(["-1", "-0.5", "0", "0.5", "1"], fontsize=9)

    ax.spines["polar"].set_linewidth(AXIS_LINEWIDTH)
    ax.grid(True, linewidth=1.0, alpha=0.45)
    ax.set_title(title, fontsize=17, fontweight="bold", pad=30)

single_group_polar(
    axes2[0, 0],
    mri_cols,
    mri_low,
    f"Regional MRI features\n{LOW_LABEL}",
    COLOR_LOW
)

single_group_polar(
    axes2[0, 1],
    mri_cols,
    mri_high,
    f"Regional MRI features\n{HIGH_LABEL}",
    COLOR_HIGH
)

single_group_polar(
    axes2[1, 0],
    lesion_cols,
    lesion_low,
    f"Regional lesion burden\n{LOW_LABEL}",
    COLOR_LOW
)

single_group_polar(
    axes2[1, 1],
    lesion_cols,
    lesion_high,
    f"Regional lesion burden\n{HIGH_LABEL}",
    COLOR_HIGH
)

fig2.suptitle(
    "Group-wise polar maps of Beam Walk-associated imaging features",
    fontsize=24,
    fontweight="bold",
    y=0.98
)

plt.tight_layout(rect=[0, 0, 1, 0.94])

fig2.savefig(
    OUT_DIR / "Beam_HighLow_PolarMaps_Separated.pdf",
    bbox_inches="tight",
    pad_inches=0.2
)

fig2.savefig(
    OUT_DIR / "Beam_HighLow_PolarMaps_Separated.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.2
)

plt.show()

print("\nDone. Outputs saved to:")
print(OUT_DIR)