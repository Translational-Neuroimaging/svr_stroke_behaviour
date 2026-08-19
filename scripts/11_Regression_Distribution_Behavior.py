from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import pearsonr


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

OUT_DIR = (
    ROOT / "CSV" / "Input" / "Normalization"
    / "Featurewise_Pearson_Distributions_MRI_vs_LesionBurden"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEHAVIORS = [
    {
        "name": "Beam Walk",
        "target": "Beam",
        "file": ROOT / "CSV" / "Output" / "Beamwalk24h.csv",
        "unit": "AU",
    },
    {
        "name": "Sticky Label",
        "target": "Sticky",
        "file": ROOT / "CSV" / "Output" / "Sticky24h.csv",
        "unit": "sec",
    },
    {
        "name": "Grip Strength",
        "target": "Grip",
        "file": ROOT / "CSV" / "Output" / "Grip24h.csv",
        "unit": "AU",
    },
]


# =============================================================================
# STYLE
# =============================================================================

matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42

TITLE_FONT_SIZE = 26
SUBTITLE_FONT_SIZE = 20
AXIS_LABEL_FONT_SIZE = 20
TICK_LABEL_FONT_SIZE = 16
ANNOTATION_FONT_SIZE = 12

AXIS_LINEWIDTH = 3.0
TICK_WIDTH = 2.3
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


def get_modality(feature_name):
    feature_name = str(feature_name)

    if feature_name.endswith("_ADC"):
        return "ADC"
    if feature_name.endswith("_T2") or feature_name.endswith("_T2w"):
        return "T2w"
    if feature_name.endswith("_NormFA") or feature_name.endswith("_FA"):
        return "FA"
    if feature_name.endswith("_NormAD") or feature_name.endswith("_AD"):
        return "AD"
    if feature_name.endswith("_NormRD") or feature_name.endswith("_RD"):
        return "RD"
    if (
        feature_name.endswith("_NormTrace")
        or feature_name.endswith("_Trace")
        or feature_name.endswith("_trace")
    ):
        return "Trace"

    return "Other"


def split_region_modality(feature_name):
    modality = get_modality(feature_name)

    if modality == "Other":
        return str(feature_name), "Regional burden"

    suffixes = [
        "_ADC", "_T2", "_T2w", "_NormFA", "_FA",
        "_NormAD", "_AD", "_NormRD", "_RD",
        "_NormTrace", "_Trace", "_trace"
    ]

    region = str(feature_name)

    for suffix in suffixes:
        if region.endswith(suffix):
            region = region[:-len(suffix)]
            break

    return region, modality


def load_feature_table(path):
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")

    if "AnimalID" not in df.columns:
        df = df.rename(columns={df.columns[0]: "AnimalID"})

    df["AnimalID"] = clean_animal_id(df["AnimalID"])

    for c in df.columns:
        if c != "AnimalID":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def load_behavior(path, target_col):
    df = pd.read_csv(path)
    df.iloc[:, 0] = clean_animal_id(df.iloc[:, 0])

    out = pd.DataFrame({
        "AnimalID": df.iloc[:, 0].values,
        target_col: pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
    })

    return out.dropna()


def compute_featurewise_pearson(feature_df, behavior_df, target_col, representation):
    merged = behavior_df.merge(feature_df, on="AnimalID", how="inner")

    feature_cols = [c for c in merged.columns if c not in ["AnimalID", target_col]]
    y = merged[target_col].values.astype(float)

    rows = []

    for feature in feature_cols:
        x = merged[feature].values.astype(float)

        valid = np.isfinite(x) & np.isfinite(y)

        x_valid = x[valid]
        y_valid = y[valid]

        region, modality = split_region_modality(feature)

        if len(x_valid) < 3 or len(np.unique(x_valid)) < 2 or len(np.unique(y_valid)) < 2:
            r = np.nan
            p = np.nan
        else:
            r, p = pearsonr(x_valid, y_valid)

        rows.append({
            "feature_representation": representation,
            "target": target_col,
            "feature": feature,
            "region": region,
            "modality": modality,
            "n": int(np.sum(valid)),
            "pearson_r": float(r) if np.isfinite(r) else np.nan,
            "pearson_p": float(p) if np.isfinite(p) else np.nan,
            "abs_pearson_r": abs(float(r)) if np.isfinite(r) else np.nan,
        })

    out = pd.DataFrame(rows)
    n_tests = out["pearson_p"].notna().sum()
    out["bonferroni_p"] = out["pearson_p"] * n_tests
    out["bonferroni_p"] = out["bonferroni_p"].clip(upper=1.0)
    out["bonferroni_significant"] = out["bonferroni_p"] < 0.05
    return out


def plot_r_distribution(ax, results, test_name, representation):
    r_values = results["pearson_r"].dropna().values
    p_values = results["pearson_p"].dropna().values

    n_features = len(r_values)
    bonf_threshold = 0.05 / n_features

    results = results.copy()
    results["bonferroni_significant"] = results["pearson_p"] < bonf_threshold

    sig = results[results["bonferroni_significant"]].copy()

    median_r = np.nanmedian(r_values)
    median_abs_r = np.nanmedian(np.abs(r_values))
    max_abs_r = np.nanmax(np.abs(r_values))
    n_sig = sig.shape[0]

    # Histogram
    counts, bins, patches = ax.hist(
        r_values,
        bins=np.linspace(-1, 1, 41),
        edgecolor="black",
        linewidth=1.0
    )

    ymax = max(counts) if len(counts) > 0 else 1

    # Zero line
    ax.axvline(
        0,
        linestyle="-",
        linewidth=2,
        color="black",
        alpha=0.8
    )
    # Median line
    ax.axvline(
        median_r,
        linestyle="--",
        linewidth=3,
        color="#b2202b",
        label=f"Median r = {median_r:.2f}"
    )

    # Bonferroni significant features
    sig05 = results[
        (results["bonferroni_p"] < 0.05) &
        (results["bonferroni_p"] >= 0.01)
        ]

    sig01 = results[
        results["bonferroni_p"] < 0.01
        ]

    if len(sig05) > 0:
        ax.scatter(
            sig05["pearson_r"],
            np.full(len(sig05), ymax * 0.07),
            color="#3cb54a",
            s=75,
            edgecolor="black",
            linewidth=0.7,
            zorder=5,
            label="p < 0.05"
        )

    if len(sig01) > 0:
        ax.scatter(
            sig01["pearson_r"],
            np.full(len(sig01), ymax * 0.14),
            color="#fce61f",
            s=90,
            edgecolor="black",
            linewidth=0.7,
            zorder=6,
            label="p < 0.01"
        )

    # Text annotation
    ax.text(
        0.05,
        0.88,
        f"Median |r| = {median_abs_r:.2f}\n"
        f"Max |r| = {max_abs_r:.2f}",
        transform=ax.transAxes,
        fontsize=ANNOTATION_FONT_SIZE,
        fontweight="bold",
        va="top"
    )

    clean_axes(ax)

    ax.set_xlim(-1, 1)

    ax.set_xlabel(
        "Pearson r",
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Feature count",
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontweight="bold"
    )

    ax.set_title(
        f"{test_name}\n{representation}",
        fontsize=SUBTITLE_FONT_SIZE,
        fontweight="bold"
    )

    ax.legend(
        frameon=False,
        fontsize=11,
        loc="center left",
        markerscale =0.8,
        handletextpad=0.5
    )


# =============================================================================
# MAIN
# =============================================================================

print("=" * 80)
print("FEATURE-WISE PEARSON CORRELATION ANALYSIS")
print("=" * 80)

df_mri = load_feature_table(MRI_FEATURE_FILE)
df_lesion = load_feature_table(LESION_FEATURE_FILE)

all_results = []

fig, axes = plt.subplots(2, 3, figsize=(20, 9))

for col_idx, info in enumerate(BEHAVIORS):
    behavior_df = load_behavior(info["file"], info["target"])

    print("\n" + "=" * 80)
    print(info["name"])
    print("Behavior file:", info["file"])
    print("N behavior rows:", len(behavior_df))
    print("=" * 80)

    # Regional MRI features
    res_mri = compute_featurewise_pearson(
        feature_df=df_mri,
        behavior_df=behavior_df,
        target_col=info["target"],
        representation="Regional MRI features"
    )

    res_mri["behavioral_test"] = info["name"]
    all_results.append(res_mri)

    plot_r_distribution(
        axes[0, col_idx],
        res_mri,
        info["name"],
        "Regional MRI features"
    )

    # Regional lesion burden
    res_lesion = compute_featurewise_pearson(
        feature_df=df_lesion,
        behavior_df=behavior_df,
        target_col=info["target"],
        representation="Regional lesion burden"
    )

    res_lesion["behavioral_test"] = info["name"]
    all_results.append(res_lesion)

    plot_r_distribution(
        axes[1, col_idx],
        res_lesion,
        info["name"],
        "Regional lesion burden"
    )

    # Save per-behavior files
    res_mri.sort_values("abs_pearson_r", ascending=False).to_csv(
        OUT_DIR / f"{info['target']}_RegionalMRI_all_pearson_correlations.csv",
        index=False
    )

    res_lesion.sort_values("abs_pearson_r", ascending=False).to_csv(
        OUT_DIR / f"{info['target']}_RegionalLesionBurden_all_pearson_correlations.csv",
        index=False
    )

    res_mri.sort_values("abs_pearson_r", ascending=False).head(10).to_csv(
        OUT_DIR / f"{info['target']}_RegionalMRI_top10_abs_pearson.csv",
        index=False
    )

    res_lesion.sort_values("abs_pearson_r", ascending=False).head(10).to_csv(
        OUT_DIR / f"{info['target']}_RegionalLesionBurden_top10_abs_pearson.csv",
        index=False
    )

combined = pd.concat(all_results, ignore_index=True)

combined = combined.sort_values(
    ["behavioral_test", "feature_representation", "abs_pearson_r"],
    ascending=[True, True, False]
)

combined.to_csv(
    OUT_DIR / "ALL_featurewise_pearson_correlations.csv",
    index=False
)

top10 = (
    combined
    .groupby(["behavioral_test", "feature_representation"], group_keys=False)
    .apply(lambda x: x.sort_values("abs_pearson_r", ascending=False).head(10))
    .reset_index(drop=True)
)

top10.to_csv(
    OUT_DIR / "ALL_top10_abs_pearson_by_behavior_and_representation.csv",
    index=False
)

fig.suptitle(
    "Distribution of feature-wise correlations with behavioral outcomes",
    fontsize=TITLE_FONT_SIZE,
    fontweight="bold",
    y=0.98
)

plt.tight_layout(rect=[0, 0, 1, 0.94])

fig.savefig(
    OUT_DIR / "Featurewise_Pearson_Distributions_MRI_vs_RegionalLesionBurden.pdf",
    bbox_inches="tight",
    pad_inches=0.2
)

fig.savefig(
    OUT_DIR / "Featurewise_Pearson_Distributions_MRI_vs_RegionalLesionBurden.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.2
)

plt.show()

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print("Outputs saved to:")
print(OUT_DIR)

print("\nTop 10 absolute Pearson correlations per behavior and feature representation:")

for (behavior, representation), group in top10.groupby(
    ["behavioral_test", "feature_representation"]
):
    print("\n" + "=" * 100)
    print(f"{behavior} | {representation}")
    print("=" * 100)

    print(
        group[
            [
                "feature",
                "region",
                "modality",
                "n",
                "pearson_r",
                "pearson_p",
                "abs_pearson_r",
            ]
        ].to_string(index=False)
    )