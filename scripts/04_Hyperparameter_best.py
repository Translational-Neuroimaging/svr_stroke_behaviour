import pandas as pd
import matplotlib.pyplot as plt
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent

csv_path = (
    ROOT / "CSV" / "Input" / "Normalization"
    / "Final_Revised_Beam_RawFoldNorm_RegionalVoxels_All_All1000"
    / "Evaluation" / "Beam"
    / "Beam_All_RegionalVoxels_final_all_model_metrics.csv"
)

df = pd.read_csv(csv_path)

params = df["best_params"].apply(ast.literal_eval)

df["kernel"] = params.apply(lambda x: x.get("svr__kernel", x.get("kernel")))
df["C"] = params.apply(lambda x: str(x.get("svr__C", x.get("C"))))
df["gamma"] = params.apply(lambda x: str(x.get("svr__gamma", x.get("gamma"))))
df["epsilon"] = params.apply(lambda x: str(x.get("svr__epsilon", x.get("epsilon"))))

r2_col = "r2_outer_test"

out_dir = csv_path.parent / "Hyperparameter_Figures"
out_dir.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

TITLE_SIZE = 15
LABEL_SIZE = 14
TICK_SIZE = 12
PANEL_SIZE = 20
AXIS_WIDTH = 1.8

plot_info = {
    "kernel": {
        "title": "Effect of kernel choice on model performance",
        "xlabel": "Kernel type",
        "panel": "A",
        "order": ["linear", "rbf"]
    },
    "C": {
        "title": r"Effect of $C$ on model performance",
        "xlabel": r"$C$",
        "panel": "B",
        "order": ["0.1", "1", "10", "100"]
    },
    "gamma": {
        "title": r"Effect of $\gamma$ on model performance",
        "xlabel": r"$\gamma$",
        "panel": "C",
        "order": ["scale", "auto", "0.01", "0.1", "1"]
    },
    "epsilon": {
        "title": r"Effect of $\epsilon$ on model performance",
        "xlabel": r"$\epsilon$",
        "panel": "D",
        "order": ["0.01", "0.1", "0.5"]
    }
}

def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(AXIS_WIDTH)
    ax.spines["left"].set_linewidth(AXIS_WIDTH)
    ax.tick_params(axis="both", labelsize=TICK_SIZE, width=AXIS_WIDTH)
    ax.grid(False)


def box_scatter(ax, parameter):
    info = plot_info[parameter]

    groups = [g for g in info["order"] if g in df[parameter].unique()]
    data = [df.loc[df[parameter] == g, r2_col].values for g in groups]

    ax.boxplot(
        data,
        labels=groups,
        showfliers=False,
        patch_artist=False,
        medianprops=dict(linewidth=1.8),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5)
    )

    for i, g in enumerate(groups, start=1):
        y = df.loc[df[parameter] == g, r2_col].values
        x = [i] * len(y)
        ax.scatter(x, y, alpha=0.22, s=10)

    ax.set_title(info["title"], fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel(info["xlabel"], fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel(r"Outer-fold $R^2$", fontsize=LABEL_SIZE, fontweight="bold")

    ax.set_ylim(-0.45, 1.0)

    clean_axes(ax)

    ax.text(
        -0.22,
        1.02,
        info["panel"],
        transform=ax.transAxes,
        fontsize=PANEL_SIZE,
        fontweight="bold",
        va="top",
        ha="left"
    )


# Individual figures
for p in ["epsilon", "gamma", "C", "kernel"]:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    box_scatter(ax, p)
    plt.tight_layout(pad=1.5)
    fig.savefig(out_dir / f"Beam_{r2_col}_by_{p}.pdf", bbox_inches="tight", pad_inches=0.2)
    fig.savefig(out_dir / f"Beam_{r2_col}_by_{p}.png", dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


# Combined vertical figure
fig, axes = plt.subplots(
    nrows=4,
    ncols=1,
    figsize=(5, 14),
    sharey=True
)
for ax, p in zip(axes, ["kernel", "C", "gamma", "epsilon"]):
    box_scatter(ax, p)

plt.tight_layout(pad=1.5)

fig.subplots_adjust(hspace=0.35)

fig.savefig(
    out_dir / "Beam_hyperparameter_combined_vertical.pdf",
    bbox_inches="tight",
    pad_inches=0.2
)

fig.savefig(
    out_dir / "Beam_hyperparameter_combined_vertical.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.2
)
plt.show()


summary = (
    df.groupby(["kernel", "C", "gamma", "epsilon"])[r2_col]
    .agg(["count", "mean", "median", "std", "max"])
    .reset_index()
    .sort_values("median", ascending=False)
)

summary.to_csv(out_dir / f"Beam_hyperparameter_summary_by_{r2_col}.csv", index=False)

print(summary.head(20))
print("Saved to:", out_dir)