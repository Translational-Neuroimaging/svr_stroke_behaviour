import nibabel as nib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path
from skimage import measure
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parent

TEST = "Beam"
TOP_N_REGIONS = 10

ATLAS_PATH = ROOT / "Atlas_neu" / "rat_Atlas12.nii"
BRAIN_MASK_PATH = ROOT / "Atlas_neu" / "full_mask_new.nii"
STROKE_PATH = ROOT / "Atlas_neu" / "joint_probability_map_GMM.nii.gz"

# Signal intensity based MRI features
FEATURE_CSV = (
    ROOT / "CSV" / "Input" / "Normalization"
    / "Final_Revised_Beam_RawFoldNorm_FeatureImportance"
    / "Evaluation" / TEST
    / f"{TEST}_final_top20_features.csv"
)

# Region lesion burden
#FEATURE_CSV = (
#    ROOT / "CSV" / "Input" / "Normalization"
#    / "Final_Revised_Beam_RawFoldNorm_RegionalVoxels_All_All1000"
#    / "Evaluation" / TEST
#    / f"{TEST}_All_RegionalVoxels_final_top20_features.csv"
#)

OUT_DIR = ROOT / "CSV" / "Visual" / TEST / "Important_Features_Top10_RankedLabels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STROKE_THRESHOLD = 0.30

SAVE_PNG = True
SAVE_PDF = True

matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42

LEGEND_FONT_SIZE = 18
TABLE_FONT_SIZE = 13
SCALEBAR_FONT_SIZE = 18
SCALEBAR_THICKNESS = 8
RANK_FONT_SIZE = 18


# =============================================================================
# ATLAS LABELS
# =============================================================================

brain_region = {
    'Amygdala contralesional': 61,
    'Amygdala ipsilesional': 61.5,
    'Antero-dorsal hippocampus contralesional': 18,
    'Antero-dorsal hippocampus ipsilesional': 18.5,
    'Auditory cortex contralesional': 43,
    'Auditory cortex ipsilesional': 43.5,
    'Central canal': 17,
    'Cingulate cortex contralesional': 57,
    'Cingulate cortex ipsilesional': 57.5,
    'Corpus callosum (Body) contralesional': 14,
    'Corpus callosum (Body) ipsilesional': 14.5,
    'Entorhinal cortex contralesional': 10,
    'Entorhinal cortex ipsilesional': 10.5,
    'External capsule contralesional': 12,
    'External capsule ipsilesional': 12.5,
    'Frontal cortex contralesional': 40,
    'Frontal cortex ipsilesional': 40.5,
    'Hypothalamus contralesional': 8,
    'Hypothalamus ipsilesional': 8.5,
    'Inferior colliculus contralesional': 38,
    'Inferior colliculus ipsilesional': 38.5,
    'Insular cortex contralesional': 65,
    'Insular cortex ipsilesional': 65.5,
    'Internal capsule contralesional': 13,
    'Internal capsule ipsilesional': 13.5,
    'Medial prefrontal cortex contralesional': 31,
    'Medial prefrontal cortex ipsilesional': 31.5,
    'Midbrain contralesional': 21,
    'Midbrain ipsilesional': 21.5,
    'Motor cortex contralesional': 52,
    'Motor cortex ipsilesional': 52.5,
    'Nucleus accumbens contralesional': 20,
    'Nucleus accumbens ipsilesional': 20.5,
    'Olfactory contralesional': 28,
    'Olfactory ipsilesional': 28.5,
    'Parietal cortex contralesional': 55,
    'Parietal cortex ipsilesional': 55.5,
    'Posterior hippocampus contralesional': 15,
    'Posterior hippocampus ipsilesional': 15.5,
    'Putamen contralesional': 25,
    'Putamen ipsilesional': 25.5,
    'Retrosplenial cortex contralesional': 36,
    'Retrosplenial cortex ipsilesional': 36.5,
    'Septum': 22,
    'Somatosensory cortex contralesional': 49,
    'Somatosensory cortex ipsilesional': 49.5,
    'Superior colliculus contralesional': 67,
    'Superior colliculus ipsilesional': 67.5,
    'Thalamus contralesional': 41,
    'Thalamus ipsilesional': 41.5,
    'Ventral tegmental area contralesional': 63,
    'Ventral tegmental area ipsilesional': 63.5,
    'Visual cortex contralesional': 44,
    'Visual cortex ipsilesional': 44.5
}
brain_region_functions = {

    # Motor
    'Motor cortex contralesional': 'Motor',
    'Motor cortex ipsilesional': 'Motor',

    'Putamen contralesional': 'Motor',
    'Putamen ipsilesional': 'Motor',

    'Cingulate cortex contralesional': 'Motor',
    'Cingulate cortex ipsilesional': 'Motor',

    # Sensory
    'Somatosensory cortex contralesional': 'Sensory',
    'Somatosensory cortex ipsilesional': 'Sensory',

    'Auditory cortex contralesional': 'Sensory',
    'Auditory cortex ipsilesional': 'Sensory',

    'Visual cortex contralesional': 'Sensory',
    'Visual cortex ipsilesional': 'Sensory',

    'Olfactory contralesional': 'Sensory',
    'Olfactory ipsilesional': 'Sensory',

    'Inferior colliculus contralesional': 'Sensory',
    'Inferior colliculus ipsilesional': 'Sensory',

    'Superior colliculus contralesional': 'Sensory',
    'Superior colliculus ipsilesional': 'Sensory',

    # Multimodal
    'Frontal cortex contralesional': 'Multimodal',
    'Frontal cortex ipsilesional': 'Multimodal',

    'Parietal cortex contralesional': 'Multimodal',
    'Parietal cortex ipsilesional': 'Multimodal',

    'Insular cortex contralesional': 'Multimodal',
    'Insular cortex ipsilesional': 'Multimodal',

    'Thalamus contralesional': 'Multimodal',
    'Thalamus ipsilesional': 'Multimodal',

    'Midbrain contralesional': 'Multimodal',
    'Midbrain ipsilesional': 'Multimodal',

    'Internal capsule contralesional': 'Multimodal',
    'Internal capsule ipsilesional': 'Multimodal',

    'Posterior hippocampus contralesional': 'Multimodal',
    'Posterior hippocampus ipsilesional': 'Multimodal',

    # Limbic/Cognitive
    'Amygdala contralesional': 'Limbic/Cognitive',
    'Amygdala ipsilesional': 'Limbic/Cognitive',

    'Entorhinal cortex contralesional': 'Limbic/Cognitive',
    'Entorhinal cortex ipsilesional': 'Limbic/Cognitive',

    'Retrosplenial cortex contralesional': 'Limbic/Cognitive',
    'Retrosplenial cortex ipsilesional': 'Limbic/Cognitive',

    'Antero-dorsal hippocampus contralesional': 'Limbic/Cognitive',
    'Antero-dorsal hippocampus ipsilesional': 'Limbic/Cognitive',

    'Medial prefrontal cortex contralesional': 'Limbic/Cognitive',
    'Medial prefrontal cortex ipsilesional': 'Limbic/Cognitive',

    'Nucleus accumbens contralesional': 'Limbic/Cognitive',
    'Nucleus accumbens ipsilesional': 'Limbic/Cognitive',

    'Hypothalamus contralesional': 'Limbic/Cognitive',
    'Hypothalamus ipsilesional': 'Limbic/Cognitive',

    'Ventral tegmental area contralesional': 'Limbic/Cognitive',
    'Ventral tegmental area ipsilesional': 'Limbic/Cognitive',

    'Septum': 'Limbic/Cognitive',

    # White matter / miscellaneous
    'Corpus callosum (Body) contralesional': 'Multimodal',
    'Corpus callosum (Body) ipsilesional': 'Multimodal',

    'External capsule contralesional': 'Multimodal',
    'External capsule ipsilesional': 'Multimodal',

    'Central canal': 'Limbic/Cognitive'
}

brain_function_color = {
    "Motor": (68/255, 120/255, 234/255),              # blue
    "Sensory": (253/255, 231/255, 37/255),           # orange
    "Multimodal": (0/255, 200/255, 0/255),        # bluish green
    "Limbic/Cognitive": (204/255, 121/255, 167/255) #  Purple
}

STROKE_COLOR = (90/255, 90/255, 90/255)


# =============================================================================
# FUNCTIONS
# =============================================================================

def save_figure(fig, save_path):
    if SAVE_PDF:
        fig.savefig(str(save_path) + ".pdf", transparent=True, bbox_inches="tight", pad_inches=0.03)
    if SAVE_PNG:
        fig.savefig(str(save_path) + ".png", transparent=True, dpi=300, bbox_inches="tight", pad_inches=0.03)


def surface_mesh(binary_image, threshold=0, color="gray", alpha=0.5):
    if np.sum(binary_image) == 0:
        return None

    verts, faces, _, _ = measure.marching_cubes(binary_image.astype(float), level=threshold)
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    mesh.set_facecolor(color)
    mesh.set_edgecolor("none")
    return mesh


def orient_standard(volume):
    volume = np.flip(volume, axis=0)
    volume = np.transpose(volume, (2, 0, 1))
    volume = volume[:, :, ::-1]
    return volume


def transform_point_standard(point, original_shape):
    """
    Transform one xyz coordinate using the same orientation as orient_standard().
    """
    x, y, z = point

    x = original_shape[0] - 1 - x
    transformed = np.array([z, x, y], dtype=float)
    transformed[2] = original_shape[1] - 1 - transformed[2]

    return transformed


def setup_3d_axis(ax, volume):
    ax.set_xlim(0, volume.shape[0])
    ax.set_ylim(0, volume.shape[1])
    ax.set_zlim(0, volume.shape[2])
    ax.set_box_aspect([volume.shape[0], volume.shape[1], volume.shape[2]])
    ax.set_axis_off()
    ax.grid(False)


def add_scalebar_standard(ax):
    scalebar_length = 23
    scalebar_origin = [25, 10, 10]

    ax.plot(
        [scalebar_origin[0], scalebar_origin[0] + scalebar_length],
        [scalebar_origin[1], scalebar_origin[1]],
        [scalebar_origin[2], scalebar_origin[2]],
        color="black",
        lw=SCALEBAR_THICKNESS
    )

    ax.text(
        scalebar_origin[0] + scalebar_length / 2,
        scalebar_origin[1] + 2,
        scalebar_origin[2] + 2,
        "5 mm",
        color="black",
        fontsize=SCALEBAR_FONT_SIZE,
        fontweight="bold"
    )


def add_scalebar_coronal(ax):
    scalebar_length = 23
    scalebar_origin = [10, 3, 3]

    ax.plot(
        [scalebar_origin[0], scalebar_origin[0] + scalebar_length],
        [scalebar_origin[1], scalebar_origin[1]],
        [scalebar_origin[2], scalebar_origin[2]],
        color="black",
        lw=SCALEBAR_THICKNESS
    )

    ax.text(
        scalebar_origin[0],
        scalebar_origin[1],
        scalebar_origin[2] + 2,
        "5 mm",
        color="black",
        fontsize=SCALEBAR_FONT_SIZE,
        fontweight="bold"
    )


def add_rank_labels_standard(
    ax,
    region_masks,
    original_shape,
    elev=18,
    azim=-65,
    offset=20
):
    """
    Place labels in front of the structures by shifting them
    toward the camera direction.
    """

    azim_rad = np.deg2rad(azim)
    elev_rad = np.deg2rad(elev)

    view_vector = np.array([
        np.cos(elev_rad) * np.cos(azim_rad),
        np.cos(elev_rad) * np.sin(azim_rad),
        np.sin(elev_rad)
    ])

    for rank, mask in enumerate(region_masks, start=1):

        coords = np.argwhere(mask)

        if coords.size == 0:
            continue

        center = coords.mean(axis=0)
        center_v = transform_point_standard(center, original_shape)

        label_pos = center_v - offset * view_vector

        ax.text(
            label_pos[0],
            label_pos[1],
            label_pos[2],
            str(rank),
            fontsize=24,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
            zorder=1000,
            bbox=dict(
                boxstyle="circle,pad=0.25",
                fc="none",
                ec="none",
                lw=3,
                alpha=0.75
            )
        )
def add_rank_labels_coronal(ax, region_masks, offset_y=-10):
    for rank, mask in enumerate(region_masks, start=1):
        coords = np.argwhere(mask)

        if coords.size == 0:
            continue

        center = coords.mean(axis=0)

        ax.text(
            center[0],
            center[1] + offset_y,   # brings label forward in coronal view
            center[2],
            str(rank),
            fontsize=24,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
            zorder=1000,
            bbox=dict(
                boxstyle="circle,pad=0.25",
                fc="none",
                ec="none",
                lw=3,
                alpha=0.75
            )
        )


def plot_regions_3d(
    brain_volume,
    region_masks,
    region_colors,
    save_path,
    view_elevation=18,
    view_azimuth=-65,
    include_scalebar=True,
    include_rank_labels=True
):
    brain_v = orient_standard(brain_volume)
    masks_v = [orient_standard(mask) for mask in region_masks]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    brain_mesh = surface_mesh(brain_v, threshold=0, color="gray", alpha=0.07)
    if brain_mesh is not None:
        ax.add_collection3d(brain_mesh)

    for mask, color in zip(masks_v, region_colors):
        mesh = surface_mesh(mask, threshold=0, color=color, alpha=0.40)
        if mesh is not None:
            ax.add_collection3d(mesh)

    setup_3d_axis(ax, brain_v)
    ax.view_init(elev=view_elevation, azim=view_azimuth)

    if include_rank_labels:
        add_rank_labels_standard(
            ax,
            region_masks,
            brain_volume.shape,
            elev=view_elevation,
            azim=view_azimuth,
            offset=12
        )

    if include_scalebar:
        add_scalebar_standard(ax)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save_figure(fig, save_path)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_alpha(0)

    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor((1, 1, 1, 0))

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.set_axis_off()

def plot_stroke_3d(
    brain_volume,
    stroke_mask,
    save_path,
    view_elevation=18,
    view_azimuth=-65,
    include_scalebar=True
):
    brain_v = orient_standard(brain_volume)
    stroke_v = orient_standard(stroke_mask)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    brain_mesh = surface_mesh(brain_v, threshold=0, color="gray", alpha=0.07)
    if brain_mesh is not None:
        ax.add_collection3d(brain_mesh)

    stroke_mesh = surface_mesh(stroke_v, threshold=0, color=STROKE_COLOR, alpha=0.50)
    if stroke_mesh is not None:
        ax.add_collection3d(stroke_mesh)

    setup_3d_axis(ax, brain_v)
    ax.view_init(elev=view_elevation, azim=view_azimuth)

    if include_scalebar:
        add_scalebar_standard(ax)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save_figure(fig, save_path)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_alpha(0)

    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor((1, 1, 1, 0))

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.set_axis_off()


def plot_regions_coronal(
    brain_volume,
    region_masks,
    region_colors,
    save_path,
    include_scalebar=True,
    include_rank_labels=True
):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    brain_mesh = surface_mesh(brain_volume, threshold=0, color="gray", alpha=0.07)
    if brain_mesh is not None:
        ax.add_collection3d(brain_mesh)

    for mask, color in zip(region_masks, region_colors):
        mesh = surface_mesh(mask, threshold=0, color=color, alpha=0.40)
        if mesh is not None:
            ax.add_collection3d(mesh)

    setup_3d_axis(ax, brain_volume)
    ax.view_init(elev=0, azim=-90)

    if include_rank_labels:
        add_rank_labels_coronal(ax, region_masks)

    if include_scalebar:
        add_scalebar_coronal(ax)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save_figure(fig, save_path)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_alpha(0)

    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor((1, 1, 1, 0))

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.set_axis_off()

def plot_stroke_coronal(
    brain_volume,
    stroke_mask,
    save_path,
    include_scalebar=True
):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    brain_mesh = surface_mesh(brain_volume, threshold=0, color="gray", alpha=0.07)
    if brain_mesh is not None:
        ax.add_collection3d(brain_mesh)

    stroke_mesh = surface_mesh(stroke_mask, threshold=0, color=STROKE_COLOR, alpha=0.50)
    if stroke_mesh is not None:
        ax.add_collection3d(stroke_mesh)

    setup_3d_axis(ax, brain_volume)
    ax.view_init(elev=0, azim=-90)

    if include_scalebar:
        add_scalebar_coronal(ax)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save_figure(fig, save_path)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_alpha(0)

    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor((1, 1, 1, 0))

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.set_axis_off()

def save_function_legend(save_path):
    legend_fig, legend_ax = plt.subplots(figsize=(12, 1.6))

    legend_elements = [
        Line2D([0], [0], color=brain_function_color["Motor"],
               lw=13, label="Motor"),

        Line2D([0], [0], color=brain_function_color["Sensory"],
               lw=13, label="Sensory"),

        Line2D([0], [0], color=brain_function_color["Multimodal"],
               lw=13, label="Multimodal"),

        Line2D([0], [0], color=brain_function_color["Limbic/Cognitive"],
               lw=13, label="Limbic/Cognitive"),

        Line2D([0], [0], color=STROKE_COLOR,
               lw=13, label="Stroke")
    ]

    legend_ax.legend(
        handles=legend_elements,
        ncol=5,
        loc="center",
        fontsize=16,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.8
    )
    legend_ax.axis("off")
    save_figure(legend_fig, save_path)
    plt.close(legend_fig)


def add_alpha(rgb_color, alpha=0.4):
    return (*rgb_color, alpha)


def save_ranked_region_table(rank_df, save_path):
    table_df = rank_df[["rank", "region", "modality", "function"]].copy()
    table_df.columns = ["Rank", "Region", "MRI", "Function"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.08, 0.48, 0.14, 0.30]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(TABLE_FONT_SIZE)
    table.scale(1.0, 1.55)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("none")
        cell.set_linewidth(0)

        if row == 0:
            cell.set_text_props(weight="bold", color="black")
            cell.set_facecolor((0.94, 0.94, 0.94, 0.6))
        else:
            cell.set_text_props(color="black")

            if col == 3:
                function = table_df.iloc[row - 1]["Function"]
                cell.set_facecolor(add_alpha(brain_function_color[function], 0.4))
            else:
                cell.set_facecolor((1, 1, 1, 0))

    save_figure(fig, save_path)
    plt.close(fig)


def stroke_overlap_check(atlas_data, stroke_mask, regions_to_check):
    print("\nStroke overlap check:")
    for base_region in regions_to_check:
        contra = f"{base_region} contralesional"
        ipsi = f"{base_region} ipsilesional"

        if contra not in brain_region or ipsi not in brain_region:
            continue

        contra_mask = atlas_data == brain_region[contra]
        ipsi_mask = atlas_data == brain_region[ipsi]

        contra_overlap = int(np.sum(stroke_mask & contra_mask))
        ipsi_overlap = int(np.sum(stroke_mask & ipsi_mask))

        print(f"{base_region}: contra overlap={contra_overlap}, ipsi overlap={ipsi_overlap}")


# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading atlas:", ATLAS_PATH)
atlas_data = nib.load(ATLAS_PATH).get_fdata()

print("Loading brain mask:", BRAIN_MASK_PATH)
brain_data = nib.load(BRAIN_MASK_PATH).get_fdata()

print("Loading stroke map:", STROKE_PATH)
stroke_data = nib.load(STROKE_PATH).get_fdata()
stroke_mask = stroke_data >= STROKE_THRESHOLD

print("Atlas shape:", atlas_data.shape)
print("Brain mask shape:", brain_data.shape)
print("Stroke map shape:", stroke_data.shape)

print("Loading feature importance:", FEATURE_CSV)
df = pd.read_csv(FEATURE_CSV)


# =============================================================================
# EXTRACT TOP UNIQUE REGIONS
# =============================================================================

raw_features = df["feature"].tolist()
raw_regions = [feature.split("_")[0] for feature in raw_features]
raw_modalities = [feature.split("_")[-1] for feature in raw_features]

top_rows = []
seen_regions = set()

for feature, region, modality in zip(raw_features, raw_regions, raw_modalities):
    if region in brain_region and region not in seen_regions:
        seen_regions.add(region)
        top_rows.append({
            "region": region,
            "modality": modality,
            "function": brain_region_functions[region],
            "atlas_label": brain_region[region],
            "source_feature": feature
        })

    if len(top_rows) >= TOP_N_REGIONS:
        break

if len(top_rows) == 0:
    raise RuntimeError("No regions from feature importance file matched the atlas dictionary.")

top_df = pd.DataFrame(top_rows)
top_df.insert(0, "rank", np.arange(1, len(top_df) + 1))

print(f"\nTop {len(top_df)} unique anatomical regions:")
for _, row in top_df.iterrows():
    print(
        f"{row['rank']}. {row['region']} | "
        f"{row['modality']} | {row['function']} | "
        f"label={row['atlas_label']}"
    )

region_masks = [atlas_data == row["atlas_label"] for _, row in top_df.iterrows()]
region_colors = [brain_function_color[row["function"]] for _, row in top_df.iterrows()]

top_df.to_csv(OUT_DIR / f"{TEST}_top{TOP_N_REGIONS}_ranked_visualized_regions.csv", index=False)


# =============================================================================
# OPTIONAL DIAGNOSTIC: IPSI/CONTRA CHECK
# =============================================================================

stroke_overlap_check(
    atlas_data,
    stroke_mask,
    [
        "Frontal cortex",
        "Somatosensory cortex",
        "Insular cortex",
        "Amygdala",
        "Motor cortex",
        "Putamen",
        "Internal capsule",
        "Parietal cortex"
    ]
)


# =============================================================================
# SAVE VIEWS
# =============================================================================

views = {
    "standard": (18, -65),
    "dorsolateral": (28, -55),
    "ipsi_to_contra_lateral": (12, -78),
    "posterior_oblique": (20, -120),
    "contra_to_ipsi_lateral":(12,78),#or (12,78)
}

for view_name, (elev, azim) in views.items():
    plot_regions_3d(
        brain_data,
        region_masks,
        region_colors,
        save_path=OUT_DIR / f"{TEST}_top{TOP_N_REGIONS}_ranked_regions_3D_{view_name}",
        view_elevation=elev,
        view_azimuth=azim,
        include_scalebar=True,
        include_rank_labels=True
    )

    plot_stroke_3d(
        brain_data,
        stroke_mask,
        save_path=OUT_DIR / f"{TEST}_stroke_3D_{view_name}",
        view_elevation=elev,
        view_azimuth=azim,
        include_scalebar=True
    )

plot_regions_coronal(
    brain_data,
    region_masks,
    region_colors,
    save_path=OUT_DIR / f"{TEST}_top{TOP_N_REGIONS}_ranked_regions_coronal",
    include_scalebar=True,
    include_rank_labels=True
)

plot_stroke_coronal(
    brain_data,
    stroke_mask,
    save_path=OUT_DIR / f"{TEST}_stroke_coronal",
    include_scalebar=True
)

save_function_legend(OUT_DIR / f"{TEST}_function_legend")
save_ranked_region_table(top_df, OUT_DIR / f"{TEST}_ranked_region_table")

print("\nDone. Files saved to:")
print(OUT_DIR)