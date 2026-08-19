import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent

amb_path = ROOT / "Ambroxol_group_24h_Median.csv"
ctl_path = ROOT / "Control_group_24h_Median.csv"
treatment_path = ROOT / "Treatment_2.xlsx"

reference_path = ROOT / "CSV/Input/Normalization/Median_neu_norm4_Beam_Grip.csv"

out_path_324 = ROOT / "CSV/Input/Normalization/Median_raw_foldnorm_Beam_Grip_matched324.csv"
out_path_325 = ROOT / "CSV/Input/Normalization/Median_raw_foldnorm_Beam_Grip_matched324_plusVoxels.csv"


def rename_feature(col):
    col = str(col)

    col = col.replace(" left_", " contralesional_")
    col = col.replace(" right_", " ipsilesional_")
    col = col.replace("_left_", "_contralesional_")
    col = col.replace("_right_", "_ipsilesional_")
    col = col.replace(" left ", " contralesional ")
    col = col.replace(" right ", " ipsilesional ")

    return col


# =============================================================================
# LOAD RAW GROUP FILES
# =============================================================================

amb = pd.read_csv(amb_path, sep=None, engine="python", encoding="utf-8-sig")
ctl = pd.read_csv(ctl_path, sep=None, engine="python", encoding="utf-8-sig")

amb = amb.rename(columns={amb.columns[0]: "AnimalID"})
ctl = ctl.rename(columns={ctl.columns[0]: "AnimalID"})

raw = pd.concat([amb, ctl], ignore_index=True)

raw["AnimalID"] = (
    raw["AnimalID"]
    .astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.replace("\xa0", "", regex=False)
    .str.replace(" ", "", regex=False)
    .str.strip()
)

raw = raw.rename(columns={c: rename_feature(c) for c in raw.columns if c != "AnimalID"})

print("Raw merged shape:", raw.shape)


# =============================================================================
# IDENTIFY VOXEL / STROKE VOLUME COLUMN BEFORE FILTERING
# =============================================================================

voxel_cols = [
    c for c in raw.columns
    if "voxel" in c.lower()
    or "voxels" in c.lower()
    or "strokevolume" in c.lower()
    or "stroke_volume" in c.lower()
    or "lesionvolume" in c.lower()
    or "lesion_volume" in c.lower()
]

print("Voxel / stroke-volume candidate columns:")
print(voxel_cols)

if len(voxel_cols) == 0:
    raise RuntimeError("No voxel/stroke-volume column found in raw table.")

if len(voxel_cols) > 1:
    print("WARNING: More than one voxel/stroke-volume column found.")
    print("Using the first one:", voxel_cols[0])

voxel_col = voxel_cols[0]


# =============================================================================
# REMOVE ONLY fMRI COLUMNS, KEEP VOXELS
# =============================================================================

drop_cols = [
    c for c in raw.columns
    if "fmri" in c.lower()
]

raw = raw.drop(columns=drop_cols, errors="ignore")

print("Raw after removing fMRI only:", raw.shape)


# =============================================================================
# LOAD REFERENCE 324 FEATURE LIST
# =============================================================================

ref = pd.read_csv(reference_path)

ref_features = [
    c for c in ref.columns
    if c not in ["Brainregion", "AnimalID"]
]

print("Reference features:", len(ref_features))

matched = [c for c in ref_features if c in raw.columns]
missing = [c for c in ref_features if c not in raw.columns]
extra = [c for c in raw.columns if c not in matched and c != "AnimalID"]

print("Matched features:", len(matched))
print("Missing features:", len(missing))
print("Extra raw features:", len(extra))

pd.DataFrame({"missing_features": missing}).to_csv(ROOT / "missing_features.csv", index=False)
pd.DataFrame({"extra_features": extra}).to_csv(ROOT / "extra_features.csv", index=False)


# =============================================================================
# MAKE 324 AND 325 RAW TABLES
# =============================================================================
if len(matched) not in [324, 325]:
    raise RuntimeError("Feature matching failed. Check missing_features.csv and extra_features.csv")

matched_no_voxels = [c for c in matched if c != voxel_col]

print("Matched without voxels:", len(matched_no_voxels))
print("Matched with voxels:", len(matched))

if len(matched_no_voxels) != 324:
    raise RuntimeError("Expected 324 non-voxel MRI features.")

if len(matched) != 325:
    raise RuntimeError("Expected 325 features including Voxels.")

raw_324 = raw[["AnimalID"] + matched_no_voxels].copy()
raw_325 = raw[["AnimalID"] + matched].copy()

raw_325 = raw_325.rename(columns={voxel_col: "StrokeVolume_Voxels"})

# =============================================================================
# LOAD TREATMENT FILE AND ORDER ANIMALS
# =============================================================================

treat = pd.read_excel(treatment_path)

if "AnimalID" not in treat.columns:
    raise RuntimeError("AnimalID column not found in Treatment_2.xlsx.")

treat["AnimalID"] = (
    treat["AnimalID"]
    .astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.replace("\xa0", "", regex=False)
    .str.replace(" ", "", regex=False)
    .str.strip()
)

wanted = treat[["AnimalID"]].drop_duplicates()

print("Animals requested:", len(wanted))
print("Overlap:", len(set(wanted["AnimalID"]) & set(raw["AnimalID"])))
print("Missing from raw:")
print(sorted(set(wanted["AnimalID"]) - set(raw["AnimalID"])))


# =============================================================================
# MERGE FINAL TABLES
# =============================================================================

final_324 = wanted.merge(raw_324, on="AnimalID", how="left")
final_325 = wanted.merge(raw_325, on="AnimalID", how="left")

missing_animals_324 = final_324.loc[
    final_324[matched_no_voxels].isna().all(axis=1),
    "AnimalID"
].tolist()

missing_animals_325 = final_325.loc[
    final_325[matched_no_voxels].isna().all(axis=1),
    "AnimalID"
].tolist()

if missing_animals_324 or missing_animals_325:
    print("Missing animals 324:", missing_animals_324)
    print("Missing animals 325:", missing_animals_325)
    raise RuntimeError("Animal matching failed.")

print("Final 324 table shape:", final_324.shape)
print("Final 325 table shape:", final_325.shape)

final_324.to_csv(out_path_324, index=False)
final_325.to_csv(out_path_325, index=False)

print("\nSUCCESS")
print("Saved 324-feature table to:")
print(out_path_324)
print("Saved 325-feature + voxel table to:")
print(out_path_325)