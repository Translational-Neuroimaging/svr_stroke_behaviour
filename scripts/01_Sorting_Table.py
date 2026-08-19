import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent

amb_path = ROOT / "Ambroxol_group_24h_Median.csv"
ctl_path = ROOT / "Control_group_24h_Median.csv"
treatment_path = ROOT / "Treatment_2.xlsx"

reference_path = ROOT / "CSV/Input/Normalization/Median_neu_norm4_Beam_Grip.csv"
out_path = ROOT / "CSV/Input/Normalization/Median_raw_foldnorm_Beam_Grip_matched324.csv"

def rename_feature(col):
    col = str(col)

    # Lesion is right-sided
    col = col.replace(" left_", " contralesional_")
    col = col.replace(" right_", " ipsilesional_")
    col = col.replace("_left_", "_contralesional_")
    col = col.replace("_right_", "_ipsilesional_")
    col = col.replace(" left ", " contralesional ")
    col = col.replace(" right ", " ipsilesional ")

    return col

# Load files
amb = pd.read_csv(amb_path, sep=None, engine="python", encoding="utf-8-sig")
ctl = pd.read_csv(ctl_path, sep=None, engine="python", encoding="utf-8-sig")
# Force first column in each file to be AnimalID
amb = amb.rename(columns={amb.columns[0]: "AnimalID"})
ctl = ctl.rename(columns={ctl.columns[0]: "AnimalID"})

raw = pd.concat([amb, ctl], ignore_index=True)

print("Raw merged shape:", raw.shape)

raw["AnimalID"] = (
    raw["AnimalID"]
    .astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.replace("\xa0", "", regex=False)
    .str.replace(" ", "", regex=False)
    .str.strip()
)
print("Amb columns:")
print(amb.columns[:10].tolist())
print(amb.head(3))

print("Ctl columns:")
print(ctl.columns[:10].tolist())
print(ctl.head(3))


if "AnimalID" not in raw.columns:
    raise RuntimeError("AnimalID column not found in merged raw CSV files.")

raw["AnimalID"] = raw["AnimalID"].astype(str).str.strip()
# Force-remove invisible characters and normalize IDs
raw["AnimalID"] = (
    raw["AnimalID"]
    .astype(str)
    .str.replace("\ufeff", "", regex=False)   # BOM
    .str.replace("\xa0", "", regex=False)     # non-breaking space
    .str.replace(" ", "", regex=False)
    .str.strip()
)
# Rename feature columns
raw = raw.rename(columns={c: rename_feature(c) for c in raw.columns if c != "AnimalID"})

# Remove fMRI and voxel columns
drop_cols = [
    c for c in raw.columns
    if "fmri" in c.lower()
]
raw = raw.drop(columns=drop_cols, errors="ignore")

print("Raw after removing fMRI/Voxels:", raw.shape)

# Load reference feature names
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

if len(matched) != 324:
    raise RuntimeError("Feature matching failed. Check missing_features.csv and extra_features.csv")

raw = raw[["AnimalID"] + matched]

# Load treatment file
treat = pd.read_excel(treatment_path)

if "AnimalID" not in treat.columns:
    raise RuntimeError("AnimalID column not found in Treatment_2.xlsx.")

# Clean treatment IDs AFTER loading treat
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

print("Raw AnimalID repr examples:")
print([repr(x) for x in raw["AnimalID"].head(20).tolist()])

print("Treatment AnimalID repr examples:")
print([repr(x) for x in wanted["AnimalID"].head(20).tolist()])

print("Overlap:", len(set(wanted["AnimalID"]) & set(raw["AnimalID"])))
print("Missing from raw:")
print(sorted(set(wanted["AnimalID"]) - set(raw["AnimalID"])))

# Merge animals
final = wanted.merge(raw, on="AnimalID", how="left")

missing_animals = final.loc[
    final[matched].isna().all(axis=1),
    "AnimalID"
].tolist()

print("Missing animals:", len(missing_animals))

if missing_animals:
    print(missing_animals)
    raise RuntimeError("Animal matching failed.")

print("Final table shape:", final.shape)

final.to_csv(out_path, index=False)

print("\nSUCCESS")
print("Saved to:")
print(out_path)