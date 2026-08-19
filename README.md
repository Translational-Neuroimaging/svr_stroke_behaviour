# Reproducibility Guide: Experimental Stroke Prediction Analysis

This repository contains the final code, data structure, and analysis order used for the revised manuscript analyses. The original code base includes many exploratory scripts; **only the final scripts listed below should be included** in the public repository, along with the final tabular input data and atlas files.

## Table of contents

- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Input data](#input-data)
- [Analysis pipeline](#analysis-pipeline)
- [Manuscript outputs](#manuscript-outputs)
- [Interpretation notes](#interpretation-notes)
- [Open items before publishing](#open-items-before-publishing)

## Repository structure

```text
Imaging-Neuroscience-main/
├── README.md
├── requirements.txt
├── scripts/
│   ├── 01_Sorting_Table.py
│   ├── 02_SVM_raw_foldnorm_FINAL_CombinedGrip_Beam_Sticky.py
│   ├── 03_SVM_Beam_raw_foldnorm_IMPORTANCE.py
│   ├── 04_hyperparameters_best.py
│   ├── 05_3D3_new_extended.py
│   ├── 06_SVM_Beam_FINAL_target_permutation_validation_CV_norm.py
│   ├── 07_Sorting_Table_w_SV.py
│   ├── 08_SVM_Beam_raw_foldnorm_IMPORTANCE_SV.py
│   ├── 09_SVM_TargetPermutation_Batch.py
│   ├── 10_SVM_Final_Cohort_FeatureSet_Analysis.py
│   ├── 11_Regression_Distribution_behavior.py
│   └── 12_Polarmaps.py
├── CSV/
│   ├── Input/
│   ├── Output/
│   └── Visual/
├── Atlas_neu/
└── Figures/
```

Scripts are renamed with numerical prefixes to reflect the run order (original filenames are noted in the [pipeline table](#analysis-pipeline) below). Do **not** upload `venv/`, temporary output folders, duplicate exploratory scripts, or old test scripts unless explicitly marked as archival.

## Requirements

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Minimum packages:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
nibabel
openpyxl
scikit-image
joblib
```

## Input data

Reproduction requires only tabular raw data and atlas files — no intermediate outputs.

### Tabular files — `CSV/Input/`

| File | Contents |
|---|---|
| `raw_behavior_and_mri_features.csv` | One row per animal: animal ID, Beam Walk score, Sticky Label score, Grip Strength score, treatment/cohort label, and regional MRI features from 54 anatomical regions × 6 contrasts (T2w, ADC, FA, AD, RD, trace) |
| `regional_lesion_burden.csv` | One row per animal, one column per anatomical region, values = percentage of region affected by ischemic injury |
| `treatment_group_table.csv` | Identifies Ambroxol-treated vs. vehicle/control animals |

### Atlas files — `Atlas_neu/`

| File | Used for |
|---|---|
| `rat_Atlas12.nii` | 3D anatomical rendering |
| `full_mask_new.nii` | 3D anatomical rendering |
| `joint_probability_map_GMM.nii.gz` | 3D anatomical rendering |

Atlas files are only required for the 3D anatomical visualizations (step 5).

## Analysis pipeline

Run the scripts in this order.

| # | Script | Original filename | Purpose | Manuscript use |
|---|---|---|---|---|
| 1 | `01_Sorting_Table.py` | `Sorting_Table.py` | Organizes raw regional MRI features and behavioral scores into the final feature table | Feeds regional MRI prediction, feature importance, correlations, polar maps |
| 2 | `02_SVM_raw_foldnorm_FINAL_CombinedGrip_Beam_Sticky.py` | `SVM_raw_foldnorm_FINAL_CombinedGrip_Beam_Sticky.py` | Repeated nested CV for Beam Walk, Sticky Label, and Grip Strength with fold-wise normalization and inner-loop hyperparameter tuning | Figure 2, main performance table |
| 3 | `03_SVM_Beam_raw_foldnorm_IMPORTANCE.py` | `SVM_Beam_raw_foldnorm_IMPORTANCE.py` | Permutation-based feature importance for Beam Walk (regional MRI features) | Top feature table, 3D visualization, polar maps |
| 4 | `04_hyperparameters_best.py` | `hyperparameters_best.py` | Summarizes most-frequently-selected SVR hyperparameters from nested CV logs | Table S1, Figure S1 |
| 5 | `05_3D3_new_extended.py` | `3D3_new_extended.py` | Projects top anatomical regions into the rat atlas using top feature-importance outputs | Main 3D anatomical figure (Figure 4) |
| 6 | `06_SVM_Beam_FINAL_target_permutation_validation_CV_norm.py` | `SVM_Beam_FINAL_target_permutation_validation_CV_norm.py` | Target-permutation validation for Beam Walk; empirical null distributions for MSE and Pearson r | Table S2, Figure S2 |
| 7 | `07_Sorting_Table_w_SV.py` | `Sorting_Table_w_SV.py` | Builds a feature table with regional MRI features + global stroke volume | Input for MRI + global lesion volume model |
| 8 | `02_SVM_raw_foldnorm_FINAL_CombinedGrip_Beam_Sticky.py` *(rerun on step 7 output)* | — | Tests whether adding global lesion volume improves prediction | Main comparison table; statement that global volume did not improve prediction |
| 9 | `08_SVM_Beam_raw_foldnorm_IMPORTANCE_SV.py` | `SVM_Beam_raw_foldnorm_IMPORTANCE_SV.py` | Checks whether global lesion volume ranks among top predictors | Statement that global volume ranked low in importance |
| 10 | `09_SVM_TargetPermutation_Batch.py` | `SVM_TargetPermutation_Batch.py` | Batch target-permutation tests across feature sets, cohorts, and outcomes | Supplementary permutation table/figure |
| 11 | `10_SVM_Final_Cohort_FeatureSet_Analysis.py` | `SVM_Final_Cohort_FeatureSet_Analysis.py` | Consolidates results across feature sets (regional MRI, MRI + volume, lesion burden) and cohorts (all animals, control-only) | Final main performance table |
| 12 | `11_Regression_Distribution_behavior.py` | `Regression_Distribution_behavior.py` | Feature-wise Pearson correlations vs. behavioral outcomes, Bonferroni-corrected | Correlation figure (Figure 5), optional supplementary tables |
| 13 | `12_Polarmaps.py` | `Polarmaps.py` | Splits animals into high/low Beam Walk groups (median split), plots normalized top-predictor values as polar maps | High/low Beam Walk phenotype figure (Figure 6) |

**Notes on specific steps:**
- **Step 3**: also used for shorter exploratory runs (e.g., 100 iterations) to check stability — confirm and document the iteration count used for the final manuscript outputs.
- **Step 5** requires `rat_Atlas12.nii`, `full_mask_new.nii`, `joint_probability_map_GMM.nii.gz`, plus the top feature tables from the regional MRI and regional lesion burden workflows.

### Example usage

```bash
python scripts/01_Sorting_Table.py
python scripts/02_SVM_raw_foldnorm_FINAL_CombinedGrip_Beam_Sticky.py
python scripts/03_SVM_Beam_raw_foldnorm_IMPORTANCE.py
# ...continue in numbered order through scripts/12_Polarmaps.py
```

## Manuscript outputs

**Main text**
- Figure 2 — Regional MRI feature prediction
- Figure 3 — Regional lesion burden prediction
- Figure 4 — 3D anatomical visualization
- Figure 5 — Feature-wise Pearson correlations
- Figure 6 — High/low Beam Walk polar maps
- Main performance table
- Top Beam Walk feature table

**Supplement**
- Table S1 — Hyperparameter selections
- Figure S1 — Hyperparameter performance distributions
- Table S2 — Target-permutation results
- Figure S2 — Permutation null distributions

## Interpretation notes

Regional lesion burden is a constrained feature representation: regions unaffected by infarction have zero or near-zero values and contribute little information to the model. This analysis should be read as testing the behavioral relevance of lesion **topology**, not as a substitute for whole-brain MRI signal features.

Regional MRI features and regional lesion burden are complementary — lesion burden identifies *where* injury occurred, while MRI features characterize regional tissue signal alterations both within and outside the infarct territory.

## Open items before publishing

- [ ] Confirm exact final filenames of the raw input tables
- [ ] Confirm exact column names (animal ID, Beam Walk, Sticky Label, Grip Strength, treatment group, global lesion volume)
- [ ] Confirm whether higher Beam Walk scores indicate worse or better performance
- [ ] Confirm exact naming format of MRI feature columns
- [ ] Confirm exact naming format of regional lesion burden columns
- [ ] Document final random seeds for nested CV and permutation testing
- [ ] Document final repetition counts (main nested CV, feature importance, target permutation)
- [ ] Document exact SVR hyperparameter grid
- [ ] Confirm all scripts perform fold-wise normalization strictly within training folds
- [ ] Confirm which treatment label corresponds to the control-only cohort
- [ ] Confirm atlas region names exactly match feature table column names
- [ ] Document approximate runtime for 1000-repeat models and permutation testing
