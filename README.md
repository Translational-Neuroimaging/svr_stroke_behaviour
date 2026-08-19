# Imaging-Based Prediction of Post-Stroke Behavioral Outcomes

Code accompanying the manuscript **Lesion topology outperforms regional MRI signal
intensity for predicting Beam Walk performance
after experimental stroke in rats**. This repository contains the analysis pipeline used to predict post-stroke behavioral recovery from regional MRI features and lesion burden in a rat model of ischemic stroke, including support vector regression, permutation-based feature importance, target-permutation validation, and 3D anatomical visualization.

## Table of contents

- [Overview](#overview)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Outputs](#outputs)
- [Interpretation notes](#interpretation-notes)

## Overview

Animals were assessed on three behavioral tests after stroke — Beam Walk, Sticky Label, and Grip Strength — and imaged with multi-contrast MRI (T2w, ADC, FA, AD, RD, trace) across 54 anatomical regions. This pipeline:

1. Builds regional MRI feature and lesion burden tables from raw data.
2. Trains support vector regression (SVR) models, with repeated nested cross-validation, to predict behavioral scores from imaging features.
3. Evaluates model significance with target-permutation testing.
4. Identifies the most predictive anatomical regions via permutation-based feature importance and projects them onto a 3D rat brain atlas.
5. Compares regional imaging features against global lesion volume and against regional lesion burden as alternative predictors.
6. Visualizes feature-behavior relationships as correlation distributions and high/low-outcome polar maps.

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

## Installation

```bash
git clone https://github.com/Translational-Neuroimaging/svr_stroke_behaviour.git
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires Python ≥ 3.9. Core dependencies:

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

## Data

### Tabular data — `CSV/Input/`

| File | Description |
|---|---|
| `raw_behavior_and_mri_features.csv` | One row per animal: animal ID, Beam Walk score, Sticky Label score, Grip Strength score, treatment/cohort label, and regional MRI features (54 regions × 6 contrasts: T2w, ADC, FA, AD, RD, trace) |
| `regional_lesion_burden.csv` | One row per animal, one column per anatomical region; values represent percentage of the region affected by ischemic injury |
| `treatment_group_table.csv` | Treatment assignment (Ambroxol-treated vs. vehicle/control) |

### Atlas data — `Atlas_neu/`

| File | Used for |
|---|---|
| `rat_Atlas12.nii` | 3D anatomical rendering |
| `full_mask_new.nii` | 3D anatomical rendering |
| `joint_probability_map_GMM.nii.gz` | 3D anatomical rendering |

Atlas files are only required to reproduce the 3D anatomical figure (step 5, below). All other analyses run from the tabular data alone.

## Usage

Run the scripts in order from the repository root:

```bash
python scripts/01_Sorting_Table.py
python scripts/02_SVM_raw_foldnorm_FINAL_CombinedGrip_Beam_Sticky.py
python scripts/03_SVM_Beam_raw_foldnorm_IMPORTANCE.py
python scripts/04_hyperparameters_best.py
python scripts/05_3D3_new_extended.py
python scripts/06_SVM_Beam_FINAL_target_permutation_validation_CV_norm.py
python scripts/07_Sorting_Table_w_SV.py
python scripts/02_SVM_raw_foldnorm_FINAL_CombinedGrip_Beam_Sticky.py   # rerun on step 7 output
python scripts/08_SVM_Beam_raw_foldnorm_IMPORTANCE_SV.py
python scripts/09_SVM_TargetPermutation_Batch.py
python scripts/10_SVM_Final_Cohort_FeatureSet_Analysis.py
python scripts/11_Regression_Distribution_behavior.py
python scripts/12_Polarmaps.py
```

| Step | Script | Purpose |
|---|---|---|
| 1 | `01_Sorting_Table.py` | Builds the regional MRI feature and behavior table used throughout the pipeline |
| 2 | `02_SVM_raw_foldnorm_FINAL_CombinedGrip_Beam_Sticky.py` | Repeated nested cross-validation for Beam Walk, Sticky Label, and Grip Strength, with fold-wise normalization and inner-loop hyperparameter tuning |
| 3 | `03_SVM_Beam_raw_foldnorm_IMPORTANCE.py` | Permutation-based feature importance for Beam Walk using regional MRI features |
| 4 | `04_hyperparameters_best.py` | Summarizes selected SVR hyperparameters across cross-validation folds |
| 5 | `05_3D3_new_extended.py` | Projects the top predictive regions onto the rat brain atlas for 3D visualization |
| 6 | `06_SVM_Beam_FINAL_target_permutation_validation_CV_norm.py` | Target-permutation validation for Beam Walk; builds empirical null distributions for MSE and Pearson r |
| 7 | `07_Sorting_Table_w_SV.py` | Builds a feature table combining regional MRI features with global lesion volume |
| 8 | *(rerun of step 2 on step 7's output)* | Tests whether adding global lesion volume improves prediction accuracy |
| 9 | `08_SVM_Beam_raw_foldnorm_IMPORTANCE_SV.py` | Checks whether global lesion volume ranks among the top predictors |
| 10 | `09_SVM_TargetPermutation_Batch.py` | Runs target-permutation tests across feature sets, cohorts, and behavioral outcomes |
| 11 | `10_SVM_Final_Cohort_FeatureSet_Analysis.py` | Consolidates results across feature representations (regional MRI, MRI + lesion volume, lesion burden) and cohorts (all animals, control-only) |
| 12 | `11_Regression_Distribution_behavior.py` | Computes Bonferroni-corrected, feature-wise Pearson correlations against behavioral outcomes |
| 13 | `12_Polarmaps.py` | Splits animals into high/low Beam Walk groups by median score and visualizes top predictors as polar maps |

## Outputs

**Main text**
- Figure 2 — Regional MRI feature prediction
- Figure 3 — Regional lesion burden prediction
- Figure 4 — 3D anatomical visualization
- Figure 5 — Feature-wise Pearson correlations
- Figure 6 — High/low Beam Walk polar maps
- Main performance table and top Beam Walk feature table

**Supplement**
- Table S1 / Figure S1 — Hyperparameter selection and performance distributions
- Table S2 / Figure S2 — Target-permutation results and null distributions

## Interpretation notes

Regional lesion burden is a constrained feature representation: regions unaffected by infarction have zero or near-zero values and contribute little information to the model. This analysis therefore tests the behavioral relevance of lesion **topology**, rather than serving as a direct substitute for whole-brain MRI signal features.

Regional MRI features and regional lesion burden are complementary — lesion burden identifies *where* injury occurred, while MRI features characterize regional tissue signal alterations both within and outside the infarct territory.
