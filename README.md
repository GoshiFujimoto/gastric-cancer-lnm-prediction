# gastric-cancer-lnm-prediction
ML-based lymph node metastasis prediction in older gastric cancer patients (TRIPOD+AI)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20376901.svg)](https://doi.org/10.5281/zenodo.20376901)

Analysis code for:
Fujimoto G, Kusanagi H. Machine learning-based prediction of lymph node metastasis for individualized surgical decision-making in older patients with gastric cancer: a retrospective simulation study compliant with TRIPOD+AI. *Ann Surg Oncol* (under revision).

## Overview
- Six algorithms (logistic regression, random forest, XGBoost, LightGBM, CatBoost, multilayer perceptron) predicting pathological lymph node metastasis (pN+) in patients aged ≥ 70 years.
- Internal cohort (1995–2020) split 80:20 into training and test sets; temporal validation cohort (2020–2025).
- Reduced-surgery simulation (thresholds 5%, 10%, 20%) and decision curve analysis in the internal test set and the temporal cohort.
- Bootstrap 95% CIs (B = 1,000), calibration (Brier score, observed-to-expected ratio, Hosmer–Lemeshow test), SHAP for all six models, and age/sex subgroup analysis.

## Usage
```bash
pip install -r requirements.txt
python lymph_node_prediction.py
```
Place the data file at `data/gastric_cancer_lnm.xlsx` (sheets `1995-2020` and `2020-2025`). Figures are saved to `output/`.

Sensitivity analysis using preoperatively available variables only: set `PREOP_ONLY = True` in the configuration section (results are saved to `output/preop_only/`).

## Data availability
Patient-level data are not publicly available because of ethical restrictions; they are available from the corresponding author upon reasonable request.

## Reproducibility
Results were obtained with Python 3.13 and the package versions in `requirements.txt` (random seed 42). XGBoost results may differ slightly with other XGBoost versions.

## Version history
- **v1.1.0** (revision): simulation and decision curve analysis evaluated in the internal test set; calibration ratio reported as O/E; preoperative-only sensitivity analysis; manuscript figures generated directly by the script; SHAP for all six models; pinned package versions; fixed a formatting error in the simulation output.
- **v1.0.0**: initial release.

## License
MIT
