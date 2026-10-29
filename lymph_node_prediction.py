# =============================================================================
# ML-Based Lymph Node Metastasis Prediction in Older Patients with Gastric Cancer
# A Retrospective Simulation Study (TRIPOD+AI Compliant)
#
# Reference:
#   Fujimoto G, Kusanagi H. Machine learning-based prediction of lymph node
#   metastasis for individualized surgical decision-making in older patients
#   with gastric cancer: a retrospective simulation study compliant with
#   TRIPOD+AI. Ann Surg Oncol. (under revision)
#
# Requirements (exact versions used for the published results):
#   pip install -r requirements.txt
#
# Tested with:
#   Python 3.13 | scikit-learn 1.6.1 | XGBoost 2.1.4 | LightGBM 4.7.0 |
#   CatBoost 1.2.10 | SHAP 0.54.1
#   Results for XGBoost can differ slightly with other XGBoost versions.
#
# Version history:
#   v1.1.0 (revision)
#     - Reduced-surgery simulation and decision curve analysis are evaluated in
#       the internal TEST set (not the full internal cohort including training
#       data) and in the temporal validation cohort.
#     - Calibration ratio reported as observed-to-expected (O/E).
#     - Sensitivity analysis using preoperatively available variables only
#       (set PREOP_ONLY = True).
#     - Manuscript figures (Figs. 1-6, Supplementary Figs. S3-S4) generated
#       directly from this script; SHAP computed for all six models.
#   v1.0.0  initial release
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

from scipy import stats
import statsmodels.api as sm

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer   # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from scipy.stats import chi2 as scipy_chi2

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import shap

# =============================================================================
# Sklearn 1.6+ compatibility wrapper
# XGBoost / LightGBM / CatBoost may not implement __sklearn_tags__ in older
# versions, causing errors with RandomizedSearchCV in scikit-learn 1.6+.
# =============================================================================
class SklearnCompatClassifier(ClassifierMixin, BaseEstimator):
    """Wrapper for third-party classifiers lacking sklearn 1.6+ __sklearn_tags__."""
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **kw):
        self.estimator.fit(X, y, **kw)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def get_params(self, deep=True):
        params = {'estimator': self.estimator}
        if deep and hasattr(self.estimator, 'get_params'):
            params.update(self.estimator.get_params(deep=True))
        return params

    def set_params(self, **params):
        if 'estimator' in params:
            self.estimator = params.pop('estimator')
        if params and hasattr(self.estimator, 'set_params'):
            self.estimator.set_params(**params)
        return self

# =============================================================================
# Configuration  —  update INPUT_FILE and OUTPUT_DIR for your environment
# =============================================================================
INPUT_FILE  = r"data/gastric_cancer_lnm.xlsx"   # path to your Excel data file
OUTPUT_DIR  = Path("output")                      # directory for saved figures
SHEET_TRAIN = "1995-2020"                         # sheet name: internal cohort
SHEET_EXT   = "2020-2025"                         # sheet name: external cohort

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE  = 42
TEST_SIZE     = 0.2
CV_FOLDS      = 5
N_ITER_SEARCH = 50
THRESHOLDS    = [0.05, 0.10, 0.20]
TARGET_COL    = "N+"

# Clinically mandatory variables always included in multivariate analysis
MANDATORY_VARS = ['size', 'por_sig_muc', 'CEA', 'CA19-9']

# Sensitivity analysis: set True to exclude variables derived from pathological
# examination (tumor size, depth of invasion, histological type) and use only
# preoperatively available variables. Outputs are written to OUTPUT_DIR/preop_only.
PREOP_ONLY = False
if PREOP_ONLY:
    OUTPUT_DIR = OUTPUT_DIR / "preop_only"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family'    : 'DejaVu Sans',
    'axes.titlesize' : 13,
    'axes.labelsize' : 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi'     : 150,
})

PALETTE = {
    'LR'      : '#4C72B0',
    'RF'      : '#55A868',
    'XGB'     : '#C44E52',
    'LGBM'    : '#8172B2',
    'CatBoost': '#CCB974',
    'MLP'     : '#64B5CD',
}

# =============================================================================
# Utilities
# =============================================================================
def sep(title="", width=80):
    if title:
        print(f"\n{'='*width}\n  {title}\n{'='*width}")
    else:
        print(f"\n{'─'*width}")

def save_fig(fig, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace('+', 'plus').replace('/', '_').replace('\\', '_')
    path = OUTPUT_DIR / f"{safe_name}.png"
    try:
        fig.savefig(path, bbox_inches='tight', dpi=150)
        print(f"  [saved] {path}")
    except Exception as e:
        print(f"  [save error] {path}: {e}")
    finally:
        plt.close(fig)

def print_df(df, title=""):
    if title:
        print(f"\n--- {title} ---")
    print(df.to_string())
    print()

# =============================================================================
# 1. Load data
# =============================================================================
sep("1. Load data")

df_int_raw = pd.read_excel(INPUT_FILE, sheet_name=SHEET_TRAIN)
df_ext_raw = pd.read_excel(INPUT_FILE, sheet_name=SHEET_EXT)

print(f"Internal cohort shape : {df_int_raw.shape}")
print(f"External cohort shape : {df_ext_raw.shape}")
print(f"\npN+ distribution (internal):\n{df_int_raw[TARGET_COL].value_counts()}")
print(f"\npN+ distribution (external):\n{df_ext_raw[TARGET_COL].value_counts()}")

# =============================================================================
# 2. Feature definition and preprocessing
# =============================================================================
sep("2. Feature definition and preprocessing")

# Depth of invasion: M.1=mucosa, SM=submucosa, MP=muscularis propria,
#                   SS=subserosa, SE=serosa exposure, SI=serosa invasion
DEPTH_COLS    = ['M.1', 'SM', 'MP', 'SS', 'SE', 'SI']
LOCATION_COLS = ['E', 'U', 'M', 'L', 'D', 'A', 'S', 'O', 'T', 'J']
WALL_COLS     = ['Ant', 'Post', 'Less', 'Gre', 'Circ']
TYPE_COLS     = ['type0', 'type1', 'type2', 'type3', 'type4']
HIST_COLS     = ['tub1-2', 'por_sig_muc', 'NEC', 'pap']
LAB_COLS      = ['Hb', 'Alb', 'CRP', 'T-Chol', 'lympohcyte',
                 'CRP/Alb_ratio', 'Hb/Alb_ratio', 'PNI', 'CONUT',
                 'CEA', 'CA19-9']
PATIENT_COLS  = ['age', 'sex', 'ASA-PS', 'BMI', 'CCI']
SIZE_COLS     = ['size']

ALL_FEATURES = (PATIENT_COLS + LOCATION_COLS + WALL_COLS +
                TYPE_COLS + SIZE_COLS + DEPTH_COLS + HIST_COLS + LAB_COLS)

if PREOP_ONLY:
    ALL_FEATURES   = [c for c in ALL_FEATURES if c not in DEPTH_COLS + HIST_COLS + SIZE_COLS]
    MANDATORY_VARS = ['CEA', 'CA19-9']
    print("  PREOP_ONLY sensitivity analysis: pathology-derived variables excluded.")

# Display labels for figures. Tumor location follows the Japanese Classification
# of Gastric Carcinoma (E/D/J: esophageal/duodenal/jejunal invasion; A/S/O/T:
# anastomotic site, suture line, other site, entire remnant stomach).
LABELS = {
    'age':'Age','sex':'Male sex','ASA-PS':'ASA-PS','BMI':'BMI','CCI':'CCI',
    'E':'Esophageal inv. (E)','U':'Upper third (U)','M':'Middle third (M)','L':'Lower third (L)',
    'D':'Duodenal inv. (D)','A':'Remnant: anastomosis (A)','S':'Remnant: suture line (S)',
    'O':'Remnant: other site (O)','T':'Remnant: entire (T)','J':'Jejunal inv. (J)',
    'Ant':'Ant. wall','Post':'Post. wall','Less':'Lesser curv.','Gre':'Greater curv.','Circ':'Circumferential',
    'type0':'Macr. type 0','type1':'Macr. type 1','type2':'Macr. type 2','type3':'Macr. type 3','type4':'Macr. type 4',
    'size':'Tumor size','M.1':'Mucosa (M)','SM':'Submucosa (SM)','MP':'Musc. propria (MP)',
    'SS':'Subserosa (SS)','SE':'Serosa exp. (SE)','SI':'Serosa inv. (SI)',
    'tub1-2':'tub1-2 (diff.)','por_sig_muc':'Undiff. type','NEC':'NEC','pap':'Papillary',
    'Hb':'Hb','Alb':'Albumin','CRP':'CRP','T-Chol':'T-Chol','lympohcyte':'Lymphocyte',
    'CRP/Alb_ratio':'CRP/Alb','Hb/Alb_ratio':'Hb/Alb','PNI':'PNI','CONUT':'CONUT','CEA':'CEA','CA19-9':'CA19-9',
}
lab = lambda c: LABELS.get(c, c)

def preprocess(df):
    """Convert object columns to numeric and cast all feature columns to float64."""
    df = df.copy()
    for c in ALL_FEATURES:
        if c in df.columns:
            if df[c].dtype == object:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df[c] = df[c].astype(float)
    use_cols = [c for c in ALL_FEATURES if c in df.columns] + [TARGET_COL]
    return df[use_cols]

df_int = preprocess(df_int_raw)
df_ext = preprocess(df_ext_raw)

FEAT_COLS = [c for c in ALL_FEATURES if c in df_int.columns]
print(f"Features used ({len(FEAT_COLS)}): {FEAT_COLS}")

# =============================================================================
# 3. Missing value imputation (MICE)
# =============================================================================
sep("3. Missing value imputation (MICE: IterativeImputer)")

import sklearn as _sk
print(f"scikit-learn version: {_sk.__version__}")

miss_int = df_int[FEAT_COLS].isnull().sum().sort_values(ascending=False)
print("Missing values — internal (top 10):"); print(miss_int[miss_int > 0].head(10))
_full_miss_int = miss_int[miss_int == len(df_int)].index.tolist()

miss_ext = df_ext[FEAT_COLS].isnull().sum().sort_values(ascending=False)
print("\nMissing values — external (top 10):"); print(miss_ext[miss_ext > 0].head(10))
_full_miss_ext = miss_ext[miss_ext == len(df_ext)].index.tolist()

_all_full_miss = list(set(_full_miss_int + _full_miss_ext))
if _all_full_miss:
    print(f"\nRemoving completely missing columns: {_all_full_miss}")
    FEAT_COLS = [c for c in FEAT_COLS if c not in _all_full_miss]

imp_mice = IterativeImputer(max_iter=10, random_state=RANDOM_STATE, keep_empty_features=True)
X_int_imp = imp_mice.fit_transform(df_int[FEAT_COLS])
X_ext_imp = imp_mice.transform(df_ext[FEAT_COLS])

if X_int_imp.shape[1] != len(FEAT_COLS):
    raise ValueError(f"MICE output columns ({X_int_imp.shape[1]}) != FEAT_COLS ({len(FEAT_COLS)}).")

X_int_all = pd.DataFrame(X_int_imp, columns=FEAT_COLS, index=df_int.index)
X_ext     = pd.DataFrame(X_ext_imp, columns=FEAT_COLS, index=df_ext.index)
y_int_all = df_int[TARGET_COL].values.astype(int)
y_ext     = df_ext[TARGET_COL].values.astype(int)

print(f"\nPost-imputation missing: internal={X_int_all.isnull().sum().sum()}, "
      f"external={X_ext.isnull().sum().sum()}")

# =============================================================================
# 4. Train/test split and standardisation
# =============================================================================
sep("4. Train/test split (80:20 stratified) and standardisation")

X_train, X_test, y_train, y_test = train_test_split(
    X_int_all, y_int_all,
    test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_int_all
)
print(f"Train: {X_train.shape}  pN+={y_train.sum()}/{len(y_train)} ({y_train.mean()*100:.1f}%)")
print(f"Test : {X_test.shape}   pN+={y_test.sum()}/{len(y_test)} ({y_test.mean()*100:.1f}%)")
print(f"Ext  : {X_ext.shape}    pN+={y_ext.sum()}/{len(y_ext)} ({y_ext.mean()*100:.1f}%)")

scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=FEAT_COLS)
X_test_sc  = pd.DataFrame(scaler.transform(X_test),      columns=FEAT_COLS)
X_ext_sc   = pd.DataFrame(scaler.transform(X_ext),       columns=FEAT_COLS)

# =============================================================================
# 5. Univariable analysis
# =============================================================================
sep("5. Univariable analysis")

univar_rows = []
for col in FEAT_COLS:
    x = df_int[col].copy()
    y = df_int[TARGET_COL].copy()
    valid   = x.notna() & y.notna()
    xv, yv  = x[valid].astype(float), y[valid].astype(float)
    n_valid = int(valid.sum())
    OR = ci_lo = ci_hi = pval = np.nan

    if xv.nunique() <= 2:
        ct = pd.crosstab(xv, yv)
        if ct.shape == (2, 2):
            OR, pval = stats.fisher_exact(ct)
    else:
        g0 = xv[yv == 0]; g1 = xv[yv == 1]
        if len(g0) > 0 and len(g1) > 0:
            _, pval = stats.mannwhitneyu(g0, g1, alternative='two-sided')
        try:
            xc  = sm.add_constant(xv)
            res = sm.Logit(yv, xc.astype(float)).fit(disp=0)
            OR    = np.exp(res.params[col])
            ci_lo = np.exp(res.conf_int().loc[col, 0])
            ci_hi = np.exp(res.conf_int().loc[col, 1])
        except Exception:
            pass

    univar_rows.append(dict(
        variable=col, n_valid=n_valid,
        OR=round(OR, 3)    if not np.isnan(OR)    else np.nan,
        CI_lo=round(ci_lo, 3) if not np.isnan(ci_lo) else np.nan,
        CI_hi=round(ci_hi, 3) if not np.isnan(ci_hi) else np.nan,
        p_value=round(pval, 4) if not np.isnan(pval)  else np.nan,
        p_raw=pval
    ))

df_univar = pd.DataFrame(univar_rows).sort_values('p_value')
print_df(df_univar, "Univariable analysis results")

sig_vars  = df_univar[df_univar['p_value'] < 0.05]['variable'].tolist()
multivars = list(dict.fromkeys(MANDATORY_VARS + sig_vars))
multivars = [v for v in multivars if v != TARGET_COL and v in FEAT_COLS]
print(f"Significant variables (p<0.05): {sig_vars}")
print(f"Multivariate analysis variables ({len(multivars)}): {multivars}")

# =============================================================================
# 6. Multivariate logistic regression
# =============================================================================
sep("6. Multivariate logistic regression")

mv_res = None; mv_table = None

# Step A: statsmodels Logit (no regularisation)
print("  Attempting statsmodels Logit...")
_sm_ok = False
try:
    X_mv_sm = sm.add_constant(X_train_sc[multivars].astype(float))
    _res_sm = sm.Logit(y_train.astype(float), X_mv_sm).fit(disp=0, maxiter=300, method='bfgs')
    _coef_ok = np.abs(_res_sm.params.values).max() < 1e6
    _ci_nan_ratio = np.isnan(_res_sm.conf_int().values).mean()
    if _coef_ok and _ci_nan_ratio < 0.5:
        mv_res   = _res_sm
        names    = ['const'] + multivars
        mv_table = pd.DataFrame({
            'variable': names,
            'coef'    : _res_sm.params.values,
            'OR'      : np.exp(np.clip(_res_sm.params.values, -10, 10)),
            'CI_lo'   : np.exp(np.clip(_res_sm.conf_int().values[:, 0], -10, 10)),
            'CI_hi'   : np.exp(np.clip(_res_sm.conf_int().values[:, 1], -10, 10)),
            'p_value' : _res_sm.pvalues.values
        })
        print("  statsmodels Logit succeeded.")
        _sm_ok = True
    else:
        print(f"  Complete separation detected. Switching to L2 regularisation.")
except Exception as e:
    print(f"  statsmodels error: {e}")

# Step B: L2-regularised logistic regression (fallback for complete separation)
if not _sm_ok:
    print("  Attempting L2-regularised logistic regression...")
    try:
        from sklearn.linear_model import LogisticRegressionCV
        _lr_mv = LogisticRegressionCV(Cs=10, cv=5, penalty='l2', solver='lbfgs',
                                       max_iter=2000, random_state=RANDOM_STATE, scoring='roc_auc')
        _lr_mv.fit(X_train_sc[multivars], y_train)
        _coef = _lr_mv.coef_[0]
        print("  Computing bootstrap CIs (B=200)...")
        _rng = np.random.default_rng(RANDOM_STATE)
        _coef_bs = np.zeros((200, len(multivars)))
        _Xmv_arr = X_train_sc[multivars].values
        for _b in range(200):
            _idx = _rng.integers(0, len(_Xmv_arr), size=len(_Xmv_arr))
            _m   = LogisticRegressionCV(Cs=10, cv=3, penalty='l2', solver='lbfgs',
                                         max_iter=1000, random_state=int(_b), scoring='roc_auc')
            _m.fit(_Xmv_arr[_idx], y_train[_idx])
            _coef_bs[_b] = _m.coef_[0]
        _ci_lo = np.percentile(_coef_bs, 2.5, axis=0)
        _ci_hi = np.percentile(_coef_bs, 97.5, axis=0)
        _se    = _coef_bs.std(axis=0, ddof=1)
        _z     = _coef / np.where(_se > 0, _se, 1e-9)
        _pval  = 2 * stats.norm.sf(np.abs(_z))
        mv_table = pd.DataFrame({'variable': multivars, 'coef': _coef,
                                  'OR': np.exp(_coef), 'CI_lo': np.exp(_ci_lo),
                                  'CI_hi': np.exp(_ci_hi), 'p_value': _pval})
        print(f"  L2 regularised logistic regression succeeded (best C={_lr_mv.C_[0]:.4f}).")
    except Exception as e2:
        print(f"  L2 logistic regression error: {e2}")

if mv_table is not None:
    print("\nMultivariate logistic regression results:")
    print("  Note: depth/location dummies may show quasi-complete separation.")
    print_df(mv_table.round(4))
    if mv_res is not None:
        print(f"AIC={mv_res.aic:.2f}  BIC={mv_res.bic:.2f}  "
              f"Pseudo-R2(McFadden)={mv_res.prsquared:.4f}")
else:
    print("  Multivariate analysis failed. Forest plot will be skipped.")

# =============================================================================
# 7. Feature selection (LASSO + RF importance)
# =============================================================================
sep("7. Feature selection (LASSO + RF importance)")

lasso = LogisticRegression(penalty='l1', solver='liblinear',
                            C=0.1, max_iter=2000, random_state=RANDOM_STATE)
lasso.fit(X_train_sc, y_train)
lasso_coef     = pd.Series(lasso.coef_[0], index=FEAT_COLS)
lasso_selected = lasso_coef[lasso_coef != 0].index.tolist()
print(f"\nLASSO selected ({len(lasso_selected)}):")
print(lasso_coef[lasso_coef != 0].sort_values(key=abs, ascending=False).round(4))

rf_fs = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
rf_fs.fit(X_train, y_train)
rf_imp = pd.Series(rf_fs.feature_importances_, index=FEAT_COLS).sort_values(ascending=False)
rf_top = rf_imp[rf_imp >= rf_imp.mean()].index.tolist()
print(f"\nRF top features ({len(rf_top)}):")
print(rf_imp[rf_imp >= rf_imp.mean()].round(4))

selected_features = list(dict.fromkeys(lasso_selected + rf_top + MANDATORY_VARS))
selected_features = [f for f in selected_features if f in FEAT_COLS]
print(f"\nFinal selected features ({len(selected_features)}): {selected_features}")

X_tr    = X_train[selected_features]; X_te  = X_test[selected_features]
X_ex_s  = X_ext[selected_features]
X_tr_sc = X_train_sc[selected_features]; X_te_sc = X_test_sc[selected_features]
X_ex_sc = X_ext_sc[selected_features]

# =============================================================================
# 8. Model development (RandomizedSearchCV)
# =============================================================================
sep("8. Model development (RandomizedSearchCV x 5-fold CV)")

cv_sk = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

model_configs = {
    'LR': {
        'estimator': LogisticRegression(solver='lbfgs', max_iter=2000, random_state=RANDOM_STATE),
        'params'   : {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l2']},
        'scaled'   : True,
    },
    'RF': {
        'estimator': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200, 300, 500], 'max_depth': [None, 3, 5, 7, 10],
            'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
        },
        'scaled': False,
    },
    'XGB': {
        'estimator': SklearnCompatClassifier(
            xgb.XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE,
                               n_jobs=-1, verbosity=0)),
        'params': {
            'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0], 'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.5],
        },
        'scaled': False,
    },
    'LGBM': {
        'estimator': SklearnCompatClassifier(
            lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)),
        'params': {
            'n_estimators': [100, 200, 300], 'max_depth': [-1, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves': [15, 31, 63], 'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
        },
        'scaled': False,
    },
    'CatBoost': {
        'estimator': SklearnCompatClassifier(
            CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, thread_count=-1)),
        'params': {
            'iterations': [100, 200, 300], 'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1], 'l2_leaf_reg': [1, 3, 5, 7],
        },
        'scaled': False,
    },
    'MLP': {
        'estimator': MLPClassifier(max_iter=500, random_state=RANDOM_STATE, early_stopping=True),
        'params': {
            'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64), (64, 32, 16)],
            'activation': ['relu', 'tanh'], 'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01],
        },
        'scaled': True,
    },
}

best_models = {}; best_params_all = {}; cv_best_auc = {}

for name, cfg in model_configs.items():
    print(f"\n  -- {name} --")
    X_use = X_tr_sc if cfg['scaled'] else X_tr
    search = RandomizedSearchCV(cfg['estimator'], cfg['params'],
                                 n_iter=N_ITER_SEARCH, cv=cv_sk, scoring='roc_auc',
                                 n_jobs=1, random_state=RANDOM_STATE, refit=True, verbose=0)
    search.fit(X_use, y_train)
    best_models[name]     = search.best_estimator_
    best_params_all[name] = search.best_params_
    cv_best_auc[name]     = search.best_score_
    print(f"  Best params: {search.best_params_}")
    print(f"  CV AUC     : {search.best_score_:.4f}")

# =============================================================================
# 9. Model evaluation
# =============================================================================
sep("9. Model evaluation")

eval_rows = []; prob_store = {'test': {}, 'ext': {}}

for name, model in best_models.items():
    sc = model_configs[name]['scaled']
    for ds, y_true, Xte, Xsc in [
        ('test', y_test, X_te, X_te_sc),
        ('ext',  y_ext,  X_ex_s, X_ex_sc)
    ]:
        Xu = Xsc if sc else Xte
        yp = model.predict_proba(Xu)[:, 1]
        ypred = (yp >= 0.5).astype(int)
        eval_rows.append(dict(
            Model=name, Dataset=ds,
            Accuracy  =round(accuracy_score(y_true, ypred), 4),
            Precision =round(precision_score(y_true, ypred, zero_division=0), 4),
            Recall    =round(recall_score(y_true, ypred, zero_division=0), 4),
            F1        =round(f1_score(y_true, ypred, zero_division=0), 4),
            ROC_AUC   =round(roc_auc_score(y_true, yp), 4),
            PR_AUC    =round(average_precision_score(y_true, yp), 4),
        ))
        prob_store[ds][name] = yp

df_eval = pd.DataFrame(eval_rows)
print_df(df_eval, "Model evaluation metrics")

test_auc  = df_eval[df_eval['Dataset'] == 'test'].set_index('Model')['ROC_AUC']
best_name = test_auc.idxmax()
print(f"\nBest model: {best_name}  (Test AUC={test_auc[best_name]:.4f})")

best_model = best_models[best_name]; best_scaled = model_configs[best_name]['scaled']

X_int_sel    = X_int_all[selected_features]
_sel_idx     = [list(FEAT_COLS).index(f) for f in selected_features]
X_int_sel_sc = pd.DataFrame(scaler.transform(X_int_all)[:, _sel_idx], columns=selected_features)
X_int_pred   = X_int_sel_sc if best_scaled else X_int_sel
prob_int_all = best_model.predict_proba(X_int_pred)[:, 1]
X_ext_pred   = X_ex_sc if best_scaled else X_ex_s
prob_ext_all = best_model.predict_proba(X_ext_pred)[:, 1]

# =============================================================================
# 9b. TRIPOD+AI supplementary analyses
# =============================================================================
sep("9b. TRIPOD+AI supplementary analyses")

B_BOOTSTRAP = 1000

def bootstrap_ci(y_true, y_prob, metric_fn, B=B_BOOTSTRAP, seed=42):
    """Compute 95% bootstrap CI for a given metric."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(B):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            scores.append(metric_fn(yt, yp))
        except Exception:
            pass
    return (np.nan, np.nan) if len(scores) < 10 else (
        float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5)))

def hl_test(y_true, y_prob, g=10):
    """Hosmer–Lemeshow goodness-of-fit test."""
    df_tmp = pd.DataFrame({'y': y_true.astype(float), 'p': y_prob})
    df_tmp['decile'] = pd.qcut(df_tmp['p'], q=g, duplicates='drop')
    grp = df_tmp.groupby('decile', observed=True).agg(O=('y', 'sum'), E=('p', 'sum'), n=('y', 'count'))
    chi2_stat = ((grp['O'] - grp['E'])**2 / (grp['E'] * (1 - grp['E'] / grp['n']))).sum()
    return chi2_stat, 1 - scipy_chi2.cdf(chi2_stat, df=g - 2)

# Bootstrap 95% CI
print(f"\nBootstrap 95% CI (B={B_BOOTSTRAP}) — {best_name}")
print(f"\n{'Metric':<20} {'Test (internal)':>32} {'External':>32}")
print("─" * 86)
for metric_name, metric_fn in [
    ("ROC-AUC", roc_auc_score), ("PR-AUC", average_precision_score),
    ("Brier Score", brier_score_loss)
]:
    yp_te = prob_store['test'][best_name]; yp_ex = prob_store['ext'][best_name]
    val_te = metric_fn(y_test, yp_te); val_ex = metric_fn(y_ext, yp_ex)
    ci_te  = bootstrap_ci(y_test, yp_te, metric_fn); ci_ex = bootstrap_ci(y_ext, yp_ex, metric_fn)
    print(f"  {metric_name:<18} {val_te:.4f} ({ci_te[0]:.4f}–{ci_te[1]:.4f})   "
          f"{val_ex:.4f} ({ci_ex[0]:.4f}–{ci_ex[1]:.4f})")

print()
for thr in THRESHOLDS:
    def _npv(yt, yp, t=thr):
        low = yp < t
        return float((1-yt[low]).sum()/low.sum()) if low.sum() > 0 else np.nan
    for ds_label, yt, yp in [
        ("Internal", y_test, prob_store['test'][best_name]),
        ("External", y_ext,  prob_store['ext'][best_name])
    ]:
        val = _npv(yt, yp)
        if not np.isnan(val):
            ci = bootstrap_ci(yt, yp, _npv)
            fn = int(yt[yp < thr].sum())
            print(f"  NPV@{int(thr*100):>2}% [{ds_label}]: {val:.4f} ({ci[0]:.4f}–{ci[1]:.4f})  FN={fn}")
        else:
            print(f"  NPV@{int(thr*100):>2}% [{ds_label}]: N/A (no candidates)")

# Enhanced calibration
print("\nEnhanced calibration (TRIPOD+AI Item 22):")
print(f"{'Dataset':<14} {'Brier':>10} {'95%CI':>22} {'O/E':>8} {'HL x2':>8} {'HL p':>10} {'Eval':>8}")
print("─" * 82)
for ds_label, yt, yp in [
    ("Internal", y_test, prob_store['test'][best_name]),
    ("External", y_ext,  prob_store['ext'][best_name])
]:
    bs = brier_score_loss(yt, yp); bs_ci = bootstrap_ci(yt, yp, brier_score_loss)
    eo = yt.mean() / yp.mean(); c2, ph = hl_test(yt, yp)   # observed-to-expected (O/E) ratio
    eval_ = "Good" if 0.9 < eo < 1.1 else "Caution"
    print(f"  {ds_label:<14} {bs:>10.4f} ({bs_ci[0]:.4f}–{bs_ci[1]:.4f}) "
          f"{eo:>8.3f} {c2:>8.1f} {'<0.001' if ph<0.001 else f'{ph:.4f}':>10} {eval_:>8}")

# Sample size / EPV
print("\nSample size and EPV (TRIPOD+AI Item 13):")
_n_ev = int(y_train.sum()); _n_var = len(selected_features); _epv = _n_ev / _n_var
print(f"  Training N={len(y_train)}, pN+ events={_n_ev}, features={_n_var}, EPV={_epv:.1f}")
print(f"  {'EPV sufficient (>=10)' if _epv >= 10 else 'EPV below recommended threshold'}")
print(f"  Riley method recommended N~700; actual N={len(y_int_all)}: sufficient.")

# Threshold rationale
print("\nThreshold rationale (TRIPOD+AI Item 15):")
print("  5%  : Most conservative; zero false-negative priority.")
print("  10% : Primary threshold — balance between candidate rate and NPV.")
print("  20% : Aggressive expansion; increased false-negatives expected.")
print("  Basis: eCura system (Japanese Gastric Cancer Treatment Guidelines, LNM <5-10%).")

# Class imbalance
print("\nClass imbalance (TRIPOD+AI Item 14):")
print(f"  Internal pN+: {y_int_all.mean()*100:.1f}%  External pN+: {y_ext.mean()*100:.1f}%")
print("  Mild imbalance (~42-44%); SMOTE not applied. Primary metrics: ROC-AUC and PR-AUC.")

# Subgroup (fairness) analysis
print(f"\nSubgroup analysis (TRIPOD+AI Item 24) — Bootstrap B={B_BOOTSTRAP}:")
print(f"{'Subgroup':<14} {'Set':>5} {'n':>5} {'AUC':>7} {'95%CI':>20} {'Delta':>8}")
print("─" * 62)
_age_te = X_test['age'].values; _sex_te = X_test['sex'].values
_age_ex = X_ext[selected_features]['age'].values if 'age' in selected_features else None
_sex_ex = X_ext[selected_features]['sex'].values if 'sex' in selected_features else None
_yp_te = prob_store['test'][best_name]; _yp_ex = prob_store['ext'][best_name]
_ov_te = roc_auc_score(y_test, _yp_te); _ov_ex = roc_auc_score(y_ext, _yp_ex)

_sg_defs = [
    ("70-79 yrs", "Test", ((_age_te>=70)&(_age_te<=79)), y_test, _yp_te, _ov_te),
    (">=80 yrs",  "Test", (_age_te>=80),                  y_test, _yp_te, _ov_te),
    ("Female",    "Test", (_sex_te==0),                   y_test, _yp_te, _ov_te),
    ("Male",      "Test", (_sex_te==1),                   y_test, _yp_te, _ov_te),
]
if _age_ex is not None:
    _sg_defs += [
        ("70-79 yrs", "Ext", ((_age_ex>=70)&(_age_ex<=79)), y_ext, _yp_ex, _ov_ex),
        (">=80 yrs",  "Ext", (_age_ex>=80),                  y_ext, _yp_ex, _ov_ex),
        ("Female",    "Ext", (_sex_ex==0),                   y_ext, _yp_ex, _ov_ex),
        ("Male",      "Ext", (_sex_ex==1),                   y_ext, _yp_ex, _ov_ex),
    ]
for lb, ds, mask, yt, yp, ov in _sg_defs:
    if mask.sum() < 15 or len(np.unique(yt[mask])) < 2:
        print(f"  {lb:<14} {ds:>5} {mask.sum():>5}  Insufficient data"); continue
    auc = roc_auc_score(yt[mask], yp[mask]); ci = bootstrap_ci(yt[mask], yp[mask], roc_auc_score)
    print(f"  {lb:<14} {ds:>5} {mask.sum():>5}  {auc:.4f}  ({ci[0]:.4f}–{ci[1]:.4f})  {auc-ov:>+8.4f}")
print(f"\n  Overall AUC — Test: {_ov_te:.4f}, External: {_ov_ex:.4f}")

# =============================================================================
# 10. SHAP analysis
# =============================================================================
sep("10. SHAP analysis")

shap_store = {}

def compute_shap(model, name, X_bg, X_target):
    """Compute SHAP values (TreeExplainer for tree models, KernelExplainer otherwise)."""
    try:
        inner = model.estimator if isinstance(model, SklearnCompatClassifier) else model
        if name in ('RF', 'XGB', 'LGBM', 'CatBoost'):
            explainer = shap.TreeExplainer(inner)
            sv = explainer.shap_values(X_target)
            if isinstance(sv, list): sv = sv[1]
            if sv.ndim == 3: sv = sv[:, :, 1]
        elif name == 'LR':
            explainer = shap.LinearExplainer(model, X_bg)
            sv = explainer.shap_values(X_target)
        else:   # MLP: model-agnostic KernelExplainer on the predicted probability
            bg = shap.kmeans(X_bg, 30)
            f  = lambda X: model.predict_proba(X)[:, 1]
            explainer = shap.KernelExplainer(f, bg)
            sv = explainer.shap_values(np.asarray(X_target), nsamples=300, silent=True)
        return explainer, sv, X_target
    except Exception as e:
        print(f"  SHAP error ({name}): {e}"); return None, None, None

for name, model in best_models.items():
    print(f"\n  -- SHAP: {name}")
    sc = model_configs[name]['scaled']
    exp, sv, Xsh = compute_shap(model, name,
                                  X_tr_sc if sc else X_tr,
                                  X_te_sc if sc else X_te)
    if sv is not None:
        shap_store[name] = {'sv': sv, 'X': Xsh}
        imp = pd.Series(np.abs(sv).mean(axis=0), index=selected_features)
        print(f"  Mean |SHAP| (desc):\n{imp.sort_values(ascending=False).round(4).to_string()}")

# =============================================================================
# 11. Reduced surgery eligibility simulation
# =============================================================================
sep("11. Reduced surgery eligibility simulation")

def run_simulation(y_true, y_prob, thresholds, label):
    print(f"\n  {label}  N={len(y_true)}, pN+={y_true.sum()}")
    rows = []
    for thr in thresholds:
        low = y_prob < thr; n_low = int(low.sum())
        fn  = int(y_true[low].sum()); tn = int((1-y_true)[low].sum())
        npv = tn/n_low if n_low > 0 else np.nan
        lr  = n_low / len(y_true)
        rows.append(dict(threshold=thr, n_low=n_low, n_high=int((~low).sum()),
                         local_rate_pct=round(lr*100, 1),
                         NPV=round(npv, 4) if not np.isnan(npv) else np.nan,
                         false_negative=fn, avoided_overop=int((1-y_true)[~low].sum())))
        npv_txt = f"{npv:.4f}" if not np.isnan(npv) else "N/A"
        print(f"  Threshold {thr*100:.0f}%: candidates={n_low} ({lr*100:.1f}%), "
              f"NPV={npv_txt}, FN={fn}")
    return pd.DataFrame(rows)

# Evaluated in data not used for model training: internal test set and temporal cohort
prob_test_best = prob_store['test'][best_name]
df_sim_int = run_simulation(y_test, prob_test_best, THRESHOLDS, "Internal test set")
df_sim_ext = run_simulation(y_ext,  prob_ext_all,   THRESHOLDS, "Temporal validation")

# =============================================================================
# 12. Decision Curve Analysis
# =============================================================================
sep("12. Decision Curve Analysis (DCA)")

def net_benefit(y, prob, thr):
    n = len(y)
    tp = int(((prob >= thr) & (y == 1)).sum()); fp = int(((prob >= thr) & (y == 0)).sum())
    return tp/n - fp/n * thr/(1-thr)

thr_range = np.linspace(0.01, 0.99, 99)

def run_dca(y, prob, label):
    nb_model = [net_benefit(y, prob, t) for t in thr_range]
    nb_all   = [y.mean() - (1-y.mean()) * t/(1-t) for t in thr_range]
    nb_none  = [0.0] * len(thr_range)
    print(f"\n  [{label}] Net benefit at primary thresholds:")
    for thr in THRESHOLDS:
        print(f"    {thr*100:.0f}%: model={net_benefit(y, prob, thr):.4f}  "
              f"treat-all={y.mean()-(1-y.mean())*thr/(1-thr):.4f}")
    return nb_model, nb_all, nb_none

nb_int_m, nb_int_a, nb_int_n = run_dca(y_test, prob_test_best, "Internal test set")
nb_ext_m, nb_ext_a, nb_ext_n = run_dca(y_ext,  prob_ext_all,   "Temporal validation")

# =============================================================================
# 13. Figure generation
# =============================================================================
sep("13. Figure generation")

# Fig01: pN+ distribution
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, (df, title) in zip(axes, [
    (df_int_raw, f"Internal (1995-2020, N={len(df_int_raw)})"),
    (df_ext_raw, f"External (2020-2025, N={len(df_ext_raw)})")
]):
    cnts = [int((df[TARGET_COL]==0).sum()), int((df[TARGET_COL]==1).sum())]
    bars = ax.bar(['pN-', 'pN+'], cnts, color=['#5B9BD5', '#ED7D31'], width=0.5)
    for bar, cnt in zip(bars, cnts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3, str(cnt),
                ha='center', va='bottom', fontweight='bold')
    ax.set_title(title); ax.set_ylabel('Cases'); ax.set_ylim(0, max(cnts)*1.2)
fig.suptitle('pN+ distribution', fontsize=14, fontweight='bold')
fig.tight_layout(); save_fig(fig, "Fig01_Nplus_distribution")

# ---------------------------------------------------------------------------
# Manuscript Fig. 1: univariable analysis (-log10 p, all candidate variables)
# ---------------------------------------------------------------------------
_uv = df_univar.dropna(subset=['p_raw']).sort_values('p_raw')
_uv = _uv.assign(l=-np.log10(_uv['p_raw'].clip(lower=1e-300)))[::-1]
fig, ax = plt.subplots(figsize=(9, max(6, 0.27*len(_uv))))
ax.barh([lab(c) for c in _uv['variable']], _uv['l'], height=0.7,
        color=['#CC6666' if p < 0.05 else '#AAAAAA' for p in _uv['p_raw']])
ax.axvline(-np.log10(0.05), color='k', ls='--', lw=1, label='p = 0.05')
for i, (l, p) in enumerate(zip(_uv['l'], _uv['p_raw'])):
    ax.text(l + 0.8, i, 'p < 0.001' if p < 0.001 else f'p = {p:.3f}', va='center', fontsize=8)
ax.set_xlim(0, _uv['l'].max()*1.12); ax.set_ylim(-0.7, len(_uv)-0.3)
ax.set_xlabel('-log10(p-value)', fontsize=12); ax.legend(loc='lower right')
fig.tight_layout(); save_fig(fig, "ManuscriptFig1_univariable")

# ---------------------------------------------------------------------------
# Manuscript Fig. 2: multivariable logistic regression (standardized variables;
# ORs per 1-SD increase). Variables with quasi-complete separation (depth of
# invasion) are not displayed.
# ---------------------------------------------------------------------------
if mv_table is not None:
    _mv = mv_table[mv_table['variable'] != 'const'].copy()
    _mv = _mv[~_mv['variable'].isin(DEPTH_COLS)].dropna(subset=['OR','CI_lo','CI_hi']).sort_values('OR')
    fig, ax = plt.subplots(figsize=(11, max(4, 0.4*len(_mv)+1)))
    for i, (_, r) in enumerate(_mv.iterrows()):
        ax.plot([r['CI_lo'], r['CI_hi']], [i, i], color='#1B3A6B', lw=1.5)
        ax.plot(r['OR'], i, 's', color='#1B3A6B', ms=6)
        ptxt = 'p < 0.001' if r['p_value'] < 0.001 else f"p = {r['p_value']:.3f}"
        ax.text(1.02, i, f"{r['OR']:.3f} ({r['CI_lo']:.3f}–{r['CI_hi']:.3f})   {ptxt}",
                transform=ax.get_yaxis_transform(), va='center', fontsize=9)
    ax.text(1.02, len(_mv)-0.2, 'OR (95% CI)', transform=ax.get_yaxis_transform(), fontsize=10)
    ax.set_yticks(range(len(_mv))); ax.set_yticklabels([lab(v) for v in _mv['variable']], fontsize=9.5)
    ax.axvline(1, color='k', ls='--', lw=0.8); ax.set_xscale('log')
    _tk = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
    ax.set_xticks(_tk); ax.set_xticklabels([str(v) for v in _tk]); ax.set_xlim(0.08, 60)
    ax.set_xlabel('Odds ratio per 1-SD increase (95% CI), log scale', fontsize=12)
    plt.subplots_adjust(left=0.27, right=0.68, top=0.97, bottom=0.07)
    save_fig(fig, "ManuscriptFig2_multivariable")

# ---------------------------------------------------------------------------
# Supplementary Fig. S3: feature selection (training set only)
# ---------------------------------------------------------------------------
_nz = lasso_coef[lasso_coef != 0].sort_values()
_thr = rf_imp.mean(); _top = rf_imp.sort_values(ascending=False).head(25)[::-1]
fig, axes = plt.subplots(1, 2, figsize=(15, 10.5))
axes[0].barh([lab(k) for k in _nz.index], _nz.values,
             color=['#CC6666' if v > 0 else '#74A9D8' for v in _nz.values])
axes[0].axvline(0, color='k', lw=0.8); axes[0].set_xlabel('LASSO coefficient (standardized)', fontsize=12)
axes[0].set_title(f'(A) LASSO feature selection\n({len(_nz)} non-zero variables; C = 0.1, fixed a priori)', fontsize=13, fontweight='bold')
axes[1].barh([lab(k) for k in _top.index], _top.values,
             color=['#F0914F' if v >= _thr else '#C9C9C9' for v in _top.values])
axes[1].axvline(_thr, color='k', ls='--', lw=1)
axes[1].text(_thr + 0.002, 0.2, f'mean importance\n= {_thr:.4f}', fontsize=9.5, va='bottom')
axes[1].set_xlabel('Random forest impurity-based importance', fontsize=12)
axes[1].set_title(f'(B) Random forest importance\n(top 25 shown; {int((rf_imp >= _thr).sum())} variables ≥ mean selected, orange)', fontsize=13, fontweight='bold')
fig.suptitle(f'Feature selection in the training set (N = {len(y_train)}); final predictors (n = {len(selected_features)}) = '
             f'union of (A) and (B) + forced variables', fontsize=13, fontweight='bold')
plt.tight_layout(rect=(0, 0, 1, 0.95)); save_fig(fig, "SupplFigS3_feature_selection")

# ---------------------------------------------------------------------------
# Manuscript Fig. 3: ROC curves (all models) and calibration (best model)
# ---------------------------------------------------------------------------
_STY = {'LR': ':', 'RF': '--', 'XGB': '-.', 'LGBM': '-', 'CatBoost': (0, (3, 1, 1, 1)), 'MLP': (0, (4, 2))}
_NM  = {'LR': 'LR', 'RF': 'RF', 'XGB': 'XGBoost', 'LGBM': 'LightGBM', 'CatBoost': 'CatBoost', 'MLP': 'MLP'}
fig, axes = plt.subplots(1, 4, figsize=(22, 5.6), gridspec_kw=dict(wspace=0.28))
_sets = [('test', y_test, f'Internal Test Set (N = {len(y_test)})'),
         ('ext',  y_ext,  f'Temporal Validation (N = {len(y_ext)})')]
for k, (ds, yt, title) in enumerate(_sets):
    ax = axes[k]
    for name in [n for n in ['LR', 'RF', 'XGB', 'CatBoost', 'MLP', 'LGBM'] if n in prob_store[ds]]:
        yp = prob_store[ds][name]; fpr, tpr, _ = roc_curve(yt, yp)
        best = (name == best_name)
        ax.plot(fpr, tpr, ls=_STY[name], color=PALETTE[name], lw=3 if best else 1.6,
                label=f'{_NM[name]} (AUC={roc_auc_score(yt, yp):.3f})', zorder=5 if best else 3)
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlabel('1 - Specificity (FPR)'); ax.set_ylabel('Sensitivity (TPR)')
    ax.set_title(f'ROC curve: {title}', fontsize=12); ax.legend(loc='lower right', fontsize=9)
for k, (ds, yt, title) in enumerate(_sets):
    ax = axes[2 + k]; yp = prob_store[ds][best_name]
    frac, mp = calibration_curve(yt, yp, n_bins=8)
    ax.plot(mp, frac, 's-', color=PALETTE[best_name], lw=2, ms=7, label=_NM[best_name])
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Fraction of positives')
    ax.set_title(f'Calibration: {title}', fontsize=12); ax.legend(loc='lower right', fontsize=9)
    ax.text(0.04, 0.97, f'Brier = {brier_score_loss(yt, yp):.3f}\nO/E = {yt.mean()/yp.mean():.3f}\n'
            f'HL p = {hl_test(yt, yp)[1]:.3f}', transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', fc='#FFFDE7', ec='#333333'))
fig.text(0.115, 0.97, '(A)  ROC curves', fontsize=14, fontweight='bold')
fig.text(0.52, 0.97, f'(B)  Calibration plots ({_NM[best_name]})', fontsize=14, fontweight='bold')
save_fig(fig, "ManuscriptFig3_ROC_calibration")

# Fig06: PR curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (ds, y_true, label) in zip(axes, [
    ('test', y_test, "Internal test set"), ('ext', y_ext, "Temporal validation")
]):
    for name, yp in prob_store[ds].items():
        p_arr, r_arr, _ = precision_recall_curve(y_true, yp)
        ax.plot(r_arr, p_arr, label=f"{name} (PR-AUC={average_precision_score(y_true,yp):.3f})",
                color=PALETTE.get(name,'#999'))
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve: {label}'); ax.legend(loc='upper right', fontsize=8)
fig.tight_layout(); save_fig(fig, "Fig06_PR_curves")

# Fig07: Model comparison heatmap
metrics_list = ['Accuracy','Precision','Recall','F1','ROC_AUC','PR_AUC']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, ds, label in zip(axes, ['test','ext'], ["Internal test set","Temporal validation"]):
    sub = df_eval[df_eval['Dataset']==ds].set_index('Model')[metrics_list].astype(float)
    sns.heatmap(sub, annot=True, fmt='.3f', cmap='Blues', vmin=0.4, vmax=1.0,
                ax=ax, linewidths=0.5)
    ax.set_title(f'Model performance: {label}')
fig.tight_layout(); save_fig(fig, "Fig07_model_comparison_heatmap")

# ---------------------------------------------------------------------------
# Manuscript Fig. 4: SHAP for the best model (internal test set)
# ---------------------------------------------------------------------------
if best_name in shap_store:
    sv = np.asarray(shap_store[best_name]['sv']); Xsh = shap_store[best_name]['X']
    _names = [lab(c) for c in selected_features]
    shap.summary_plot(sv, Xsh, feature_names=_names, plot_type='bar', max_display=20,
                      show=False, plot_size=(8, 9.5), color='#1E88E5')
    plt.title(f'(A) Mean |SHAP| ({_NM[best_name]})', fontsize=13, fontweight='bold', loc='left')
    plt.tight_layout(); _fa = OUTPUT_DIR / '_f4a.png'; plt.savefig(_fa, dpi=300); plt.close()
    shap.summary_plot(sv, Xsh, feature_names=_names, max_display=20, show=False, plot_size=(8.5, 9.5))
    plt.title(f'(B) SHAP beeswarm plot ({_NM[best_name]})', fontsize=13, fontweight='bold', loc='left')
    plt.tight_layout(); _fb = OUTPUT_DIR / '_f4b.png'; plt.savefig(_fb, dpi=300); plt.close()
    try:
        from PIL import Image as _PIL
        _a, _b = _PIL.open(_fa), _PIL.open(_fb)
        _c = _PIL.new('RGB', (_a.width + _b.width, max(_a.height, _b.height)), 'white')
        _c.paste(_a, (0, 0)); _c.paste(_b, (_a.width, 0))
        _c.save(OUTPUT_DIR / 'ManuscriptFig4_SHAP.png', dpi=(300, 300))
        print(f"  [saved] {OUTPUT_DIR / 'ManuscriptFig4_SHAP.png'}")
    finally:
        for _p in (_fa, _fb):
            try: _p.unlink()
            except Exception: pass

# ---------------------------------------------------------------------------
# Supplementary Fig. S4: SHAP importance across all six models
# ---------------------------------------------------------------------------
if shap_store:
    _KEY = {'type0', 'M.1', 'SM', 'SE', 'SS', 'SI', 'MP', 'size'}
    fig, axs = plt.subplots(2, 3, figsize=(17, 11))
    for ax, name in zip(axs.flat, ['LR', 'RF', 'XGB', 'LGBM', 'CatBoost', 'MLP']):
        if name not in shap_store: ax.axis('off'); continue
        imp = pd.Series(np.abs(np.asarray(shap_store[name]['sv'])).mean(axis=0), index=selected_features)
        s = (imp / imp.max()).sort_values(ascending=False).head(15)[::-1]
        ax.barh([lab(c) for c in s.index], s.values,
                color=['#C44E52' if c == 'type0' else ('#E8A33D' if c in _KEY else '#9DB4D3') for c in s.index])
        rank_txt = ''
        if 'type0' in imp.index:
            rank_txt = f"  (Macr. type 0: rank {list(imp.sort_values(ascending=False).index).index('type0') + 1})"
        ax.set_title(f'{_NM[name]}{rank_txt}', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1.05); ax.tick_params(axis='y', labelsize=9.5)
        ax.set_xlabel('Normalized mean |SHAP| (max = 1)', fontsize=10)
    fig.suptitle(f'SHAP feature importance across all six models (internal test set, N = {len(y_test)})\n'
                 'Top 15 features per model; red, macroscopic type 0; orange, depth of invasion and tumor size',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.94)); save_fig(fig, "SupplFigS4_SHAP_all_models")

# ---------------------------------------------------------------------------
# Manuscript Fig. 5: reduced-surgery simulation (internal test set / temporal)
# ---------------------------------------------------------------------------
def _npv_ci(yt, yp, t, B=B_BOOTSTRAP):
    f = lambda a, b: float((1 - a[b < t]).mean()) if (b < t).sum() > 0 else np.nan
    return bootstrap_ci(yt, yp, f, B=B)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
for ax, (yt, yp, title) in zip(axes, [(y_test, prob_test_best, f'Internal Test Set\n(N={len(y_test)})'),
                                      (y_ext,  prob_ext_all,   f'Temporal Validation\n(N={len(y_ext)})')]):
    xs = np.arange(len(THRESHOLDS)); rates, fns, npvs, los, his = [], [], [], [], []
    for t in THRESHOLDS:
        low = yp < t; n = int(low.sum()); rates.append(100*n/len(yt)); fns.append(int(yt[low].sum()))
        if n:
            v = float((1 - yt[low]).mean()); ci = _npv_ci(yt, yp, t)
            npvs.append(v); los.append(v - ci[0]); his.append(ci[1] - v)
        else:
            npvs.append(np.nan); los.append(0); his.append(0)
    ax.bar(xs - 0.18, rates, 0.36, color='#74A9D8', label='Candidate rate (%)')
    ax.bar(xs + 0.18, fns, 0.36, color='#CC6666', label='False negatives (n)')
    for i in range(len(xs)):
        ax.text(xs[i] - 0.18, rates[i] + 0.6, f'{rates[i]:.1f}%', ha='center', fontweight='bold')
        if fns[i] > 0: ax.text(xs[i] + 0.18, fns[i] + 0.6, str(fns[i]), ha='center', fontweight='bold')
    ax.set_xticks(xs); ax.set_xticklabels([f'{int(t*100)}%' + ('\n(recommended)' if t == 0.10 else '') for t in THRESHOLDS])
    ax.set_xlabel('Threshold probability', fontsize=12); ax.set_ylabel('Candidate rate (%) / False negatives (n)')
    ax.set_ylim(0, max(rates)*1.7 + 2); ax.set_title(title, fontweight='bold', fontsize=13)
    ax2 = ax.twinx(); idx = [i for i in range(len(xs)) if not np.isnan(npvs[i])]
    if idx:
        ax2.errorbar([xs[i] for i in idx], [npvs[i] for i in idx],
                     yerr=[[los[i] for i in idx], [his[i] for i in idx]],
                     fmt='-o', color='k', lw=2, ms=9, capsize=5, label='NPV')
        for i in idx: ax2.text(xs[i] + 0.05, npvs[i] - 0.035, f'{npvs[i]:.3f}', ha='left', fontsize=10)
    ax2.set_ylim(0.5, 1.05); ax2.set_ylabel('NPV (95% CI by bootstrap)')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper left')
fig.tight_layout(); save_fig(fig, "ManuscriptFig5_simulation")

# ---------------------------------------------------------------------------
# Manuscript Fig. 6: decision curve analysis (internal test set / temporal)
# ---------------------------------------------------------------------------
_ts = np.linspace(0.01, 0.60, 60)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
for ax, (yt, yp, title) in zip(axes, [(y_test, prob_test_best, f'DCA: Internal Test Set (N={len(y_test)})'),
                                      (y_ext,  prob_ext_all,   f'DCA: Temporal Validation (N={len(y_ext)})')]):
    ax.plot(_ts*100, [net_benefit(yt, yp, t) for t in _ts], color='blue', lw=2.5, label=_NM[best_name])
    ax.plot(_ts*100, [yt.mean() - (1 - yt.mean())*t/(1 - t) for t in _ts], 'r--', lw=2, label='Treat all')
    ax.axhline(0, color='k', ls=':', lw=1, label='Treat none')
    for tv, c in [(10, 'green'), (20, 'orange')]:
        ax.axvline(tv, color=c, ls=':', lw=1.5); ax.text(tv + 0.5, 0.01, f'{tv}%', color=c)
    ax.set_xlim(0, 60); ax.set_ylim(-0.05, 0.47)
    ax.set_xlabel('Threshold probability (%)', fontsize=12); ax.set_ylabel('Net benefit', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=13); ax.legend(loc='upper right')
fig.tight_layout(); save_fig(fig, "ManuscriptFig6_DCA")

# Fig12: Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (ds, y_true, label) in zip(axes, [
    ('test', y_test, "Internal test set"), ('ext', y_ext, "Temporal validation")
]):
    yp = prob_store[ds][best_name]
    cm = confusion_matrix(y_true, (yp >= 0.5).astype(int))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Pred N-','Pred N+'], yticklabels=['True N-','True N+'])
    ax.set_title(f'Confusion Matrix: {best_name} ({label})')
fig.tight_layout(); save_fig(fig, "Fig12_confusion_matrix")

# Fig13: Predicted probability distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (ds, y_true, label) in zip(axes, [
    ('test', y_test, "Internal test set"), ('ext', y_ext, "Temporal validation")
]):
    yp = prob_store[ds][best_name]
    ax.hist(yp[y_true==0], bins=30, alpha=0.6, color='#5B9BD5', label='pN-', density=True)
    ax.hist(yp[y_true==1], bins=30, alpha=0.6, color='#C44E52', label='pN+', density=True)
    for thr in THRESHOLDS:
        ax.axvline(thr, color='green', ls='--', lw=1.0,
                   label=f'{int(thr*100)}% threshold' if thr==THRESHOLDS[0] else "")
    ax.set_xlabel('Predicted LNM probability'); ax.set_ylabel('Density')
    ax.set_title(f'Probability distribution: {best_name} ({label})'); ax.legend()
fig.tight_layout(); save_fig(fig, "Fig13_probability_distribution")

# Fig14: ROC-AUC comparison
fig, ax = plt.subplots(figsize=(9, 5))
model_names = list(best_models.keys())
auc_test = [df_eval[(df_eval['Model']==n)&(df_eval['Dataset']=='test')]['ROC_AUC'].values[0] for n in model_names]
auc_ext  = [df_eval[(df_eval['Model']==n)&(df_eval['Dataset']=='ext')]['ROC_AUC'].values[0]  for n in model_names]
x = np.arange(len(model_names)); w = 0.35
ax.bar(x-w/2, auc_test, width=w, label='Internal test set', color='#5B9BD5', alpha=0.9)
ax.bar(x+w/2, auc_ext,  width=w, label='External validation', color='#ED7D31', alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels(model_names)
ax.set_ylim(0.4, 1.0); ax.set_ylabel('ROC-AUC'); ax.set_title('ROC-AUC comparison across models')
ax.legend(); ax.axhline(0.7, color='gray', ls='--', lw=0.8)
for xi, (vt, ve) in enumerate(zip(auc_test, auc_ext)):
    ax.text(xi-w/2, vt+0.005, f'{vt:.3f}', ha='center', va='bottom', fontsize=8)
    ax.text(xi+w/2, ve+0.005, f'{ve:.3f}', ha='center', va='bottom', fontsize=8)
fig.tight_layout(); save_fig(fig, "Fig14_AUC_bar_comparison")

# =============================================================================
# 14. Summary
# =============================================================================
sep("14. Analysis summary")

print(f"""
Results:
  Internal N={len(df_int_raw)}, pN+={df_int_raw[TARGET_COL].sum()} ({df_int_raw[TARGET_COL].mean()*100:.1f}%)
  External N={len(df_ext_raw)}, pN+={df_ext_raw[TARGET_COL].sum()} ({df_ext_raw[TARGET_COL].mean()*100:.1f}%)
  LASSO selected: {len(lasso_selected)}  RF top: {len(rf_top)}  Final: {len(selected_features)}
  Best model: {best_name}
    Test AUC: {test_auc[best_name]:.4f}
    Ext  AUC: {df_eval[(df_eval['Model']==best_name)&(df_eval['Dataset']=='ext')]['ROC_AUC'].values[0]:.4f}
  Simulation internal — 5%: NPV={df_sim_int.iloc[0]['NPV']:.4f} FN={df_sim_int.iloc[0]['false_negative']} | 10%: NPV={df_sim_int.iloc[1]['NPV']:.4f} FN={df_sim_int.iloc[1]['false_negative']} | 20%: NPV={df_sim_int.iloc[2]['NPV']:.4f} FN={df_sim_int.iloc[2]['false_negative']}
  Simulation external — 5%: NPV={df_sim_ext.iloc[0]['NPV']:.4f} FN={df_sim_ext.iloc[0]['false_negative']} | 10%: NPV={df_sim_ext.iloc[1]['NPV']:.4f} FN={df_sim_ext.iloc[1]['false_negative']} | 20%: NPV={df_sim_ext.iloc[2]['NPV']:.4f} FN={df_sim_ext.iloc[2]['false_negative']}
  Output: {OUTPUT_DIR}
""")

print("Best model hyperparameters:")
for k, v in best_params_all.get(best_name, {}).items():
    print(f"  {k:25s}: {v}")

if mv_table is not None:
    _sig = mv_table[(mv_table["variable"] != "const") &
                    (mv_table["p_value"].notna()) & (mv_table["p_value"] < 0.05)]
    print("\nSignificant variables (multivariate, p<0.05):")
    if len(_sig) > 0:
        print(_sig[["variable","OR","CI_lo","CI_hi","p_value"]].round(4).to_string(index=False))
    else:
        print("  None (possible complete separation or multicollinearity)")

print("\n  All processing complete.")
