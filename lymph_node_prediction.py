# =============================================================================
# ML-Based Lymph Node Metastasis Prediction in Older Patients with Gastric Cancer
# A Retrospective Simulation Study (TRIPOD+AI Compliant)
#
# Reference:
#   Fujimoto G, Kusanagi H. Machine learning-based prediction of lymph node
#   metastases for individualized surgical decision-making in older patients
#   with gastric cancer. Gastric Cancer. (under review)
#
# Requirements:
#   pip install pandas numpy matplotlib seaborn scikit-learn \
#               xgboost lightgbm catboost shap scipy statsmodels openpyxl pillow
#
# Tested with:
#   Python 3.13 | scikit-learn 1.6.1 | LightGBM 4.x | XGBoost 2.x
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
        p_value=round(pval, 4) if not np.isnan(pval)  else np.nan
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
print(f"{'Dataset':<14} {'Brier':>10} {'95%CI':>22} {'E/O':>8} {'HL x2':>8} {'HL p':>10} {'Eval':>8}")
print("─" * 82)
for ds_label, yt, yp in [
    ("Internal", y_test, prob_store['test'][best_name]),
    ("External", y_ext,  prob_store['ext'][best_name])
]:
    bs = brier_score_loss(yt, yp); bs_ci = bootstrap_ci(yt, yp, brier_score_loss)
    eo = yt.mean() / yp.mean(); c2, ph = hl_test(yt, yp)
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
print("  10% : Recommended — NPV>=0.99 with meaningful eligibility expansion.")
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
        else:
            bg = shap.sample(X_bg, 50, random_state=RANDOM_STATE)
            explainer = shap.KernelExplainer(model.predict_proba, bg, link='logit')
            Xsamp = shap.sample(X_target, 50, random_state=RANDOM_STATE)
            sv = explainer.shap_values(Xsamp, nsamples=100)
            if isinstance(sv, list): sv = sv[1]
            if sv.ndim == 3: sv = sv[:, :, 1]
            X_target = Xsamp
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
        print(f"  Threshold {thr*100:.0f}%: candidates={n_low} ({lr*100:.1f}%), "
              f"NPV={npv:.4f if not np.isnan(npv) else 'N/A'}, FN={fn}")
    return pd.DataFrame(rows)

df_sim_int = run_simulation(y_int_all, prob_int_all, THRESHOLDS, "Internal (all)")
df_sim_ext = run_simulation(y_ext,     prob_ext_all, THRESHOLDS, "External validation")

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

nb_int_m, nb_int_a, nb_int_n = run_dca(y_int_all, prob_int_all, "Internal")
nb_ext_m, nb_ext_a, nb_ext_n = run_dca(y_ext,     prob_ext_all, "External")

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

# Fig02: Univariable forest plot (top 20)
df_uv_p = df_univar.dropna(subset=['OR','CI_lo','CI_hi']).sort_values('p_value').head(20)
fig, ax  = plt.subplots(figsize=(8, 8))
for i, (_, r) in enumerate(df_uv_p.iterrows()):
    c = '#C44E52' if r['p_value'] < 0.05 else '#A0A0A0'
    ax.errorbar(r['OR'], i, xerr=[[r['OR']-r['CI_lo']], [r['CI_hi']-r['OR']]],
                fmt='o', color=c, capsize=4, markersize=5)
ax.axvline(1, color='k', ls='--', lw=0.8)
ax.set_yticks(range(len(df_uv_p))); ax.set_yticklabels(df_uv_p['variable'].tolist(), fontsize=8)
ax.set_xlabel('Odds Ratio (95% CI)'); ax.set_title('Univariable Forest Plot (top 20)')
ax.legend(handles=[mpatches.Patch(color='#C44E52', label='p<0.05'),
                    mpatches.Patch(color='#A0A0A0', label='p>=0.05')])
fig.tight_layout(); save_fig(fig, "Fig02_univariate_forest_plot")

# Fig03: Multivariable forest plot
if mv_table is not None:
    mv_p = mv_table[mv_table['variable'] != 'const'].dropna(subset=['OR','CI_lo','CI_hi'])
    fig, ax = plt.subplots(figsize=(8, max(4, len(mv_p)*0.45)))
    for i, (_, r) in enumerate(mv_p.iterrows()):
        pval = r['p_value'] if not np.isnan(r['p_value']) else 1.0
        c = '#C44E52' if pval < 0.05 else '#A0A0A0'
        ax.errorbar(r['OR'], i,
                    xerr=[[max(r['OR']-r['CI_lo'],0)], [max(r['CI_hi']-r['OR'],0)]],
                    fmt='s', color=c, capsize=4, markersize=6)
    ax.axvline(1, color='k', ls='--', lw=0.8)
    ax.set_yticks(range(len(mv_p))); ax.set_yticklabels(mv_p['variable'].tolist(), fontsize=9)
    ax.set_xlabel('Odds Ratio (95% CI)'); ax.set_title('Multivariable Logistic Regression Forest Plot')
    ax.legend(handles=[mpatches.Patch(color='#C44E52', label='p<0.05'),
                        mpatches.Patch(color='#A0A0A0', label='p>=0.05')])
    fig.tight_layout(); save_fig(fig, "Fig03_multivariate_forest_plot")

# Fig04: Feature selection
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
lasso_nz = lasso_coef[lasso_coef != 0].sort_values(key=abs, ascending=True)
axes[0].barh(lasso_nz.index, lasso_nz.values,
             color=['#C44E52' if v > 0 else '#4C72B0' for v in lasso_nz.values])
axes[0].axvline(0, color='k', lw=0.8)
axes[0].set_title('LASSO coefficients'); axes[0].set_xlabel('Coefficient')
rf_top20 = rf_imp.head(20).sort_values()
axes[1].barh(rf_top20.index, rf_top20.values, color='#55A868')
axes[1].set_title('RF feature importance (top 20)'); axes[1].set_xlabel('Importance')
fig.tight_layout(); save_fig(fig, "Fig04_feature_selection")

# Fig05: ROC curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (ds, y_true, label) in zip(axes, [
    ('test', y_test, "Internal test set"), ('ext', y_ext, "External validation")
]):
    for name, yp in prob_store[ds].items():
        fpr, tpr, _ = roc_curve(y_true, yp)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true,yp):.3f})",
                color=PALETTE.get(name,'#999'))
    ax.plot([0,1],[0,1],'k--',lw=0.8)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title(f'ROC Curve: {label}')
    ax.legend(loc='lower right', fontsize=8)
fig.tight_layout(); save_fig(fig, "Fig05_ROC_curves")

# Fig06: PR curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (ds, y_true, label) in zip(axes, [
    ('test', y_test, "Internal test set"), ('ext', y_ext, "External validation")
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
for ax, ds, label in zip(axes, ['test','ext'], ["Internal test set","External validation"]):
    sub = df_eval[df_eval['Dataset']==ds].set_index('Model')[metrics_list].astype(float)
    sns.heatmap(sub, annot=True, fmt='.3f', cmap='Blues', vmin=0.4, vmax=1.0,
                ax=ax, linewidths=0.5)
    ax.set_title(f'Model performance: {label}')
fig.tight_layout(); save_fig(fig, "Fig07_model_comparison_heatmap")

# Fig08: Calibration plots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, (ds, y_true, label) in zip(axes, [
    ('test', y_test, "Internal test set"), ('ext', y_ext, "External validation")
]):
    yp = prob_store[ds][best_name]
    frac, mp = calibration_curve(y_true, yp, n_bins=8)
    ax.plot(mp, frac, 's-', label=best_name, color=PALETTE.get(best_name,'#333'))
    ax.plot([0,1],[0,1],'k--', label='Perfect calibration')
    ax.set_xlabel('Mean Predicted Probability'); ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'Calibration Plot: {label}'); ax.legend()
fig.tight_layout(); save_fig(fig, "Fig08_calibration_plot")

# Fig09: SHAP (best model)
if best_name in shap_store:
    sv = shap_store[best_name]['sv']; Xsh = shap_store[best_name]['X']
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(sv, Xsh, plot_type='bar', show=False, feature_names=selected_features)
    plt.title(f'SHAP Feature Importance: {best_name}')
    plt.tight_layout(); save_fig(fig, f"Fig09a_SHAP_bar_{best_name}")

    fig = plt.figure(figsize=(8, 7))
    shap.summary_plot(sv, Xsh, show=False, feature_names=selected_features)
    plt.title(f'SHAP Beeswarm: {best_name}')
    plt.tight_layout(); save_fig(fig, f"Fig09b_SHAP_beeswarm_{best_name}")

# Fig09c: SHAP all models (combined)
if shap_store:
    _paths = []
    for name, sr in shap_store.items():
        _fig = plt.figure(figsize=(5, 5))
        shap.summary_plot(sr['sv'], sr['X'], plot_type='bar', show=False, feature_names=selected_features)
        _cur = plt.gcf(); _cur.axes[0].set_title(name, fontsize=11); _cur.tight_layout()
        _tmp = OUTPUT_DIR / f"_tmp_shap_{name}.png"
        _cur.savefig(_tmp, bbox_inches='tight', dpi=150); plt.close(_cur)
        _paths.append((_tmp, name))
    try:
        from PIL import Image as _PIL
        _imgs = [_PIL.open(str(p)) for p, _ in _paths]
        _w = sum(im.width for im in _imgs); _h = max(im.height for im in _imgs)
        _canvas = _PIL.new('RGB', (_w, _h), (255, 255, 255))
        _x = 0
        for _im in _imgs:
            _canvas.paste(_im, (_x, 0)); _x += _im.width
        _out = OUTPUT_DIR / "Fig09c_SHAP_all_models.png"
        _canvas.save(str(_out), dpi=(150, 150)); print(f"  [saved] {_out}")
    except Exception as _e:
        print(f"  [PIL merge skipped: {_e}]")
    for _p, _ in _paths:
        try: _p.unlink()
        except Exception: pass

# Fig10: Simulation results
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (df_sim, title) in zip(axes, [
    (df_sim_int, "Internal (all)"), (df_sim_ext, "External validation")
]):
    x_labels = [f"{int(t*100)}%" for t in df_sim['threshold']]
    ax2 = ax.twinx(); bw = 0.35; xs = np.arange(len(x_labels))
    ax.bar(xs-bw/2, df_sim['local_rate_pct'], width=bw, color='#5B9BD5', alpha=0.8, label='Candidate rate (%)')
    ax.bar(xs+bw/2, df_sim['false_negative'],  width=bw, color='#C44E52', alpha=0.8, label='False negatives')
    ax2.plot(xs, df_sim['NPV'], 'ko-', lw=2, ms=8, label='NPV')
    ax.set_xticks(xs); ax.set_xticklabels(x_labels)
    ax.set_ylabel('Cases / Rate (%)'); ax2.set_ylabel('NPV'); ax2.set_ylim(0.5, 1.05)
    ax.set_title(f'Simulation: {title}')
    l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, lb1+lb2, loc='upper left', fontsize=8)
fig.tight_layout(); save_fig(fig, "Fig10_simulation_results")

# Fig11: DCA
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (nb_m, nb_a, nb_n, label) in zip(axes, [
    (nb_int_m, nb_int_a, nb_int_n, "Internal (all)"),
    (nb_ext_m, nb_ext_a, nb_ext_n, "External validation")
]):
    ax.plot(thr_range, nb_m, 'b-', lw=2, label=f'Model ({best_name})')
    ax.plot(thr_range, nb_a, 'r--', lw=1.5, label='Treat all')
    ax.plot(thr_range, nb_n, 'k-', lw=1.0, label='Treat none')
    for thr in THRESHOLDS: ax.axvline(thr, color='gray', ls=':', lw=0.8)
    ax.set_xlim(0, 0.5); ax.set_ylim(-0.05, max(nb_a)*1.3)
    ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net Benefit')
    ax.set_title(f'Decision Curve Analysis: {label}'); ax.legend()
fig.tight_layout(); save_fig(fig, "Fig11_DCA")

# Fig12: Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (ds, y_true, label) in zip(axes, [
    ('test', y_test, "Internal test set"), ('ext', y_ext, "External validation")
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
    ('test', y_test, "Internal test set"), ('ext', y_ext, "External validation")
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
