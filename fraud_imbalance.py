"""Part G: imbalanced-learning arms, built so that every re-balancing step lives INSIDE the training pipeline.

Pipeline = features -> feature-set selector -> impute/encode/scale -> [sampler] -> model   (imbalanced-learn Pipeline)
imblearn's Pipeline applies the sampler only while fitting. predict / predict_proba / transform skip it, so validation and test rows are never
resampled, and in cross-validation the sampler only ever sees the training fold. The sampler sits AFTER encoding because SMOTE needs numbers.

Arms:  none | class_weight (balanced) | rus_<r> random undersampling | ros_<r> random oversampling | smote_<r> synthetic oversampling
where <r> is the target minority:majority ratio after resampling (natural ratio here is about 0.14; 1.0 = fully balanced).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn import FunctionSampler
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import fraud_features as ff

SEED = 42
HERE = Path(__file__).resolve().parent


def final_rf_params():
    """Hyper-parameters of the Part D final Random Forest (kept fixed here so only the imbalance treatment changes between arms)."""
    info = json.load(open(HERE / "outputs" / "models" / "D_final_models.json"))["models"]["RF"]["best_params"]
    return {k.split("__", 1)[1]: v for k, v in info.items() if k.startswith("model__")}, info["select__feature_set"]


def make_model(name, class_weight=None):
    if name == "LR":
        return LogisticRegression(C=1.0, max_iter=3000, random_state=SEED, class_weight=class_weight), True, "base"
    if name == "RF":
        p, fs = final_rf_params()
        return RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=SEED, class_weight=class_weight, **p), False, fs
    if name == "RF_deep":     # unrestricted trees, used only to demonstrate how leakage inflates scores for a flexible model
        return RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=SEED, class_weight=class_weight, max_features="sqrt"), False, "base"
    raise ValueError(name)


def make_sampler(kind, ratio, seed=SEED):
    if kind == "rus": return RandomUnderSampler(sampling_strategy=ratio, random_state=seed)
    if kind == "ros": return RandomOverSampler(sampling_strategy=ratio, random_state=seed)
    if kind == "smote": return SMOTE(sampling_strategy=ratio, k_neighbors=5, random_state=seed)
    raise ValueError(kind)


def parse_arm(arm):
    """'none' | 'class_weight' | 'rus_0.5' ... -> (kind, ratio)"""
    if arm in ("none", "class_weight"): return arm, None
    kind, r = arm.split("_"); return kind, float(r)


def build_arm(model_name, arm, sampler=None):
    """imblearn Pipeline for one (model, imbalance treatment). `sampler` may be injected (used by the integrity test)."""
    kind, ratio = parse_arm(arm)
    est, scale, fs = make_model(model_name, class_weight="balanced" if kind == "class_weight" else None)
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    der_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        num_steps.append(("scale", StandardScaler())); der_steps.append(("scale", StandardScaler()))
    ct = ColumnTransformer([("num", SkPipeline(num_steps), ff.NUM_BASE),
                            ("derived", SkPipeline(der_steps), ff._derived_selector),
                            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ff.CAT_BASE)], remainder="drop", verbose_feature_names_out=False)
    steps = [("features", ff.FraudFeatureEngineer(groups=tuple(ff.FEATURE_GROUPS))), ("select", ff.FeatureSetSelector(fs)), ("prep", ct)]
    if sampler is not None: steps.append(("sampler", sampler))
    elif kind in ("rus", "ros", "smote"): steps.append(("sampler", make_sampler(kind, ratio)))
    steps.append(("model", est))
    return ImbPipeline(steps)


# ------------------------------------------------------------------ evaluation on natural-prevalence test folds
def fold_metrics(y, p, top_frac=0.10, thr=0.5):
    y, p = np.asarray(y), np.asarray(p, float); k = int(np.ceil(top_frac * len(y))); top = np.argsort(-p, kind="stable")[:k]; pred = (p >= thr).astype(int)
    return dict(roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p), brier=brier_score_loss(y, p), log_loss=log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
                recall_top10=y[top].sum() / y.sum(), precision_top10=y[top].mean(),
                precision_at_05=precision_score(y, pred, zero_division=0), recall_at_05=recall_score(y, pred, zero_division=0), f1_at_05=f1_score(y, pred, zero_division=0),
                flagged_at_05=pred.mean(), mean_p=p.mean())


def _one(model_name, arm, X, y, tr, te, i):
    m = build_arm(model_name, arm).fit(X.iloc[tr], y.iloc[tr]); p = m.predict_proba(X.iloc[te])[:, 1]
    return i, te, p, dict(fold=i, **fold_metrics(y.iloc[te].values, p))


def run_arm(model_name, arm, X, y, splits, n_jobs=2):
    """Returns (per-fold metrics, [OOF vector per CV repeat]). Splits come from fraud_eval.make_splits (identical to Part D's outer folds)."""
    res = Parallel(n_jobs=n_jobs)(delayed(_one)(model_name, arm, X, y, tr, te, i) for i, (tr, te) in enumerate(splits))
    n_rep = len(splits) // 5; oof = [np.full(len(y), np.nan) for _ in range(n_rep)]
    for i, te, p, _ in res: oof[i // 5][te] = p
    folds = pd.DataFrame([r[3] for r in res]); folds.insert(0, "arm", arm); folds.insert(0, "model", model_name)
    return folds, oof


# ------------------------------------------------------------------ integrity test: the sampler sees training rows only
_SEEN: list = []


def _recording_undersample(X, y, ratio=0.5):
    _SEEN.append((len(y), int(np.sum(y)), np.asarray(X).shape[1]))
    return RandomUnderSampler(sampling_strategy=ratio, random_state=SEED).fit_resample(X, y)


def integrity_test(X, y, splits):
    """For every fold: the sampler is called exactly once per fit, with exactly the training rows (count and positives), and is NOT called by predict."""
    rows = []
    for i, (tr, te) in enumerate(splits[:5]):
        _SEEN.clear()
        pipe = build_arm("LR", "none", sampler=FunctionSampler(func=_recording_undersample, kw_args={"ratio": 0.5}, validate=False))
        pipe.fit(X.iloc[tr], y.iloc[tr]); n_calls_fit = len(_SEEN); seen = _SEEN[0] if _SEEN else (None, None, None)
        pipe.predict_proba(X.iloc[te]); n_calls_after_predict = len(_SEEN)
        rows.append(dict(fold=i, train_rows=len(tr), test_rows=len(te), sampler_calls_during_fit=n_calls_fit, rows_seen_by_sampler=seen[0], frauds_seen_by_sampler=seen[1],
                         frauds_in_train_fold=int(y.iloc[tr].sum()), sampler_calls_during_predict=n_calls_after_predict - n_calls_fit,
                         ok=bool(n_calls_fit == 1 and seen[0] == len(tr) and seen[1] == int(y.iloc[tr].sum()) and n_calls_after_predict == n_calls_fit)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ the WRONG way, for contrast
def leaky_oversampling_cv(model_name, X, y, splits_orig_n=5, repeats=2, ratio=0.5, seed=SEED):
    """INVALID protocol shown only to measure how much it flatters the score: oversample the WHOLE development set first, then cross-validate.
    Copies of the same fraud land in both training and test folds. Scores are computed on the ORIGINAL rows of each test fold only
    (the appended copies are dropped), so they are directly comparable with the honest numbers."""
    from sklearn.model_selection import RepeatedStratifiedKFold
    n = len(y)
    ros = RandomOverSampler(sampling_strategy=ratio, random_state=seed); Xo, yo = ros.fit_resample(X, y)
    is_orig = np.zeros(len(yo), bool); is_orig[:n] = True        # imblearn keeps the originals first
    rows = []
    for i, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=splits_orig_n, n_repeats=repeats, random_state=seed).split(np.zeros(len(yo)), yo)):
        m = build_arm(model_name, "none").fit(Xo.iloc[tr], yo.iloc[tr]); te_o = te[is_orig[te]]
        p = m.predict_proba(Xo.iloc[te_o])[:, 1]; rows.append(dict(fold=i, **fold_metrics(yo.iloc[te_o].values, p)))
    out = pd.DataFrame(rows); out.insert(0, "protocol", "WRONG: oversample first, then split"); out.insert(0, "model", model_name); return out


def honest_cv(model_name, arm, X, y, splits):
    rows = []
    for i, (tr, te) in enumerate(splits):
        m = build_arm(model_name, arm).fit(X.iloc[tr], y.iloc[tr]); p = m.predict_proba(X.iloc[te])[:, 1]; rows.append(dict(fold=i, **fold_metrics(y.iloc[te].values, p)))
    out = pd.DataFrame(rows); out.insert(0, "protocol", "RIGHT: split first, oversample inside training folds"); out.insert(0, "model", model_name); return out
