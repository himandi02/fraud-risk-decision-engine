"""Shared evaluation helpers: fixed folds, fold metrics, paired comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

SEED = 42


def make_splits(y, n_splits=5, n_repeats=3, seed=SEED):
    """Identical folds for every configuration, so comparisons are paired."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y))


def fold_metrics(y, p, top_frac=0.10):
    y, p = np.asarray(y), np.asarray(p)
    k = int(np.ceil(top_frac * len(y)))
    top = np.argsort(-p)[:k]
    return dict(pr_auc=average_precision_score(y, p), roc_auc=roc_auc_score(y, p),
                brier=brier_score_loss(y, p), log_loss=log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
                recall_at_10pct=y[top].sum() / y.sum())


def _fit_eval(pipe, X, y, tr, te, i):
    m = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
    p = m.predict_proba(X.iloc[te])[:, 1]
    return dict(fold=i, **fold_metrics(y.iloc[te].values, p))


def run_cv(pipe, X, y, splits, label, n_jobs=2):
    rows = Parallel(n_jobs=n_jobs)(delayed(_fit_eval)(pipe, X, y, tr, te, i) for i, (tr, te) in enumerate(splits))
    out = pd.DataFrame(rows)
    out.insert(0, "config", label)
    return out


def corrected_ttest(d, n_train, n_test):
    """Nadeau-Bengio corrected resampled t-test for paired CV differences d (one per fold)."""
    d = np.asarray(d, float)
    k = len(d)
    se = np.sqrt((1 / k + n_test / n_train) * d.var(ddof=1))
    mean = d.mean()
    t = mean / se if se > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), k - 1)
    h = stats.t.ppf(.975, k - 1) * se
    return dict(mean_diff=mean, se=se, ci_low=mean - h, ci_high=mean + h, p_value=p)


def default_models():
    """Fixed, untuned settings. Tuning happens in Part D."""
    return {
        "LR": (LogisticRegression(C=1.0, max_iter=3000, random_state=SEED), True),
        "HGB": (HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                               min_samples_leaf=20, random_state=SEED), False),
        "RF": (RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt", n_jobs=1,
                                      random_state=SEED), False),
    }
