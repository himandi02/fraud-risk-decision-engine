"""Part D: model zoo, hyper-parameter search spaces, nested cross-validation and final fitting.

Design
  * Every model is a full pipeline: features -> feature-set selector -> impute/encode(/scale) -> model.
    Nothing learns from data outside the pipeline, so nothing leaks across folds.
  * NESTED CV: outer 5-fold x 2 repeats gives honest performance estimates. Inside each outer-training set a
    RandomizedSearchCV (5-fold inner CV, scoring = average precision) tunes the hyper-parameters AND the
    feature set. The outer test fold never influences any choice.
  * Only the DEVELOPMENT set is used. The hold-out test set is never read here.
  * Seeds: SEED = 42 for outer folds/models; inner CV and random search use SEED + outer fold id.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump
from scipy.stats import loguniform, randint
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold

from fraud_features import FEATURE_SET_COLUMNS, build_search_pipeline

SEED = 42
N_OUTER_SPLITS, N_OUTER_REPEATS, INNER_FOLDS = 5, 2, 5
FEATURE_SETS = list(FEATURE_SET_COLUMNS)                      # base / recommended / recommended_no_red_flags / all
HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------- data
def load_dev(data_csv=None, split_csv=None):
    """Development rows only (original ID order). The hold-out rows are dropped here and never returned."""
    data_csv = data_csv or HERE / "transaction_fraud_risk_advanced_dataset.csv"
    split_csv = split_csv or HERE / "outputs" / "split_assignment.csv"
    df, sp = pd.read_csv(data_csv), pd.read_csv(split_csv)
    assert (sp.transaction_id.values == df.transaction_id.values).all()
    dev = df[sp.split.values == "development"].reset_index(drop=True)
    return dev.drop(columns=["transaction_id", "fraud"]), dev["fraud"], dev["transaction_id"]


# ----------------------------------------------------------------------------- models
SPACE_DOC = {
    "LR": "C ~ loguniform(1e-3, 1e2); penalty in {l1, l2} (liblinear); feature_set in 4 options; 20 draws",
    "RF": "300 trees; max_depth in {3,5,8,12,None}; min_samples_leaf ~ randint(2,25); max_features in {sqrt,0.2,0.4,0.6}; "
          "max_samples in {None,0.6,0.8}; feature_set in 4 options; 16 draws",
    "HGB": "learning_rate ~ loguniform(0.01,0.2); max_iter ~ randint(80,400); max_depth in {2,3,4,6,None}; max_leaf_nodes ~ randint(4,32); "
           "min_samples_leaf ~ randint(10,80); l2_regularization ~ loguniform(0.01,20); max_features in {0.4,0.7,1.0}; feature_set in 4 options; 24 draws",
}


def model_specs():
    fs = {"select__feature_set": FEATURE_SETS}
    return {
        "Dummy": dict(estimator=DummyClassifier(strategy="prior"), scale=False, space=None, n_iter=0,
                      fixed={"select__feature_set": "base"}, kind="reference"),
        "LR_baseline": dict(estimator=LogisticRegression(C=1.0, max_iter=3000, random_state=SEED), scale=True, space=None, n_iter=0,
                            fixed={"select__feature_set": "base"}, kind="baseline"),
        "LR_tuned": dict(estimator=LogisticRegression(solver="liblinear", max_iter=3000, random_state=SEED), scale=True, n_iter=20,
                         space={**fs, "model__C": loguniform(1e-3, 1e2), "model__penalty": ["l1", "l2"]}, fixed={}, kind="tuned"),
        "RF": dict(estimator=RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=SEED), scale=False, n_iter=16,
                   space={**fs, "model__max_depth": [3, 5, 8, 12, None], "model__min_samples_leaf": randint(2, 25),
                          "model__max_features": ["sqrt", 0.2, 0.4, 0.6], "model__max_samples": [None, 0.6, 0.8]}, fixed={}, kind="tuned"),
        "HGB": dict(estimator=HistGradientBoostingClassifier(early_stopping=False, random_state=SEED), scale=False, n_iter=24,
                    space={**fs, "model__learning_rate": loguniform(0.01, 0.2), "model__max_iter": randint(80, 400),
                           "model__max_depth": [2, 3, 4, 6, None], "model__max_leaf_nodes": randint(4, 32),
                           "model__min_samples_leaf": randint(10, 80), "model__l2_regularization": loguniform(0.01, 20),
                           "model__max_features": [0.4, 0.7, 1.0]}, fixed={}, kind="tuned"),
    }


def _build(spec):
    pipe = build_search_pipeline(clone(spec["estimator"]), spec["scale"])
    return pipe.set_params(**spec["fixed"]) if spec["fixed"] else pipe


def _search(spec, seed, inner_folds):
    return RandomizedSearchCV(_build(spec), spec["space"], n_iter=spec["n_iter"], scoring="average_precision",
                              cv=StratifiedKFold(inner_folds, shuffle=True, random_state=seed), random_state=seed,
                              n_jobs=1, refit=True, error_score="raise")


# ----------------------------------------------------------------------------- metrics
def full_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    k = int(np.ceil(0.10 * len(y)))
    top = np.argsort(-p, kind="stable")[:k]
    pred = (p >= 0.5).astype(int)
    return dict(pr_auc=average_precision_score(y, p), roc_auc=roc_auc_score(y, p),
                recall_at_10pct=y[top].sum() / y.sum(), precision_at_10pct=y[top].mean(),
                brier=brier_score_loss(y, p), log_loss=log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
                precision_at_0p5=precision_score(y, pred, zero_division=0), recall_at_0p5=recall_score(y, pred, zero_division=0),
                f1_at_0p5=f1_score(y, pred, zero_division=0))


# ----------------------------------------------------------------------------- nested CV
def _outer_fold(name, spec, X, y, tr, te, fold_id):
    seed, t0 = SEED + fold_id, time.time()
    if spec["space"] is None:
        est = _build(spec).fit(X.iloc[tr], y.iloc[tr]); best, inner = {}, np.nan
    else:
        s = _search(spec, seed, INNER_FOLDS).fit(X.iloc[tr], y.iloc[tr])
        est, best, inner = s.best_estimator_, s.best_params_, s.best_score_
    p = est.predict_proba(X.iloc[te])[:, 1]
    row = dict(model=name, fold=fold_id, repeat=fold_id // N_OUTER_SPLITS, outer_fold=fold_id % N_OUTER_SPLITS,
               n_train=len(tr), n_test=len(te), inner_best_pr_auc=inner, seconds=round(time.time() - t0, 1),
               feature_set=best.get("select__feature_set", spec["fixed"].get("select__feature_set", "")),
               best_params=json.dumps({k: (None if v is None else (float(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (int, np.integer)) else v))) for k, v in best.items()}),
               **full_metrics(y.iloc[te], p))
    pim = None
    if name != "Dummy":
        pi = permutation_importance(est, X.iloc[te], y.iloc[te], scoring="average_precision", n_repeats=3, random_state=seed, n_jobs=1)
        pim = pd.DataFrame({"model": name, "fold": fold_id, "feature": X.columns, "importance": pi.importances_mean})
    return row, te, p, pim


def _code_hash():
    h = hashlib.md5()
    for f in ("fraud_features.py", "fraud_models.py"):
        h.update((HERE / f).read_bytes())
    h.update(json.dumps(SPACE_DOC, sort_keys=True).encode())
    return h.hexdigest()


def run_nested_cv(X, y, ids, out_dir, n_jobs=2, force=False, verbose=True):
    """Runs (or loads from cache) the nested CV for every model. Returns fold table, OOF predictions, permutation importances."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "D_nested_cache.pkl"; key = _code_hash()
    if cache.exists() and not force:
        c = pickle.load(open(cache, "rb"))
        if c["key"] == key:
            if verbose: print("loaded cached nested-CV results (code hash matches).")
            return c["folds"], c["oof"], c["perm"]
        if verbose: print("cache is stale (code changed): recomputing.")
    cv = RepeatedStratifiedKFold(n_splits=N_OUTER_SPLITS, n_repeats=N_OUTER_REPEATS, random_state=SEED)
    splits = list(cv.split(np.zeros(len(y)), y))
    rows, perms = [], []
    oof = pd.DataFrame({"transaction_id": ids.values, "fraud": y.values})
    for name, spec in model_specs().items():
        t0 = time.time()
        res = Parallel(n_jobs=n_jobs)(delayed(_outer_fold)(name, spec, X, y, tr, te, i) for i, (tr, te) in enumerate(splits))
        for rep in range(N_OUTER_REPEATS): oof[f"p_{name}_r{rep}"] = np.nan
        for row, te, p, pim in res:
            rows.append(row); oof.loc[te, f"p_{name}_r{row['repeat']}"] = p
            if pim is not None: perms.append(pim)
        if verbose: print(f"{name}: {len(res)} outer folds in {time.time() - t0:.0f}s | mean PR-AUC "
                          f"{np.mean([r[0]['pr_auc'] for r in res]):.4f}", flush=True)
    folds, perm = pd.DataFrame(rows), pd.concat(perms, ignore_index=True)
    pickle.dump(dict(key=key, folds=folds, oof=oof, perm=perm), open(cache, "wb"))
    folds.to_csv(out_dir / "D_nested_fold_results.csv", index=False)
    oof.to_csv(out_dir / "D_oof_predictions.csv", index=False)
    perm.to_csv(out_dir / "D_permutation_importance_by_fold.csv", index=False)
    return folds, oof, perm


# ----------------------------------------------------------------------------- final fits
def fit_final_models(X, y, out_dir, force=False, verbose=True):
    """Refits every model on the full development set. Tuned models re-run the same search with 5-fold inner CV."""
    out_dir = Path(out_dir); mdir = out_dir / "models"; mdir.mkdir(parents=True, exist_ok=True)
    meta = mdir / "D_final_models.json"
    if meta.exists() and not force and json.load(open(meta)).get("key") == _code_hash():
        if verbose: print("loaded cached final models.")
        return json.load(open(meta)), {n: pd.read_csv(mdir / f"D_search_{n}.csv") for n in ("LR_tuned", "RF", "HGB")}
    info, searches = {"key": _code_hash(), "seed": SEED, "models": {}}, {}
    for name, spec in model_specs().items():
        t0 = time.time()
        if spec["space"] is None:
            est, best, score = _build(spec).fit(X, y), {}, None
        else:
            s = _search(spec, SEED, INNER_FOLDS).fit(X, y)
            est, best, score = s.best_estimator_, s.best_params_, float(s.best_score_)
            cvr = pd.DataFrame(s.cv_results_)
            cvr["params_json"] = cvr["params"].apply(lambda d: json.dumps({k: (None if v is None else (float(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (int, np.integer)) else v))) for k, v in d.items()}))
            keep = ["params_json", "mean_test_score", "std_test_score", "rank_test_score"] + [c for c in cvr.columns if c.startswith("param_")]
            cvr[keep].to_csv(mdir / f"D_search_{name}.csv", index=False); searches[name] = cvr[keep]
        dump(est, mdir / f"D_{name}.joblib")
        clean = {k: (None if v is None else (float(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (int, np.integer)) else v))) for k, v in best.items()}
        info["models"][name] = dict(best_params=clean, inner_cv_pr_auc=score, seconds=round(time.time() - t0, 1))
        if verbose: print(f"final {name}: {info['models'][name]}", flush=True)
    json.dump(info, open(meta, "w"), indent=2)
    return info, searches
