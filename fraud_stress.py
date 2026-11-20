"""Part J: robustness and stress testing of the fraud decision engine.

Everything is out-of-fold: the production Random Forest pipeline is refitted on the training part of each outer fold (5 x 2, same seed as Part D), and
every stress is applied to the TEST part only, so training stays clean and no row is scored by a model that saw it. The hold-out set is never read.

Two families of stress, which answer different questions:
  * population shifts (importance weights): who arrives changes (more large amounts, more velocity, another channel mix, another fraud rate) but the
    relationship between a transaction's inputs and fraud stays the same. Labels and inputs are untouched; rows are re-weighted (exponential tilting
    for continuous fields, post-stratification for channel, class weights for prevalence).
  * input perturbations: the model is fed distorted inputs (units drift, counting changes, missing values, outliers, fraudsters adapting). Labels are
    kept as they were, so the effect on scores and money is measured exactly.
Metrics are weighted so that both families use the same code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

import fraud_calibration as fcal
import fraud_costs as fc
import fraud_eval as fe
import fraud_ops as fo

SEED = 42


# ------------------------------------------------------------------ out-of-fold models
def fit_folds(template, X, y, splits, train_filter=None):
    """One fitted pipeline per outer fold. `train_filter(Xtr) -> boolean mask of training rows to KEEP` is used by the extreme-observation refit test."""
    models = []
    for tr, te in splits:
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        if train_filter is not None:
            keep = np.asarray(train_filter(Xtr)); Xtr, ytr = Xtr[keep], ytr[keep]
        models.append(clone(template).fit(Xtr, ytr))
    return models


def platt(p_raw, cal):
    p = np.clip(np.asarray(p_raw, float), 1e-4, 1 - 1e-4); return fcal.sigmoid(cal["intercept"] + cal["slope"] * fcal.logit(p))


def oof_predict(models, splits, X, y, amount_true, n_repeats=2, perturb=None, cal=None, seed=SEED, keep_cols=()):
    """Out-of-fold scores for every row and repeat. `perturb(Xte, yte, rng, Xtr, at) -> (X_perturbed, true_amount_of_those_rows)` is applied to each test fold.
    Returns dict(p=(R, n) calibrated probabilities, amount_seen=(R, n), amount_true=(R, n))."""
    n = len(y); R = n_repeats; P = np.zeros((R, n)); AS = np.zeros((R, n)); AT = np.zeros((R, n)); XS = pd.DataFrame(index=range(n), columns=list(keep_cols), dtype=object)
    for i, ((tr, te), m) in enumerate(zip(splits, models)):
        r = i // 5; rng = np.random.default_rng(seed + 1000 * i); Xte = X.iloc[te]; at = np.asarray(amount_true)[te].copy()
        if perturb is not None: Xte, at = perturb(Xte.copy(), y.iloc[te], rng, X.iloc[tr], at)
        raw = m.predict_proba(Xte)[:, 1]; P[r, te] = platt(raw, cal) if cal is not None else raw
        AS[r, te] = Xte["transaction_amount"].to_numpy(); AT[r, te] = at
        if r == 0 and keep_cols: XS.loc[te, list(keep_cols)] = Xte[list(keep_cols)].to_numpy()
    return dict(p=P, amount_seen=AS, amount_true=AT, X_seen=XS.infer_objects())


# ------------------------------------------------------------------ weights
def tilt_weights(x, target_mean, base_w=None):
    """Exponential tilting: w = exp(beta * z) with beta chosen so the weighted mean of x equals target_mean (bisection). Returns weights (mean 1)."""
    x = np.asarray(x, float); ok = ~np.isnan(x); z = np.zeros(len(x)); z[ok] = (x[ok] - x[ok].mean()) / x[ok].std(); b0 = np.ones(len(x)) if base_w is None else base_w
    def wm(beta): w = b0 * np.exp(beta * z); return (w[ok] * x[ok]).sum() / w[ok].sum()
    lo, hi = -8.0, 8.0                                              # the weighted mean rises with beta, so plain bisection works
    for _ in range(80):
        mid = (lo + hi) / 2
        if wm(mid) < target_mean: lo = mid
        else: hi = mid
    w = b0 * np.exp(((lo + hi) / 2) * z); return w / w.mean()


def poststrat_weights(cat, target_share):
    """Weights that move the category mix to target_share (dict category -> share; unspecified categories share the remainder in proportion)."""
    cat = pd.Series(cat).astype(str).reset_index(drop=True); base = cat.value_counts(normalize=True); rest = 1 - sum(target_share.values()); other = [c for c in base.index if c not in target_share]
    tgt = dict(target_share); tgt.update({c: rest * base[c] / base[other].sum() for c in other}); w = cat.map(lambda c: tgt[c] / base[c]).to_numpy(); return w / w.mean()


def prevalence_weights(y, target_rate):
    y = np.asarray(y); base = y.mean(); w = np.where(y == 1, target_rate / base, (1 - target_rate) / (1 - base)); return w / w.mean()


def ess(w):
    w = np.asarray(w, float); return float(w.sum() ** 2 / (w ** 2).sum())


# ------------------------------------------------------------------ weighted metrics
def w_recall_top(y, p, w, frac=0.10):
    y, p, w = map(np.asarray, (y, p, w)); o = np.argsort(-p, kind="stable"); cw = np.cumsum(w[o]) / w.sum(); sel = np.zeros(len(y), bool); sel[o[: np.searchsorted(cw, frac) + 1]] = True
    return float((w * y * sel).sum() / (w * y).sum())


def w_calibration(y, p, w, n_bins=10):
    y, p, w = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4), np.asarray(w, float); lg = fcal.logit(p)
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(lg[:, None], y, sample_weight=w); o = np.argsort(p, kind="stable"); cw = np.cumsum(w[o]) / w.sum(); b = np.minimum((cw * n_bins).astype(int), n_bins - 1)
    ece = sum((w[o][b == k].sum() / w.sum()) * abs(np.average(p[o][b == k], weights=w[o][b == k]) - np.average(y[o][b == k], weights=w[o][b == k])) for k in range(n_bins) if w[o][b == k].sum() > 0)
    return dict(mean_pred=float(np.average(p, weights=w)), observed=float(np.average(y, weights=w)), cal_slope=float(lr.coef_[0, 0]), cal_intercept=float(lr.intercept_[0]), ece=float(ece),
                brier=float(np.average((p - y) ** 2, weights=w)))


def capped_bayes_weighted(p, amount, sc, w, rounds=6, iters=40):
    """Label-free shadow-price rule of Part F, with weighted shares (used to re-price the policy for a shifted population)."""
    w = np.asarray(w, float) / np.sum(w); E = fcal.expected_costs(p, amount, sc)
    def acts(lam, mu): M = E.copy(); M[:, 1] += lam; M[:, 2] += mu; return np.argmin(M, axis=1)
    def smallest(share_fn, cap, hi=1e5):
        if share_fn(0.0) <= cap + 1e-12: return 0.0
        lo = 0.0
        for _ in range(iters):
            mid = (lo + hi) / 2
            if share_fn(mid) <= cap + 1e-12: hi = mid
            else: lo = mid
        return hi
    lam = mu = 0.0
    for _ in range(rounds):
        lam = smallest(lambda x: float((w * (acts(x, mu) == 1)).sum()), sc.review_capacity); mu = smallest(lambda x: float((w * (acts(lam, x) == 2)).sum()), sc.max_block_rate)
    return acts(lam, mu), lam, mu


def policy_eval(actions, y, amount_true, w, sc):
    y = np.asarray(y); w = np.asarray(w, float) / np.sum(w); c_app, c_rev, c_blk = fc.row_costs(amount_true, y, sc); a = np.asarray(actions)
    cost = np.where(a == 2, c_blk, np.where(a == 1, c_rev, c_app)); wy = (w * y).sum(); caught = (w * y * (a == 2)).sum() + sc.review_detect * (w * y * (a == 1)).sum()
    return dict(cost=float((w * cost).sum()), cost_approve_all=float((w * c_app).sum()), review_share=float((w * (a == 1)).sum()), block_share=float((w * (a == 2)).sum()),
                capture=float(caught / wy) if wy > 0 else np.nan, legit_blocked_share=float((w * (1 - y) * (a == 2)).sum() / (w * (1 - y)).sum()),
                review_precision=float((w * y * (a == 1)).sum() / max((w * (a == 1)).sum(), 1e-12)))


def evaluate_scenario(S, y, w, sc, lam, mu, scale=100_000, adaptive=True, p_override=None):
    """S = output of oof_predict. Averages over repeats. Returns one flat dict of metrics."""
    y = np.asarray(y); rows = []
    for r in range(S["p"].shape[0]):
        p = S["p"][r] if p_override is None else p_override[r]; a_seen = np.where(np.isnan(S["amount_seen"][r]), np.nan, S["amount_seen"][r]); a_seen = np.where(np.isnan(a_seen), np.nanmedian(a_seen), a_seen)
        m = dict(pr_auc=average_precision_score(y, p, sample_weight=w), roc_auc=roc_auc_score(y, p, sample_weight=w), recall_top10=w_recall_top(y, p, w)); m.update(w_calibration(y, p, w))
        pf = policy_eval(fo.decide(p, a_seen, sc, lam, mu), y, S["amount_true"][r], w, sc); m.update({f"frozen_{k}": v for k, v in pf.items()})
        if adaptive:
            act, l2, m2 = capped_bayes_weighted(p, a_seen, sc, w); pa = policy_eval(act, y, S["amount_true"][r], w, sc); m.update({f"adaptive_{k}": v for k, v in pa.items()}); m["adaptive_shadow_review"] = l2
        rows.append(m)
    out = pd.DataFrame(rows).mean().to_dict(); out["fraud_rate"] = float(np.average(y, weights=w)); out["ess"] = ess(w)
    for k in ("frozen", "adaptive"):
        if f"{k}_cost" in out: out[f"{k}_cost_per_100k"] = out[f"{k}_cost"] * scale; out[f"{k}_saving_pct"] = 100 * (1 - out[f"{k}_cost"] / out[f"{k}_cost_approve_all"])
    out["approve_all_per_100k"] = out["frozen_cost_approve_all"] * scale; return out


def bootstrap_metric(S, y, w, fn, B=200, seed=SEED):
    """Row-bootstrap interval of a statistic fn(y, p, w) averaged over the repeats (weights carried with the rows)."""
    rng = np.random.default_rng(seed); y = np.asarray(y); n = len(y); vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        try: vals.append(np.mean([fn(y[i], S["p"][r][i], np.asarray(w)[i]) for r in range(S["p"].shape[0])]))
        except Exception: pass
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


# ------------------------------------------------------------------ monitoring statistics
def psi(base, new, w_new=None, bins=10):
    """Population stability index of `new` against `base`, with bin edges from base quantiles (continuous) or its categories (non-numeric)."""
    base = pd.Series(base).dropna(); new = pd.Series(new); w_new = np.ones(len(new)) if w_new is None else np.asarray(w_new, float); ok = ~new.isna().to_numpy(); new = new[ok]; w_new = w_new[ok]
    if not pd.api.types.is_numeric_dtype(base):
        cats = sorted(set(base.astype(str)) | set(new.astype(str))); b = base.astype(str).value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
        n_ = pd.Series(w_new).groupby(new.astype(str).to_numpy()).sum().reindex(cats).fillna(0).to_numpy(); n_ = n_ / n_.sum()
    else:
        new = pd.to_numeric(new, errors="coerce"); ok2 = ~new.isna().to_numpy(); new, w_new = new[ok2], w_new[ok2]
        edges = np.unique(np.quantile(base, np.linspace(0, 1, bins + 1))); edges[0], edges[-1] = -np.inf, np.inf
        b = np.histogram(base, edges)[0] / len(base); n_ = np.histogram(new.to_numpy(float), edges, weights=w_new)[0]; n_ = n_ / n_.sum()
    b, n_ = np.clip(b, 1e-4, None), np.clip(n_, 1e-4, None); return float(np.sum((n_ - b) * np.log(n_ / b)))


# ------------------------------------------------------------------ fold-level variability
def fold_table(P, y, splits, amount_true, sc, lam, mu):
    """Per outer fold: ranking, calibration and policy cost of the frozen policy, from out-of-fold probabilities P (R, n)."""
    y = np.asarray(y); rows = []
    for i, (tr, te) in enumerate(splits):
        p = P[i // 5][te]; yy = y[te]; at = np.asarray(amount_true)[te]; act = fo.decide(p, at, sc, lam, mu); pe = policy_eval(act, yy, at, np.ones(len(te)), sc); cal = w_calibration(yy, p, np.ones(len(te)))
        rows.append(dict(fold=i, repeat=i // 5, n=len(te), frauds=int(yy.sum()), pr_auc=average_precision_score(yy, p), roc_auc=roc_auc_score(yy, p), recall_top10=w_recall_top(yy, p, np.ones(len(te))),
                         brier=cal["brier"], mean_pred=cal["mean_pred"], observed=cal["observed"], cal_slope=cal["cal_slope"], ece=cal["ece"], cost_per_1000=pe["cost"] * 1000, saving_pct=100 * (1 - pe["cost"] / pe["cost_approve_all"]),
                         review_share=pe["review_share"], block_share=pe["block_share"], capture=pe["capture"]))
    return pd.DataFrame(rows)


def nadeau_bengio_se(values, n_train, n_test):
    """Standard error of a mean over k resampled folds, corrected for the overlap of training sets (Nadeau and Bengio, 2003)."""
    v = np.asarray(values, float); k = len(v); return float(np.sqrt((1 / k + n_test / n_train) * v.var(ddof=1)))


# ------------------------------------------------------------------ input perturbations: f(Xte, yte, rng, Xtr, at) -> (X', true amount)
def p_inflate_amount(k):
    """Real inflation: amounts (what the model sees AND what is at stake) are multiplied by k."""
    def f(X, y, rng, Xtr, at): X["transaction_amount"] = X["transaction_amount"] * k; return X, at * k
    return f


def p_scale_velocity(k):
    """Counting change: the recorded 1h and 24h transaction counts are multiplied by k (what really happened is unchanged)."""
    def f(X, y, rng, Xtr, at):
        for c in ("transactions_last_1h", "transactions_last_24h"): X[c] = np.rint(X[c] * k).astype(int)
        return X, at
    return f


def p_relabel_channel(frac, src="POS", dst="E-commerce"):
    def f(X, y, rng, Xtr, at):
        m = (X["channel"].astype(str) == src).to_numpy() & (rng.random(len(X)) < frac); ch = X["channel"].astype(object).copy(); ch[m] = dst; X["channel"] = ch; return X, at
    return f


def p_evasion(frac, fields=("new_device", "billing_address_mismatch", "failed_logins_24h", "prior_chargebacks")):
    """Fraudsters adapt: for a share of the frauds the visible red flags are switched off."""
    def f(X, y, rng, Xtr, at):
        m = (np.asarray(y) == 1) & (rng.random(len(X)) < frac)
        for c in fields: X.loc[m, c] = 0
        return X, at
    return f


def p_missing(cols, rate):
    """Values go missing completely at random."""
    def f(X, y, rng, Xtr, at):
        for c in cols:
            m = rng.random(len(X)) < rate
            if not pd.api.types.is_numeric_dtype(X[c]): X[c] = X[c].astype(object).where(~m, np.nan)
            else: X[c] = X[c].astype(float).where(~m, np.nan)
        return X, at
    return f


def p_missing_informative(cols, rate):
    """Values go missing preferentially when they are high (a failing risk-score feed that drops its worst readings)."""
    def f(X, y, rng, Xtr, at):
        for c in cols:
            hi = (X[c] > Xtr[c].median()).to_numpy(); m = hi & (rng.random(len(X)) < rate); X[c] = X[c].astype(float).where(~m, np.nan)
        return X, at
    return f


def p_outliers(col, mult, frac, cap_q=None):
    """A share of rows get an extreme value (col x mult), for example a unit error. cap_q: clip the input at that training quantile before scoring (mitigation)."""
    def f(X, y, rng, Xtr, at):
        m = rng.random(len(X)) < frac; X[col] = X[col].astype(float).where(~m, X[col] * mult)
        if cap_q is not None: X[col] = X[col].clip(upper=Xtr[col].quantile(cap_q))
        return X, at
    return f
