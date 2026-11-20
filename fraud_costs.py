"""Cost framework (Part E): cost matrix, cost scenarios, threshold search and honest (cross-validated) evaluation.

All amounts are hypothetical and every parameter is explicit (see SCENARIOS).  Thresholds are searched on a quantile grid of
the out-of-fold scores; the honest evaluation chooses thresholds on training folds only and applies them to unseen rows.
"""
from __future__ import annotations

from dataclasses import dataclass, replace, asdict

import numpy as np
from sklearn.model_selection import StratifiedKFold

GRID_POINTS = 301          # quantile grid: 300 equal steps in score rank


@dataclass(frozen=True)
class Scenario:
    name: str = "S1_base"
    loss_share: float = 1.0
    chargeback_fee: float = 15.0
    fp_margin: float = 0.05
    fp_friction: float = 20.0
    review_cost: float = 6.0
    review_detect: float = 0.85
    review_wrong_decline: float = 0.03
    review_delay: float = 1.5
    review_capacity: float = 0.10
    max_block_rate: float = 0.10

    def with_(self, **kw):
        return replace(self, **kw)

    def as_dict(self):
        return asdict(self)


SCENARIOS = {s.name: s for s in [
    Scenario(),
    Scenario("S2_fraud_averse", chargeback_fee=40.0, fp_friction=10.0),
    Scenario("S3_customer_first", fp_margin=0.08, fp_friction=60.0, review_delay=3.0),
    Scenario("S4_review_expensive", review_cost=15.0),
    Scenario("S5_review_cheap_big_team", review_cost=2.5, review_capacity=0.20),
    Scenario("S6_partial_recovery", loss_share=0.6, chargeback_fee=10.0),
    Scenario("S7_no_operational_limits", review_capacity=1.0, max_block_rate=1.0)]}
BASE = SCENARIOS["S1_base"]


def row_costs(amt, y, sc):
    """Per-transaction cost of (approve, review, block)."""
    y = np.asarray(y); amt = np.asarray(amt, float)
    c_app = np.where(y == 1, sc.loss_share * amt + sc.chargeback_fee, 0.0)
    c_blk = np.where(y == 0, sc.fp_margin * amt + sc.fp_friction, 0.0)
    c_rev = sc.review_cost + np.where(y == 1, (1 - sc.review_detect) * c_app, sc.review_delay + sc.review_wrong_decline * c_blk)
    return c_app, c_rev, c_blk


def make_grid(scores, n=GRID_POINTS):
    a = np.concatenate([np.asarray(s, float) for s in scores])
    return np.append(np.unique(np.quantile(a, np.linspace(0, 1, n))), np.inf)


def _cum(score, vals, g):
    o = np.argsort(score, kind="stable"); s = score[o]; pos = np.searchsorted(s, g, side="left")
    cs = np.concatenate([[0.0], np.cumsum(vals[o])]); return cs[pos], pos


def surface(S, costs, g):
    """cost[i, j] of 'approve below g[i], review g[i]..g[j], block from g[j]'; also review and block shares."""
    cA, cR, cB = costs; n = len(S[0]); k = len(S)
    CA = CR = CB = CNT = 0
    for s in S:
        a, pos = _cum(s, cA, g); r, _ = _cum(s, cR, g); b, _ = _cum(s, cB, g)
        CA = CA + a / k; CR = CR + r / k; CB = CB + b / k; CNT = CNT + pos / k
    cost = CA[:, None] + (CR[None, :] - CR[:, None]) + (cB.sum() - CB)[None, :]
    return cost, (CNT[None, :] - CNT[:, None]) / n, (n - CNT) / n


def best_pair(surf, g, kind, rev_cap, blk_cap):
    cost, rev, blk = surf; G = len(g)
    valid = np.triu(np.ones((G, G), bool))
    if kind == "block": valid = np.eye(G, dtype=bool)
    if kind == "review_only": valid = valid & (np.arange(G)[None, :] == G - 1)
    ok = valid & (rev <= rev_cap + 1e-9) & (blk[None, :] <= blk_cap + 1e-9)
    c = np.where(ok, cost, np.inf); i, j = np.unravel_index(np.argmin(c), c.shape)
    return g[i], g[j], c[i, j]


def _caps(kind, sc):
    return {"block": (1.0, sc.max_block_rate), "review_only": (sc.review_capacity, 1.0),
            "three_way": (sc.review_capacity, sc.max_block_rate)}[kind]


def chooser_cost(kind, sc, amt, y):
    """Returns f(S, idx) -> (t_review, t_block) minimising expected cost on rows idx, inside the operating limits."""
    costs = row_costs(amt, y, sc); rc, bc = _caps(kind, sc)

    def choose(S, idx):
        Si = [s[idx] for s in S]; ci = tuple(c[idx] for c in costs); g = make_grid(Si)
        t1, t2, _ = best_pair(surface(Si, ci, g), g, kind, rc, bc)
        return (t2, t2) if kind == "block" else (t1, t2)
    return choose


def chooser_metric(metric, y):
    """Single threshold maximising accuracy or F1 on rows idx (review and block thresholds equal)."""
    def choose(S, idx):
        yi = np.asarray(y)[idx]; Si = [s[idx] for s in S]; g = make_grid(Si)[:-1]; best, bt = -1, g[0]
        for t in g:
            vals = []
            for s in Si:
                pred = s >= t; tp = (pred & (yi == 1)).sum(); fp = (pred & (yi == 0)).sum(); fn = ((~pred) & (yi == 1)).sum(); tn = len(yi) - tp - fp - fn
                vals.append((tp + tn) / len(yi) if metric == "accuracy" else (2 * tp / (2 * tp + fp + fn) if tp else 0.0))
            v = float(np.mean(vals))
            if v > best + 1e-15: best, bt = v, t
        return bt, bt
    return choose


def apply_thresholds(score, t1, t2):
    return np.where(score < t1, 0, np.where(score < t2, 1, 2))


def fixed_row_costs(S, y, amt, sc, t1, t2):
    c = row_costs(amt, y, sc)
    return np.mean([np.choose(apply_thresholds(s, t1, t2), c) for s in S], axis=0)


def honest_row_costs(S, y, amt, sc, chooser, n_cv_repeats=10, k=5, seed=42):
    """Thresholds chosen on training folds only and applied to the unseen fold; averaged over CV repeats and score repeats."""
    n = len(y); c = row_costs(amt, y, sc); acc = np.zeros(n); ths = []
    for r in range(n_cv_repeats):
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed + r).split(np.zeros(n), y):
            t1, t2 = chooser(S, tr); ths.append((t1, t2))
            acc[te] += np.mean([np.choose(apply_thresholds(s[te], t1, t2), tuple(x[te] for x in c)) for s in S], axis=0)
    return acc / n_cv_repeats, ths


def policy_summary(S, y, amt, sc, t1, t2):
    y = np.asarray(y); c = row_costs(amt, y, sc); n = len(y); out = dict(cost=0.0, review_rate=0.0, block_rate=0.0, fraud_capture=0.0, legit_blocked=0.0, frauds_missed=0.0, block_precision=0.0)
    bp = []
    for s in S:
        a = apply_thresholds(s, t1, t2)
        out["cost"] += np.choose(a, c).mean() * 1000 / len(S); out["review_rate"] += (a == 1).mean() / len(S); out["block_rate"] += (a == 2).mean() / len(S)
        out["fraud_capture"] += (((a == 2) & (y == 1)).sum() + sc.review_detect * ((a == 1) & (y == 1)).sum()) / y.sum() / len(S); out["legit_blocked"] += ((a == 2) & (y == 0)).sum() / n * 1000 / len(S)
        out["frauds_missed"] += ((a == 0) & (y == 1)).sum() / n * 1000 / len(S)
        bp.append(((a == 2) & (y == 1)).sum() / max((a == 2).sum(), 1) if (a == 2).any() else np.nan)
    out["block_precision"] = float(np.nanmean(bp)) if not np.all(np.isnan(bp)) else np.nan
    return out


def paired_bootstrap(a, b, n_boot=2000, seed=0):
    """mean(a-b)*1000 with a 95% interval, resampling transactions; negative = a cheaper."""
    d = np.asarray(a) - np.asarray(b); rng = np.random.default_rng(seed); idx = rng.integers(0, len(d), (n_boot, len(d))); bs = d[idx].mean(1) * 1000
    return d.mean() * 1000, np.quantile(bs, .025), np.quantile(bs, .975), 2 * min((bs <= 0).mean(), (bs >= 0).mean())
