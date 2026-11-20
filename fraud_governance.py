"""Part K helpers: an independent re-implementation of the Part E cost framework (validated against the notebook's
printed results), plus monitoring / drift utilities. Reads only development-set files; the hold-out set is never touched."""
from dataclasses import dataclass, replace
import numpy as np, pandas as pd

@dataclass(frozen=True)
class Scenario:
    name: str = "S1_base"; loss_share: float = 1.0; chargeback_fee: float = 15.0; fp_margin: float = 0.05; fp_friction: float = 20.0
    review_cost: float = 6.0; review_detect: float = 0.85; review_wrong_decline: float = 0.03; review_delay: float = 1.5
    review_capacity: float = 0.10; max_block_rate: float = 0.10
    def with_(self, **kw): return replace(self, **kw)

SCENARIOS = {s.name: s for s in [
    Scenario(), Scenario("S2_fraud_averse", chargeback_fee=40.0, fp_friction=10.0),
    Scenario("S3_customer_first", fp_margin=0.08, fp_friction=60.0, review_delay=3.0), Scenario("S4_review_expensive", review_cost=15.0),
    Scenario("S5_review_cheap_big_team", review_cost=2.5, review_capacity=0.20), Scenario("S6_partial_recovery", loss_share=0.6, chargeback_fee=10.0),
    Scenario("S7_no_operational_limits", review_capacity=1.0, max_block_rate=1.0)]}

def row_costs(amt, y, sc):
    """per-row cost of (approve, review, block)"""
    y = np.asarray(y); amt = np.asarray(amt, float)
    c_app = np.where(y == 1, sc.loss_share * amt + sc.chargeback_fee, 0.0)
    c_blk = np.where(y == 0, sc.fp_margin * amt + sc.fp_friction, 0.0)
    c_rev = sc.review_cost + np.where(y == 1, (1 - sc.review_detect) * c_app, sc.review_delay + sc.review_wrong_decline * c_blk)
    return c_app, c_rev, c_blk

def make_grid(scores, n=None):
    u = np.unique(np.concatenate(scores)); 
    if n and len(u) > n: u = np.unique(np.quantile(np.concatenate(scores), np.linspace(0, 1, n)))
    return np.append(u, np.inf)

def _cum(score, vals, g):
    o = np.argsort(score, kind="stable"); s = score[o]; pos = np.searchsorted(s, g, side="left")  # rows with score < g[i]
    cs = np.concatenate([[0.0], np.cumsum(vals[o])]); return cs[pos], pos

def surface(S, costs, g):
    cA, cR, cB = costs; n = len(S[0]); tot = 0; rr = 0; 
    CA = CR = CB = 0; CNT = 0
    for s in S:
        a, pos = _cum(s, cA, g); r, _ = _cum(s, cR, g); b, _ = _cum(s, cB, g); CA = CA + a / len(S); CR = CR + r / len(S); CB = CB + b / len(S); CNT = CNT + pos / len(S)
    CBtot = cB.sum()
    cost = CA[:, None] + (CR[None, :] - CR[:, None]) + (CBtot - CB)[None, :]
    rev = (CNT[None, :] - CNT[:, None]) / n; blk = (n - CNT) / n
    return cost, rev, blk

def best_pair(surf, g, kind, rev_cap, blk_cap):
    cost, rev, blk = surf; G = len(g)
    valid = np.triu(np.ones((G, G), bool))                    # t_review <= t_block
    if kind == "block": valid = np.eye(G, dtype=bool)       # single block threshold: review band empty
    if kind == "review_only": valid = valid & (np.arange(G)[None, :] == G - 1)
    ok = valid & (rev <= rev_cap + 1e-9) & (blk[None, :] <= blk_cap + 1e-9)
    c = np.where(ok, cost, np.inf); i, j = np.unravel_index(np.argmin(c), c.shape); return g[i], g[j], c[i, j]

def fixed_costs(score, costs, t1, t2):
    cA, cR, cB = costs; return np.where(score < t1, cA, np.where(score < t2, cR, cB))

# ------------------------------------------------------------------ honest (cross-validated) policy cost
from sklearn.model_selection import StratifiedKFold
def choose(S, costs, sc, idx, kind="three_way", grid_n=500, rev_cap=None, blk_cap=None):
    rev_cap = sc.review_capacity if rev_cap is None else rev_cap; blk_cap = sc.max_block_rate if blk_cap is None else blk_cap
    Si = [s[idx] for s in S]; ci = tuple(c[idx] for c in costs); g = make_grid(Si, grid_n)
    return best_pair(surface(Si, ci, g), g, kind, rev_cap, blk_cap)[:2]

def honest_row_costs(S, y, costs, sc, kind="three_way", n_rep=10, k=5, seed=42, grid_n=500):
    """thresholds chosen on training rows only, applied to unseen rows; per-row cost averaged over n_rep CV repeats and the score repeats"""
    n = len(y); acc = np.zeros(n)
    for r in range(n_rep):
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed + r).split(np.zeros(n), y):
            t1, t2 = choose(S, costs, sc, tr, kind, grid_n)
            acc[te] += np.mean([fixed_costs(s[te], tuple(c[te] for c in costs), t1, t2) for s in S], axis=0)
    return acc / n_rep

def paired_bootstrap(a, b, B=2000, seed=0):
    """mean(a-b)*1000 with 95% interval, resampling transactions; negative = a cheaper"""
    d = np.asarray(a) - np.asarray(b); rng = np.random.default_rng(seed); idx = rng.integers(0, len(d), (B, len(d))); bs = d[idx].mean(1) * 1000
    return d.mean() * 1000, np.quantile(bs, .025), np.quantile(bs, .975), 2 * min((bs <= 0).mean(), (bs >= 0).mean())

def load_dev(oof_path, raw_path):
    """out-of-fold scores of the four candidates + development-row amounts (40 missing amounts filled with the dev median, label-free, as in Part E)"""
    oof = pd.read_csv(oof_path); raw = pd.read_csv(raw_path, keep_default_na=False, na_values=[''])
    amt = oof.transaction_id.map(raw.set_index("transaction_id").transaction_amount); AMT = amt.fillna(amt.median()).to_numpy(); Y = oof.fraud.to_numpy()
    S = {m: [oof[f"p_{m}_r0"].to_numpy(), oof[f"p_{m}_r1"].to_numpy()] for m in ["LR_baseline", "LR_tuned", "RF", "HGB"]}
    return oof, raw, AMT, Y, S

# ------------------------------------------------------------------ Part K evidence functions (each returns a DataFrame)
import glob, itertools
from pathlib import Path
from sklearn.metrics import average_precision_score
MODELS = ["LR_baseline", "LR_tuned", "RF", "HGB"]

def find(name, *dirs):
    for d in dirs:
        hit = sorted(Path(d).glob(f"*{name}"))
        if hit: return str(hit[0])
    raise FileNotFoundError(name)

def value_weighted_capture(S, Y, AMT, tops=(.05, .10, .20, .30)):
    """share of fraud COUNT and fraud VALUE found in the top k% of scores (what a capacity-limited policy can reach)"""
    N = len(Y); rows = []
    for m in S:
        for k in tops:
            r, v = [], []
            for s in S[m]:
                sel = np.zeros(N, bool); sel[np.argsort(-s, kind="stable")[:int(round(k * N))]] = True
                r.append(Y[sel].sum() / Y.sum()); v.append(AMT[sel & (Y == 1)].sum() / AMT[Y == 1].sum())
            rows.append(dict(model=m, top_share=k, fraud_count_recall=np.mean(r), fraud_value_recall=np.mean(v)))
    return pd.DataFrame(rows)

def reliability_bins(S, Y, models=("LR_baseline", "RF"), bins=10):
    rows = []
    for m in models:
        p = np.concatenate(S[m]); y = np.concatenate([Y, Y]); q = pd.qcut(p, bins, labels=False, duplicates="drop")
        for b in np.unique(q): rows.append(dict(model=m, bin=int(b) + 1, n=int((q == b).sum()) // 2, mean_predicted=p[q == b].mean(), observed_rate=y[q == b].mean()))
    return pd.DataFrame(rows)

def workload_vs_recall(S, Y, targets=(.2, .3, .4, .5, .6, .7, .8, .9), monthly=100000, reviews_per_analyst=1072):
    N = len(Y); rows = []
    for m in S:
        rec, prec = [], []
        for s in S[m]:
            o = np.argsort(-s, kind="stable"); cf = np.cumsum(Y[o]); rec.append(cf / Y.sum()); prec.append(cf / np.arange(1, N + 1))
        rec, prec = np.mean(rec, 0), np.mean(prec, 0)
        for tr in targets:
            k = int(np.argmax(rec >= tr)) + 1
            rows.append(dict(model=m, target_recall=tr, flagged_share=k / N, reviews_per_month=monthly * k / N, analysts=monthly * k / N / reviews_per_analyst, precision=prec[k - 1], flagged_per_fraud_found=k / (tr * Y.sum())))
    W = pd.DataFrame(rows)
    W["extra_reviews_per_extra_fraud"] = W.groupby("model", group_keys=False).apply(lambda g: (g.flagged_share.diff() * N) / (g.target_recall.diff() * Y.sum()))
    return W

def interaction_pairs(d, seed=42, B=1000):
    """empirical pairwise interaction on the DEVELOPMENT rows: fraud rate when both flags present vs the additive expectation (risk-difference scale)"""
    rng = np.random.default_rng(seed); y = d.fraud.to_numpy(); base = y.mean()
    F = pd.DataFrame({"new_device": d.new_device == 1, "billing_mismatch": d.billing_address_mismatch == 1, "auth_None": d.authentication_level.isna(), "prior_chargeback": d.prior_chargebacks >= 1,
        "failed_logins_2plus": d.failed_logins_24h >= 2, "merchant_risk_top_quartile": d.merchant_risk_score >= d.merchant_risk_score.quantile(.75),
        "ip_risk_top_quartile": d.ip_risk_score >= d.ip_risk_score.quantile(.75), "velocity_1h_3plus": d.transactions_last_1h >= 3, "amount_top_10pct": d.transaction_amount >= d.transaction_amount.quantile(.90)}).astype(int)
    def cells(a, b, yy):
        out = {}
        for i in (0, 1):
            for j in (0, 1):
                m = (a == i) & (b == j); out[(i, j)] = (yy[m].mean() if m.sum() else np.nan, int(m.sum()))
        return out
    rows = []
    for ca, cb in itertools.combinations(F.columns, 2):
        a, b = F[ca].to_numpy(), F[cb].to_numpy(); r = cells(a, b, y); ex = r[(1, 1)][0] - (r[(1, 0)][0] + r[(0, 1)][0] - r[(0, 0)][0]); bs = []
        for _ in range(B):
            ix = rng.integers(0, len(y), len(y)); rr = cells(a[ix], b[ix], y[ix]); bs.append(rr[(1, 1)][0] - (rr[(1, 0)][0] + rr[(0, 1)][0] - rr[(0, 0)][0]))
        lo, hi = np.nanquantile(bs, [.025, .975])
        rows.append(dict(flag_a=ca, flag_b=cb, n_both=r[(1, 1)][1], rate_neither=r[(0, 0)][0], rate_a_only=r[(1, 0)][0], rate_b_only=r[(0, 1)][0], rate_both=r[(1, 1)][0], joint_lift_vs_base=r[(1, 1)][0] / base,
                         excess_over_additive=ex, ci_low=lo, ci_high=hi, excludes_zero=bool((lo > 0) | (hi < 0))))
    nf = (F.new_device + F.billing_mismatch + F.prior_chargeback + F.auth_None + F.failed_logins_2plus).clip(upper=3)
    acc = pd.DataFrame({"flags": nf, "y": y}).groupby("flags").y.agg(n="size", fraud_rate="mean").reset_index()
    return pd.DataFrame(rows).sort_values("rate_both", ascending=False).reset_index(drop=True), acc

TORNADO_RANGES = {"loss_share": (.5, 1.0), "chargeback_fee": (5, 60), "fp_margin": (0, .10), "fp_friction": (5, 60), "review_cost": (2, 15), "review_detect": (.6, .95),
                  "review_wrong_decline": (0, .10), "review_delay": (0, 5), "review_capacity": (.05, .20), "max_block_rate": (.02, .20)}
def tornado(scores, AMT, Y, base=None, ranges=TORNADO_RANGES):
    base = base or SCENARIOS["S1_base"]; N = len(Y); g = make_grid(scores)
    def opt(sc): c = row_costs(AMT, Y, sc); t1, t2, cc = best_pair(surface(scores, c, g), g, "three_way", sc.review_capacity, sc.max_block_rate); return t1, t2, cc / N * 1000
    def at(sc, t1, t2): c = row_costs(AMT, Y, sc); return np.mean([fixed_costs(s, c, t1, t2).mean() for s in scores]) * 1000
    T1, T2, CB = opt(base); rows = []
    for p, (lo, hi) in ranges.items():
        row = dict(parameter=p, low=lo, high=hi)
        for tag, v in (("low", lo), ("high", hi)):
            sc = base.with_(**{p: v}); t1, t2, c = opt(sc); row.update({f"t_review_{tag}": t1, f"t_block_{tag}": t2, f"cost_{tag}": c, f"regret_{tag}": at(sc, T1, T2) - c})
        row["swing"] = abs(row["cost_high"] - row["cost_low"]); rows.append(row)
    return pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True), (T1, T2, CB)

def capacity_sweep(S, AMT, Y, caps=(0, .025, .05, .10, .15, .20, .30, .40, .50, .75, 1.0), block_cap=.10):
    rows = []; base = SCENARIOS["S1_base"]
    for cap in caps:
        sc = base.with_(name=f"cap{cap}", review_capacity=cap, max_block_rate=block_cap); costs = row_costs(AMT, Y, sc); R = {m: honest_row_costs(S[m], Y, costs, sc) for m in S}
        for m in S: rows.append(dict(review_capacity=cap, model=m, cost_per_1000=R[m].mean() * 1000))
        for m in ("LR_baseline", "LR_tuned", "HGB"):
            d, lo, hi, p = paired_bootstrap(R[m], R["RF"]); rows.append(dict(review_capacity=cap, model=f"{m}_minus_RF", cost_per_1000=d, ci_low=lo, ci_high=hi, p=p))
    return pd.DataFrame(rows)

def cost_structure_monte_carlo(S, AMT, Y, K=300, seed=42, n_rep=4):
    """draw K cost structures from wide ranges (fixed before running); honest cost of every model; paired bootstrap RF vs best alternative"""
    rng = np.random.default_rng(seed); lu = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b)))); out = []
    for k in range(K):
        p = dict(loss_share=rng.uniform(.4, 1.0), chargeback_fee=lu(5, 80), fp_margin=rng.uniform(0, .15), fp_friction=lu(5, 300), review_cost=lu(1, 25), review_detect=rng.uniform(.5, .98),
                 review_wrong_decline=rng.uniform(0, .15), review_delay=rng.uniform(0, 6), review_capacity=float(rng.choice([.02, .05, .10, .15, .20, .30, .50])), max_block_rate=float(rng.choice([.02, .05, .10, .20, 1.0])))
        sc = Scenario(name=f"mc{k}", **p); costs = row_costs(AMT, Y, sc); R = {m: honest_row_costs(S[m], Y, costs, sc, n_rep=n_rep) for m in S}
        aa = costs[0].mean() * 1000; c = {m: R[m].mean() * 1000 for m in S}; alt = min([m for m in S if m != "RF"], key=c.get); d, lo, hi, _ = paired_bootstrap(R["RF"], R[alt], B=500, seed=k)
        out.append(dict(k=k, **p, fn_fp_ratio=(p["loss_share"] * AMT[Y == 1].mean() + p["chargeback_fee"]) / (p["fp_margin"] * AMT[Y == 0].mean() + p["fp_friction"]), approve_all=aa,
                        **{f"cost_{m}": c[m] for m in S}, cheapest=min(c, key=c.get), best_alternative=alt, RF_minus_best_alt=d, ci_low=lo, ci_high=hi))
    MC = pd.DataFrame(out); MC["model_saving_vs_approve_all_pct"] = 100 * (MC.approve_all - MC.cost_RF) / MC.approve_all; return MC

def summarise_monte_carlo(MC, by, bins, labels):
    g = MC.assign(bucket=pd.cut(MC[by], bins, labels=labels))
    def f(x): return pd.Series(dict(structures=len(x), RF_cheapest_pct=100 * (x.cheapest == "RF").mean(), HGB_cheapest_pct=100 * (x.cheapest == "HGB").mean(), LR_cheapest_pct=100 * x.cheapest.isin(["LR_tuned", "LR_baseline"]).mean(),
        RF_significantly_cheaper_pct=100 * (x.ci_high < 0).mean(), alternative_significantly_cheaper_pct=100 * (x.ci_low > 0).mean(), median_RF_minus_best_alt_per1000=x.RF_minus_best_alt.median(), median_model_saving_vs_approve_all_pct=x.model_saving_vs_approve_all_pct.median()))
    return g.groupby("bucket", observed=True).apply(f)

def psi(expected, actual, edges):
    e = np.clip(np.histogram(expected, edges)[0] / len(expected), 1e-4, None); a = np.clip(np.histogram(actual, edges)[0] / len(actual), 1e-4, None); return float(((a - e) * np.log(a / e)).sum())

def psi_null(fields, windows=(("day", 4545), ("week", 22727), ("month", 100000)), reps=300, seed=42):
    """PSI you would see from sampling noise alone when the live population equals the reference"""
    rng = np.random.default_rng(seed); rows = []
    for f, x in fields.items():
        edges = np.unique(np.quantile(x, np.linspace(0, 1, 11))); edges[0], edges[-1] = -np.inf, np.inf
        for lab, n in windows:
            v = [psi(x, rng.choice(x, n, replace=True), edges) for _ in range(reps)]
            rows.append(dict(field=f, window=lab, n=n, psi_null_median=np.median(v), psi_null_p95=np.quantile(v, .95), psi_null_p99=np.quantile(v, .99)))
    return pd.DataFrame(rows)

def prauc_window_noise(score, Y, frauds=(30, 60, 100, 200, 400), reps=400, seed=42):
    rng = np.random.default_rng(seed); pos, neg = np.where(Y == 1)[0], np.where(Y == 0)[0]; rows = []
    for F in frauds:
        n = int(round(F / Y.mean())); v = []
        for _ in range(reps):
            ix = np.concatenate([rng.choice(pos, F, replace=True), rng.choice(neg, n - F, replace=True)]); v.append(average_precision_score(Y[ix], score[ix]))
        sd = np.std(v); rows.append(dict(frauds_in_window=F, rows_at_dev_rate=n, pr_auc_sd=sd, min_detectable_relative_drop=2.49 * sd / np.mean(v)))
    return pd.DataFrame(rows)
