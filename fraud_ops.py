"""Part I: fraud operations simulation.

Turns calibrated fraud probabilities into the three operational decisions (APPROVE 0, MANUAL REVIEW 1, BLOCK 2), and measures what each decision
policy would do to volumes, fraud capture, customer friction, analyst workload and money, per month of hypothetical traffic.

All money and staffing figures are HYPOTHETICAL assumptions stated in the notebook (OPS dict) on top of the Part E cost scenarios (fraud_costs.py).
Everything here is label-free at decision time: a policy sees only a calibrated probability, the amount and (for the legacy rules) two observable fields.
Labels are used only afterwards to score the decisions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import fraud_calibration as fcal
import fraud_costs as fc

APPROVE, REVIEW, BLOCK = fc.APPROVE, fc.REVIEW, fc.BLOCK

# Staffing assumptions (hypothetical). 8 minutes x 45 per hour = 6.0 per review, which is the review_cost of the base scenario.
OPS = dict(monthly_transactions=100_000, days_per_month=30, working_days=22, minutes_per_review=8.0, analyst_cost_per_hour=45.0, productive_hours_per_day=6.5)


def reviews_per_analyst_month(ops=OPS):
    return ops["productive_hours_per_day"] * 60.0 / ops["minutes_per_review"] * ops["working_days"]


# ------------------------------------------------------------------ actions
def honest_actions(score_list, y, amount, sc, choose, n_cv_repeats=5, k=5, seed=0):
    """Thresholds are chosen on training folds only and applied to the unseen fold. Returns an (R, n) array of actions (R = n_cv_repeats x len(score_list))
    and a table of the thresholds chosen in each fold."""
    y = np.asarray(y); n = len(y); out, chosen = [], []
    for r in range(n_cv_repeats):
        a = np.zeros((len(score_list), n), int)
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed + r).split(np.zeros(n), y):
            t1, t2 = choose(score_list, tr); chosen.append((t1, t2))
            for j, s in enumerate(score_list): a[j, te] = fc.apply_thresholds(np.asarray(s)[te], t1, t2)
        out.extend(list(a))
    return np.array(out), pd.DataFrame(chosen, columns=["t_review", "t_block"])


def amount_aware_actions(prob_list, amount, sc):
    """The proposed rule: cheapest expected action per transaction under the cost table, with label-free shadow prices that keep the review share
    and block share within the operating limits. One action vector per probability vector."""
    acts, lam, mu = [], [], []
    for p in prob_list:
        a, l, m = fcal.capped_bayes_actions(p, amount, sc); acts.append(a); lam.append(l); mu.append(m)
    return np.array(acts), float(np.mean(lam)), float(np.mean(mu))


def static_actions(rule_actions, n_reps):
    return np.tile(np.asarray(rule_actions, int), (n_reps, 1))


# ------------------------------------------------------------------ money
def cost_components(actions, y, amount, sc):
    """Per-row expected cost split into four parts, averaged over the action repeats.
    fraud_loss: money lost on frauds that get through (approved, or missed by the reviewer) incl. chargeback fee
    review_labour: analyst cost of every review
    review_friction: delay and wrong declines of legitimate customers in review
    block_friction: lost margin and friction of legitimate customers who are blocked"""
    y = np.asarray(y); a_ = np.asarray(amount, float); fn = sc.loss_share * a_ + sc.chargeback_fee; fp = sc.fp_margin * a_ + sc.fp_friction
    comp = dict(fraud_loss=[], review_labour=[], review_friction=[], block_friction=[])
    for a in np.atleast_2d(actions):
        appr, rev, blk = a == APPROVE, a == REVIEW, a == BLOCK
        comp["fraud_loss"].append(y * fn * (appr + (1 - sc.review_detect) * rev))
        comp["review_labour"].append(rev * sc.review_cost * np.ones_like(fn))
        comp["review_friction"].append(rev * (1 - y) * (sc.review_delay + sc.review_wrong_decline * fp))
        comp["block_friction"].append(blk * (1 - y) * fp)
    out = pd.DataFrame({k: np.mean(v, axis=0) for k, v in comp.items()}); out["total"] = out.sum(axis=1); return out


def row_cost_matrix(actions, y, amount, sc):
    """(R, n) realised expected cost of each action vector; matches fraud_costs.realised_cost."""
    c = fc.row_costs(amount, y, sc); return np.array([np.where(a == BLOCK, c[2], np.where(a == REVIEW, c[1], c[0])) for a in np.atleast_2d(actions)])


# ------------------------------------------------------------------ operational metrics
def ops_metrics(actions, y, amount, sc, monthly=None, ops=OPS):
    """Everything scaled to `monthly` transactions (default OPS), averaged over action repeats."""
    monthly = monthly or ops["monthly_transactions"]; y = np.asarray(y); a_ = np.asarray(amount, float); n = len(y); k = monthly / n; rows = []
    fn = sc.loss_share * a_ + sc.chargeback_fee
    for a in np.atleast_2d(actions):
        appr, rev, blk = a == APPROVE, a == REVIEW, a == BLOCK; d = sc.review_detect
        caught = (y * blk).sum() + d * (y * rev).sum(); stopped_amt = (a_ * y * blk).sum() + d * (a_ * y * rev).sum()
        rows.append(dict(approve=appr.sum() * k, review=rev.sum() * k, block=blk.sum() * k,
                         frauds_in_approve=(y * appr).sum() * k, frauds_in_review=(y * rev).sum() * k, frauds_in_block=(y * blk).sum() * k,
                         frauds_stopped=caught * k, frauds_leaking=(y.sum() - caught) * k, fraud_capture=caught / y.sum(),
                         fraud_value_total=(a_ * y).sum() * k, fraud_value_stopped=stopped_amt * k, fraud_value_leaking=((a_ * y).sum() - stopped_amt) * k,
                         legit_blocked=((1 - y) * blk).sum() * k, legit_reviewed=((1 - y) * rev).sum() * k,
                         legit_wrongly_declined_after_review=sc.review_wrong_decline * ((1 - y) * rev).sum() * k,
                         legit_blocked_share=((1 - y) * blk).sum() / (1 - y).sum(), legit_touched_share=((1 - y) * (blk | rev)).sum() / (1 - y).sum(),
                         block_precision=(y[blk].mean() if blk.any() else np.nan), review_precision=(y[rev].mean() if rev.any() else np.nan),
                         flagged_per_fraud_stopped=((blk | rev).sum() / caught) if caught > 0 else np.nan))
    m = pd.DataFrame(rows).mean(); m["review_share"] = m.review / monthly; m["block_share"] = m.block / monthly
    m["reviews_per_working_day"] = m.review / ops["working_days"]; m["analysts_needed_fte"] = m.review / reviews_per_analyst_month(ops)
    return m


def policy_table(actions_by_policy, y, amount, sc, ops=OPS):
    """One row per policy: volumes, capture, friction, workload, monthly money."""
    rows = {}
    for name, A in actions_by_policy.items():
        m = ops_metrics(A, y, amount, sc, ops=ops); comp = cost_components(A, y, amount, sc).mean() * ops["monthly_transactions"] / 1.0
        m["cost_total"] = comp.total; m["cost_fraud_loss"] = comp.fraud_loss; m["cost_review_labour"] = comp.review_labour; m["cost_review_friction"] = comp.review_friction; m["cost_block_friction"] = comp.block_friction
        rows[name] = m
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------ effective probability thresholds of the amount-aware rule
def effective_thresholds(sc, lam, mu, amounts, grid=None):
    """For a given amount: the probability above which review beats approve, and above which block beats review, under the shadow-priced costs.
    Returns NaN where the action never wins (for example block when amounts are small)."""
    grid = np.linspace(0, 1, 4001) if grid is None else grid; rows = []
    for amt in amounts:
        E = fcal.expected_costs(grid, np.full(len(grid), amt), sc); E[:, 1] += lam; E[:, 2] += mu; best = E.argmin(1)
        t_rev = grid[np.argmax(best >= 1)] if (best >= 1).any() else np.nan; t_blk = grid[np.argmax(best == 2)] if (best == 2).any() else np.nan
        rows.append(dict(amount=amt, review_from_p=t_rev, block_from_p=t_blk))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Monte Carlo of daily operations
def simulate_days(actions, y, amount, sc, n_days=2000, tx_per_day=3333, seed=0, d_staff=None):
    """Each simulated day draws `tx_per_day` transactions with replacement from the development rows and applies one of the action vectors.
    Returns one row per day: review queue, blocks, legitimate customers blocked, frauds leaking, cost."""
    rng = np.random.default_rng(seed); y = np.asarray(y); a_ = np.asarray(amount, float); A = np.atleast_2d(actions); R, n = A.shape; C = row_cost_matrix(A, y, a_, sc)
    fn = sc.loss_share * a_ + sc.chargeback_fee; rows = []
    for _ in range(n_days):
        idx = rng.integers(0, n, tx_per_day); r = rng.integers(0, R); a = A[r, idx]; yy = y[idx]
        rows.append(dict(reviews=(a == REVIEW).sum(), blocks=(a == BLOCK).sum(), legit_blocked=((a == BLOCK) & (yy == 0)).sum(), frauds_leaking=(yy * ((a == APPROVE) + (1 - sc.review_detect) * (a == REVIEW))).sum(),
                         frauds_total=yy.sum(), cost=C[r, idx].sum()))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ the deployable rule
ACTION_NAMES = {APPROVE: "Approve", REVIEW: "Manual review", BLOCK: "Block"}


def decide(p, amount, sc, lam, mu):
    """Production decision for calibrated probabilities `p` and transaction amounts: the cheapest expected action under the cost table, where `lam` and `mu`
    are the shadow prices (extra cost per review and per block) stored in outputs/I_decision_policy.json. Returns 0 / 1 / 2 (Approve / Manual review / Block)."""
    E = fcal.expected_costs(np.atleast_1d(p), np.atleast_1d(amount), sc); E[:, 1] += lam; E[:, 2] += mu
    return np.argmin(E, axis=1)
