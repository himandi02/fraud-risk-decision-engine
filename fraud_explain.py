"""Part H: explainability for the fraud decision engine (SHAP on the production Random Forest).

SHAP (TreeExplainer) splits one prediction into additive contributions from the model's inputs:
    model probability = base rate + sum of contributions        (checked numerically for every row)
The model sees 74 encoded inputs (one-hot columns, missing-value flags, engineered features). For people these are regrouped, WITHOUT changing the maths
(contributions simply add up inside a group), into
    * units    (41): a one-hot block becomes its field, a missing flag joins its field, engineered features keep their own name
    * families (9) : units that measure the same business idea (e.g. all risk-score features)
Everything here describes what the MODEL does. None of it is a causal statement about fraud (see Part H of the notebook).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.base import clone

import fraud_features as ff

FAMILIES = {
    "Merchant and IP risk scores": ["merchant_risk_score", "ip_risk_score", "risk_mean", "risk_max", "risk_product", "high_risk_both", "logamt_x_merchrisk", "newdev_x_iprisk"],
    "Device and authentication": ["new_device", "device_age_days", "device_type", "authentication_level", "authmiss_x_digital", "newdev_x_logdist", "billmis_x_newdev", "billing_address_mismatch"],
    "Chargeback history and red flags": ["prior_chargebacks", "failed_logins_24h", "n_red_flags"],
    "Transaction velocity": ["transactions_last_1h", "transactions_last_24h", "velocity_ratio_1h_24h", "velocity_surge", "log_tx_24h", "velocity_x_logamt"],
    "Amount": ["transaction_amount"],
    "Account and customer profile": ["account_age_days", "email_age_days", "login_to_transaction_minutes", "customer_tier"],
    "Location": ["distance_from_usual_location_km", "country"],
    "Merchant and channel context": ["merchant_category", "channel", "card_presence"],
    "Time of day": ["hour", "day_of_week", "is_night", "is_business_hours", "hour_sin", "hour_cos"],
}
FAMILY_OF = {u: f for f, us in FAMILIES.items() for u in us}

LABELS = {
    "new_device": "New device", "billing_address_mismatch": "Billing address mismatch", "prior_chargebacks": "Prior chargebacks", "failed_logins_24h": "Failed logins (24h)",
    "authentication_level": "Authentication level", "merchant_risk_score": "Merchant risk score", "ip_risk_score": "IP risk score", "transactions_last_1h": "Transactions in last hour",
    "transactions_last_24h": "Transactions in last 24h", "transaction_amount": "Transaction amount", "distance_from_usual_location_km": "Distance from usual location (km)",
    "account_age_days": "Account age (days)", "device_age_days": "Device age (days)", "email_age_days": "Email age (days)", "login_to_transaction_minutes": "Minutes from login to transaction",
    "merchant_category": "Merchant category", "channel": "Channel", "country": "Country", "device_type": "Device type", "customer_tier": "Customer tier", "card_presence": "Card presence",
    "hour": "Hour of day", "day_of_week": "Day of week", "n_red_flags": "Number of red flags", "risk_mean": "Average of merchant and IP risk", "risk_max": "Higher of merchant and IP risk",
    "risk_product": "Merchant x IP risk", "high_risk_both": "Both risk scores in top quartile", "velocity_ratio_1h_24h": "Share of 24h activity in last hour", "velocity_surge": "Velocity surge (1h minus hourly average)",
    "log_tx_24h": "Transactions in 24h (log)", "velocity_x_logamt": "Velocity x amount", "logamt_x_merchrisk": "Amount x merchant risk", "newdev_x_logdist": "New device x distance",
    "newdev_x_iprisk": "New device x IP risk", "authmiss_x_digital": "Missing authentication on a digital channel", "billmis_x_newdev": "Billing mismatch on a new device",
    "is_night": "Night-time (00:00 to 05:59)", "is_business_hours": "Business hours", "hour_sin": "Hour (cyclic sine)", "hour_cos": "Hour (cyclic cosine)",
}


def unit_of(encoded_name: str) -> str:
    if encoded_name.startswith("missingindicator_"): return encoded_name[len("missingindicator_"):]
    for c in ff.CAT_BASE:
        if encoded_name.startswith(c + "_"): return c
    return encoded_name


@dataclass
class Explanation:
    base: float                 # average model output (the starting point of every explanation)
    prob: np.ndarray            # model probability for each row
    phi_enc: np.ndarray         # (n, 74) SHAP values of the encoded inputs
    phi: pd.DataFrame           # (n, 41) SHAP values of the units
    values: pd.DataFrame        # (n, 41) readable value of each unit (category text for categorical fields)
    Z: pd.DataFrame             # (n, 74) encoded model inputs


class FraudExplainer:
    """Wraps a fitted scoring pipeline (features -> select -> prep -> RandomForest)."""

    def __init__(self, pipeline):
        self.pipe = pipeline
        self.enc_names = list(pipeline["prep"].get_feature_names_out())
        self.units = list(dict.fromkeys(unit_of(n) for n in self.enc_names))
        self.M = np.zeros((len(self.enc_names), len(self.units)))
        for i, n in enumerate(self.enc_names): self.M[i, self.units.index(unit_of(n))] = 1.0
        self.tree = shap.TreeExplainer(pipeline["model"])
        self.base = float(self.tree.expected_value[1])

    def encode(self, X):
        F = self.pipe["select"].transform(self.pipe["features"].transform(X)); Z = pd.DataFrame(self.pipe["prep"].transform(F), index=F.index, columns=self.enc_names); return F, Z

    def explain(self, X, background=None) -> Explanation:
        F, Z = self.encode(X)
        if background is None: sv = self.tree.shap_values(Z.values, check_additivity=False)[:, :, 1]; base = self.base
        else:
            ex = shap.TreeExplainer(self.pipe["model"], data=background.values, feature_perturbation="interventional"); sv = ex.shap_values(Z.values, check_additivity=False)[:, :, 1]; base = float(ex.expected_value[1])
        prob = self.pipe["model"].predict_proba(Z.values)[:, 1]
        assert np.allclose(base + sv.sum(1), prob, atol=1e-6 if background is None else 3e-3), "SHAP additivity violated"   # interventional mode shows a tiny numerical residual (<0.2 pts)
        phi = pd.DataFrame(sv @ self.M, index=F.index, columns=self.units)
        return Explanation(base, prob, sv, phi, F[self.units], Z)

    def interaction_units(self, Z: pd.DataFrame):
        """Mean absolute SHAP interaction between every pair of units (probability points; the pair total counts both directions)."""
        iv = self.tree.shap_interaction_values(Z.values)[:, :, :, 1]                      # (n, 74, 74)
        U = len(self.units); out = np.zeros((U, U)); full = np.einsum("nij,iu,jv->nuv", iv, self.M, self.M)  # block-sum to units
        return np.abs(full).mean(0), full


def reason_text(units_phi: pd.Series, values: pd.Series, k=3, min_abs=0.003):
    """Plain-English reason codes: the k largest risk-increasing and risk-decreasing contributions (in probability points)."""
    def fmt(u):
        v = values[u]; lab = LABELS.get(u, u)
        if isinstance(v, str): return f"{lab} = {v}"
        if pd.isna(v): return f"{lab} missing"
        if u in ("new_device", "billing_address_mismatch", "is_night", "is_business_hours", "high_risk_both", "authmiss_x_digital", "billmis_x_newdev"): return f"{lab}: {'yes' if v >= .5 else 'no'}"
        return f"{lab} = {v:,.2f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{lab} = {v:,.0f}"
    pos = units_phi[units_phi >= min_abs].sort_values(ascending=False).head(k); neg = units_phi[units_phi <= -min_abs].sort_values().head(k)
    up = "; ".join(f"{fmt(u)} ({p * 100:+.1f} pts)" for u, p in pos.items()) or "none above 0.3 pts"; dn = "; ".join(f"{fmt(u)} ({p * 100:+.1f} pts)" for u, p in neg.items()) or "none above 0.3 pts"
    return up, dn


# ------------------------------------------------------------------ stability: refit the whole pipeline on bootstrap resamples, explain the same rows
def _boot_one(template, X, y, X_eval, seed):
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(y), len(y)); m = clone(template).fit(X.iloc[idx], y.iloc[idx])
    ex = FraudExplainer(m); e = ex.explain(X_eval); return e.phi.values, e.prob


def bootstrap_explanations(template, X, y, X_eval, B=30, seed=0, n_jobs=2):
    """Returns (phi array B x n x 41, probabilities B x n). `template` is an UNFITTED clone of the production pipeline."""
    res = Parallel(n_jobs=n_jobs)(delayed(_boot_one)(template, X, y, X_eval, seed + b) for b in range(B))
    return np.stack([r[0] for r in res]), np.stack([r[1] for r in res])
