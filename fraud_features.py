"""Feature engineering for the fraud-risk pipeline (Part C).

Rules
-----
* No label is ever used. ``fit`` ignores ``y``; every learned constant (percentiles, category medians,
  robust distance centre/scale) is computed on the rows passed to ``fit`` only (the training rows).
* Missing inputs stay missing in the derived columns (median imputation happens later, inside the model
  pipeline, fitted on the training folds). Exception: risk_mean and risk_max use whichever of the two risk
  scores is present, so they are missing only when both are.
* ``authentication_level`` missing (the literal string "None" in the source file) becomes its own level
  "Missing" (decision T2).  It is NOT imputed with the mode.
* Unknown categories at prediction time fall back to the global median (amount ratios) or to 0 (flags).

The class and attribute names are those stored inside the fitted pipelines
(``fraud_features.FraudFeatureEngineer`` and ``fraud_features.FeatureSetSelector``), so
``joblib.load("03_fraud_model_pipeline.joblib")`` needs this module on the Python path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ID = "transaction_id"
TARGET = "fraud"

NUM_BASE = ["hour", "day_of_week", "transaction_amount", "distance_from_usual_location_km", "transactions_last_1h",
            "transactions_last_24h", "account_age_days", "failed_logins_24h", "prior_chargebacks", "device_age_days",
            "merchant_risk_score", "ip_risk_score", "email_age_days", "login_to_transaction_minutes",
            "new_device", "billing_address_mismatch"]
CAT_BASE = ["card_presence", "channel", "country", "device_type", "merchant_category", "authentication_level",
            "customer_tier"]
# column order used inside the fitted pipelines (and in 02_processed_dataset.csv)
RAW_COLUMNS = ["hour", "day_of_week", "transaction_amount", "distance_from_usual_location_km", "transactions_last_1h",
               "transactions_last_24h", "account_age_days", "failed_logins_24h", "prior_chargebacks", "device_age_days",
               "merchant_risk_score", "ip_risk_score", "email_age_days", "login_to_transaction_minutes",
               "card_presence", "channel", "country", "device_type", "merchant_category", "authentication_level",
               "new_device", "billing_address_mismatch", "customer_tier"]

FEATURE_GROUPS = {
    "velocity": ["velocity_ratio_1h_24h", "velocity_surge", "log_tx_24h"],
    "amount": ["log_amount", "amount_to_category_median", "amount_to_tier_median", "is_amount_tail"],
    "risk": ["risk_mean", "risk_max", "risk_product", "high_risk_both", "n_red_flags"],
    "time": ["is_night", "is_business_hours", "hour_sin", "hour_cos"],
    "behavior": ["log_device_account_gap", "log_email_account_gap", "is_young_account", "is_quick_login",
                 "log_login_minutes", "distance_robust_z", "is_far_distance"],
    "interaction": ["newdev_x_logdist", "logamt_x_merchrisk", "authmiss_x_digital", "newdev_x_iprisk",
                    "billmis_x_newdev", "velocity_x_logamt"],
}
ALL_DERIVED = [f for g in FEATURE_GROUPS.values() for f in g]
DERIVED_GROUP = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}

RECOMMENDED = ["velocity", "risk", "time", "interaction"]          # Part C pre-set rule
FEATURE_SET_COLUMNS = {
    "base": [],
    "recommended": [f for g in RECOMMENDED for f in FEATURE_GROUPS[g]],
    "recommended_no_red_flags": [f for g in RECOMMENDED for f in FEATURE_GROUPS[g] if f != "n_red_flags"],
    "all": list(ALL_DERIVED),
}
SENSITIVITY_GROUPS = FEATURE_GROUPS


def derived_columns(groups=None):
    groups = tuple(FEATURE_GROUPS) if groups is None else groups
    return [f for g in FEATURE_GROUPS if g in groups for f in FEATURE_GROUPS[g]]


def _num(s):
    return pd.to_numeric(s, errors="coerce").astype(float)


def _flag(cond, *inputs):
    """1.0 / 0.0 indicator that is NaN wherever any input is missing."""
    out = pd.Series(np.where(cond, 1.0, 0.0), index=cond.index)
    miss = pd.concat([i.isna() for i in inputs], axis=1).any(axis=1)
    return out.mask(miss)


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds the 29 derived features to the 23 raw columns. Learns constants on the fitting rows only."""

    def __init__(self, groups=tuple(FEATURE_GROUPS)):
        self.groups = groups

    # ------------------------------------------------------------------ fit
    def fit(self, X, y=None):                                   # y is deliberately ignored
        X = pd.DataFrame(X).copy()
        amt = _num(X["transaction_amount"]); dist = _num(X["distance_from_usual_location_km"])
        self.amount_p95_ = float(amt.quantile(0.95))
        self.global_amount_median_ = float(amt.median())
        self.cat_median_ = {str(k): float(v) for k, v in amt.groupby(X["merchant_category"]).median().items()}
        self.tier_median_ = {str(k): float(v) for k, v in amt.groupby(X["customer_tier"]).median().items()}
        self.login_p10_ = float(_num(X["login_to_transaction_minutes"]).quantile(0.10))
        self.merchant_p75_ = float(_num(X["merchant_risk_score"]).quantile(0.75))
        self.ip_p75_ = float(_num(X["ip_risk_score"]).quantile(0.75))
        ld = np.log1p(dist)
        self.dist_log_median_ = float(ld.median())
        self.dist_log_mad_ = float(1.4826 * (ld - ld.median()).abs().median())
        self.dist_p95_ = float(dist.quantile(0.95))
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if "authentication_level" in X:                          # decision T2: explicit level, never imputed
            X["authentication_level"] = X["authentication_level"].astype(object).where(
                X["authentication_level"].notna(), "Missing")
            X.loc[X["authentication_level"].isin(["None", "none", "nan", "NaN", ""]), "authentication_level"] = "Missing"
        out = X[[c for c in RAW_COLUMNS if c in X.columns]].copy()
        idx = X.index
        f = {}
        amt, dist = _num(X["transaction_amount"]), _num(X["distance_from_usual_location_km"])
        t1, t24 = _num(X["transactions_last_1h"]), _num(X["transactions_last_24h"])
        mr, ir = _num(X["merchant_risk_score"]), _num(X["ip_risk_score"])
        hour = _num(X["hour"]); login = _num(X["login_to_transaction_minutes"])
        acc, dev, mail = _num(X["account_age_days"]), _num(X["device_age_days"]), _num(X["email_age_days"])
        nd, bm = _num(X["new_device"]), _num(X["billing_address_mismatch"])
        pc, fl = _num(X["prior_chargebacks"]), _num(X["failed_logins_24h"])
        auth_missing = (X["authentication_level"] == "Missing").astype(float)
        digital = X["channel"].isin(["E-commerce", "Mobile App"]).astype(float)
        logamt, logdist = np.log1p(amt), np.log1p(dist)

        # velocity
        f["velocity_ratio_1h_24h"] = t1 / (t24 + 1)
        f["velocity_surge"] = t1 - t24 / 24
        f["log_tx_24h"] = np.log1p(t24)
        # amount
        f["log_amount"] = logamt
        cm = X["merchant_category"].astype(str).map(self.cat_median_).fillna(self.global_amount_median_)
        tm = X["customer_tier"].astype(str).map(self.tier_median_).fillna(self.global_amount_median_)
        f["amount_to_category_median"] = amt / cm
        f["amount_to_tier_median"] = amt / tm
        f["is_amount_tail"] = _flag(amt >= self.amount_p95_, amt)
        # risk
        both = pd.concat([mr, ir], axis=1)
        f["risk_mean"] = both.mean(axis=1)           # uses whichever score is present (NaN only if both missing)
        f["risk_max"] = both.max(axis=1)
        f["risk_product"] = mr * ir
        f["high_risk_both"] = _flag((mr >= self.merchant_p75_) & (ir >= self.ip_p75_), mr, ir)
        f["n_red_flags"] = nd + bm + (pc >= 1).astype(float) + (fl >= 2).astype(float) + auth_missing
        # time
        f["is_night"] = (hour <= 5).astype(float)
        f["is_business_hours"] = ((hour >= 9) & (hour <= 17)).astype(float)
        f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        # behavioural deviation
        f["log_device_account_gap"] = np.log1p(dev) - np.log1p(acc)
        f["log_email_account_gap"] = np.log1p(mail) - np.log1p(acc)
        f["is_young_account"] = (acc <= 30).astype(float)
        f["is_quick_login"] = _flag(login <= self.login_p10_, login)
        f["log_login_minutes"] = np.log1p(login)
        f["distance_robust_z"] = (logdist - self.dist_log_median_) / self.dist_log_mad_
        f["is_far_distance"] = _flag(dist >= self.dist_p95_, dist)
        # interactions
        f["newdev_x_logdist"] = nd * logdist
        f["logamt_x_merchrisk"] = logamt * mr
        f["authmiss_x_digital"] = auth_missing * digital
        f["newdev_x_iprisk"] = nd * ir
        f["billmis_x_newdev"] = bm * nd
        f["velocity_x_logamt"] = t1 * logamt

        keep = derived_columns(self.groups)
        for c in keep:
            out[c] = pd.Series(f[c], index=idx).astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(RAW_COLUMNS + derived_columns(self.groups))


class FeatureSetSelector(BaseEstimator, TransformerMixin):
    """Keeps the raw columns plus the derived columns of one named feature set (searched inside CV)."""

    def __init__(self, feature_set="recommended"):
        self.feature_set = feature_set

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        keep = set(FEATURE_SET_COLUMNS[self.feature_set])
        cols = [c for c in X.columns if c not in ALL_DERIVED or c in keep]
        return X[cols]


def _derived_selector(X):
    """Column selector for the ColumnTransformer: every derived column that survived the feature-set selector."""
    return [c for c in X.columns if c in ALL_DERIVED]


def build_search_pipeline(model, feature_set="recommended"):
    """Features -> feature set -> preprocessing -> model (all learning inside the pipeline)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True))]), NUM_BASE),
        ("derived", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True))]), _derived_selector),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_BASE),
    ], verbose_feature_names_out=False)
    return Pipeline([("features", FraudFeatureEngineer()), ("select", FeatureSetSelector(feature_set)),
                     ("prep", prep), ("model", model)])


build_pipeline = build_search_pipeline
