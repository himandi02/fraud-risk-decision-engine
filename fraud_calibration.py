"""Probability calibration layer (Part F).

Platt scaling: p_cal = sigmoid(intercept + slope * logit(raw score)), fitted by unregularised logistic regression on
OUT-OF-FOLD scores only (never on scores a model produced for rows it had trained on).
Platt scaling is monotone, so it cannot change the ranking of transactions; it only repairs the probability scale.

``CalibratedFraudModel`` wraps the fitted preprocessing + Random Forest pipeline (03_fraud_model_pipeline.joblib)
and the calibrator, and is what ``04_calibrated_model.joblib`` stores.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-12


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


class PlattCalibrator:
    def fit(self, scores, y):
        lr = LogisticRegression(C=1e10, max_iter=100000, tol=1e-12).fit(logit(scores).reshape(-1, 1), np.asarray(y, int))
        self.a_ = float(lr.intercept_[0]); self.b_ = float(lr.coef_[0, 0])
        return self

    def predict(self, scores):
        return sigmoid(self.a_ + self.b_ * logit(scores))


class IsotonicCalibrator:
    def fit(self, scores, y):
        self.iso_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(np.asarray(scores, float), np.asarray(y, int))
        return self

    def predict(self, scores):
        return self.iso_.predict(np.asarray(scores, float))


CALIBRATORS = {"platt": PlattCalibrator, "isotonic": IsotonicCalibrator}


class CalibratedFraudModel:
    """predict_proba(X) returns [P(genuine), P(fraud)] with P(fraud) on the calibrated scale."""

    def __init__(self, base, calibrator, meta=None):
        self.base = base
        self.calibrator = calibrator
        self.meta = dict(meta or {})
        self.classes_ = np.array([0, 1])

    def raw_score(self, X):
        return self.base.predict_proba(X)[:, 1]

    def predict_proba(self, X):
        p = self.calibrator.predict(self.raw_score(X))
        return np.column_stack([1 - p, p])

    def predict(self, X, threshold=None):
        t = self.meta.get("thresholds_calibrated", {}).get("t_review", 0.5) if threshold is None else threshold
        return (self.predict_proba(X)[:, 1] >= t).astype(int)
