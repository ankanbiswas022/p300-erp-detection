"""Evaluation: within-subject, cross-subject, and decision-level aggregation.

Two validation regimes that answer different product questions:

* **within-subject** (stratified 5-fold inside each subject): "how good is the detector
  after a calibration session for this person?" This is how P300 spellers and
  concealed-information tests are actually deployed.
* **cross-subject** (leave-one-subject-out): "how good is it with zero calibration?"

**Decision-level aggregation.** A single-trial P300 detector is noisy (AUC ~0.8). Real
systems never decide on one flash: they repeat the stimulus and average the evidence.
``aggregate_curve`` draws k target trials and k non-target trials from the same subject,
averages the detector's scores, and measures how the AUC of the *aggregated* decision
grows with k. This is the curve that tells a product team how many repetitions a
decision needs to reach a target reliability, and it is the same logic behind
"the test completes in under 8 minutes".
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .models import build

log = logging.getLogger(__name__)


def _metrics(y, p):
    return {"auc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, (p >= 0.5).astype(int)))}


def within_subject(ds, kind: str, n_splits: int = 5, seed: int = 7) -> tuple[pd.DataFrame, np.ndarray]:
    oof = np.full(len(ds.y), np.nan)
    rows = []
    for s in np.unique(ds.subject):
        m = np.where(ds.subject == s)[0]
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        for tr, te in skf.split(m, ds.y[m]):
            clf = build(kind, ds.X.shape[1], ds.X.shape[2], seed)
            clf.fit(ds.X[m[tr]], ds.y[m[tr]])
            oof[m[te]] = clf.predict_proba(ds.X[m[te]])[:, 1]
        rows.append({"subject": int(s), **_metrics(ds.y[m], oof[m])})
        log.info("[%s] within-subject S%02d AUC %.3f", kind, s, rows[-1]["auc"])
    return pd.DataFrame(rows), oof


def cross_subject(ds, kind: str, seed: int = 7) -> tuple[pd.DataFrame, np.ndarray]:
    oof = np.full(len(ds.y), np.nan)
    rows = []
    for s in np.unique(ds.subject):
        te, tr = ds.subject == s, ds.subject != s
        clf = build(kind, ds.X.shape[1], ds.X.shape[2], seed)
        clf.fit(ds.X[tr], ds.y[tr])
        oof[te] = clf.predict_proba(ds.X[te])[:, 1]
        rows.append({"subject": int(s), **_metrics(ds.y[te], oof[te])})
        log.info("[%s] leave-one-subject-out S%02d AUC %.3f", kind, s, rows[-1]["auc"])
    return pd.DataFrame(rows), oof


def aggregate_curve(ds, oof: np.ndarray, ks=(1, 2, 3, 4, 6, 8, 10), n_draws: int = 400,
                    seed: int = 0) -> pd.DataFrame:
    """AUC of the mean score over k trials of the same class, per subject."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in np.unique(ds.subject):
        m = ds.subject == s
        pos, neg = oof[m & (ds.y == 1)], oof[m & (ds.y == 0)]
        for k in ks:
            a = rng.choice(pos, (n_draws, k)).mean(1)
            b = rng.choice(neg, (n_draws, k)).mean(1)
            rows.append({"subject": int(s), "k": k,
                         "auc": float(roc_auc_score(np.r_[np.ones(n_draws), np.zeros(n_draws)], np.r_[a, b]))})
    return pd.DataFrame(rows)
