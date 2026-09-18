"""Three P300 detectors with one interface (``fit(X, y)`` / ``predict_proba(X)``).

* ``xdawn_lda``   - xDAWN spatial filtering (Rivet 2009) + vectorised epoch + shrinkage LDA.
                    The classic, fast, and very hard to beat on 8 channels.
* ``riemann``     - xDAWN covariances + tangent-space projection + logistic regression
                    (Barachant / Congedo). Winner of most P300 benchmarks; robust across subjects.
* ``eegnet``      - compact EEGNet (Lawhern 2018) on raw epochs, PyTorch, class-weighted loss.

All three are wrapped in scikit-learn Pipelines so the same evaluation code applies.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from torch import nn


class Xdawn(BaseEstimator):
    """Minimal xDAWN: spatial filters maximising evoked (target) signal-to-signal-plus-noise."""

    def __init__(self, n_filters: int = 4):
        self.n_filters = n_filters

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        evoked = X[y == 1].mean(0)                                # (n_ch, n_t)
        Cs = evoked @ evoked.T / evoked.shape[1]
        Cx = np.mean([x @ x.T / x.shape[1] for x in X], axis=0)
        Cx += 1e-8 * np.trace(Cx) / len(Cx) * np.eye(len(Cx))
        # generalised eigenproblem Cs w = l Cx w  via whitening
        d, V = np.linalg.eigh(Cx)
        W = V @ np.diag(1 / np.sqrt(np.maximum(d, 1e-12))) @ V.T
        _, U = np.linalg.eigh(W @ Cs @ W)
        self.filters_ = (W @ U[:, ::-1][:, :self.n_filters]).T   # (n_filters, n_ch)
        return self

    def transform(self, X):
        return np.einsum("fc,nct->nft", self.filters_, np.asarray(X, dtype=np.float64))


def build(kind: str, n_ch: int, n_t: int, seed: int = 7):
    if kind == "xdawn_lda":
        return make_pipeline(Xdawn(4), FunctionTransformer(lambda Z: Z.reshape(len(Z), -1)),
                             StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    if kind == "riemann":
        return make_pipeline(XdawnCovariances(nfilter=4, estimator="lwf"), TangentSpace(metric="riemann"),
                             LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    if kind == "eegnet":
        return Pipeline([("net", EEGNetClassifier(n_ch, n_t, seed=seed))])
    raise ValueError(kind)


# --------------------------------------------------------------------------- EEGNet
class _EEGNet(nn.Module):
    def __init__(self, n_ch, n_t, F1=8, D=2, F2=16, kern=32, drop=0.5):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False),
                                nn.BatchNorm2d(F1),
                                nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
                                nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(drop))
        self.b2 = nn.Sequential(nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
                                nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
                                nn.AvgPool2d((1, 8)), nn.Dropout(drop))
        with torch.no_grad():
            n = self.b2(self.b1(torch.zeros(1, 1, n_ch, n_t))).numel()
        self.head = nn.Linear(n, 2)

    def forward(self, x):
        return self.head(self.b2(self.b1(x.unsqueeze(1))).flatten(1))


class EEGNetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_ch, n_t, epochs=40, batch_size=128, lr=1e-3, weight_decay=1e-4, seed=7,
                 device=None):
        self.n_ch, self.n_t, self.epochs, self.batch_size = n_ch, n_t, epochs, batch_size
        self.lr, self.weight_decay, self.seed = lr, weight_decay, seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        self.scale_ = np.median(np.abs(X)) * 1.4826 + 1e-12
        Xt = torch.from_numpy((np.asarray(X) / self.scale_).astype(np.float32))
        yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
        self.net_ = _EEGNet(self.n_ch, self.n_t).to(self.device)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        w = torch.tensor(np.bincount(y, minlength=2), dtype=torch.float32)
        w = (w.sum() / (2 * w)).to(self.device)
        dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt, yt),
                                         batch_size=self.batch_size, shuffle=True)
        for _ in range(self.epochs):
            self.net_.train()
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                loss = F.cross_entropy(self.net_(xb), yb, weight=w)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
        self.classes_ = np.array([0, 1])
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        self.net_.eval()
        Xt = torch.from_numpy((np.asarray(X) / self.scale_).astype(np.float32))
        out = [F.softmax(self.net_(Xt[i:i + 512].to(self.device)), -1).cpu() for i in range(0, len(Xt), 512)]
        return torch.cat(out).numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(1)
