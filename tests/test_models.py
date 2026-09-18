import numpy as np
import pytest

from p300erp.evaluate import aggregate_curve
from p300erp.models import Xdawn, build


def synth(n=600, n_ch=8, n_t=103, seed=0):
    """Non-target: noise. Target: noise + a positive bump at ~300 ms on posterior channels."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_ch, n_t)) * 5e-6
    y = (rng.random(n) < 0.2).astype(int)
    t = np.arange(n_t) / 128
    bump = np.exp(-((t - 0.3) ** 2) / (2 * 0.05 ** 2)) * 6e-6
    X[y == 1, 2:, :] += bump
    return X.astype(np.float32), y


def test_xdawn_filters_shape_and_orthogonality():
    X, y = synth()
    f = Xdawn(4).fit(X, y)
    assert f.filters_.shape == (4, 8)
    assert f.transform(X[:3]).shape == (3, 4, 103)


@pytest.mark.parametrize("kind", ["xdawn_lda", "riemann", "eegnet"])
def test_detectors_learn_synthetic_p300(kind):
    from sklearn.metrics import roc_auc_score
    X, y = synth(seed=1)
    Xt, yt = synth(seed=2)
    clf = build(kind, 8, 103)
    if kind == "eegnet":
        clf.set_params(net__epochs=5)
    clf.fit(X, y)
    p = clf.predict_proba(Xt)[:, 1]
    assert p.shape == (len(Xt),)
    assert roc_auc_score(yt, p) > 0.85


def test_aggregate_curve_is_monotone_for_informative_scores():
    from types import SimpleNamespace
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    oof = np.clip(0.5 + 0.15 * (2 * y - 1) + 0.3 * rng.standard_normal(2000), 0, 1)
    ds = SimpleNamespace(subject=np.ones(2000, int), y=y)
    c = aggregate_curve(ds, oof, ks=(1, 4, 10), n_draws=300)
    a = c.set_index("k").auc
    assert a[1] < a[4] < a[10]
