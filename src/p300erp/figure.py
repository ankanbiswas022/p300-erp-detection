"""Overview figure: grand-average ERP (per-subject robust scaling) + aggregation curves."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_overview(ds, curves: pd.DataFrame, kinds: list[str], out: Path) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    t = np.arange(ds.X.shape[2]) / ds.sfreq * 1000
    iz = ds.ch_names.index("Pz") if "Pz" in ds.ch_names else 2
    # MOABB returns microvolts. Scale each subject by its own MAD so one large-amplitude
    # recording does not dominate the grand average, then average the per-subject ERPs.
    erps = {0: [], 1: []}
    for s in np.unique(ds.subject):
        m = ds.subject == s
        scale = np.median(np.abs(ds.X[m, iz])) * 1.4826 + 1e-12
        for lab in (0, 1):
            erps[lab].append(ds.X[m & (ds.y == lab), iz].mean(0) / scale)
    for lab, name in ((1, "target"), (0, "non-target")):
        ax[0].plot(t, np.mean(erps[lab], 0), label=name)
    ax[0].plot(t, np.mean(erps[1], 0) - np.mean(erps[0], 0), "k--", lw=1, label="difference")
    ax[0].axvspan(250, 500, color="k", alpha=.07)
    ax[0].set(xlabel="ms after flash", ylabel="amplitude (subject-scaled)",
              title="Grand-average ERP at Pz (8 subjects)")
    ax[0].legend(frameon=False)
    for i, regime in enumerate(("within_subject", "cross_subject")):
        for kind in kinds:
            g = curves[(curves.model == kind) & (curves.regime == regime)].groupby("k").auc.mean()
            ax[i + 1].plot(g.index, g.values, "o-", label=kind)
        ax[i + 1].axhline(0.95, color="k", ls=":", lw=1)
        ax[i + 1].set(xlabel="repetitions averaged (k)", ylabel="decision AUC", ylim=(0.5, 1.01),
                      title=f"Decision reliability vs repetitions ({regime.replace('_', '-')})")
        ax[i + 1].legend(frameon=False)
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
