"""Run all detectors in both regimes and write reports/.

    python -m p300erp.run [--prepare] [--skip-eegnet]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from .data import DATA, load, prepare
from .evaluate import aggregate_curve, cross_subject, within_subject

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports"
log = logging.getLogger("p300erp")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true", help="download via MOABB and write the .npz")
    ap.add_argument("--skip-eegnet", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if a.prepare or not DATA.exists():
        prepare()
    ds = load()
    log.info(ds.summary())
    REP.mkdir(exist_ok=True)
    kinds = ["xdawn_lda", "riemann"] + ([] if a.skip_eegnet else ["eegnet"])
    results, curves, t0 = {}, [], time.time()
    for kind in kinds:
        for regime, fn in (("within_subject", within_subject), ("cross_subject", cross_subject)):
            per_subj, oof = fn(ds, kind)
            pooled = {"auc": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(ds.y, oof))}
            results[f"{kind}/{regime}"] = {"per_subject_mean_auc": float(per_subj.auc.mean()),
                                           "per_subject_sd_auc": float(per_subj.auc.std()),
                                           "per_subject_mean_ap": float(per_subj.ap.mean()),
                                           "pooled_auc": pooled["auc"],
                                           "per_subject": per_subj.to_dict("records")}
            log.info("== %s %s: mean per-subject AUC %.3f ± %.3f", kind, regime,
                     per_subj.auc.mean(), per_subj.auc.std())
            c = aggregate_curve(ds, oof); c["model"], c["regime"] = kind, regime
            curves.append(c)
    curves = pd.concat(curves, ignore_index=True)
    curves.to_csv(REP / "aggregation_curves.csv", index=False)
    results["aggregation"] = (curves.groupby(["model", "regime", "k"]).auc.mean()
                              .round(3).reset_index().to_dict("records"))
    results["seconds"] = round(time.time() - t0, 1)
    (REP / "metrics.json").write_text(json.dumps(results, indent=2))

    # figure: grand-average ERP + aggregation curves
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    t = np.arange(ds.X.shape[2]) / ds.sfreq * 1000
    iz = ds.ch_names.index("Pz") if "Pz" in ds.ch_names else 2
    for lab, name in ((1, "target"), (0, "non-target")):
        ax[0].plot(t, ds.X[ds.y == lab, iz].mean(0) * 1e6, label=name)
    ax[0].axvspan(250, 500, color="k", alpha=.07); ax[0].set(xlabel="ms after flash", ylabel="µV", title="Grand-average ERP at Pz"); ax[0].legend(frameon=False)
    for i, regime in enumerate(("within_subject", "cross_subject")):
        for kind in kinds:
            g = curves[(curves.model == kind) & (curves.regime == regime)].groupby("k").auc.mean()
            ax[i + 1].plot(g.index, g.values, "o-", label=kind)
        ax[i + 1].axhline(0.95, color="k", ls=":", lw=1)
        ax[i + 1].set(xlabel="repetitions averaged (k)", ylabel="decision AUC", ylim=(0.5, 1.01),
                      title=f"Decision reliability vs repetitions ({regime.replace('_', '-')})")
        ax[i + 1].legend(frameon=False)
    for x in ax: x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(REP / "overview.png", dpi=150)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_subject"} if isinstance(v, dict) else v
                      for k, v in results.items() if k != "aggregation"}, indent=2))


if __name__ == "__main__":
    main()
