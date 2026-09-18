"""Run all detectors in both regimes and write reports/.

    python -m p300erp.run [--prepare] [--skip-eegnet] [--figure-only]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from .data import DATA, load, prepare
from .evaluate import aggregate_curve, cross_subject, within_subject
from .figure import make_overview

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports"
log = logging.getLogger("p300erp")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true", help="download via MOABB and write the .npz")
    ap.add_argument("--skip-eegnet", action="store_true")
    ap.add_argument("--figure-only", action="store_true", help="rebuild overview.png from saved reports")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if a.prepare or not DATA.exists():
        prepare()
    ds = load()
    log.info(ds.summary())
    REP.mkdir(exist_ok=True)
    kinds = ["xdawn_lda", "riemann"] + ([] if a.skip_eegnet else ["eegnet"])

    if a.figure_only:
        curves = pd.read_csv(REP / "aggregation_curves.csv")
        make_overview(ds, curves, [k for k in kinds if k in set(curves.model)], REP / "overview.png")
        return

    results, curves, t0 = {}, [], time.time()
    for kind in kinds:
        for regime, fn in (("within_subject", within_subject), ("cross_subject", cross_subject)):
            per_subj, oof = fn(ds, kind)
            results[f"{kind}/{regime}"] = {"per_subject_mean_auc": float(per_subj.auc.mean()),
                                           "per_subject_sd_auc": float(per_subj.auc.std()),
                                           "per_subject_mean_ap": float(per_subj.ap.mean()),
                                           "pooled_auc": float(roc_auc_score(ds.y, oof)),
                                           "per_subject": per_subj.to_dict("records")}
            log.info("== %s %s: mean per-subject AUC %.3f +/- %.3f", kind, regime,
                     per_subj.auc.mean(), per_subj.auc.std())
            c = aggregate_curve(ds, oof)
            c["model"], c["regime"] = kind, regime
            curves.append(c)
    curves = pd.concat(curves, ignore_index=True)
    curves.to_csv(REP / "aggregation_curves.csv", index=False)
    results["aggregation"] = (curves.groupby(["model", "regime", "k"]).auc.mean()
                              .round(3).reset_index().to_dict("records"))
    results["seconds"] = round(time.time() - t0, 1)
    (REP / "metrics.json").write_text(json.dumps(results, indent=2))
    make_overview(ds, curves, kinds, REP / "overview.png")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_subject"} if isinstance(v, dict) else v
                      for k, v in results.items() if k != "aggregation"}, indent=2))


if __name__ == "__main__":
    main()
