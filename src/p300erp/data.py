"""Load the BNCI 2014-008 P300 speller dataset (via MOABB) into plain arrays.

BNCI 2014-008 (Riccio et al. 2013): 8 participants with ALS, 8 EEG channels
(Fz, Cz, Pz, Oz, P3, P4, PO7, PO8), 256 Hz, row/column speller with 10 repetitions per
character. MOABB's P300 paradigm gives stimulus-locked epochs labelled Target / NonTarget
(ratio 1:5). We resample to 128 Hz, band-pass 1-24 Hz, and keep 0-0.8 s post-stimulus.

The download is cached by MOABB (~100 MB). ``prepare()`` writes a compact .npz so the
rest of the pipeline never touches MOABB again.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "bnci2014008_p300.npz"


@dataclass
class ERPDataset:
    X: np.ndarray          # (n_trials, n_ch, n_times) Volts
    y: np.ndarray          # (n_trials,) 1 = target, 0 = non-target
    subject: np.ndarray    # (n_trials,)
    session: np.ndarray    # (n_trials,)
    sfreq: float
    ch_names: list[str]

    def summary(self) -> str:
        return (f"{len(self.y)} trials | {self.X.shape[1]} ch x {self.X.shape[2]} samples @ "
                f"{self.sfreq:g} Hz | {len(np.unique(self.subject))} subjects | "
                f"target fraction {self.y.mean():.3f}")


def prepare(out: Path = DATA, resample: int = 128, fmin: float = 1.0, fmax: float = 24.0,
            tmax: float = 0.8) -> Path:
    import mne
    import moabb
    from moabb.datasets import BNCI2014_008
    from moabb.paradigms import P300
    mne.set_log_level("ERROR"); moabb.set_log_level("warning")
    ds = BNCI2014_008()
    para = P300(resample=resample, fmin=fmin, fmax=fmax, tmin=0.0, tmax=tmax)
    X, y, meta = para.get_data(ds, subjects=ds.subject_list)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X.astype(np.float32), y=(y == "Target").astype(np.int8),
                        subject=meta.subject.to_numpy(), session=meta.session.astype(str).to_numpy(),
                        sfreq=float(resample))
    return out


def load(path: Path = DATA) -> ERPDataset:
    z = np.load(path, allow_pickle=True)
    default = ["Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8"]        # BNCI 2014-008 montage
    ch = list(z["ch_names"]) if "ch_names" in z else default[: z["X"].shape[1]]
    return ERPDataset(z["X"], z["y"].astype(int), z["subject"], z["session"], float(z["sfreq"]), ch)
