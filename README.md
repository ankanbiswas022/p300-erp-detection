# p300-erp-detection

[![ci](https://github.com/ankanbiswas022/p300-erp-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/ankanbiswas022/p300-erp-detection/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11-blue)
![license](https://img.shields.io/badge/license-MIT-green)

**Single-trial P300 detection on a public speller dataset, and how many repetitions a
decision needs.** Three detectors (xDAWN + LDA, Riemannian tangent space, EEGNet) under
one interface, evaluated within-subject and cross-subject, plus a decision-level
aggregation curve: the number that a P300-based product actually cares about.

<p align="center"><img src="reports/overview.png" width="95%"></p>

## Results

| detector | within-subject AUC | within-subject AP | cross-subject (LOSO) AUC | cross-subject AP |
|---|---|---|---|---|
| xDAWN + shrinkage LDA | 0.850 ± 0.049 | 0.60 | 0.764 ± 0.051 | 0.40 |
| Riemannian (xDAWN cov → tangent space → logreg) | **0.859** ± 0.045 | **0.62** | 0.781 ± 0.035 | 0.44 |
| EEGNet (PyTorch) | 0.856 ± 0.043 | 0.61 | **0.797** ± 0.033 | **0.47** |

Chance AP is 0.17. The three detectors tie within-subject; EEGNet transfers best across
subjects, the Riemannian pipeline is close and ~50x cheaper to train. Whole run (three
detectors, two regimes, aggregation curves): 33 min on a laptop CPU, of which EEGNet is 30.

Per-subject means over 8 subjects, single-trial AUC. Target : non-target ratio is 1 : 5, so
average precision (AP) is reported alongside AUC. Full per-subject numbers in
[reports/metrics.json](reports/metrics.json).

**Decision-level reliability.** Averaging the detector's score over k repeated
presentations of the same item:

| decision AUC after k repetitions | k=1 | k=2 | k=3 | k=4 | k=6 | k=8 | k=10 |
|---|---|---|---|---|---|---|---|
| Riemannian, within-subject | 0.86 | 0.93 | 0.97 | 0.98 | 0.99 | 0.997 | 0.999 |
| EEGNet, within-subject | 0.86 | 0.93 | 0.96 | 0.98 | 0.99 | 0.997 | 0.999 |
| EEGNet, cross-subject | 0.79 | 0.87 | 0.92 | 0.95 | 0.97 | 0.99 | 0.99 |
| Riemannian, cross-subject | 0.78 | 0.86 | 0.91 | 0.93 | 0.96 | 0.98 | 0.99 |

This is the curve that sets a product's session length. With a per-person calibration,
three repetitions already give a decision AUC above 0.95; with no calibration at all it
takes four to six. The same detector that looks mediocre at the single-trial level (0.8)
becomes reliable once evidence is aggregated, which is how every deployed P300 system works.

## Why these three models

| | xDAWN + LDA | Riemannian (xDAWN cov → tangent space → logreg) | EEGNet |
|---|---|---|---|
| Idea | supervised spatial filters that maximise the evoked response, then a linear classifier on the filtered epoch | covariance of [evoked prototype ; epoch] mapped to a Euclidean tangent space; geometry handles scale/impedance differences | learned temporal + depthwise spatial filters |
| Calibration data needed | a few hundred trials | a few hundred trials | thousands |
| Cross-subject robustness | moderate | best in most benchmarks | fragile on small cohorts |
| Cost | ms | ms | seconds to train, ms to run |

## Validation regimes

* **Within-subject** (stratified 5-fold per subject): performance after a short calibration
  session for that person. This is how P300 spellers and concealed-information tests are
  deployed.
* **Cross-subject** (leave-one-subject-out): zero-calibration performance.
* **Aggregation curve**: draw k target and k non-target trials of the same subject, average
  the scores, and measure the AUC of the aggregated decision as k grows (400 draws per k).

## Quick start

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install -e ".[dev]"
pytest -q                          # synthetic-ERP tests, no download
python -m p300erp.run --prepare    # downloads BNCI 2014-008 via MOABB (~100 MB), runs everything
```

## Data

BNCI 2014-008 (Riccio et al. 2013, *Frontiers in Human Neuroscience*): 8 participants with
ALS, 8-channel EEG (Fz, Cz, Pz, Oz, P3, P4, PO7, PO8), 256 Hz, row/column P300 speller, 10
repetitions per character, 4200 stimulus-locked trials per subject. Accessed through
[MOABB](https://moabb.neurotechx.com/). Here: resampled to 128 Hz, 1–24 Hz band-pass,
0–0.8 s post-stimulus.

## Author

Ankan Biswas — PhD (Neuroscience), IISc Bengaluru. ankanbiswas0804@gmail.com ·
[eeg-cognitive-scoring](https://github.com/ankanbiswas022/eeg-cognitive-scoring) ·
[Google Scholar](https://scholar.google.com/citations?user=oG28KRIAAAAJ) ·
[LinkedIn](https://www.linkedin.com/in/ankan-biswas-45357685/)
