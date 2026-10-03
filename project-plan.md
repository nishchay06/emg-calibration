# How Much Data Does a New User Cost?
### Data-Efficient Personalization of Surface-EMG Typing Decoders


*Project plan, v0.2 (October 2026). Living document: updated as results come in.*

## Summary

Neuromotor interfaces such as wrist-worn surface electromyography (sEMG) decoders work well for the people they were trained on and noticeably worse for a new person. The usual fix is to collect calibration data from each new user and fine-tune, but calibration time is the real cost a user pays before the interface becomes useful. This project measures that cost directly on the public **emg2qwerty** benchmark: *for a new user, how quickly does decoding error fall as a function of minutes of calibration data, and which adaptation method gets there with the least data and the fewest updated parameters?* The output is a reproducible set of data-efficiency curves, an open-source harness, and a short technical report.

## Motivation

Interfaces that couple directly to the body have to adapt to the person wearing them. sEMG signals vary across people because of anatomy, electrode placement and typing style, so a model trained on one population transfers imperfectly to the next user. The emg2qwerty benchmark (Sivakumar et al., NeurIPS 2024) shows this clearly: a generic model trained on many users has much higher character error rate (CER) on unseen users than a model fine-tuned on each user’s own data.

That comparison uses the user’s full set of training sessions. In practice, nobody will record hours of data before using a new device. The question that matters for deployment is the shape of the curve in between: how much of the personalization gain is recovered with 1, 2, 5 or 10 minutes of data, and whether lighter-weight adaptation (updating a small fraction of parameters) is as data-efficient as full fine-tuning. If a few minutes and a few thousand parameters are enough, per-user adaptation becomes cheap enough to run on the device and to repeat as the user’s state changes.

## Related work and novelty check (Phase 0, 2026-10-02)

Searched the 41 papers citing emg2qwerty (Semantic Scholar). **None reports CER as a function of calibration minutes, or compares parameter-efficient adaptation methods, on emg2qwerty.** Every paper uses the benchmark's two fixed settings: zero-shot generic, and full per-user fine-tuning. The gap looks open. Closest work:

| Paper | What it does | Relevance |
|---|---|---|
| **SplashNet** (arXiv 2506.12356, 2025) | Rolling Time Normalization aligns input statistics across users; channel masking; split-hand encoder. Zero-shot CER 51.8 → 36.4, fine-tuned 7.0 → 5.9. | **Closest.** RTN is label-free input-level adaptation, adjacent to RQ3. Must cite; include RTN as a baseline or comparison for RQ3. No public code found. |
| **Open, Accurate, and Calibration-Free MCIs** (CHI 2026, doi:10.1145/3772318.3790689) | Calibration-free 5-class gesture recognition from a 612-user Myo dataset. | Supports the framing: calibration-free works for a few gestures, while full-alphabet typing still has ~50% zero-shot CER. Calibration cost still matters for typing. |
| **Typing Reinvented** (arXiv 2511.18213, 2025) | Attention models on emg2qwerty; personalized CER 10.86 → 10.10. | Architecture work; uses the standard fixed settings. |
| **Scaling and Distilling Transformers for sEMG** (arXiv 2507.22094, 2025) | Scales to 110M params for cross-user performance, distills 50×. | Generic-model side of the same trade-off. |
| **Meta neuromotor interface** (Nature 2025) | Generic models from thousands of users; small personalization gains. | Main motivation reference; read the personalization section. |
| **EMGBlend** (arXiv 2609.25582, 2026) | Multi-dataset self-supervised EMG pretraining. | Possible stronger base model; out of scope for v1. |

Caveat: the citation list may be incomplete, and the broader EMG gesture literature (e.g. Ninapro) has few-shot calibration work. Positioning: **first data-efficiency and adaptation-method study on open-vocabulary sEMG typing**, not first few-shot EMG work.

### Numbers to reproduce (paper Table 2, mean over 8 test users, test CER %)

| Setting | No LM | With 6-gram LM |
|---|---|---|
| Generic (zero-shot) | 55.38 (sd 4.38) | 51.78 (sd 4.93) |
| Personalized, random init | 15.38 (sd 6.28) | 9.55 (sd 5.52) |
| Personalized, fine-tuned from generic | 11.28 (sd 4.76) | 6.95 (sd 3.86) |

Per-user values are in `scripts/experimental_results.py` in the repo. Variation across users is large (fine-tuned no-LM ranges from 5.8 to 20.6), so report per-user curves, not just means.

### Reproduction progress (2026-10-03)

The first zero-shot sanity check is complete. On `user0`, the released generic
checkpoint with greedy CTC produced **60.082565% validation CER** and
**61.509636% test CER**, compared with the upstream per-user references of
60.07% and 61.48%. The gaps are +0.012565 and +0.029636 percentage points,
respectively. This validates the pinned environment, checkpoint, selective data
staging, and evaluation path for one user; the remaining seven-user and
personalized baselines are still pending.

### Data and compute

- The full dataset is a single **308 GB** tarball. Only the 8 benchmark test users are needed (100 sessions total: train/val/test per `config/user/user{0..7}.yaml`, roughly 9% of sessions, likely ~25–35 GB).
- Plan: stream the tarball and extract only those sessions (`curl … | tar -xz` with a file list), so disk use stays small even though the whole file passes through the network once.
- Released checkpoints (`models/generic.ckpt`, `models/personalized-finetuned/user*.ckpt`) come via git-lfs, so no generic-model training is needed.
- Use **personal** hardware and accounts only (no work laptop GPU, cloud desktop or work AWS accounts).

## Research questions

- **RQ1. Data efficiency.** For a held-out user, how does CER change as a function of minutes of labeled calibration data, starting from the released generic model?

- **RQ2. Method comparison.** At each data budget, how do adaptation methods that update different amounts of the model compare: full fine-tuning, final-layer only, normalization-layer only, and low-rank adapters (LoRA)?

- **RQ3. Label-free adaptation** *(stretch)*. How much of the gain can be recovered with no labels at all, by adapting normalization statistics to the new user’s unlabeled signal (test-time adaptation)? Compare against SplashNet’s Rolling Time Normalization, the closest prior label-free method.

### Hypotheses, and what would refute them

- **H1.** Most of the personalization gain is recovered within the first few minutes of data, so the curve flattens early. *Refuted if* CER keeps falling roughly linearly in log-minutes up to the full-data setting.

- **H2.** At small budgets, parameter-efficient methods (normalization-only, LoRA) match or beat full fine-tuning because they overfit less; at large budgets, full fine-tuning wins. *Refuted if* full fine-tuning is best at every budget.

- **H3.** Unlabeled normalization adaptation closes a non-trivial part of the gap. *Refuted if* it is indistinguishable from the zero-shot generic model.

A negative result on any of these is still a result worth writing down.

## Experimental setup

|  |  |
|:---|:---|
| **Dataset** | emg2qwerty: 1,135 sessions, 108 users, 346 hours of two-wristband sEMG recorded during touch typing, with keylogger ground truth. Official benchmark splits. |
| **Base model** | Released generic checkpoint and baseline architecture from the emg2qwerty repository (CTC-trained convolutional encoder). I do not retrain the generic model, which keeps the project within single-GPU compute. |
| **Calibration budgets** | 0 (zero-shot), 1, 2, 5, 10 minutes, and the full per-user training set (reproduces the paper’s personalized setting). Budgets are sampled as contiguous time windows from the user’s training sessions. |
| **Methods** | (a) zero-shot generic; (b) full fine-tuning; (c) final-layer only; (d) normalization layers only; (e) LoRA adapters on convolutional and linear layers; (f) label-free adaptation (RQ3): re-estimating normalization statistics on the user’s unlabeled signal (AdaBN-style), and entropy minimization on normalization parameters (Tent-style). |
| **Metric** | Character error rate on each held-out user’s test sessions, with greedy CTC decoding (primary) and beam search with the released character language model (secondary). Also reported: number of updated parameters and adaptation wall-clock time. |
| **Rigor** | 3 random seeds per (user, budget, method), with different calibration windows per seed. Per-user results, mean and spread. Hyperparameters chosen on validation sessions only, never on test. |
| **Compute** | One GPU (Colab or a single rented A100/L40S). |

### Sanity checks before any new experiment

1.  Reproduce the paper’s generic and personalized CER numbers from the released checkpoints.

2.  Confirm my full-data fine-tuning reproduces the personalized result, so that every other point on the curve shares a validated pipeline.

## Planned deliverables

- **Main figure:** CER vs. minutes of calibration data (log scale), one line per method, with per-user spread.

- **Table:** CER at each budget, parameters updated, adaptation time.

- **Open-source harness** on GitHub: one command per experiment, configs checked in, results regenerable from scratch.

- **Technical report** (4–6 pages): question, setup, results, and limitations, including any hypotheses that turned out wrong.

## Timeline

| **Dates (2026)** | **Milestone** |
|:---|:---|
| Oct 5 – Oct 18 | Environment, data download, reproduce generic and personalized baselines. |
| Oct 19 – Nov 8 | Calibration-budget sampler; methods (b)–(e); first full curve on a subset of users. |
| Nov 9 – Nov 22 | All users and seeds; RQ3 if on schedule; ablations (window choice, learning rate). |
| Nov 23 – Dec 6 | Figures, technical report, repository cleanup. |
| By Dec 15 | Publish the report and reproducibility repository. |

## Risks and mitigations

- **Cannot reproduce baseline numbers.** Report the gap honestly, and contact the authors via the repository; all comparisons are made within my pipeline either way.

- **Compute or storage limits.** Run the full grid on a subset of users first; prioritize the smallest budgets, which are the cheapest and the most interesting.

- **Very short windows make training unstable.** Fix the number of update steps across budgets and report variance across seeds rather than only best runs.

- **Scope creep.** RQ3 is explicitly a stretch goal and is dropped if RQ1–RQ2 are not done by mid-November.

## Broader motivation

This project treats calibration time as a measurable systems cost. The immediate
goal is to build a reproducible evaluation pipeline for personalization under
small data budgets. Beyond this study, the same framework can support questions
about session drift, fatigue, electrode shift, and when a decoder should adapt
or request recalibration.

## References

- V. Sivakumar et al. *emg2qwerty: A Large Dataset with Baselines for Touch Typing using Surface Electromyography.* NeurIPS 2024, Datasets and Benchmarks Track. [arXiv:2410.20081](https://arxiv.org/abs/2410.20081). Code: [facebookresearch/emg2qwerty](https://github.com/facebookresearch/emg2qwerty).

- CTRL-labs at Reality Labs. *A generic non-invasive neuromotor interface for human-computer interaction.* Nature, 2025. Code: [facebookresearch/generic-neuromotor-interface](https://github.com/facebookresearch/generic-neuromotor-interface).

- E. J. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.

- Y. Li et al. *Revisiting Batch Normalization for Practical Domain Adaptation* (AdaBN). ICLR Workshop 2017.

- D. Wang et al. *Tent: Fully Test-Time Adaptation by Entropy Minimization.* ICLR 2021.
