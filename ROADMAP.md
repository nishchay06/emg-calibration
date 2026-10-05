# Roadmap

Path from the completed generic baseline (M4) to a written result. Each
milestone ends in an **acceptance gate**: a measurable check that must pass
before the next milestone spends money. Results are captured as structured JSON
under `results/`, as in M3 and M4.

**Drafted:** 2026-10-04 · **Target:** report and repository finished by Dec 1,
with publication-ready figures, evidence and reproducibility instructions.

## Where we are

| Milestone | Status |
|---|---|
| M0–M2: preflight, upstream audit, Runpod smoke test | Done |
| M3: one-user generic baseline | Done |
| M4: eight-user generic baseline (55.38% ± 4.38 test CER, greedy) | Done 2026-10-04 |
| M5a: released personalized checkpoints (both families, all users) | Done 2026-10-04 |
| M5b: own full-data training (user0/user5, within 1.0 pp) | Done 2026-10-05 |

The eight-user subset contains 100 sessions and occupies 28.42 GB (26.47 GiB)
after selective extraction. Dataset and checkpoints are obtained from upstream
and retained outside Git.

**Storage update, 2026-10-05 IST:** the attempt to migrate the working data from
400 GB to 50 GB stopped at the transfer gate. The original source remains
authoritative; the incomplete candidate must not be used for training. Both
temporary CPU Pods were deleted, and both volumes were retained. See
`results/storage-migration-attempt-20261005.json`. M6 remains the next research
gate and can proceed locally.

## Upstream facts that shape the design

Audited at the pinned commit `3200d91`:

- **Fine-tuning recipe.** Upstream personalization is `emg2qwerty.train` with
  `checkpoint=generic.ckpt` and `user=userN`: Adam at 1e-3, 10 warm-up epochs,
  cosine decay, **150 epochs**, batch size 32, 4-second windows (8,000 samples
  at 2 kHz) with 900 ms past and 100 ms future padding.
- **Model selection uses the user's validation sessions.** Upstream keeps the
  checkpoint with the best `val/CER`, so it uses labeled validation data from the
  new user. That is fine for reproducing the paper, but at a 1-minute budget it
  would bring in far more labeled data than the budget allows. The budget
  experiments must not select checkpoints on per-user validation data (see M6).
- **Epochs do not scale across budgets.** One minute of EMG is about 15
  four-second windows, which is less than one batch. "150 epochs" would mean 150
  optimizer steps at 1 minute but thousands at full data. The budget
  experiments use a **fixed number of optimizer steps** instead.
- **Normalization layers.** The model has one `BatchNorm2d` (`SpectrogramNorm`,
  at the input) and `LayerNorm`s inside every TDS block. "Normalization-only"
  adaptation therefore means BatchNorm plus LayerNorm affine parameters. The
  label-free AdaBN variant (RQ3) can only touch the single input BatchNorm. That
  makes it conceptually close to SplashNet's Rolling Time Normalization, which is
  worth saying explicitly in the report.

## Milestones

### M5a: Released personalized checkpoints (evaluation only)

- Evaluate `models/personalized-finetuned/user{0..7}.ckpt` with greedy
  decoding, reusing the M4 sweep with a personalized-checkpoint mode.
- **Gate:** every validation and test CER within 0.10 pp of the upstream
  reference (fine-tuned, no LM: mean test 11.28% ± 4.76).
- Optional, same Pod: `personalized-randominit` (mean test 15.38%), which shows
  how much the generic pre-training contributes.
- **Cost:** similar to M4 (under $1).

### M5b: Own fine-tuning reproduces the paper

Build the adaptation harness **once**, with the budget and method parameters
from the start, so that this reproduction is simply the `budget=full,
method=full` point of the final grid.

- `scripts/adapt.py` (wrapping upstream `train.py`/Lightning): arguments
  `--user`, `--budget-minutes {1,2,5,10,30,60,full}`, `--method
  {full,head,norm,lora}`, `--seed`, `--steps`, `--select {upstream,fixed}`.
- Run `user0` at `budget=full, method=full, --select upstream` (the exact
  upstream recipe). If it passes, run `user5` (the user with the lowest CER).
- **Gate:** test CER within **1.0 pp** of the upstream per-user value (`user0`
  20.57%, `user5` 5.81%). Training is not bit-for-bit deterministic, so the
  tolerance is wider than M4/M5a.
- **Measure:** wall-clock time and cost per full-data fine-tune. This number
  sets the M9 budget.
- If the gate fails, stop and diagnose before building anything else. Every
  later result depends on this pipeline.

**Passed 2026-10-05 IST:** both users completed 150 epochs; user0 test CER
21.209850% (+0.639850 pp), user5 6.130137% (+0.319137 pp). See
`results/m5b-full-upstream-summary.json` and latest `RESEARCH_STATUS.md` for time,
provisional costs and verified cleanup.

### M6: Calibration-budget sampler and training protocol

No GPU needed; local tests only.

- **Budget definition.** *N* minutes means *N* × 120,000 samples of the user's
  **training** sessions, taken as one contiguous window from a single
  seeded-random session, which mimics one calibration sitting. If *N* minutes
  exceeds one session, take consecutive sessions.
- **Budgets:** 1, 2, 5, 10, 30, 60 minutes and full. Adding 30 and 60 is cheap,
  and without them the curve has a gap between 10 minutes and about 3 hours.
- **Protocol for budgeted runs (`--select fixed`):** a fixed step count and a
  fixed learning-rate schedule per method, with no per-user validation-based
  checkpoint selection. Evaluate the final checkpoint on the user's test
  sessions.
- **Hyperparameters:** choose steps and learning rate per method on `user0`
  and `user1` only (using their validation sessions), then freeze them. Report
  all 8 users, and also the 6 untouched users separately as a check.
- **Gate (unit tests):** exact sample counts; same seed gives the same window;
  windows never overlap validation or test sessions; the `full` budget
  reproduces the upstream training set exactly.

### M7: First curve (first real result)

- `method=full`, budgets 1 → full, users 0–2, 1 seed.
- **Gate:** CER falls (roughly) monotonically with budget; the `full` point
  agrees with M5b; 0 minutes equals the M4 generic result.
- **Output:** first figure, CER vs minutes (log x-axis) per user. This is
  already a result worth putting in the lab interest form and the email to the
  professor.

### M8: Adaptation methods

- `head`: final linear layer only.
- `norm`: `SpectrogramNorm` BatchNorm plus all `LayerNorm` affine parameters.
- `lora`: low-rank adapters on the TDS `Conv2d` and `Linear` layers and the
  input MLP (rank 4 and 8). Base weights stay frozen.
- Log the trainable-parameter count for every method.
- **Gate:** each method runs on `user0` at 5 minutes and full; trainable counts
  match expectations; `lora` with rank → large approaches `full`.

### M9: Full grid

- 8 users × 7 budgets × 4 methods × 3 seeds (the full budget needs 1 seed,
  since it has no window choice): about 600 runs, most of them short.
- **Spend gate:** project the cost from the M5b timing before launching. If the
  projection exceeds the cap (see Budget), cut seeds to 2 or drop the 60-minute
  budget first.
- Run budgets in ascending order so the cheap, most interesting points land
  first.
- **Gate:** every run produces a result JSON; seed-to-seed spread is reported.

### M10 (stretch): Label-free adaptation

Drop this if M9 has not finished by Nov 15.

- AdaBN: re-estimate `SpectrogramNorm` statistics on the user's unlabeled
  signal at each budget.
- Tent: entropy minimization on normalization parameters.
- Compare both against zero-shot and against `norm` at the same budget.

### M11: Language-model decoding for headline numbers

- Build or install KenLM and re-decode the final checkpoints with the 6-gram
  beam decoder. The literature quotes with-LM numbers (51.78 generic, 6.95
  fine-tuned), so the report needs both.
- Evaluation only; no retraining.

### M12: Write-up

- Main figure (CER vs minutes, one line per method, per-user spread), a results
  table, and parameter counts and adaptation time.
- A 4–6 page report (Markdown → PDF), with limitations and any refuted
  hypotheses.
- README rewritten to lead with the question and the figure. Move the cost and
  experimental measurements in `results/`; keep operational records local.
- Delete the Runpod volume once all artifacts are copied off.

## Timeline

| Dates | Milestones | Deliverable |
|---|---|---|
| Oct 5 – Oct 8 | M5a, M5b | Personalized baselines reproduced; cost per fine-tune known |
| Oct 9 – Oct 15 | M6, M7 | **First curve** (method=full, 3 users) |
| Oct 16 – Oct 25 | M8 | All four methods working |
| Oct 26 – Nov 8 | M9 | Full grid |
| Nov 9 – Nov 15 | M10 (stretch) | Label-free comparison |
| Nov 16 – Nov 20 | M11 | With-LM numbers |
| Nov 21 – Dec 1 | M12 | Report, figures, public repo |
| Dec 1 – Dec 15 | Buffer | Review and revisions |

## Budget

- Spent through M4: about $4 (compute and storage).
- Storage: the migration cleanup read-back recorded about $0.044/hour
  ($1.05/day) for the retained 400 GB source and incomplete 50 GB candidate.
  This is a historical estimate, not a live billing check or final charge.
- Compute: full-data runs took 27.93–34.44 minutes per user on RTX 4090. **Planning cap: $75 in total compute.**
  Re-check after M5b and M7.

## Fixed experimental decisions

1. Budget runs use fixed optimizer steps and final-checkpoint evaluation;
   per-user validation cannot select a checkpoint.
2. Tune steps and learning rates only on user0/user1, then freeze.
3. Calibration budgets are 1, 2, 5, 10, 30, 60 minutes and full.
4. Recheck the $75 compute plan after M5b and M7 using attributable charges;
   keep pending estimates separate from itemized costs.

## Risks

| Risk | Mitigation |
|---|---|
| M5b misses the 1.0 pp gate | Diagnose before M6. Likely causes: seed, number of workers, Lightning version. Record it in `RESEARCH_STATUS.md`. |
| Full-data fine-tuning is slow (hours per user) | Run the full budget only once per user (1 seed), in parallel Pods if needed. Short budgets are cheap. |
| Very small budgets are unstable | Fixed steps, 3 seeds, and report the spread. Consider 2× steps at 1–2 minutes only if tuned on user0/1. |
| Loss of ephemeral compute or stored artifacts | Copy evidence after every milestone and verify archive/file hashes before teardown. |
| Scope creep | M10 is the first thing dropped; M11 is evaluation only. |
