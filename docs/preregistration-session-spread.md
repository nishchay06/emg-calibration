# Pre-registration: one calibration sitting vs several sessions

**Locked:** 2026-10-24, before any run of this experiment.
**Status:** locked. Changes after this commit go in *Deviations* with a reason.

## Question

At a fixed amount of labelled calibration data, does taking it from several
recording sessions (separate band don/doff) give a personalized sEMG typing
decoder a different test CER than taking it from one session?

- **H1 (primary, two-sided):** spread and single-session calibration differ in
  test CER at equal minutes. Expected direction: spread is lower.
- **H0:** no difference.

## Motivation and prior work

- In the M7 curve, user0's CER barely changes from 5 to 60 single-sitting
  minutes and then drops at the full multi-session budget. All three M7 users
  are worse at 10 minutes than at 5 minutes.
- Kaifosh et al. (*Nature* 2025, doi:10.1038/s41586-025-09255-w) personalize
  handwriting decoders with 1–20 minutes sampled uniformly across all training
  sessions, so the contrast with a single sitting is never made.
- Myoelectric gesture studies show that training across conditions improves
  robustness, but the multi-session condition usually adds data. Burrello et
  al. (IEEE SAS 2021, doi:10.1109/sas51076.2021.9530007) compare one session
  with training on sessions 1–10 and never match the amount of data. Fougner et
  al. (IEEE TNSRE 2011, doi:10.1109/tnsre.2011.2163529) give a size-matched
  comparison, but across limb positions within one sitting.
- To our knowledge, no prior work holds the calibration budget fixed while
  varying the number of recording sessions for a pretrained decoder.

## Fixed (same as M7)

- Released generic checkpoint as initialization; method `full`.
- Frozen profile: 3,000 optimizer steps, peak learning rate 3e-4, 300 warmup
  steps, cosine schedule (`configs/m7-frozen-protocol.json`).
- Final checkpoint evaluated; no validation-based selection.
- Validation and test sessions are never used for calibration.
- Outcome: test CER (%), greedy decoding, no language model.

## Conditions

| Condition | Algorithm | Definition |
|---|---|---|
| **Single** | `training-contiguous-v1` (existing, unchanged) | One contiguous window of the full budget from one seeded-random training session |
| **Spread** | `training-spread-v1` (new) | 5 distinct training sessions drawn uniformly without replacement (seeded); one contiguous chunk of budget ÷ 5 from each, at a seeded uniform-random offset |

Chunk sizes: 5 minutes → 5 × 1 minute (120,000 samples each); 10 minutes →
5 × 2 minutes (240,000 samples each). Every test user has at least 8 training
sessions, so 5 distinct sessions are always available.

## Grid

- **Users:** 0–7. Users 0 and 1 were the M7 tuning users; users 2–7 are untouched.
- **Budgets:** 5 and 10 minutes. Both fit inside one session for every user,
  so *single* is truly one session.
- **Seeds:** 1501 and 1502. The seed sets both the data draw and the training
  randomness, as in M7.
- **Reused:** M7's single-session seed-1501 results for users 0–2 at 5 and
  10 minutes (6 runs, same protocol).
- **New runs:** 64 − 6 = **58**, plus full-data runs for users 3, 4, 6 and 7
  (user5's exists from M7) = **62 runs**.
- **Estimate:** about 12 GPU-hours.

## Primary analysis

For each user *u*:

> D_u = mean over budgets {5, 10} and seeds {1501, 1502} of (CER_single − CER_spread)

Two-sided Wilcoxon signed-rank test on D_0 … D_7 (n = 8), α = 0.05.
Report the median D with a 95% bootstrap CI: resample users, 10,000
resamples, fixed seed. With n = 8 the smallest possible two-sided p is 0.0078.

**Interpretation, fixed in advance:**
- p < 0.05 and median D > 0: spread helps.
- p < 0.05 and median D < 0: spread hurts.
- Otherwise: no detectable difference at this sample size. Report the CI and
  make no equivalence claim.

## Secondary analyses (always reported)

1. The primary test restricted to untouched users 2–7 (n = 6; smallest p 0.031).
2. Per-budget differences at 5 and 10 minutes, descriptive.
3. Fraction of the full-personalization gain recovered,
   (generic − CER) ÷ (generic − full), for each condition.
4. Seed-to-seed spread of single-session CER, which measures session luck.
5. Whether the 10-minute single-session inversion of users 0–2 persists with seed 1502.

## Exploratory (labelled as such)

- Days between the calibration sessions and the test sessions.
- Keystrokes in the calibration data.
- Relation of both to the user's single-minus-spread difference.

## Exclusions

A run is invalid only if its loss is NaN/inf after one rerun, or its data
can't be read. Invalid runs are reported, never silently dropped. No other
exclusions, and no reruns to get better numbers.

## Required sampler tests (before any run)

- Exact total sample count for each budget.
- Exactly 5 distinct sessions, all from the training split.
- Chunks stay inside session bounds and never overlap validation or test sessions.
- Same seed gives the same draw; different seeds give different draws.
- `training-contiguous-v1` draws are unchanged by the new code.

## Deviations

None yet.
