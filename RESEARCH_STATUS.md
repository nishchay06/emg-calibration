# Research status

Upstream is pinned to `3200d91eeb952cbed1f278e47d0cc56928334fd1`.
Research evidence belongs in `results/`; raw data, checkpoints and machine-local
session records stay outside version control.

## Completed gates

- M3: user0 generic greedy-decoder baseline reproduced.
- M4: generic released checkpoint reproduced for all eight users; all
  validation/test CER checks within 0.10 percentage points of reference.
- M5a: both released personalized checkpoint families reproduced for all eight
  users, with the same 0.10 pp per-split tolerance.
- M5b: pinned adaptation harness and CPU checkpoint-restoration smoke passed.
  User0 full-data reproduction passed: 150 epochs, test CER 21.209850%,
  +0.639850 pp from reference, 34.44 minutes. User5 also passed: 150 epochs,
  test CER 6.130137%, +0.319137 pp from reference, 27.93 minutes.

## Evidence policy

Publish metrics, unsuccessful experiments, recipe/version pins, checkpoint and
archive hashes, measured runtime, and clearly qualified experiment costs.
Keep account balances, access coordinates, resource identifiers and live-session
handoffs in ignored local records. Retaining a digest establishes identity;
it does not make a private raw archive publicly downloadable.

Raw participant data and model checkpoints are not redistributed. Refer to the
upstream project for permitted access and licensing. See `README.md` for the
research objective and `ROADMAP.md` for acceptance criteria.

## Interrupted user5 attempt

The first user5 attempt stopped after four epochs: 77–95 seconds per epoch
projected beyond the planned session. It produced no final test CER and has no
accuracy-gate verdict. Matching packages/recipes and a 6.842× slower synthetic
CUDA fit suggested an execution-environment regression; driver differences and
shared cuDNN warnings did not establish causality. See
`results/m5b-user5-interrupted-upstream.json` and
`results/m5b-user5-runtime-diagnosis.json`.

## Bounded performance diagnostic

All six synthetic/real train/validation phases completed without recipe changes.
The historical slowdown was not reproduced. User5 warmed training computation
was 79.08 ms/batch; loader wait varied. CPU quota/thread and stage measurements
are in `results/m5b-performance-diagnostic-20261005.json`. The exact original
cause remains unresolved. Short-batch projections are scenarios, not bounds.

## M5b completion and next gate

Both full-data runs passed the 1.0 pp test CER tolerance; selected and last
checkpoints were preserved and independently verified. All 150 epochs are
recorded for both users. `results/m5b-full-upstream-summary.json` consolidates
metrics and runtime. The full runs use upstream validation checkpoint selection
for reproduction; this must not be used for calibration-budget experiments.

M6's local deterministic calibration-window sampler, exact sample counts,
training-only windows, no validation/test overlap, and full-training-set identity
have now passed their local acceptance checks (see below).
Budget experiments use fixed optimizer steps and final-checkpoint evaluation;
hyperparameters are tuned only on user0/user1, then frozen. No calibration-minute
curve or head/norm/LoRA result is claimed yet.

## Incomplete storage migration

The 400 GB to 50 GB storage migration stopped at its file-transfer gate.
Source checks verified 100 expected HDF5 sessions (28,417,553,136 bytes), the
generic checkpoint digest, and a preservation inventory of 1,012 files
(30,539,725,849 bytes). The large dataset archive was excluded from the copy;
the original source was retained.

A bounded network-only probe passed, but all eight parallel rsync file streams
timed out. This does not establish a filesystem throughput fix; the cause
remains unresolved. The destination contains partial files and has not passed
digest, HDF5-readability or mounted-configuration verification. It must not be
used for training.

Both temporary CPU Pods were deleted; read-back at 2026-10-04 22:50:25 UTC
confirmed zero Pods and endpoints, with both volumes retained. CPU cost is an
estimate of $0.05189 through cleanup, with itemized billing pending. Raw evidence
was preserved locally and its archive digest verified. See
`results/storage-migration-attempt-20261005.json`. Training recipes and the
passed M5b gate were unchanged; M6 was the next gate at that time.

## M6 local acceptance

The deterministic sampler, bounded HDF5 dataset adapter and fixed-update runner
passed local tests. All seven budgets and eight pinned split configurations
were checked with synthetic lengths; padding, jitter and labels stay inside
selected ranges, and full-data loading delegates to upstream.

The real 5,293,315-parameter upstream model completed a synthetic CPU check:
exactly three optimizer updates, batches of 2/1/2, and learning rates
1e-8/1e-3/1e-6. No validation batches ran during fitting. Final checkpoint state
and evaluation outputs matched after reload; ordinary test and tuning-only
validation paths both completed. These checks use seeded random initialization,
not the released generic checkpoint, and establish no participant CER result.

All 52 local unit tests passed with no skips, along with shell syntax, CLI guards
and parallel archive-staging checks. Ordinary calibration runs require a frozen
profile supported by user0/user1 final-checkpoint tuning receipts; no research
profile has been tuned or frozen yet. See `M6_RUNBOOK.md` and
`results/m6-local-acceptance.json`.

Next: prepare the bounded M7 tuning/first-curve proposal. Before real training,
verify actual session coverage/allocation, generic initialization and CUDA runtime,
then tune only on user0/user1 and freeze the selected full-method profile. The
storage migration remains incomplete; the original source remains authoritative.

## M7 offline work during the capacity wait

Session A is approved but its initial start stopped before provisioning at the
location-specific GPU availability gate. No participant tuning or freeze has
occurred; Session B remains a separate approval. The user will signal when
capacity is available, so no availability polling is running.

Offline inspection found a receipt-count mismatch: the tuning worker supplies
four winning receipts (users0/1 × 5/full), while protocol validation previously
required exactly two. The fix retains matched budget/seed contexts, rejects
duplicates and incomplete pairs, and passes a full selection-to-freeze regression.
The fixed-step training recipe and approved tuning grid are unchanged.

The offline reporting pipeline validates a complete 21-point full-method curve
and user5 full sentinel, replays selected allocations against pinned splits,
checks protocol/initialization/final-checkpoint provenance, and exports JSON,
CSV, Markdown and PNG/PDF/SVG figures. It separates tuning users, reuses M4
generic baselines, places full points at measured recording minutes, and retains
failed scientific checks and adjacent inversions. Synthetic fixtures exercise
the pipeline without recordings or participant training; their outputs are
explicitly fictional and remain ignored. See `M7_ANALYSIS_RUNBOOK.md` and
`results/m7-offline-analysis-preparation.json` for scope and local verification.

Next: wait for the user's availability signal, refresh the exact approved quote
and balance once, then perform Session A within its existing limits. Offline
analysis does not pass M7, freeze a real profile, or authorize the curve session.

## M7 measurements — 2026-10-18

Completed the required preflight and smoke run (user0, 100 updates; validation CER 48.38%; final-checkpoint save/reload verified). All 16 tuning runs completed; selected and froze the full-method profile at 3,000 steps and learning rate 3e-4, with mean final-checkpoint validation CER 21.11%. Completed all seven user0 curve budgets: test CER was 51.62% (1 min), 45.16% (2), 41.30% (5), 79.12% (10; inversion), 40.82% (30), 41.76% (60), and 21.43% (full). Estimated session spend was $4.42; the Pod was terminated and zero Pods remain. Remaining M7: user1/user2 curves, user5 full sentinel, plot and inversion summary.

## M7 continuation attempt — 2026-10-18

Committed and pushed the completed smoke, 16-run tuning set, frozen profile, and seven user0 curve records as `da75666`. A single continuation Pod create for the first approved GPU (RTX 4090, Community, US-IL-1; catalog showed Low stock) was rejected with Runpod's 400 “There are no longer any instances available with the requested specifications.” Per the fixed start rule, no retry, GPU substitution, location move, or training run was made; a fresh Pod listing confirms zero Pods. No new measurements or Pod compute were produced this attempt. Resume M7 at user1/user2 curves, user5 full sentinel, plot and inversion summary when the approved GPU can be placed.


## M7 measurements — 2026-10-18 (continued)

Completed the 1-minute full-method final-checkpoint test evaluations for user1 and user2 with the frozen 3,000-step, 3e-4 profile on an RTX 4090. Test CER was 37.16% for user1 (105.03 min) and 27.40% for user2 (105.33 min); both used 3,000 optimizer steps and completed with finite losses and gradients. The estimated session compute spend was about $3.8, and the Pod was terminated after these runs; no running Pods remain. User0’s recorded 10-minute inversion remains visible in the curve. Next: run the remaining user1/user2 budgets, then the user5 full-data report-only point, and generate the M7 plot and inversion summary.

At the user's request, ran the next M7 curve point on an RTX A5000: user1, 2-minute budget, full method, seed 1501, frozen profile (3,000 steps, lr 3e-4). Final-checkpoint test CER was 33.88%; training and test evaluation completed with finite losses and gradients in 30.54 minutes. The per-run GPU cost estimate is $0.14; total Pod GPU spend including setup was about $0.18 (about $0.21 including the existing volume during Pod uptime). A5000 availability in US-IL-1 was low and required CUDA 13.0 host placement; the image's CUDA 12.1 runtime worked. The earlier RTX 4090 comparison point was user1 at 1 minute and 105.03 minutes, so the different budgets do not establish a card-speed comparison. Pod terminated, zero Pods remain, and volume `ni0dpvtday` remains intact. Next: continue user1 at 5 minutes, then remaining user1/user2 budgets, user5 full sentinel, and the M7 plot/inversion summary.

## M7 continuation blocked — 2026-10-19

Attempted to start the next M7 curve point (user1, 5-minute budget, full method, seed 1501) on RTX 4090 Community in US-IL-1 ($0.34/hour), because the project’s GPU order places it first; the live catalog showed Low availability, but Pod creation returned HTTP 400, “There are no longer any instances available with the requested specifications.” The user clarified that only RTX A5000 should be used. The live US-IL-1 availability read did not list an A5000, so no A5000 Pod was created; per the create-failure rule, no second placement attempt or other GPU substitution was made. The post-attempt Pod listing confirmed zero Pods, so no compute was spent and no M7 measurement was produced. Account balance at session start was $1.0494981135. Next: run this same curve point on an A5000 once it is listed as available in US-IL-1.

## M7 measurement — 2026-10-19

Completed user1’s 5-minute full-method curve point (seed 1501; frozen profile, 3,000 steps at lr 3e-4) on an RTX A5000. Final-checkpoint test CER was 22.93% in 14.82 minutes, improving 10.95 percentage points over user1’s 2-minute result (33.88%); losses and gradients remained finite. Estimated run GPU cost was $0.067; total Pod GPU spend including setup was about $0.118 ($0.135 including the existing volume). The Pod was terminated and zero Pods remain. Next: user1 at 10 minutes.

## M7 measurement — 2026-10-19 (continued)

Completed user1’s 10-minute full-method curve point (seed 1501; frozen profile, 3,000 steps at lr 3e-4) on an RTX 4090. Final-checkpoint test CER was 32.41% in 9.98 minutes, a 9.47-point inversion from user1’s 5-minute CER of 22.93%; losses and gradients were finite. Estimated run GPU cost was $0.148 and total Pod GPU spend including setup was about $0.45; the Pod was deleted and zero Pods remain. Next: user1 at 30, 60 and full budgets, then user2’s remaining budgets, user5 full sentinel, and the M7 plot/inversion summary.

## M7 measurements — 2026-10-22

Completed user1’s 30-minute, 60-minute and full-budget points on an RTX 4090: final-checkpoint test CERs were 17.27%, 12.94% and 11.19% (3,000 steps, lr 3e-4; runtimes 8.21, 7.73 and 7.15 minutes). Completed user2 at 2, 5 and 10 minutes: test CERs were 23.25%, 19.91% and 21.75% (runtimes 34.82, 13.91 and 10.29 minutes; RTX 4090, 3,000 steps, lr 3e-4). User2’s 5-to-10-minute point is a 1.84 pp inversion; the run completed with finite losses and gradients and exit code 0. Next: user2 at 30, 60 and full budgets, then the user5 full-data sentinel and M7 plot/inversion summary.

## M7 measurement — 2026-10-24

Completed user2’s 30-minute full-method curve point (seed 1501; frozen profile, 3,000 steps at lr 3e-4) on an RTX 4090. Final-checkpoint test CER was 14.87% in 7.05 minutes; loss and gradients were finite and the run exited successfully. CER improved 6.88 pp from user2’s 10-minute point (21.75%); the earlier 5-to-10-minute inversion remains. Next: user2 at 60 and full budgets, then user5’s full-data sentinel and the M7 plot/inversion summary.

## M7 measurement — 2026-10-24 (continued)

Completed user2’s 60-minute full-method curve point (seed 1501; frozen profile, 3,000 steps at lr 3e-4) on an RTX 4090. Final-checkpoint test CER was 11.09% in 7.89 minutes; loss and gradients were finite and the run exited successfully. CER improved 3.78 pp over user2’s 30-minute point (14.87%). Next: user2 at the full-data budget (running), then user5’s full-data sentinel and the M7 plot/inversion summary.

## M7 measurement — 2026-10-24 (full budget)

Completed user2’s full-budget full-method curve point (seed 1501; frozen profile, 3,000 steps at lr 3e-4) on an RTX 4090. Final-checkpoint test CER was 9.28% in 6.85 minutes; loss and gradients were finite and the run exited successfully. CER improved 1.81 pp over user2’s 60-minute point (11.09%); the earlier 5-to-10-minute inversion remains. Next: user5’s full-data sentinel (running), then the M7 plot/inversion summary.

## M7 measurement — 2026-10-24 (user5 full-data sentinel)

Completed user5’s full-data comparison (seed 1501; frozen profile, 3,000 steps at lr 3e-4) on an RTX 4090. Final-checkpoint test CER was 5.81% in 6.92 minutes; loss and gradients were finite and the run exited successfully. User5’s M7 CER is 0.32 pp below M5b’s 6.13%, despite the frozen profile’s 3,000-update cap versus upstream’s 14,550; this point is report-only. Next: generate the CER-versus-minutes plot and list the curve inversions.

## M7 curve analysis — 2026-10-24

The validated M7 report passed its scientific checks; the CER-versus-minutes plot and 21-point CSV are saved under results/. CER generally fell as calibration time increased (Spearman rho: user0 -0.64, user1 -0.96, user2 -0.96; median CER 37.16% at 1 minute versus 12.94% at 60 minutes). Adjacent inversions: user0 +37.83 pp from 5 to 10 minutes and +0.93 pp from 30 to 60; user1 +9.47 pp from 5 to 10; user2 +1.84 pp from 5 to 10. User5 full-data sentinel was 5.81%, 0.32 pp below M5b’s 6.13%; this remains report-only. M7 first curve is complete; next: M8 methods.

## Session-spread preflight — 2026-10-24

No session-spread runs began: Runpod’s Pod capacity read showed no RTX 4090 capacity in US-IL-1; Secure inventory listed 4090 locations only in Europe, and Community inventory showed none. No Pod was created and no project data changed. Next: resume the preregistered run sequence when the requested GPU is available in US-IL-1.

## Session-spread preflight update — 2026-10-24

No session-spread runs began. A manually deployed RTX 4090 in US-IL-1 was available with the established data mount, but the account balance did not meet the required session-plus-storage reserve. Next: restore the required balance and resume with user0, 10 minutes, contiguous allocation, seed 1502.

## Session-spread measurement — 2026-10-28

Completed user0 at 10 minutes, contiguous allocation, seed 1502, using the frozen full-method profile (3,000 steps; learning rate 3e-4). Final test CER was 35.17% after 9.89 minutes; losses and gradients were finite. Validation CER was not measured for this test-only run; estimated run cost was $0.147 on an RTX 4090. Next: user1 at 10 minutes, contiguous allocation, seed 1502.

Completed user1 at 10 minutes, contiguous allocation, seed 1502, using the frozen full-method profile (3,000 steps; learning rate 3e-4) on an RTX 4090. Final test CER was 21.12% after 9.78 minutes; losses and gradients were finite. Validation CER was not measured for this test-only run; estimated run cost was $0.145. Next: user2 at 10 minutes, contiguous allocation, seed 1502.

Completed user2 at 10 minutes, contiguous allocation, seed 1502, using the frozen full-method profile (3,000 steps; learning rate 3e-4) on an RTX 4090. Final test CER was 17.22% after 10.00 minutes; losses and gradients were finite. Validation CER was not measured for this test-only run; estimated run cost was $0.148. The first three user 10-minute contiguous seed-1502 checks are complete; next: user0's remaining 5- and 10-minute session-spread grid points.

Completed user0 at 5 minutes, contiguous allocation, seed 1502, using the frozen full-method profile (3,000 steps; learning rate 3e-4) on an RTX 4090. Final test CER was 79.25% after 13.07 minutes; losses and gradients were finite and the run exited successfully. This is a large difference from the existing user0 curve; retain and report the preregistered result without rerunning. Validation CER was not measured; estimated run cost was $0.194. Next: user0 at 5 minutes, spread allocation, seed 1501.

Completed user0 at 5 minutes, spread allocation, seed 1501, using the frozen full-method profile (3,000 steps; learning rate 3e-4) on an RTX 4090. The selection algorithm was `training-spread-v1`; final test CER was 49.25% after 12.62 minutes, with finite losses and gradients. Validation CER was not measured; estimated run cost was $0.187. The available contiguous comparison is seed 1502, so a same-seed comparison is not yet available. Next: user0 at 5 minutes, spread allocation, seed 1502.

Completed user0 at 5 minutes, spread allocation, seed 1502, using the frozen full-method profile (3,000 steps; learning rate 3e-4) on an RTX 4090. The selection algorithm was `training-spread-v1`; final test CER was 51.04% after 14.41 minutes, with finite losses and gradients. Validation CER was not measured; estimated run cost was $0.214. In the single matched user0/5-minute/seed-1502 pair, spread CER was 28.21 pp lower than contiguous; treat this as preliminary pending the full grid. Next: user0 at 10 minutes, spread allocation, seed 1501.

Completed user0 at 10 minutes, spread allocation, seed 1501, using the frozen full-method profile (3,000 steps; learning rate 3e-4) on an RTX 4090. The selection algorithm was `training-spread-v1`; final test CER was 38.30% after 9.97 minutes, with finite losses and gradients. Validation CER was not measured; estimated run cost was $0.148. Next: user0 at 10 minutes, spread allocation, seed 1502.
