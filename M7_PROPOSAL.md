# M7 proposal: first full-adaptation curve

## Status and scope

Draft for review. M6's local gate passed; no participant calibration run or
research-profile freeze has occurred. This proposal keeps the approved seven
budgets, seed 1501, training-only allocation, fixed updates and final checkpoint.
Head, normalization and LoRA remain M8 work.

## Session A: mounted-data preflight and tuning

Use the complete retained data directory and pinned M5b runtime. Verify the
100 expected nonempty HDF5 files against the preserved inventory, upstream pin,
generic checkpoint SHA-256, both M5b acceptance records and CUDA/packages.
Index actual recording lengths and require all 56 user/budget allocations to
pass, with positive window capacity. Refuse the incomplete migration copy.
Run 100-update user0 smoke fits at one minute and full, with validation only;
exclude these from profile selection. Check finite training, exact updates,
bounded access, saved/reloaded final weights and measured wall time.

Then run the declared matrix in `configs/m7-proposed-matrix.json`:

| Variable | Candidates |
|---|---|
| Updates | 1,000; 3,000 |
| Peak learning rate | 0.0003; 0.001 |
| Tuning users | user0; user1 |
| Tuning budgets | 5 minutes; full |
| Schedule | 10% update warmup, cosine decay to 0.000001 |
| Initialization | Verified generic checkpoint, fresh for every run |

Four profiles yield 16 runs and 32,000 updates, plus 200 smoke updates.
Select the lowest unweighted mean final-checkpoint validation CER over the
four user/budget pairs. Within 0.1 percentage points of that minimum, prefer
fewer updates, then lower learning rate. No test result enters selection.
Freeze one shared full-method profile with all four winning tuning receipts.
Preserve every candidate result, including failures.

Session A's proposed limit is **3 hours from Pod creation and $2.50 in Pod
charges**, including setup, smoke, tuning, evidence copying and deletion.
At 0.14–0.25 seconds per update, fitting is about 75–134 minutes; allow the
remaining time for setup/evaluation. Actual small-budget timings decide whether
the complete matrix fits. Stop rather than reduce steps, skip candidates,
freeze an incomplete matrix or retry.

## Session B: separately approved first curve

Only propose this session after A passes and the profile is frozen.
First evaluate frozen-profile full-data sentinels on user0 and user5.
Require absolute CER differences of at most **1.0 percentage point from our
M5b results**: 21.209850% and 6.130137%, respectively. These are different
training protocols; this is a proposed empirical compatibility gate, not a
guarantee. A miss stops the session before the remaining curve runs and
requires review. Do not retune from these test outcomes.

Run all seven budgets on users0–2 with the frozen profile. Reuse the user0
full sentinel as its curve point: 21 curve runs plus one user5 sentinel.
No per-user, per-budget or checkpoint selection is allowed. Preserve one result
and final checkpoint per run, plus allocation and protocol digests.

Use the verified M4 generic CERs as explicitly reused zero-calibration
references, not new measurements. Verify their checkpoint, user/split and result
provenance. Plot them as per-user horizontal references; zero is not on the
logarithmic minutes axis. Plot full at its actual training-sample minutes and
identify tuning users separately from untouched user2.

Proposed trend gate: finite complete results, negative CER-versus-budget
Spearman correlation for each user, and median 60-minute CER below median
one-minute CER. Report every inversion rather than requiring exact
monotonicity. A single seed is exploratory; no uncertainty or significance
claim is supported. Failure preserves the curve and stops progression to M8.

At the maximum candidate size this is 66,000 updates, about 2.6–4.6 hours
at the planning rates, plus evaluation/setup. The provisional limit is
**5.5 hours and $4.50 in Pod charges**, to be resized from Session A measurements
and separately approved. Output: CSV/table, per-user log-axis figure, frozen
protocol and public-safe evidence JSON.

## Costs, supervision and cleanup

Propose one Secure RTX 4090, existing volume, no archive transfer or parallel
Pods. Refresh balance, location-specific availability and the actual all-in
quote before creation. The planning GPU catalog rate is $0.74/hour.
Network storage is ongoing and separate from Pod limits. No new volume,
migration retry, data deletion or push is included.

A deadline-limited worker stops launching jobs when its remaining budget
cannot cover a run and reserved evidence/cleanup time. A scoped local watchdog
attempts Pod deletion on completion or deadline; verify zero Pods/endpoints
and retained volume read-back after copying and hashing evidence. Check only
readiness, phase completion, an error or a requested timer check.

The installed CLI has no verified automatic termination flag. A worker timeout
does not stop Pod billing; the local deletion guard depends on the controlling
machine and API/network availability. These are operational limits, not a
provider-enforced dollar cap. Keep the controller awake and arrange a deadline
check; failure to delete can accrue charges beyond the proposal. Do not start
unless this supervision risk is accepted.

