# M7 Session A: approved tuning execution

Session A was approved on 2026-10-05: mounted-data/CUDA preflight, 200 unscored
smoke updates and the complete 16-run user0/user1 validation grid. Its original
review packet is `M7_PROPOSAL.md` at commit `5daffcd`. Candidate/matrix files
retain that draft snapshot; approval is operational authorization, not a tuned
or frozen research profile. Session B remains separately gated.

## Before creation

Run the repository tests with the complete pinned Python 3.10 environment.
The seven M7 selection tests reject incomplete grids, test-informed selection,
different allocations and invalid training/checkpoint proofs. Refresh balance,
retained-source identity and the exact region/Secure-GPU availability quote.
Do not substitute a GPU, region or incomplete migration copy. The personal
controller and deletion guard remain under ignored operational artifacts.

The approved session limit is three hours from the creation request and $2.50
in Pod charges. Reserve the last 15 minutes for copying and deletion. Worker
timeouts do not enforce cloud billing; the independently launched local guard
is a best-effort backup, dependent on the controller's machine and network.

## Worker contract

`scripts/run_m7_tuning.py` requires fresh output and an explicit work deadline.
It verifies all 100 session checksums/bytes, the generic checkpoint and upstream
pin, CUDA/package pins and all 56 allocations from actual recording lengths.
Then it runs 100-update user0 smoke fits at one minute and full, using validation
only. These receipts never enter candidate selection. The fitter records elapsed
fit time without changing optimization or data loading.

Run with the approved mounted paths and deadline supplied by the controller:

```bash
python scripts/run_m7_tuning.py \
  --upstream-dir /workspace/pinned-upstream \
  --data-dir /workspace/data --checkpoint /workspace/models/generic.ckpt \
  --source-inventory /workspace/inputs/source-inventory.json \
  --output-dir artifacts/m7-session-a \
  --work-deadline-epoch <approved-work-deadline-in-unix-seconds>
```

Before every candidate run, project the entire remaining declared matrix using
measured fit rates and startup/evaluation overhead. Stop at a failed gate or
projected overrun, without reducing steps, skipping candidates or resuming.
All candidate runs use fresh generic initialization, seed 1501 and final
checkpoints; no test evaluation or validation during fitting occurs.

## Completion and evidence

Require all 16 receipts with matched allocations, profile/initialization pins,
finite training, exact update counts and saved/reloaded checkpoint identity.
Select the declared unweighted validation mean, with the reviewed tie rule.
Freeze the winner using all four winning user/budget receipts; retain every
candidate's result and checkpoint. Emit `session-final.json`, the score table
and frozen protocol. A partial matrix cannot freeze.

Copy and verify the evidence archive, then delete only this session's Pod and
read back zero Pods/endpoints with retained-source identity. If copying cannot
finish before the deadline, preserve evidence on the retained volume and delete
the Pod anyway. Record failures as well as success. A completed tuning session
does not establish M7's curve/compatibility gate or authorize Session B.
