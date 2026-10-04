# Agent handoff

Read this first in every new session. It is the standing context for this
project, so nothing important depends on chat history.

## Goal

Measure how much labeled calibration data a new user needs before a
surface-EMG typing decoder becomes useful: **CER vs minutes of calibration
data** on emg2qwerty, comparing full fine-tuning, last-layer, normalization-only
and LoRA adaptation. Deliverables: a per-user data-efficiency figure, a results
table, this reproducible repo, and a 4–6 page report by **Dec 1, 2026** (hard
deadline: graduate applications on Dec 15).

## Sources of truth

| File | Use it for |
|---|---|
| `ROADMAP.md` | Milestones M5–M12, acceptance gates, timeline, budget, approved decisions |
| `STATUS.md` | What has actually been done, with evidence; its last "Next acceptance test" section is the current next step |
| `project-plan.md` | Research questions, hypotheses, related work |
| `results/*.json` | Structured evidence for every reproduced number |

**Current next step:** the first milestone in `ROADMAP.md` whose gate is not
yet recorded as passed in `STATUS.md`. As of 2026-10-04, M5a has passed for
both released checkpoint families and all eight users. M5b local harness and
configuration checks are complete; next verify the Python 3.10 training runtime
with a no-cost CPU smoke test (see `M5B_RUNBOOK.md`), then prepare the exact paid
proposal for full-data/full-method upstream reproduction. Do not begin calibration-budget experiments
until M5b reproduces `user0` and `user5` test CER within 1.0 pp.

## Approved decisions (2026-10-04). Do not re-open them without the user.

1. Budgeted runs use a **fixed number of optimizer steps** and **no per-user
   validation checkpoint selection** (upstream selection leaks labeled data).
2. Hyperparameters are tuned on `user0`/`user1` only, then frozen.
3. Budgets: 1, 2, 5, 10, 30, 60 minutes and full.
4. Compute cap: **$75 in total**, re-checked after M5b and M7.
5. M5b (own fine-tuning reproduces upstream within 1.0 pp) must pass before
   any budget experiment.

## Working rules

- **Personal resources only.** Personal Runpod and GitHub accounts; never
  employer hardware, accounts or cloud.
- **Gate before spend.** Dry-run first; one milestone per Pod session; stop at
  the first failed gate and diagnose.
- **Clean up.** Delete every Pod after copying evidence; verify with read-back
  (zero Pods, zero endpoints). Keep the 400 GB volume `ni0dpvtday`
  (`US-IL-1`, data at `/workspace/data`, archive at `/workspace/archive/`) until
  the project ends. Check the Runpod balance before each run (~$0.93/day
  storage).
- **Prefer new files** over large edits to existing ones.
- Pin upstream to `3200d91eeb952cbed1f278e47d0cc56928334fd1`. Never commit
  dataset files or checkpoints (CC BY-NC-SA 4.0 upstream terms).

## Session ritual

**Start:** read this file, then `ROADMAP.md`, then the end of `STATUS.md`. Run
`git log --oneline -10`. State the current milestone and its gate before doing
anything.

**End (or before context runs low):** record what passed or failed in
`STATUS.md` (numbers, cost, cleanup read-back), add result JSON under
`results/`, update the "Current next step" line above, then commit and push.
The next session must be able to continue from the repo alone.
