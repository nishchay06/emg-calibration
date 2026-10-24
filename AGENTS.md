# LEAN MODE: operating instructions (these override all earlier gating rules)

You are running a small personal research project. The research question,
the protocol decisions and the data/compute setup are **settled**. Your job is
to **produce measurements**, not to design processes. Every session must end
with new numbers in `results/` or a clear, specific reason why it couldn't.

Read this file completely at the start of every session. Follow it literally.
If an older file (ROADMAP.md, runbooks, proposals, RESEARCH_STATUS.md)
conflicts with this file, **this file wins**.

---

## 1. Start of session (at most 5 minutes)

1. Read this file.
2. Read **only the last section** of `RESEARCH_STATUS.md`.
3. Run `git log --oneline -5`.
4. List the result files that already exist for the current §5 experiment.
   Those runs are done; resume from the first missing one.
5. Write one sentence to the user: *"This session's deliverable is: …"*
6. Start working. **Do not** read runbooks, proposals or old results unless
   you need one specific command or number, and then read only that part.

## 2. Standing authorization: do these WITHOUT asking

- Create, use and delete Runpod Pods, up to **$5 and 6 hours per session**,
  while total project compute stays under **$75**. Check the balance **once**
  at the start.
- **GPU choice:** in data center `US-IL-1` (where the volume is), take the
  **first available** of: RTX A5000 → RTX 4090 → L40S → A40 → RTX A6000 →
  RTX 3090. Secure or Community cloud are both fine. Never wait or poll for a
  specific GPU. Record the GPU in the result JSON; results don't depend on it.
  If none is available in `US-IL-1`, tell the user and stop. Do not move data.
- Run any experiment in §5 end to end, including back-to-back sessions. There
  are **no separate session approvals**.
- Fix bugs in our own scripts. Re-run a failed run **once**.
- Commit and push to `nishchay06/emg-calibration` (§7).

## 3. Ask the user ONLY for these (otherwise never stop to ask)

1. Spending more than $5 in a session, or more than $75 in total.
2. Deleting or changing the volume `ni0dpvtday`, `/workspace/data`, the
   archive or any checkpoint.
3. Changing an approved research decision: anything in
   `configs/m7-frozen-protocol.json` or in a pre-registration under `docs/`
   (conditions, budgets, users, seeds, methods, analysis).
4. Anything public beyond `git push`: arXiv, emails, posts, forms.

## 4. Forbidden (do not do these, even if they seem helpful)

- Do **not** write proposals, approval records, receipts, "pre-provision"
  files, session runbooks or new milestone runbooks.
- Do **not** add guard scripts, dry-run modes, wrapper layers, watchdog
  redesigns or quote/capacity checkers. Use the existing scripts as they are.
- Do **not** write tests for one-off or diagnostic scripts. Add a test only if
  you change the sampler, the protocol or the metric code.
- Do **not** build analysis, plotting or reporting features before the data
  they need exists. One analysis script, written after the numbers exist, is
  enough.
- Do **not** re-verify what has already passed (M3–M7). Pre-flight is exactly:
  `/workspace/data` contains 100 `.hdf5` files, the generic checkpoint file
  exists, and CUDA is available. No inventory hashing, no SHA re-checks, no
  re-reading the M5b records.
- Do **not** edit the locked sections of a pre-registration. Only append to
  its *Deviations* section, with the reason.
- Do **not** add runs, seeds, users or budgets beyond the pre-registered grid,
  and never re-run a completed run to get a better number.
- Do **not** change anything in response to interim results. Inversions and
  surprises are reported, not fixed.
- Do **not** start side quests: storage migration, performance root-causing,
  price or capacity polling, repo restructuring, documentation cleanup.
- Do **not** run unattended "while waiting" work. If you are blocked, report
  and stop.
- Do **not** open PRs or branches for this solo project. Commit to `main`.

## 5. What to run, in order

### Done

- **M7 first curve (2026-10-24).** Frozen profile
  `configs/m7-frozen-protocol.json` (3,000 steps, peak lr 3e-4). Results in
  `results/m7-*`. Do not re-run.

### Session-spread experiment (current)

Specification: `docs/preregistration-session-spread.md`. Run exactly what it
says. The 62 runs take about 12 GPU-hours, so expect 2–3 sessions.

1. **Pre-flight** (§4 definition).
2. **Runs**, all with the frozen profile and `method=full`, in this order:
   1. Users 0, 1, 2 at 10 minutes, `contiguous`, seed 1502 (the 10-minute
      inversion check).
   2. All remaining runs for users 0–7 × budgets 5 and 10 × allocations
      `contiguous` and `spread` × seeds 1501 and 1502, user by user.
   3. Full budget, seed 1501, for users 3, 4, 6 and 7.

   Six runs already exist from M7 and count as `contiguous`, seed 1501. Don't
   re-run them: `results/m7-curve-user0-5.json`, `m7-curve-user0-10.json`,
   `m7-curve-user1-5-a5000.json`, `m7-curve-user1-10-4090.json`,
   `m7-curve-user2-5-4090.json`, `m7-curve-user2-10-4090.json`.

   Command per run:
   `python scripts/adapt.py --run --select fixed --protocol configs/m7-frozen-protocol.json --accelerator gpu --user userN --budget-minutes 5 --allocation spread --seed 1501 --upstream-dir <upstream> --data-dir /workspace/data --output-dir <fresh dir>`

   In each result, `calibration_selection.algorithm` must be
   `training-spread-v1` for `spread` and `training-contiguous-v1` for
   `contiguous`. If it isn't, stop: that is a bug (§6.4).
3. **Result files:** `results/ss-<user>-<budget>-<contiguous|spread>-s<seed>.json`
   with the §7 fields. Full-data runs: `results/ss-<user>-full-s1501.json`.
4. **Covariates (exploratory):** after the runs, on the Pod, write
   `results/ss-covariates.json` with, for each run, the mean days between its
   calibration sessions and the user's test sessions, and the keystroke count
   inside its calibration ranges.
5. **Analysis:** once all 62 runs exist, write one script,
   `scripts/analyze_session_spread.py`, that computes the pre-registered primary
   and secondary analyses (bootstrap seed 1501). Inputs:
   - generic CER: `results/m4-userN-generic-greedy.json`
   - full-data CER: `results/m7-curve-user0-full.json`,
     `m7-curve-user1-full-4090.json`, `m7-curve-user2-full-4090.json`,
     `m7-curve-user5-full-4090.json` and the new `ss-userN-full-s1501.json`

   Save `results/ss-summary.json` and one figure. State the primary result in
   exactly the words the pre-registration's interpretation rules give.

### Later (postponed until the session-spread experiment is written up)

Follow `ROADMAP.md` (M8 methods → M9 grid → M10 stretch → M11 LM → M12
write-up) under these same rules. Before M9, compute
`runs × measured minutes per run × GPU $/hour` and ask the user only if it
exceeds the $75 total.

## 6. Stop conditions: the ONLY reasons to stop early

1. The budget would be exceeded (§3.1).
2. Data or checkpoint missing or unreadable.
3. Loss is NaN/inf again after one re-run.
4. A bug in our code that you can't fix in 30 minutes.
5. Continuing would change an approved decision (§3.3).
6. No allowed GPU is available in `US-IL-1`.
7. The account balance can't cover this session's $5 cap plus 7 days of
   volume storage (about $12). Ask the user to top up; the volume is at risk
   if the balance runs out.

When you stop: say which condition, what you tried, and the one thing you need
from the user. Then clean up (§7).

## 7. End of session (always)

1. Copy result files off the Pod. **Delete the Pod.** Confirm zero running
   Pods. **Never delete the volume.**
2. Write **one result JSON per run** in `results/`: user, budget, method,
   allocation, seed, steps, learning rate, val CER, test CER, GPU, runtime
   minutes, cost estimate, and the run's `calibration_selection`. No other
   process files.
3. Append **one paragraph** (at most 8 lines) to `RESEARCH_STATUS.md`: what
   ran, key numbers, any inversions or surprises, what's next. Write it as a
   neutral lab-notebook entry for a public repository: no account balances, no
   prices except run cost estimates, no Pod or access details, and no "the user
   said/asked". Record decisions as decisions.
4. Commit as `Nishchay Mago <77321421+nishchay06@users.noreply.github.com>`
   (set it with `git config user.name` / `git config user.email` in the clone;
   never commit with any other email). Use a plain message, e.g.
   `Session spread: users 0–2 at 10 min, seed 1502`, and push to `main`.
5. Tell the user in at most 5 lines: deliverable done or not, key numbers,
   money spent this session, next step.

## 8. Self-check before any action

Ask yourself: **"Does this produce or record a measurement for the current
experiment today?"** If not, and it isn't in §1 or §7, don't do it.
