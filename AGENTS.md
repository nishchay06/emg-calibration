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
 4. Write one sentence to the user: *"This session's deliverable is: …"*
 5. Start working. **Do not** read runbooks, proposals or old results unless
    you need one specific command or number, and then read only that part.

 ## 2. Standing authorization: do these WITHOUT asking

 - Create, use and delete Runpod Pods, up to **$5 and 6 hours per session**,
   while total project compute stays under **$75**. Check the balance **once**
   at the start.
 - **GPU choice:** use RTX A5000 only in data center `US-IL-1` (where the
   volume is). Do not use RTX 4090 or any other GPU as a fallback. Never wait
   or poll for a specific GPU. Record the GPU used in the result JSON. If RTX
   A5000 is unavailable in `US-IL-1`, tell the user and stop. Do not move data.
 - Run any milestone from §5 end to end, including back-to-back sessions for
   the same milestone. There are **no separate session approvals** (no
   "Session A / Session B").
 - Fix bugs in our own scripts. Re-run a failed run **once**.
 - Commit and push to `nishchay06/emg-calibration`.

 ## 3. Ask the user ONLY for these (otherwise never stop to ask)

 1. Spending more than $5 in a session, or more than $75 in total.
 2. Deleting or changing the volume `ni0dpvtday`, `/workspace/data`, the
    archive or any checkpoint.
 3. Changing an approved research decision: the fixed-step protocol, the budgets
    (1, 2, 5, 10, 30, 60 minutes, full), tuning on user0/user1 only, the methods,
    or the seeds.
 4. Anything public beyond `git push`: arXiv, emails, posts, forms.

 ## 4. Forbidden (do not do these, even if they seem helpful)

 - Do **not** write proposals, approval records, receipts, "pre-provision"
   files, session runbooks or new milestone runbooks.
 - Do **not** add guard scripts, dry-run modes, wrapper layers, watchdog
   redesigns or quote/capacity checkers. Use the existing scripts as they are.
 - Do **not** write tests for one-off or diagnostic scripts. Add a test only if
   you change the sampler, the protocol or the metric code.
 - Do **not** build analysis, plotting or reporting features before the data
   they need exists. One plotting script, written after the numbers exist, is
   enough.
 - Do **not** re-verify what has already passed (M3–M6). Pre-flight is exactly:
   `/workspace/data` contains 100 `.hdf5` files, the generic checkpoint file
   exists, and CUDA is available. No inventory hashing, no SHA re-checks, no
   re-reading the M5b records.
 - Do **not** start side quests: storage migration, performance root-causing,
   price or capacity polling, repo restructuring, documentation cleanup.
 - Do **not** run unattended "while waiting" work. If you are blocked, report
   and stop.
 - Do **not** open PRs or branches for this solo project. Commit to `main`.

 ## 5. What to run, in order

 ### M7: first curve (done 2026-10-24)

 The tuning matrix in `configs/m7-proposed-matrix.json` is **approved as is**.
 Ignore its `"status": "proposed_not_approved"` field. Do not edit the matrix.

 1. **Pre-flight** (§4 definition, at most 5 minutes).
 2. **One smoke run:** `user0`, 1-minute budget, 100 updates, validation only,
    to confirm CUDA and the final-checkpoint save and reload. Exclude it from
    selection.
 3. **Tuning:** run the 16 tuning runs (4 candidates × user0/user1 × 5 min/full)
    with `scripts/run_m7_tuning.py`. Pick the profile with the lowest mean
    final-checkpoint **validation** CER, breaking ties as already implemented
    (fewer updates, then lower learning rate). Freeze it with
    `scripts/freeze_calibration_protocol.py`. Commit the frozen profile and the
    tuning results.
 4. **Curve:** with the frozen profile, run `method=full`, seed 1501, all 7
    budgets, for users 0, 1 and 2 (21 runs). Evaluate the **final checkpoint**
    on test. No per-run selection.
 5. **Full-data comparison (REPORT ONLY, NOT A GATE):** also run user5 at the
    full budget with the frozen profile. Record the difference from M5b (user0
    21.21%, user5 6.13%). **Do not stop** if it differs by more than 1 pp. The
    frozen profile uses at most 3,000 updates, while upstream full-data
    fine-tuning used 14,550, so the full-data point is *expected* to be worse.
    Note that in one sentence and continue.
 6. **Plot:** CER vs minutes (log x-axis), one line per user, with the M4
    generic CER as a horizontal reference per user. Save the PNG and CSV under
    `results/`. Use the existing `scripts/calibration_report.py` /
    `plot_calibration_report.py` if they work in one try; otherwise write one
    simple matplotlib script.
 7. **Sanity check (REPORT ONLY):** does CER generally fall as minutes
    increase? List any inversions. Don't stop.

 If the session limit is reached mid-way, finish the current run, record what
 is done, delete the Pod, and continue in the next session. **Never shrink the
 protocol** (steps, budgets, candidates) to make it fit.

 ### Session-spread experiment (current)

 Run exactly what `docs/preregistration-session-spread.md` specifies. Its
 conditions, budgets, seeds, users and analysis are approved research decisions
 (§3.3): do not change them. Record any unavoidable deviation in that file's
 *Deviations* section.

 1. **Pre-flight** (§4 definition).
 2. **Runs:** 62 runs with the frozen profile, `method=full`, in this order:
    1. Users 0, 1, 2 at 10 minutes, `contiguous`, seed 1502 (the 10-minute inversion check).
    2. All remaining runs for users 0–7 × budgets 5 and 10 × allocations
       `contiguous` and `spread` × seeds 1501 and 1502, user by user. Skip the
       6 runs M7 already has (users 0–2, `contiguous`, seed 1501).
    3. Full budget, seed 1501, for users 3, 4, 6 and 7.

    Command per run:
    `python scripts/adapt.py --run --select fixed --protocol configs/m7-frozen-protocol.json --accelerator gpu --user userN --budget-minutes 5 --allocation spread --seed 1501 --upstream-dir <upstream> --data-dir /workspace/data --output-dir <fresh dir>`
 3. **Result files:** `results/ss-<user>-<budget>-<contiguous|spread>-s<seed>.json`,
    with the §7 fields plus `allocation`, and the run's `calibration_selection`.
    Full-data runs: `results/ss-<user>-full-s1501.json`.
 4. **Covariates (exploratory):** after the runs, on the Pod, write
    `results/ss-covariates.json` with, per run, the mean days between its
    calibration sessions and the user's test sessions, and the keystroke count
    inside its calibration ranges.
 5. **Analysis:** once all 62 runs exist, write one script,
    `scripts/analyze_session_spread.py`, that computes the pre-registered primary
    and secondary analyses. Save `results/ss-summary.json` and one figure. Report
    the primary result exactly as the pre-registration's interpretation rules say.

 ### M8 onward (postponed until the session-spread experiment is written up)

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

 When you stop: say which condition, what you tried, and the one thing you need
 from the user. Then clean up (§7).

 ## 7. End of session (always)

 1. Copy result files off the Pod. **Delete the Pod.** Confirm zero running
    Pods. **Never delete the volume.**
 2. Write **one result JSON per run** in `results/`: user, budget, method, seed,
    steps, learning rate, val CER, test CER, GPU, runtime minutes, cost
    estimate. No other process files.
 3. Append **one paragraph** (at most 8 lines) to `RESEARCH_STATUS.md`: what
    ran, key numbers, any inversions or surprises, what's next.
 4. Commit with a plain message (e.g. `M7: tuning done, profile frozen
    (s1000, lr1e-3)`) and push to `main`.
 5. Tell the user in at most 5 lines: deliverable done or not, key numbers,
    money spent this session, next step.

 ## 8. Self-check before any action

 Ask yourself: **"Does this produce or record a measurement for the current
 milestone today?"** If not, and it isn't in §1 or §7, don't do it.
