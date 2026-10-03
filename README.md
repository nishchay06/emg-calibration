# Data-Efficient Personalization of sEMG Typing Decoders

This repository contains the research plan and working notes for studying how
much labeled calibration data a new user needs before a surface-EMG typing
decoder becomes useful.

The project is built around the public `emg2qwerty` benchmark. The first
milestone is to reproduce the benchmark's released generic and personalized
baselines. Only after that baseline is validated will we compare calibration
budgets and adaptation methods.

## Current state

- Research question and experiment plan: documented in [`project-plan.md`](project-plan.md)
- Working checklist and application timeline: [`GUIDE.md`](GUIDE.md)
- One-user baseline harness: implemented and validated; adaptation methods are not implemented
- Dataset and checkpoints: used ephemerally for M3; not retained locally
- Baseline reproduction: `user0` generic greedy baseline matched within 0.03
  percentage points; seven more generic users and the personalized sweep remain
- Local preflight, upstream audit, and GPU environment smoke test: complete and
  recorded in [`STATUS.md`](STATUS.md)
- One-user baseline preparation: [`M3_RUNBOOK.md`](M3_RUNBOOK.md)
- Captured one-user metrics: [`results/m3-user0-generic-greedy.json`](results/m3-user0-generic-greedy.json)

## Small-milestone sequence

1. **M0 — Preflight:** confirm storage, Python, GPU, and dependency constraints.
2. **M1 — Upstream audit:** inspect the official `emg2qwerty` code, configs, checkpoints, and evaluation command.
3. **M2 — Environment:** create a reproducible environment without downloading the full dataset.
4. **M3 — Baseline:** evaluate the released generic checkpoint on one test user.
5. **M4 — Baseline sweep:** reproduce the eight-user generic and personalized numbers.
6. **M5 — First experiment:** run one calibration budget and one adaptation method on a small user subset.

Each milestone should leave behind a command, an artifact, or a recorded result
that another person can inspect.

## Project documents

- [`project-plan.md`](project-plan.md) — research question, hypotheses, setup, and risks
- [`GUIDE.md`](GUIDE.md) — living checklist for the research project and applications
- [`research-statement.md`](research-statement.md) — older statement to retarget after results exist
