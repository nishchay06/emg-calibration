# Repository Guidelines

## Project Structure & Module Organization

`scripts/` contains staging, evaluation, adaptation and diagnostic tools.
`tests/` holds Python unit tests and shell integration checks; `tests/fixtures/`
contains small console fixtures. `requirements/` pins dependencies, `manifests/`
identifies dataset sessions, and `references/` stores upstream benchmark values.
Publish small result records in `results/`; keep data, checkpoints and raw logs
in ignored `data/` or `artifacts/` directories.

## Build, Test, and Development Commands

There is no application build. Use Python 3.10 for training and Linux with CUDA
for GPU runs. Install lightweight validation dependencies with
`python -m pip install -r requirements/m4-staging.txt -r requirements/m5b-config.txt`.

- `bash -n scripts/*.sh`: validate shell syntax.
- `python -m unittest discover -s tests -v`: run unit tests.
- `./tests/test_cli_guards.sh`: check command guards and mutation-free previews.
- `./tests/test_parallel_staging.sh`: verify archive integrity and extraction.
- `python scripts/adapt.py --dry-run --user user0 --upstream-dir upstream/emg2qwerty --data-dir data --output-dir artifacts/m5b-preview`: preview reproduction.

Configuration tests additionally need the pinned upstream checkout at
`upstream/emg2qwerty`; see `.github/workflows/validate.yml` for complete checks.
M6 CPU integration tests need the full training dependencies; lightweight CI
alone skips those tests. See `M6_RUNBOOK.md` for calibration checks and commands.

## Coding Style & Naming Conventions

Use four-space Python indentation, descriptive `snake_case` names, and
`test_*.py` test modules. Shell entry points use Bash with strict error handling.
Match surrounding code; no formatter or linter is currently configured.

## Testing Guidelines

Use `unittest` and the existing shell checks. Test scientific invariants and
failure paths: split isolation, deterministic selection, artifact integrity,
acceptance tolerances and process cleanup. No numerical coverage threshold is
configured. Keep dataset and GPU requirements out of ordinary unit tests.

## Commit & Pull Request Guidelines

History uses imperative descriptions such as “Add guarded adaptation harness”
and “Reproduce user5 full-data fine-tuning.” Keep commits focused on one
verifiable step. Describe the research question or behavioral change, link
issues when applicable, report validation and limitations, and include the
result record for numerical claims. Screenshots are optional for figures.

## Research Evidence

Pin upstream versions and checkpoint hashes. Preserve unsuccessful attempts.
Distinguish measured CER from projections and settled costs from estimates.
Keep credentials, account balances and session access details in ignored local
records. Follow upstream data/model licensing; never commit participant data.
