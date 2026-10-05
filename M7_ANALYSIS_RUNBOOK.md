# M7 offline analysis

Generate the first full-fine-tuning curve after separately approved Session B
produces 21 results (users0–2 × seven budgets) and the user5 full-data sentinel.
These tools do not provision, train, tune, or approve either paid session.

## Collect measured receipts

Use the frozen protocol and session index emitted by Session A. Supply exactly
22 completed adaptation `result.json` files, three accepted M4 generic baselines,
and both own M5b reproductions. Choose a fresh output directory:

```bash
python scripts/calibration_report.py \
  --upstream-dir upstream/emg2qwerty \
  --protocol artifacts/m7-tuning/frozen-protocol.json \
  --session-index artifacts/m7-tuning/session-index.json \
  --results artifacts/m7-curve/run-*/result.json \
  --baselines results/m4-user{0,1,2}-generic-greedy.json \
  --m5b-references results/m5b-user{0,5}-full-upstream.json \
  --output-dir artifacts/m7-analysis/measured
```

Paths above are examples; keep original receipts intact. The collector checks
all four winning tuning contexts, the shared frozen profile, exact updates,
successful GPU training, final-checkpoint reload hashes, finite training,
zero validation during fitting, and complete coverage. It replays training-only
allocations against all 100 indexed sessions and the pinned official splits.
It rejects duplicate, incomplete, mixed-protocol or test-informed inputs.

Outputs: `summary.json`, `curve.csv`, and `table.md`. Exit **0** means analysis
gates pass; **1** retains a valid complete report whose scientific gate failed;
**2** refuses invalid inputs. No existing output directory is overwritten.

## Plot and interpret

Install `requirements/analysis.txt` in a separate analysis environment; leave
the pinned training environment unchanged. Matplotlib uses a headless backend.

```bash
python scripts/plot_calibration_report.py \
  --report artifacts/m7-analysis/measured/summary.json \
  --output-dir artifacts/m7-analysis/measured/figures
```

Exports PNG, PDF and SVG. Each user has a log-x panel; full-data positions use
selected samples / 120,000 samples per minute. Zero minutes reuses the accepted
M4 generic CER as a horizontal reference because zero cannot appear on log-x.
Users0/1 are tuning participants; user2 is the sole untouched curve participant.
The user5 full sentinel is reported separately in the table's gate section.

The declared checks are user0/user5 full CER within 1.0 pp of own M5b, negative
Spearman correlation for each curve user, and median 60-minute CER below median
1-minute CER. Adjacent inversions remain visible. Full and 60 minutes may coincide
if their recording durations match. Single-seed results remain exploratory;
do not claim uncertainty estimates or significance. Receipt/digest checks do
not reread signals/checkpoints or remeasure CER. Milestone acceptance also needs
session evidence, costs and cleanup; analysis alone does not pass M7.

## Exercise the pipeline offline

```bash
python scripts/calibration_report_demo.py \
  --upstream-dir upstream/emg2qwerty \
  --output-dir artifacts/m7-analysis/demo
python scripts/plot_calibration_report.py \
  --report artifacts/m7-analysis/demo/report/summary.json \
  --output-dir artifacts/m7-analysis/demo/figures
python -m unittest discover -s tests -p test_calibration_report.py -v
python tests/check_calibration_plot.py -v
```

**All demo lengths, checkpoint receipts, timings and CER values are fictional.**
No participant recordings are read. Demo inputs require explicit synthetic mode;
ordinary collection rejects them. CSV rows, Markdown, JSON and figures identify
the synthetic provenance. Keep demo artifacts ignored; never cite them as
research results. On restricted machines set `MPLCONFIGDIR` and `XDG_CACHE_HOME`
to writable temporary directories before plotting.
