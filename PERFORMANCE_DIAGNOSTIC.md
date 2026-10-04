# M5b performance diagnostic

## Question and measurements

The interrupted user5 attempt had 77–95-second epochs and a 6.842× slower
synthetic CUDA update than user0. Its packages and training recipe matched.
Measure execution behavior before interpreting this as an accuracy failure.

`scripts/diagnose_m5b_performance.py` checks warmed synthetic batches (sizes 2
and 32), then short user0/user5 train and validation batches. It records loader
wait, transfers, forward/CTC, decode/metrics, backward/optimizer stages, CPU quota
and throttling deltas, thread settings, and GPU telemetry. It does not read test
sessions or save checkpoints. Stage profiling adds overhead; unprofiled timings
are reported separately. See the script's `--help` for bounded-run arguments.

The supervisor accepts a deadline no more than 660 seconds away, then terminates
and reaps its worker. This limits process execution; it does not terminate paid
infrastructure. The optional shell launcher uses a previously prepared workspace
and enforces separate setup/measurement cutoffs. Neither script provisions GPUs.

## Observed results

All six phases completed. Warmed synthetic batch-2/batch-32 medians were
15.69/74.25 ms; training computation was 81.54 ms/batch for user0 and 79.08 ms
for user5. CPU quota was 17.85 core-equivalents, with 128 visible CPUs and
PyTorch 64 intra-op/128 inter-op threads. No throttling increments occurred
within measured phases. User5 training loader wait was variable: median
29.05 ms and p90 318.95 ms.

The historical slowdown was not reproduced. Its cause remains unresolved;
no thread, driver, worker or training-recipe change was made. Ten-batch runtime
projections (23.63 minutes using medians, 42.52 using means) omit Trainer,
checkpoint, final evaluation and setup/copy overhead; they are scenarios, not
bounds or evidence that a 150-epoch run will finish in that time.

Full measurements, caveats and raw-archive digest are in
`results/m5b-performance-diagnostic-20261005.json`. Local finite-update and
supervisor checks are in `results/m5b-performance-local-preflight.json`.
