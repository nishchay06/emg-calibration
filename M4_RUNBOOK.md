# M4 runbook — multi-user baseline preparation

## Goal

Reproduce generic greedy-decoder baselines on a small held-out-user subset,
then expand to all eight users only after the subset matches upstream values.
No paid resource is provisioned by the commands in the planning section.

All inputs remain pinned to upstream commit
`3200d91eeb952cbed1f278e47d0cc56928334fd1`.

## Inventory

The eight official held-out-user configs require 100 unique sessions:

- 68 training sessions
- 16 validation sessions
- 16 test sessions

Counts and provenance are recorded in
[`manifests/test-users.json`](manifests/test-users.json). Regeneration and
validation are documented in [`manifests/README.md`](manifests/README.md).

## No-cost planning

Plan the first three-user subset without creating a directory or starting a
download:

```bash
./scripts/stage_test_users_data.sh \
  --dry-run /workspace/data user0 user1 user2
```

Expected plan:

```text
Selected users: user0,user1,user2
Required sessions: 40
Minimum free space: 23 GiB
Source archive: 308382645571 bytes
Dry run only; no directories were created and no data was downloaded.
```

Plan all eight users with the explicit list:

```bash
./scripts/stage_test_users_data.sh \
  --dry-run /workspace/data \
  user0 user1 user2 user3 user4 user5 user6 user7
```

That plan resolves 100 sessions and requires at least 48 GiB free space.

The remaining `user3` through `user7` subset resolves 60 sessions and requires
at least 33 GiB free space. The successful three-user run streamed at 29.96
MB/s, so a worst-case traversal of the full 308,382,645,571-byte archive would
take about 2 hours 52 minutes. The staging command therefore accepts a validated
`EMG_STAGE_TIMEOUT_SECONDS` override while retaining the original two-hour
default. Preview the four-hour guard without downloading data:

```bash
EMG_STAGE_TIMEOUT_SECONDS=14400 \
  ./scripts/stage_test_users_data.sh \
  --dry-run /workspace/data \
  user0 user1 user2 user3 user4 user5 user6 user7
```

## Paid staging command

Run this only after live compute and storage prices have been checked and a
hard spending ceiling has been agreed:

```bash
./scripts/stage_test_users_data.sh \
  --ack-stream-308gb /workspace/data user0 user1 user2
```

For the post-gate all-user staging run, use the explicit four-hour guard:

```bash
EMG_STAGE_TIMEOUT_SECONDS=14400 \
  ./scripts/stage_test_users_data.sh \
  --ack-stream-308gb /workspace/data \
  user0 user1 user2 user3 user4 user5 user6 user7
```

The script requires GNU tar and GNU timeout. It builds a unique manifest,
streams the public archive until every requested member is found, accepts curl
exit 23 only when tar completed successfully, validates every non-empty HDF5
file, and refuses to overwrite an existing destination.

When the complete upstream archive has already been downloaded to temporary
Pod storage, avoid a second network traversal with the local-archive mode:

```bash
./scripts/stage_test_users_data.sh \
  --archive-file /tmp/emg2qwerty-data-2021-08.tar.gz \
  /workspace/data \
  user0 user1 user2 user3 user4 user5 user6 user7
```

This mode requires the archive to match the pinned 308,382,645,571-byte size
before creating the destination. Install the pinned parallel decoder first:

```bash
python -m pip install -r requirements/m4-staging.txt
```

Local-archive mode uses `rapidgzip==0.16.0` with automatic parallelism and
explicit CRC32 verification. Unlike the network-stream path, it deliberately
does not ask GNU tar to stop after the first occurrence: the decoder and tar
scan the complete archive so truncation or a corrupt gzip trailer cannot be
hidden by an early successful member match. The destination is accepted only
after the decoder, tar, per-file, and exact-count checks all pass. The archive
must never be committed.

## Per-user evaluation

```bash
EMG_ACCELERATOR=gpu ./scripts/evaluate_generic_greedy.sh \
  user0 \
  /workspace/emg2qwerty \
  /workspace/data \
  /workspace/emg2qwerty/models/generic.ckpt \
  /workspace/results/user0-generic-greedy
```

Use a distinct output directory for each user. The evaluator verifies the
upstream commit, checkpoint digest, complete per-user manifest, and output-path
nonexistence before invoking the official test path.

After a run completes, convert its console output into a small structured
artifact while preserving the raw log:

```bash
./scripts/capture_generic_result.py \
  user0 \
  /workspace/results/user0-generic-greedy/console.log \
  /workspace/results/user0-generic-greedy/result.json
```

The capture command requires all five validation and test metrics, refuses to
overwrite an existing result, and calculates CER differences from the pinned
references in [`references/generic-greedy.json`](references/generic-greedy.json).
It records whether each absolute CER difference is at most **0.10 percentage
points**. A failed comparison is still written to `result.json` as evidence,
then the command exits nonzero.

## Three-user sweep

Preview the complete evaluation plan without checking inputs, creating output
directories, or starting an evaluation:

```bash
./scripts/evaluate_generic_sweep.sh \
  --dry-run \
  /workspace/emg2qwerty \
  /workspace/data \
  /workspace/emg2qwerty/models/generic.ckpt \
  /workspace/results/generic-greedy \
  user0 user1 user2
```

After staging and input verification, replace `--dry-run` with `--run`. The
sweep evaluates users sequentially, gives each user a separate output
directory, captures `result.json` immediately after each successful run, and
stops on the first execution, capture, or CER-acceptance failure. It performs a
full overwrite preflight before creating the output root.

The first-subset greedy references from upstream
`scripts/experimental_results.py` are:

| User | Validation CER | Test CER |
|---|---:|---:|
| `user0` | 60.07% | 61.48% |
| `user1` | 55.59% | 59.96% |
| `user2` | 47.38% | 48.00% |

## Acceptance gate

1. The three-user sweep dry-run resolves three distinct output directories
   without creating them.
2. `user0`, `user1`, and `user2` validation/test runs complete without training.
3. Each observed CER is captured in `result.json`; both validation and test
   must be within 0.10 percentage points of their pinned upstream per-user
   values. The sweep stops after recording the first failed comparison.
4. Raw console logs and Hydra configs are copied out before ephemeral compute is
   terminated.
5. The all-user sweep is not started until the three-user subset passes.
6. Every paid Pod and unused persistent volume is explicitly deleted and then
   verified absent.

## Observed three-user gate

The gate passed on 2026-10-03 using the pinned upstream commit and generic
checkpoint on a Secure Cloud RTX 4090 in `US-IL-1`.

| User | Validation CER | Reference | Test CER | Reference | Accepted |
|---|---:|---:|---:|---:|---|
| `user0` | 60.082565% | 60.07% | 61.509636% | 61.48% | yes |
| `user1` | 55.591190% | 55.59% | 59.945858% | 59.96% | yes |
| `user2` | 47.390659% | 47.38% | 48.010944% | 48.00% | yes |

Selective staging found all 40 sessions after streaming 16.9% of the source
archive; the extracted data occupied 13 GB. The successful Pod ran for 1,927
seconds at `$0.74/hour`, an estimated `$0.3961` in compute. The total observed
Runpod balance decrease for this milestone was `$0.4607`, including short
failed transfer attempts and temporary storage.

Raw logs and Hydra configurations were copied locally before cleanup. The
checked-in results are listed by
[`results/m4-three-user-gate-summary.json`](results/m4-three-user-gate-summary.json).
Post-cleanup read-back returned zero Pods and zero network volumes.

## Observed all-user staging attempt

The first post-gate all-user staging attempt ran on 2026-10-04 using a Secure
Cloud L4 in `EU-RO-1`. It deliberately stopped before evaluation because the
source-archive transfer did not meet the agreed cost gate:

| Measurement | Observed | Required |
|---|---:|---:|
| First rate window | 1,641,654 bytes/s | 28,000,000 bytes/s |
| Additional 61-second window | 1,668,363 bytes/s | 28,000,000 bytes/s |
| Archive progress at stop | 0.4% | n/a |
| Staging runtime | 11m42s | n/a |

The stream was intentionally terminated, so GNU tar exited 2 and
`/workspace/data.partial` remained unvalidated. This is transfer-failure
evidence, not a completed dataset. The Pod (`uevnc74dw12z7b`) was deleted after
copying raw logs to ignored `artifacts/m4-runpod-2026-10-04/`. The associated
60 GB Standard volume (`6jpri33smy`) was later deleted after the
evidence-preservation check.

Read-only investigation established that the original 308,382,645,571-byte
object is hosted in AWS `us-east-1` and accepts byte-range requests. A local
64 MiB range probe reached 7,190,777 bytes/s, while the previously successful
`US-IL-1` RTX 4090 stream reached 29.96 MB/s. Public alternatives found during
the audit do not preserve the required input path: the Hugging Face entry has
no dataset files, and NEMAR distributes a converted BDF/TSV representation
rather than the upstream HDF5 sessions.

That finding motivated the approved `US-IL-1` parallel-range download and
colocated-volume preservation attempt documented below.

## Observed US-IL-1 archive-preservation attempt

The approved follow-up used a Secure RTX 4090 in `US-IL-1`. Sixteen HTTP range
connections passed a corrected ten-minute gate at 59.8 MB/s and completed the
308,382,645,571-byte archive at an aria2-reported average of 51 MiB/s.

Staging then exposed a different bottleneck: `tar -xzf` launched one `gzip -d`
process at effectively one CPU core. After 1,680 seconds that process had read
25,225,068,544 compressed bytes, implying roughly five to six hours for a full
traversal. The staging job was stopped before evaluation; this run produced no
CER results.

The exact-size archive was copied to the 400 GB Standard volume
`ni0dpvtday` in `US-IL-1`, and the GPU Pod was deleted. Exact size and
successful range writes were confirmed, but the original source SHA-256 pass
was too slow and was stopped. Treat the persistent copy as pending integrity
validation until a complete gzip CRC32-verified traversal succeeds.

The preserved file is
`/workspace/archive/emg2qwerty-data-2021-08.tar.gz`. Before rerunning staging,
inspect and remove the unaccepted `/workspace/data.partial` directory and the
empty archive `.source.sha256` placeholder left by the stopped attempt. Do not
mistake either artifact for validated evidence. The guarded staging command is:

```bash
python -m pip install -r requirements/m4-staging.txt
./scripts/stage_test_users_data.sh \
  --archive-file /workspace/archive/emg2qwerty-data-2021-08.tar.gz \
  /workspace/data \
  user0 user1 user2 user3 user4 user5 user6 user7
```

Billing evidence from Runpod records `$2.3493767390` for the Pod and its
temporary disk. The retained volume had accrued `$0.2508333419` through the
subsequent audit and continues at `$0.0388888903/hour`.

The local-archive implementation now pins `rapidgzip==0.16.0` and streams
`rapidgzip --verify -P 0` into tar. It intentionally scans the complete stream
instead of using tar's early-exit `--occurrence=1` behavior. Synthetic tests
cover successful selected-member extraction, missing members, a truncated gzip
trailer, and invalid parallelism. These synthetic checks established the
preconditions for the guarded production run below.

## Observed all-user completion

The guarded production rerun completed on 2026-10-04 using Secure RTX 4090 Pod
`t2a5o7zxoqp098` in `US-IL-1`. It attached the retained 400 GB Standard volume,
used the pinned PyTorch image and a 20 GB ephemeral container disk, and armed an
absolute 2h50m self-termination guard before staging.

The ten-minute decoder gate measured 283,533,834,240 bytes consumed by tar in
601 seconds, or 471,770,106 bytes/second, against the required 50,000,000
bytes/second. The full archive traversal then completed with `rapidgzip=0` and
`tar=0`, which verifies the gzip CRC32 and selective tar extraction. Exactly
100 nonempty HDF5 files were accepted at `/workspace/data`, occupying 27 GB.

The preserved upstream checkout and checkpoint were reverified before use:

- upstream commit: `3200d91eeb952cbed1f278e47d0cc56928334fd1`
- checkpoint SHA-256:
  `338afa55f2ad5dd23abe3900e8047068bf8ee9893e75b54e1c6e6ab91c0d1a81`
- runtime smoke marker: `GREEDY_DECODER_OK` with PyTorch and Torchaudio
  `2.3.0+cu121`

The sequential `user3`-`user7` sweep passed every 0.10-point CER check:

| User | Validation CER | Reference | Test CER | Reference | Accepted |
|---|---:|---:|---:|---:|---|
| `user3` | 59.027016% | 59.03% | 54.689388% | 54.69% | yes |
| `user4` | 58.939510% | 58.93% | 58.236763% | 58.24% | yes |
| `user5` | 56.035351% | 56.01% | 53.847031% | 53.86% | yes |
| `user6` | 58.067543% | 58.08% | 54.661217% | 54.66% | yes |
| `user7` | 49.451645% | 49.45% | 52.170109% | 52.17% | yes |

The eight-user test CER aggregate is 55.383868% mean with 4.383906 sample
standard deviation, reproducing the upstream 55.38% ± 4.38 result.

The local evidence archive has SHA-256
`05449a3a6b6a1cf724066c185fdd2c836b748ebd9a14fd02d28488eedffc70a1`
and contains all five raw console logs, Hydra configuration triplets,
structured results, the environment freeze, and staging logs. The Pod was
deleted after local verification. CLI and Runpod MCP read-back found zero Pods
and zero endpoints. The reusable 400 GB volume `ni0dpvtday` remains in
`US-IL-1` at `$0.0388888903/hour`.

The observed balance decrease for this guarded run was `$0.3824294963`, from
`$11.3544687955` to `$10.9720392992`, below the `$2.25` ceiling. Itemized Pod
billing had not posted at the final audit; treat the balance delta as the
current cost evidence and update it only when a matching billing row appears.
