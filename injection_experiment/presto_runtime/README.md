# DRAFTS PRESTO blind single-pulse baseline

This directory contains the PRESTO baseline for the injection experiment.  The
formal recall/precision figures use a blind DM search, not truth-DM windows.
The searched FITS live in the experiment-level `simdata/` directory, and the
truth manifests live in the experiment-level `truth_archive/` directory.  DL
`runs/` directories are search outputs only.

## Files

| Path | Purpose |
| --- | --- |
| `run_search.py` | Run the full blind PRESTO search, cluster candidates into events, and write aggregate tables. |
| `export_threshold_data.py` | Export a compact local threshold-sweep package from the full event table. |
| `sweep_thresholds.py` | Recompute metrics and figures for sigma thresholds from the compact package. |
| `search_utils.py` | Shared parameter bins, CSV helpers, and the only PRESTO plotting implementation. |
| `../matching.py` | Sparse maximum-cardinality/minimum-cost truth-event matching shared with the DRAFTS analysis. |

## Requirements

Run this workflow in an environment where PRESTO is installed and the following
commands are available on `PATH`:

```text
rfifind
prepsubband
single_pulse_search.py
```

Install the DRAFTS Python dependencies from the repository root. The campaign
also needs a writable scratch directory with enough space for temporary
`.dat`, `.inf`, mask, and single-pulse products. Use `--scratch-root` or the
`PRESTO_SCRATCH_ROOT` environment variable to place those files on suitable
local storage.

## PRESTO command pattern

The workflow follows the conventional PRESTO single-pulse sequence:

```bash
rfifind left.fits center.fits right.fits -o prefix -time 1

prepsubband -nobary -numout 393216 -nsub 1024 \
  -lodm 100 -dmstep 1 -numdms 300 -downsamp 1 \
  -mask prefix_rfifind.mask -o prefix left.fits center.fits right.fits

ls *.dat | xargs -n 408 -P 12 python single_pulse_search.py -b -m 2 -t 3.0 -p
```

The full campaign repeats the `prepsubband` block from DM 100 to 2000 in
300-DM chunks, filters all-padding `.dat` files, and merges PRESTO
`.singlepulse` rows into source-level events.  All FITS files in each selected
batch/quantization directory are searched, so events from files without
injected truth are counted as false positives.

Event clustering enforces the configured DM/time tolerance across the full
event diameter, preventing single-link bridges. Truth and events are then
assigned one-to-one with maximum cardinality first and minimum normalized
distance second; input ordering cannot reduce an otherwise achievable match
count.

Each searched FITS is the center of a short context window.  The default
`--context-left-files 1 --context-right-files 1` gives PRESTO one neighboring
segment on both sides so high-DM de-dispersion is not truncated at the
6.44-second file boundary.  After de-dispersion the `.dat`/`.inf` products are
cropped back to the center FITS by default, so `single_pulse_search.py` runs on
the same 131072-sample interval that is scored.

By default the script keeps only event-level JSONL files.  Raw PRESTO
`.singlepulse` candidate rows can be very large at `-t 3.0`; use
`--keep-candidates` only for focused debugging.

## Full blind-search run

Run the full blind baseline directly through the Python entrypoint:

```bash
cd DRAFTS/injection_experiment/presto_runtime
source /path/to/miniforge3/etc/profile.d/conda.sh
conda activate presto_gpu

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-$PWD/results/presto_blind_full_$STAMP}"
mkdir -p logs

python run_search.py \
  --sim-root ../simdata \
  --truth-root ../truth_archive \
  --output-root "$OUT_ROOT" \
  --scratch-root "${SCRATCH_ROOT:-/path/to/presto_scratch}" \
  --mode blind \
  --batches 0-19 \
  --quantizations raw8,packed2 \
  --workers "${WORKERS:-8}" \
  --gpu-ids "${GPU_IDS:-0,1,2,3,4,5,6,7}" \
  --search-all-files \
  --dm-min 100 \
  --dm-max 2000 \
  --dm-step 1 \
  --dm-block-size 300 \
  --nsub 1024 \
  --retry-nsub 4096 \
  --numout 131072 \
  --context-left-files "${CONTEXT_LEFT_FILES:-1}" \
  --context-right-files "${CONTEXT_RIGHT_FILES:-1}" \
  --crop-dat-to-center \
  --downsamp 1 \
  --rfifind-time 1 \
  --xargs-chunk 408 \
  --xargs-procs "${XARGS_PROCS:-12}" \
  --maxwidth 2 \
  --threshold 3.0 \
  --source-dm-tolerance 60 \
  --source-time-tolerance-ms 30 \
  --event-dedup-dm-tolerance 60 \
  --event-dedup-time-tolerance-ms 30 \
  --overwrite 2>&1 | tee "logs/presto_blind_full_$STAMP.log"
```

Key outputs:

- `aggregate/summary.csv`
- `aggregate/all_matches.csv`
- `aggregate/all_false_positives.csv`
- `aggregate/all_events.csv`
- `events/<quantization>/bXX/*.jsonl`
- `analysis/cells_*.csv`
- `publication_figures/parameter_maps/*_recall_precision.png`
- `publication_figures/summary/snr_recall_precision.png`
- `run_summary.json`

## Export and sweep sigma thresholds

The full campaign does not create a threshold-sweep package automatically.
Export a compact package from a completed result root:

```bash
python export_threshold_data.py \
  --result-root "$OUT_ROOT" \
  --output-dir "$OUT_ROOT/threshold_sweep_package" \
  --source-dm-tolerance 60 \
  --source-time-tolerance-ms 30
```

The package contains slim truth rows, near-truth events, false-positive
histograms, and metadata. Recompute metrics and figures without rerunning
PRESTO:

```bash
python sweep_thresholds.py \
  --package-dir "$OUT_ROOT/threshold_sweep_package" \
  --output-dir "$OUT_ROOT/threshold_sweeps" \
  --thresholds 3,5,7 \
  --source-dm-tolerance 60 \
  --source-time-tolerance-ms 30 \
  --localize-dm-tolerance 25 \
  --localize-time-tolerance-ms 30
```

Use a new `--output-dir` when comparing threshold grids so one sweep does not
overwrite another.
