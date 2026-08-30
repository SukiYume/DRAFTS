#!/usr/bin/env python3
# fmt: off

"""Finalize a completed PRESTO run from its aggregate threshold package.

Use this after the expensive search and aggregate CSV export have completed but
the in-memory detailed-analysis stage was interrupted.  The script never
reruns PRESTO and never rewrites the large aggregate event/false-positive CSVs.
It validates the sigma-threshold reconstruction against the authoritative
primary summary, installs the reconstructed tables/figures, and writes the
missing ``run_summary.json``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import shutil
from collections import Counter
from pathlib import Path

COUNT_FIELDS = (
    "truth_count",
    "detected_count",
    "localized_count",
    "false_positive_count",
)
RATE_FIELDS = ("recall", "localized_fraction", "precision_proxy")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def typed_summary(rows: list[dict[str, str]]) -> list[dict]:
    typed: list[dict] = []
    for row in rows:
        out: dict[str, str | int | float] = {"quantization": row["quantization"]}
        for field in COUNT_FIELDS:
            out[field] = int(row[field])
        for field in RATE_FIELDS:
            out[field] = float(row[field])
        typed.append(out)
    return typed


def assert_summaries_equal(primary: list[dict], reconstructed: list[dict]) -> None:
    primary_by_quant       = {row["quantization"]: row for row in primary}
    reconstructed_by_quant = {row["quantization"]: row for row in reconstructed}
    if primary_by_quant.keys() != reconstructed_by_quant.keys():
        raise RuntimeError(
            "Primary and reconstructed summaries have different quantizations"
        )
    for quant, expected in primary_by_quant.items():
        actual = reconstructed_by_quant[quant]
        for field in COUNT_FIELDS:
            if int(actual[field]) != int(expected[field]):
                raise RuntimeError(
                    f"{quant} {field} mismatch: "
                    f"primary={expected[field]} reconstructed={actual[field]}"
                )
        for field in RATE_FIELDS:
            if not math.isclose(
                float(actual[field]),
                float(expected[field]),
                rel_tol = 0.0,
                abs_tol = 1e-12,
            ):
                raise RuntimeError(
                    f"{quant} {field} mismatch: "
                    f"primary={expected[field]} reconstructed={actual[field]}"
                )


def load_statuses(result_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted((result_root / "status").glob("*/*/*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid status JSON: {path}") from exc
    return rows


def threshold_slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--threshold-sweep-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=3.0)
    args = parser.parse_args()

    result_root = args.result_root.resolve()
    sweep_root  = args.threshold_sweep_dir.resolve()
    package_dir = (
        args.package_dir.resolve()
        if args.package_dir is not None
        else (result_root / "threshold_sweep_package").resolve()
    )
    reconstructed_root = sweep_root / f"sigma_ge_{threshold_slug(args.threshold)}"

    primary_summary = typed_summary(read_csv(result_root / "aggregate" / "summary.csv"))
    reconstructed_summary = typed_summary(
        read_csv(reconstructed_root / "aggregate" / "summary.csv")
    )
    assert_summaries_equal(primary_summary, reconstructed_summary)

    metadata      = json.loads((package_dir / "metadata.json").read_text(encoding="utf-8"))
    statuses      = load_statuses(result_root)
    status_counts = Counter(str(row.get("status", "unknown")) for row in statuses)

    truth_count = sum(int(row["truth_count"]) for row in primary_summary)
    false_positive_count = sum(
        int(row["false_positive_count"]) for row in primary_summary
    )
    event_count        = int(metadata["events_total"])
    status_event_count = sum(int(row.get("event_count", 0) or 0) for row in statuses)
    if int(metadata["truth_count"]) != truth_count:
        raise RuntimeError(
            "Threshold package truth count does not match primary summary"
        )
    if int(metadata["false_positive_events"]) != false_positive_count:
        raise RuntimeError(
            "Threshold package false-positive count does not match primary summary"
        )
    if status_event_count != event_count:
        raise RuntimeError(
            "Threshold package event count does not match summed job statuses: "
            f"package={event_count} statuses={status_event_count}"
        )

    source_analysis = reconstructed_root / "analysis"
    source_figures  = reconstructed_root / "figures"
    if not source_analysis.is_dir() or not source_figures.is_dir():
        raise RuntimeError(
            f"Reconstructed analysis is incomplete under {reconstructed_root}"
        )
    shutil.copytree(
        source_analysis,
        result_root / "analysis",
        dirs_exist_ok=True,
    )
    shutil.copytree(
        source_figures,
        result_root / "publication_figures",
        dirs_exist_ok=True,
    )

    config = json.loads(
        (result_root / "campaign_config.json").read_text(encoding="utf-8")
    )
    run_summary = {
        "output_root": str(result_root),
        "sim_root": config.get("sim_root"),
        "truth_root": config.get("truth_root"),
        "candidate_count": int(
            sum(int(row.get("candidate_count", 0) or 0) for row in statuses)
        ),
        "event_count": event_count,
        "job_count": len(statuses),
        "failed_jobs": int(status_counts.get("failed", 0)),
        "matches": truth_count,
        "false_positives": false_positive_count,
        "summary": primary_summary,
        "mode": config.get("mode"),
        "dm_window": config.get("dm_window"),
        "dm_step": config.get("dm_step"),
        "threshold": config.get("threshold"),
        "maxwidth": config.get("maxwidth"),
        "context_left_files": config.get("context_left_files"),
        "context_right_files": config.get("context_right_files"),
        "crop_dat_to_center": config.get("crop_dat_to_center"),
        "center_numout": config.get("numout"),
        "source_dm_tolerance": config.get("source_dm_tolerance"),
        "source_time_tolerance_ms": config.get("source_time_tolerance_ms"),
        "status_counts": dict(sorted(status_counts.items())),
        "recovered_from_existing_aggregate": True,
        "recovery_completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "threshold_package": str(package_dir),
        "threshold_sweep_dir": str(sweep_root),
        "reconstruction_threshold": args.threshold,
        "reconstruction_matches_primary_summary": True,
        "large_aggregate_csvs_rewritten": False,
    }
    destination = result_root / "run_summary.json"
    temporary   = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(run_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)

    validation = {
        "result_root": str(result_root),
        "primary_summary": primary_summary,
        "reconstructed_summary": reconstructed_summary,
        "event_count": event_count,
        "job_count": len(statuses),
        "status_counts": dict(sorted(status_counts.items())),
        "analysis_source": str(reconstructed_root),
        "run_summary": str(destination),
    }
    (result_root / "analysis" / "recovery_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

# fmt: on
