# fmt: off

"""Regression checks for PRESTO post-processing and threshold recovery."""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
RUNTIME_DIR   = BENCHMARK_DIR / "presto_runtime"
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(RUNTIME_DIR))

from finalize_existing_run import assert_summaries_equal  # noqa: E402
from run_search import detailed_tables                    # noqa: E402
from sweep_thresholds import compute_threshold            # noqa: E402


class ThresholdAssignmentTest(unittest.TestCase):
    def test_threshold_sweep_maximizes_number_of_matches(self) -> None:
        specs = {"dm_pc_cm3": {"bins": [90.0, 120.0]}}
        common = {
            "quantization": "raw8",
            "batch": 0,
            "output_file_stem": "segment",
            "time_reso_seconds": 0.001,
            "fluence_snr": 4.0,
            "_fluence_snr_bin": 0,
            "_bin_dm_pc_cm3": 0,
        }
        truth = [
            {
                **common,
                "injection_id": "a",
                "highest_freq_toa_file_raw_sample": 0,
                "dm_pc_cm3": 100.0,
            },
            {
                **common,
                "injection_id": "b",
                "highest_freq_toa_file_raw_sample": 10,
                "dm_pc_cm3": 110.0,
            },
        ]
        events = {
            ("raw8", 0, "segment"): [
                {
                    "event_id": "shared",
                    "quantization": "raw8",
                    "batch": 0,
                    "file_stem": "segment",
                    "sigma": 5.0,
                    "sigma_centi": 500,
                    "dm_pc_cm3": 101.0,
                    "sample": 1,
                    "dt_seconds": 0.001,
                },
                {
                    "event_id": "a_only",
                    "quantization": "raw8",
                    "batch": 0,
                    "file_stem": "segment",
                    "sigma": 5.0,
                    "sigma_centi": 500,
                    "dm_pc_cm3": 109.0,
                    "sample": -9,
                    "dt_seconds": 0.001,
                },
            ]
        }

        result = compute_threshold(
            3.0,
            truth,
            specs,
            events,
            Counter(),
            Counter(),
            source_dm_tol        = 10.0,
            source_time_tol_ms   = 10.0,
            localize_dm_tol      = 20.0,
            localize_time_tol_ms = 20.0,
        )
        raw8 = next(row for row in result["summary"] if row["quantization"] == "raw8")
        self.assertEqual(raw8["truth_count"], 2)
        self.assertEqual(raw8["detected_count"], 2)


class LinearAggregationTest(unittest.TestCase):
    def test_detailed_tables_count_matches_and_false_positives(self) -> None:
        specs = {"dm_pc_cm3": {"bins": [100.0, 200.0, 300.0]}}
        matches = [
            {
                "quantization": "raw8",
                "fluence_snr": 4.0,
                "dm_pc_cm3": 150.0,
                "detected": True,
            },
            {
                "quantization": "raw8",
                "fluence_snr": 4.0,
                "dm_pc_cm3": 150.0,
                "detected": False,
            },
        ]
        false_positives = [
            {
                "assigned_quantization": "raw8",
                "assigned_fluence_snr": 4.0,
                "assigned_dm_pc_cm3": 150.0,
            }
        ]
        tables, snr_rows = detailed_tables(matches, false_positives, specs)
        cell = next(
            row
            for row in tables["dm_pc_cm3"]
            if row["quantization"] == "raw8"
            and row["param_bin"] == 0
            and row["fluence_snr_bin"] == 0
        )
        self.assertEqual(cell["n_injected"], 2)
        self.assertEqual(cell["n_detected"], 1)
        self.assertEqual(cell["n_false_positive_assigned"], 1)
        snr = next(
            row
            for row in snr_rows
            if row["quantization"] == "raw8" and row["fluence_snr_low"] == 3
        )
        self.assertEqual(snr["n_injected"], 2)
        self.assertEqual(snr["n_detected"], 1)
        self.assertEqual(snr["n_false_positive_assigned"], 1)


class RecoveryValidationTest(unittest.TestCase):
    def test_recovery_rejects_count_mismatch(self) -> None:
        primary = [
            {
                "quantization": "raw8",
                "truth_count": 10,
                "detected_count": 8,
                "localized_count": 7,
                "false_positive_count": 3,
                "recall": 0.8,
                "localized_fraction": 0.7,
                "precision_proxy": 8 / 11,
            }
        ]
        reconstructed = [dict(primary[0], detected_count=7)]
        with self.assertRaisesRegex(RuntimeError, "detected_count mismatch"):
            assert_summaries_equal(primary, reconstructed)


if __name__ == "__main__":
    unittest.main()

# fmt: on
