"""Regression tests for the generic PBS section submitter."""

from __future__ import annotations

import importlib.util
import io
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "s-pbsspt.py"
SPEC = importlib.util.spec_from_file_location("s_pbsspt", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {SCRIPT_PATH}")
PBS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PBS)


class NodeConfigTests(unittest.TestCase):
    def test_parse_node_config(self):
        self.assertEqual(PBS.parse_node_config("1:8, 2:4"), {1: 8, 2: 4})

    def test_parse_node_config_rejects_invalid_values(self):
        for value in ("", "1", "1:0", "0:1", "x:1", "1:x", "1:1,1:2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                PBS.parse_node_config(value)


class EntryScriptTests(unittest.TestCase):
    def test_project_entries_expose_one_literal_section_count(self):
        for name in ("d-center-binary-gate.py", "d-dm-time-predown.py"):
            with self.subTest(name=name):
                self.assertEqual(PBS.read_section_num(SCRIPT_PATH.parent / name), 32)

    def test_resolve_entry_script(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "gate.py"
            script.write_text("section_num = 1,\n", encoding="utf-8")

            resolved_root, resolved_script = PBS.resolve_entry_script(root, "gate.py")

            self.assertEqual(resolved_root, root.resolve())
            self.assertEqual(resolved_script, script.resolve())

    def test_resolve_entry_script_rejects_escape_and_missing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(ValueError):
                PBS.resolve_entry_script(root, "../outside.py")
            with self.assertRaises(FileNotFoundError):
                PBS.resolve_entry_script(root, "missing.py")

    def test_read_section_num(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            script = Path(temp_dir) / "gate.py"
            script.write_text(
                "config = ProcessConfig(\n    section_num             = 32,\n)\n",
                encoding="utf-8",
            )
            self.assertEqual(PBS.read_section_num(script), 32)

    def test_read_section_num_requires_one_positive_literal(self):
        cases = {
            "missing": "value = 32\n",
            "duplicate": "section_num = 1,\nsection_num = 2,\n",
            "zero": "section_num = 0,\n",
            "expression": "section_num = 4 * 8,\n",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            script = Path(temp_dir) / "gate.py"
            for name, content in cases.items():
                with self.subTest(name=name):
                    script.write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        PBS.read_section_num(script)


class AssignmentTests(unittest.TestCase):
    def test_assign_sections_evenly_across_gpu_jobs(self):
        self.assertEqual(
            PBS.assign_sections(8, {1: 2, 2: 1}),
            [
                (1, (0, 1, 2)),
                (1, (3, 4, 5)),
                (2, (6, 7)),
            ],
        )

    def test_assign_sections_uses_only_needed_gpu_slots(self):
        self.assertEqual(
            PBS.assign_sections(2, {1: 4}),
            [(1, (0,)), (1, (1,))],
        )

    def test_assign_sections_rejects_empty_or_nonpositive_inputs(self):
        with self.assertRaises(ValueError):
            PBS.assign_sections(0, {1: 1})
        with self.assertRaises(ValueError):
            PBS.assign_sections(1, {})

    def test_section_tag(self):
        self.assertEqual(PBS.section_tag((3,), 2), "03")
        self.assertEqual(PBS.section_tag((3, 4, 5), 2), "03-05")


class JobFileTests(unittest.TestCase):
    def test_render_pbs_script_contains_runtime_contract(self):
        content = PBS.render_pbs_script(
            root_path=Path("/tmp/runtime"),
            script_path=Path("/tmp/runtime/gate.py"),
            pbs_dir=Path("/tmp/runtime/pbsspt"),
            node=3,
            sections=(4, 5, 6),
            section_width=2,
            job_name="drafts",
            conda_env="pytorch",
            threads=8,
            workers_per_gpu=2,
        )

        self.assertIn("#PBS -l nodes=gpu03", content)
        self.assertIn("#PBS -W x=GRES:gpu@1", content)
        self.assertIn("sections=(4 5 6)", content)
        self.assertIn("workers_per_gpu=2", content)
        self.assertIn('"$PYTHON_BIN" -u "$SCRIPT_PATH" "$section"', content)
        self.assertIn("wait_batch", content)

    def test_create_job_files_distributes_all_sections(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "gate.py"
            script.write_text("section_num = 5,\n", encoding="utf-8")

            jobs = PBS.create_job_files(
                root_path=root,
                script_path=script,
                node_config={1: 2},
                section_num=5,
                job_name="drafts",
                conda_env="pytorch",
                threads=4,
                workers_per_gpu=1,
            )

            self.assertEqual(
                [(node, sections) for node, sections, _ in jobs],
                [(1, (0, 1, 2)), (1, (3, 4))],
            )
            self.assertTrue(all(path.is_file() for _, _, path in jobs))
            self.assertEqual(
                [path.name for _, _, path in jobs],
                [
                    "drafts-node-01-section-00-02.pbs",
                    "drafts-node-01-section-03-04.pbs",
                ],
            )

    def test_create_job_files_validates_resource_arguments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "gate.py"
            script.write_text("section_num = 1,\n", encoding="utf-8")
            base = {
                "root_path": root,
                "script_path": script,
                "node_config": {1: 1},
                "section_num": 1,
                "job_name": "drafts",
                "conda_env": "pytorch",
                "threads": 1,
                "workers_per_gpu": 1,
            }
            for field, value in (
                ("job_name", "bad name"),
                ("threads", 0),
                ("workers_per_gpu", 0),
            ):
                with self.subTest(field=field):
                    args = dict(base)
                    args[field] = value
                    with self.assertRaises(ValueError):
                        PBS.create_job_files(**args)


class SubmissionTests(unittest.TestCase):
    def test_submit_jobs_returns_failure_count(self):
        jobs = [
            (1, (0,), Path("first.pbs")),
            (1, (1,), Path("second.pbs")),
        ]
        completed = [
            subprocess.CompletedProcess(["qsub"], 0, stdout="123.server\n", stderr=""),
            subprocess.CompletedProcess(["qsub"], 1, stdout="", stderr="rejected\n"),
        ]
        with (
            mock.patch.object(PBS.subprocess, "run", side_effect=completed) as run,
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            failures = PBS.submit_jobs(jobs, submit_delay_s=0)

        self.assertEqual(failures, 1)
        self.assertEqual(run.call_count, 2)

    def test_submit_jobs_handles_missing_qsub(self):
        jobs = [(1, (0,), Path("first.pbs"))]
        with (
            mock.patch.object(PBS.subprocess, "run", side_effect=FileNotFoundError),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(PBS.submit_jobs(jobs, submit_delay_s=0), 1)

    def test_main_dry_run_generates_files_without_qsub(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "gate.py"
            script.write_text("section_num = 3,\n", encoding="utf-8")
            with (
                mock.patch.object(PBS, "submit_jobs") as submit,
                redirect_stdout(io.StringIO()),
            ):
                result = PBS.main(
                    [
                        "--script",
                        "gate.py",
                        "--nodes",
                        "1:2",
                        "--job-name",
                        "drafts",
                        "--root",
                        str(root),
                        "--dry-run",
                    ]
                )

            self.assertEqual(result, 0)
            submit.assert_not_called()
            self.assertEqual(len(list((root / "pbsspt").glob("*.pbs"))), 2)


if __name__ == "__main__":
    unittest.main()
