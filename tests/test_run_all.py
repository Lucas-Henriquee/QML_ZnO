"""Coordination tests using mocks only; never execute scientific stages."""

import contextlib
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.workflows import run_all


def arguments():
    return ["--run-id", "organization_plan_check", "--source-results", "results",
            "--dataset-config", "configs/datasets/zno_existing_results.yaml",
            "--model-config", "configs/models/zno_local.yaml",
            "--analysis-config", "configs/analysis/zno_current.yaml",
            "--model-dataset", "multitech/md_window_unique_dataset.csv"]


class CombinedLauncherTests(unittest.TestCase):
    def test_plan_preserves_configs_and_routes_new_data(self):
        args = run_all.parse_args(arguments())
        paths = [run_all.resolve(getattr(args, name + "_config")) for name in ["dataset", "model", "analysis"]]
        before = [p.read_bytes() for p in paths]
        commands, _ = run_all.build_plan(args)
        self.assertEqual([name for name, _ in commands], ["science", "datasets", "models", "model_analysis"])
        self.assertIn("local", commands[2][1])
        self.assertIn(str(run_all.ROOT / "data/generated/organization_plan_check/multitech/md_window_unique_dataset.csv"), commands[2][1])
        self.assertFalse(any("--submit-ibm" in command for _, command in commands))
        self.assertEqual(before, [p.read_bytes() for p in paths])

    def test_default_does_not_execute(self):
        with patch("sys.argv", ["run_all", *arguments()]), patch.object(run_all, "execute_plan") as execute:
            with contextlib.redirect_stdout(io.StringIO()):
                run_all.main()
            execute.assert_not_called()

    def test_simulation_configuration_is_explicit(self):
        args = run_all.parse_args(arguments())
        args.source_results = None
        args.simulate = True
        with self.assertRaisesRegex(ValueError, "requires --simulation-config"):
            run_all.build_plan(args)
        args.simulation_config = Path("configs/simulations/zno_production.yaml")
        commands, _ = run_all.build_plan(args)
        self.assertEqual(commands[0][0], "simulations")
        self.assertNotIn("--resume", commands[0][1])

    def test_model_dataset_cannot_escape_run(self):
        args = run_all.parse_args(arguments())
        args.model_dataset = Path("../unrelated.csv")
        with self.assertRaises(ValueError):
            run_all.build_plan(args)

    def test_nonzero_stage_stops_sequence(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(run_all, "ROOT", Path(directory)):
            with patch.object(run_all.subprocess, "run", return_value=subprocess.CompletedProcess(["first"], 1)) as process:
                with self.assertRaises(subprocess.CalledProcessError):
                    run_all.execute_plan([("first", ["first"]), ("second", ["second"])], {}, "test")
                self.assertEqual(process.call_count, 1)
            with self.assertRaises(FileExistsError):
                run_all.execute_plan([], {}, "test")


if __name__ == "__main__":
    unittest.main()
