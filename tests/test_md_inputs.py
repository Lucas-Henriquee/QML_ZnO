"""Check current and archived MD filenames without changing feature calculations."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.common.md_inputs import find_md_summaries
from scripts.datasets.prepare_multitech_dataset import load_md_summary_features, load_md_windows
from scripts.workflows.run_datasets import collect_inputs


class MDInputTests(unittest.TestCase):
    def test_current_and_archived_names_produce_identical_descriptors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "md/ZnO/top_metal"
            directory.mkdir(parents=True)
            current = directory / "md_summary.csv"
            pd.DataFrame({"time_ps": [0, 1, 2, 3],
                          "c_surface_distance_ang": [2.0, 2.2, 2.4, 2.6],
                          "oco_angle_deg": [170, 171, 172, 173]}).to_csv(current, index=False)
            summary = load_md_summary_features(root)
            windows = load_md_windows(root, 2)
            config = {"input": {}, "methods": {"multitech": {"targets": ["md_windows"]}}}
            self.assertEqual(collect_inputs("multitech", config, root, None), [current])
            archived = directory / "md_summary_original.csv"
            current.rename(archived)
            pd.testing.assert_frame_equal(load_md_summary_features(root), summary)
            pd.testing.assert_frame_equal(load_md_windows(root, 2), windows)
            self.assertEqual(collect_inputs("multitech", config, root, None), [archived])

    def test_multiple_trajectories_require_explicit_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "md/ZnO/bridge"
            directory.mkdir(parents=True)
            (directory / "md_summary.csv").touch()
            (directory / "md_summary_old.csv").touch()
            with self.assertRaisesRegex(ValueError, "Multiple MD summaries"):
                find_md_summaries(root)

    def test_missing_trajectory_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                find_md_summaries(Path(tmp))
