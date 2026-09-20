from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


class PipelineSmokeTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "runs").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="contract-smoke-", dir=ROOT / "runs")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.inputs = self.base / "inputs"
        self.inputs.mkdir()
        x = np.linspace(0, 3, 64)
        train = pd.DataFrame({
            "x": x, "wave": np.cos(x), "constant": 7.0,
            "empty": np.nan, "label": 1.5 * x - 0.2 * np.cos(x),
        })
        train.loc[4, "x"] = np.nan
        train.to_csv(self.inputs / "train.csv", index=False)
        pd.DataFrame({
            "ID": [501, 502, 503, 504, 505], "x": np.linspace(1, 2, 5),
            "wave": np.cos(np.linspace(1, 2, 5)), "constant": 7.0, "empty": np.nan,
        }).to_csv(self.inputs / "test.csv", index=False)
        pd.DataFrame({"ID": [501, 502, 503, 504, 505], "prediction": 0.0}).to_csv(
            self.inputs / "sample_submission.csv", index=False,
        )

    def run_script(self, script, config, name, *arguments):
        path = self.base / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--config", str(path), *arguments],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        )

    def baseline_config(self, name):
        config = yaml.safe_load((ROOT / "configs/01_main_baseline_full.yaml").read_text())
        config["data"]["raw_dir"] = self.inputs.relative_to(ROOT).as_posix()
        config["models"]["ridge"]["alphas"] = [1.0]
        config["ensemble"]["enabled"] = False
        config["output"] = {
            "run_dir": (self.base / name / "run").relative_to(ROOT).as_posix(),
            "submission_dir": (self.base / name / "submissions").relative_to(ROOT).as_posix(),
            "prediction_col": "prediction",
        }
        return config

    def test_ridge_cli_with_identifiers_and_id_free_benchmark_input(self):
        for mode in ("ids", "positional"):
            with self.subTest(mode=mode):
                if mode == "positional":
                    test = pd.read_csv(self.inputs / "test.csv").drop(columns="ID")
                    test.to_csv(self.inputs / "test.csv", index=False)
                config = self.baseline_config(mode)
                result = self.run_script("run_main_baseline.py", config, mode, "--only-models", "ridge")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                for filename in ("submission_ridge.csv", "submission_best.csv"):
                    submission = pd.read_csv(ROOT / config["output"]["submission_dir"] / filename)
                    np.testing.assert_array_equal(submission["ID"], np.arange(501, 506))
                    self.assertTrue(np.isfinite(submission["prediction"]).all())
                summary = json.loads((ROOT / config["output"]["run_dir"] / "data_summary.json").read_text())
                self.assertEqual(summary["id_col"], "ID" if mode == "ids" else None)
                self.assertEqual(summary["feature_count"], 4)

    def test_ridge_cli_rejects_misaligned_submission_before_export(self):
        template = pd.read_csv(self.inputs / "sample_submission.csv").iloc[::-1]
        template.to_csv(self.inputs / "sample_submission.csv", index=False)
        config = self.baseline_config("bad_ids")
        result = self.run_script("run_main_baseline.py", config, "bad_ids", "--only-models", "ridge")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("same order", result.stderr)
        self.assertEqual(list((ROOT / config["output"]["submission_dir"]).glob("*.csv")), [])

    def test_rolling_cli_rejects_overlap_and_empty_embargo_before_training(self):
        for mode in ("overlap", "empty_embargo"):
            with self.subTest(mode=mode):
                config = yaml.safe_load((ROOT / "configs/03_temporal_rolling_validation.yaml").read_text())
                config["data"]["raw_dir"] = self.inputs.relative_to(ROOT).as_posix()
                config["folds"] = {
                    "embargo_rows": 32 if mode == "empty_embargo" else 0,
                    "definitions": [{
                        "name": mode, "train_start": 0.0,
                        "train_end": 0.6 if mode == "overlap" else 0.5,
                        "valid_start": 0.5, "valid_end": 0.8,
                    }],
                }
                config["output"] = {
                    "run_dir": (self.base / mode / "run").relative_to(ROOT).as_posix(),
                    "figure_dir": (self.base / mode / "figures").relative_to(ROOT).as_posix(),
                }
                result = self.run_script("run_rolling_validation.py", config, mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Rolling windows", result.stderr)
                self.assertFalse((ROOT / config["output"]["run_dir"] / "metrics_rolling_validation.csv").exists())


if __name__ == "__main__":
    unittest.main()
