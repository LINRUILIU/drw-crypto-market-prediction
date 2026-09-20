from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drw_crypto.validation import purged_group_time_series_splits, rolling_time_window_indices


class ValidationTests(unittest.TestCase):
    def window(self, **kwargs):
        parameters = dict(n_rows=100, train_start=0.0, train_end=0.5, valid_start=0.5, valid_end=0.8)
        parameters.update(kwargs)
        return rolling_time_window_indices(**parameters)

    def test_forward_order_disjointness_and_embargo(self):
        train, valid = self.window(embargo_rows=3)
        np.testing.assert_array_equal(train, np.arange(47))
        np.testing.assert_array_equal(valid, np.arange(50, 80))
        self.assertLess(train.max(), valid.min())
        self.assertEqual(np.intersect1d(train, valid).size, 0)

    def test_existing_rolling_configuration_retains_its_indices(self):
        config = yaml.safe_load((ROOT / "configs/03_temporal_rolling_validation.yaml").read_text())
        for fold in config["folds"]["definitions"]:
            parameters = {key: fold[key] for key in ("train_start", "train_end", "valid_start", "valid_end")}
            for n_rows in (100, 101):
                with self.subTest(fold=fold["name"], n_rows=n_rows):
                    train, valid = rolling_time_window_indices(n_rows, **parameters)
                    np.testing.assert_array_equal(train, np.arange(int(n_rows * fold["train_start"]), int(n_rows * fold["train_end"])))
                    np.testing.assert_array_equal(valid, np.arange(int(n_rows * fold["valid_start"]), int(n_rows * fold["valid_end"])))

    def test_overlapping_future_nonfinite_and_out_of_range_windows_fail(self):
        invalid = (
            {"train_end": 0.6}, {"train_start": 0.7, "train_end": 0.9},
            {"train_start": -0.1}, {"valid_end": 1.1}, {"train_end": np.nan},
            {"valid_end": np.inf}, {"train_end": 0.0}, {"valid_end": 0.5},
        )
        for parameters in invalid:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                self.window(**parameters)

    def test_empty_windows_or_invalid_embargo_do_not_silently_proceed(self):
        for parameters in (
            {"embargo_rows": 50}, {"embargo_rows": 51}, {"embargo_rows": -1},
            {"embargo_rows": 1.5}, {"n_rows": 0}, {"n_rows": 2.5}, {"n_rows": "100"},
            {"n_rows": 1}, {"train_end": 0.001}, {"valid_end": 0.501},
        ):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                self.window(**parameters)

    def test_historical_purged_group_analysis_remains_two_sided(self):
        folds = purged_group_time_series_splits(60, n_groups=6, gap=1)
        np.testing.assert_array_equal(folds[0].valid_indices, np.arange(10))
        np.testing.assert_array_equal(folds[0].train_indices, np.arange(20, 60))
        for fold in folds:
            self.assertEqual(np.intersect1d(fold.train_indices, fold.valid_indices).size, 0)


if __name__ == "__main__":
    unittest.main()
