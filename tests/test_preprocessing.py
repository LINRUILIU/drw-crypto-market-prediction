from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drw_crypto.preprocessing import TabularPreprocessor, infer_id_column, infer_target_column


class PreprocessingTests(unittest.TestCase):
    def test_missing_infinite_and_constant_columns_produce_finite_float32(self):
        train = pd.DataFrame({
            "x": [1.0, 3.0, np.nan, np.inf, -np.inf],
            "empty": [np.nan] * 5,
            "constant": [7.0] * 5,
        })
        for fill in ("median", "mean", "zero"):
            with self.subTest(fill=fill):
                prep = TabularPreprocessor(list(train.columns)).fit(train, missing_fill=fill)
                result = prep.transform(train)
                self.assertEqual(result.dtype, np.float32)
                self.assertTrue(np.isfinite(result).all())
                np.testing.assert_array_equal(result[:, 1:], np.zeros((5, 2)))
                self.assertEqual(prep.scales["constant"], 1.0)

    def test_validation_uses_training_statistics_without_refitting(self):
        train = pd.DataFrame({"x": [1.0, 3.0]})
        prep = TabularPreprocessor(["x"]).fit(train)
        stats = [series.copy() for series in (prep.fill_values, prep.means, prep.scales)]
        valid = pd.DataFrame({"x": [100.0, np.nan]})
        np.testing.assert_allclose(prep.transform(valid).ravel(), [98.0, 0.0])
        for before, after in zip(stats, (prep.fill_values, prep.means, prep.scales)):
            pd.testing.assert_series_equal(before, after)
        np.testing.assert_array_equal(train["x"], [1.0, 3.0])

    def test_feature_order_and_nonstandardized_fill(self):
        prep = TabularPreprocessor(["b", "a"]).fit(
            pd.DataFrame({"a": [1.0, 3.0], "b": [4.0, 6.0]}), standardize=False,
        )
        result = prep.transform(pd.DataFrame({"a": [np.nan], "b": [10.0]}))
        np.testing.assert_array_equal(result, [[10.0, 2.0]])
        self.assertEqual(prep.transform(pd.DataFrame({"b": [], "a": []})).shape, (0, 2))

    def test_unfitted_empty_duplicate_and_missing_features_fail(self):
        train = pd.DataFrame({"x": [1.0, 3.0]})
        with self.assertRaises(RuntimeError):
            TabularPreprocessor(["x"]).transform(train)
        with self.assertRaisesRegex(ValueError, "empty training"):
            TabularPreprocessor(["x"]).fit(train.iloc[:0])
        for columns in ([], ["x", "x"]):
            with self.subTest(columns=columns), self.assertRaises(ValueError):
                TabularPreprocessor(columns).fit(train)
        with self.assertRaisesRegex(ValueError, "duplicate column"):
            TabularPreprocessor(["x"]).fit(train[["x", "x"]])
        prep = TabularPreprocessor(["x"]).fit(train)
        with self.assertRaises(KeyError):
            prep.transform(pd.DataFrame({"other": [1.0]}))
        with self.assertRaisesRegex(ValueError, "Unsupported missing_fill"):
            prep.fit(train, missing_fill="unknown")

    def test_float32_overflow_is_rejected_and_failed_fit_keeps_old_statistics(self):
        prep = TabularPreprocessor(["x"]).fit(pd.DataFrame({"x": [1.0, 3.0]}))
        with self.assertRaisesRegex(ValueError, "float32"):
            prep.transform(pd.DataFrame({"x": [1e300]}))
        with self.assertRaisesRegex(ValueError, "float32"):
            prep.fit(pd.DataFrame({"x": [1e300, 1e300]}), standardize=False)
        self.assertTrue(prep.standardize)
        np.testing.assert_array_equal(prep.transform(pd.DataFrame({"x": [2.0]})), [[0.0]])

    def test_target_and_test_only_identifier_configuration(self):
        train = pd.DataFrame({"x": [1.0], "label": [2.0]})
        test = pd.DataFrame({"x": [3.0], "ID": [42]})
        self.assertEqual(infer_id_column(train, test, None, "label", None), "ID")
        self.assertEqual(infer_id_column(train, test, None, "label", "ID"), "ID")
        self.assertEqual(infer_id_column(train, None, None, "label", "ID"), "ID")
        self.assertIsNone(infer_id_column(train, test.drop(columns="ID"), None, "label", None))
        with self.assertRaisesRegex(ValueError, "Configured target"):
            infer_target_column(train, test, "typo")
        for configured in ("typo", "label"):
            with self.subTest(configured=configured), self.assertRaisesRegex(ValueError, "Configured ID"):
                infer_id_column(train, test, None, "label", configured)


if __name__ == "__main__":
    unittest.main()
