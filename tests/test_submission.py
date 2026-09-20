from __future__ import annotations

import runpy
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drw_crypto.pipeline import make_submission


class SubmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test = pd.DataFrame({"ID": [10, 20], "x": [1.0, 2.0]})
        self.template = pd.DataFrame({"ID": [10, 20], "prediction": [0.0, 0.0]})
        self.pred = np.array([0.1, 0.2])

    def submit(self, *, template=None, test=None, pred=None, id_col=None):
        return make_submission(
            self.template if template is None else template,
            self.test if test is None else test,
            self.pred if pred is None else pred,
            "prediction",
            id_col,
        )

    def test_valid_ids_and_metadata_are_preserved_without_mutating_inputs(self):
        self.template["note"] = ["a", "b"]
        before = self.template.copy(deep=True)
        result = self.submit()
        np.testing.assert_array_equal(result["prediction"], self.pred)
        pd.testing.assert_frame_equal(result.drop(columns="prediction"), before.drop(columns="prediction"))
        pd.testing.assert_frame_equal(self.template, before)

    def test_reordered_or_different_ids_are_rejected(self):
        for ids in ([20, 10], [10, 99]):
            for id_col in (None, "ID"):
                with self.subTest(ids=ids, id_col=id_col), self.assertRaisesRegex(ValueError, "same order"):
                    self.submit(template=self.template.assign(ID=ids), id_col=id_col)

    def test_duplicate_and_missing_ids_are_rejected_on_both_sides(self):
        for ids in ([10, 10], [10, None]):
            for side in ("test", "template"):
                frame = getattr(self, side).assign(ID=ids)
                with self.subTest(side=side, ids=ids), self.assertRaisesRegex(ValueError, "unique and non-missing"):
                    self.submit(**{side: frame})

    def test_prediction_shape_count_and_values_are_checked(self):
        invalid = (
            np.array([[0.1], [0.2]]), np.array([0.1]), np.array(0.1),
            np.array([np.nan, 0.2]), np.array([np.inf, 0.2]), np.array([-np.inf, 0.2]),
            np.array(["0.1", "0.2"]), np.array([0.1 + 1j, 0.2]),
        )
        for pred in invalid:
            with self.subTest(pred=repr(pred)), self.assertRaises(ValueError):
                self.submit(pred=pred)
        with self.assertRaisesRegex(ValueError, "test row count"):
            self.submit(test=self.test.iloc[:1])
        with self.assertRaisesRegex(ValueError, "sample submission length"):
            self.submit(template=self.template.iloc[:1])

    def test_prediction_and_id_column_conflicts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "not in sample"):
            self.submit(template=self.template.drop(columns="prediction"))
        for id_col in (None, "ID"):
            with self.subTest(id_col=id_col), self.assertRaisesRegex(ValueError, "overwrite"):
                make_submission(self.template, self.test, self.pred, "ID", id_col)
        with self.assertRaisesRegex(ValueError, "generated row ID"):
            make_submission(None, self.test.drop(columns="ID"), self.pred, "row_id", None)

    def test_duplicate_columns_and_ambiguous_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate column"):
            self.submit(template=self.template[["ID", "prediction", "prediction"]])
        with self.assertRaisesRegex(ValueError, "duplicate column"):
            self.submit(test=self.test[["ID", "ID", "x"]])
        with self.assertRaisesRegex(ValueError, "Multiple ID columns"):
            self.submit(test=self.test.assign(row_id=[10, 20]))
        with self.assertRaisesRegex(ValueError, "not found"):
            self.submit(id_col="missing")

    def test_custom_identifier_is_checked(self):
        template = self.template.rename(columns={"ID": "sample_key"})
        test = self.test.rename(columns={"ID": "sample_key"})
        result = self.submit(template=template, test=test)
        np.testing.assert_array_equal(result["sample_key"], [10, 20])
        with self.assertRaisesRegex(ValueError, "same order"):
            self.submit(template=template.iloc[::-1], test=test, id_col="sample_key")

    def test_id_free_benchmark_uses_documented_positional_alignment(self):
        result = self.submit(test=self.test.drop(columns="ID"))
        np.testing.assert_array_equal(result["ID"], [10, 20])
        np.testing.assert_array_equal(result["prediction"], self.pred)
        only_prediction = self.submit(template=self.template[["prediction"]])
        self.assertEqual(list(only_prediction.columns), ["prediction"])

    def test_without_template_preserves_available_ids_or_generates_row_ids(self):
        result = make_submission(None, self.test, self.pred, "prediction", None)
        np.testing.assert_array_equal(result["ID"], [10, 20])
        result = make_submission(None, self.test.drop(columns="ID"), self.pred, "prediction", None)
        np.testing.assert_array_equal(result["row_id"], [0, 1])
        np.testing.assert_array_equal(result["prediction"], self.pred)

    def test_historical_entrypoints_reject_misaligned_outputs(self):
        names = (
            "run_main_baseline", "run_pearson_topk", "run_beta1_signal_blend",
            "run_beta2_alpha_variants", "run_beta2_intermediate_ridge",
            "run_beta2_ridge_topk", "run_beta3_new_feature_signals", "run_stability_blend",
        )
        for name in names:
            with self.subTest(entrypoint=name):
                helper = runpy.run_path(str(ROOT / "scripts" / f"{name}.py"))["make_submission"]
                with self.assertRaisesRegex(ValueError, "same order"):
                    helper(self.template.iloc[::-1], self.test, self.pred, "prediction", None)


if __name__ == "__main__":
    unittest.main()
