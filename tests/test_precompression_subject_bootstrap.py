import unittest

from tools.summarize_precompression_subject_bootstrap import summarize


class SubjectBootstrapTests(unittest.TestCase):
    def test_paired_changes_and_subject_aggregation(self):
        rows = []
        for index, (subject, old, new) in enumerate(((1, 1, 0), (1, 0, 0), (2, 0, 1), (2, 1, 1)), 1):
            rows.append({"sample": f"S001C001P{subject:03}R001A{index:03}_rgb.npz", "label": "0",
                         "clean_unmodified_prediction": "0", "clean_max5_prediction": "0",
                         "pre_dropout_unmodified_prediction": str(old), "pre_dropout_max5_prediction": str(new)})
        result = summarize(rows, repeats=200, seed=42)
        self.assertEqual(result, summarize(rows, repeats=200, seed=42))
        self.assertEqual(result["subjects"], 2)
        self.assertEqual(result["conditions"]["clean"]["difference_pp"], 0)
        self.assertEqual(result["conditions"]["pre_dropout"]["corrected"], 1)
        self.assertEqual(result["conditions"]["pre_dropout"]["new_errors"], 1)
        self.assertEqual(result["conditions"]["pre_dropout"]["difference_pp"], 0)
        with self.assertRaises(ValueError):
            summarize(rows + [rows[0]], repeats=10, seed=42)


if __name__ == "__main__":
    unittest.main()
