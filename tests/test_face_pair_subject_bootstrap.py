"""Check subject-level paired uncertainty for the repair evaluation."""
import unittest

from tools.summarize_face_pair_subject_bootstrap import summarize


def row(subject, action, before, after):
    result = {"sample": f"S001C001P{subject:03}R001A{action:03}_rgb.npz", "label": "0"}
    for condition in ("clean", "point05", "point10"):
        result[f"{condition}_unmodified"] = str(before)
        result[f"{condition}_repaired"] = str(after)
    return result


class SubjectBootstrapTests(unittest.TestCase):
    def test_resamples_whole_subjects_and_preserves_pairing(self):
        rows = [row(1, 1, 1, 0), row(1, 2, 1, 0),
                row(2, 1, 0, 1), row(2, 2, 0, 1)]
        result = summarize(rows, replicates=10000, seed=9)
        condition = result["conditions"]["point05"]
        self.assertEqual(result["subjects"], 2)
        self.assertEqual((condition["gain"], condition["loss"]), (2, 2))
        self.assertEqual(condition["delta_percentage_points"], 0)
        self.assertEqual(condition["subject_bootstrap_95_percent_ci_pp"], [-100, 100])

    def test_rejects_duplicate_samples_and_missing_subject(self):
        sample = row(1, 1, 1, 0)
        with self.assertRaises(ValueError):
            summarize([sample, sample])
        with self.assertRaises(ValueError):
            summarize([{**sample, "sample": "missing-subject"}, row(2, 1, 0, 1)])


if __name__ == "__main__":
    unittest.main()
