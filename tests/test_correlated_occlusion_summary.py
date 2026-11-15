import numpy as np
import torch

from tools.evaluate_face_pair_correlated_occlusion import scores
from tools.summarize_correlated_occlusion import METHODS, summarize


def test_subject_paired_differences_and_eligibility():
    rows = []
    for index, (subject, eligible) in enumerate(((1, True), (1, False), (2, True), (2, True))):
        row = {"sample": f"S001C001P{subject:03}R{index + 1:03}A001", "label": 0, "eligible": str(eligible)}
        for condition in ("clean", "occlusion"):
            for method in METHODS:
                row[f"{condition}_{method}"] = 0
        rows.append(row)
    rows[0]["occlusion_original_arithmetic"] = 1
    rows[1]["occlusion_original_arithmetic"] = 1
    rows[2]["clean_max5_arithmetic"] = 1
    result = summarize(rows, repeats=1000, seed=5)
    all_rows = result["subsets"]["all"]
    entry = all_rows["conditions"]["occlusion"]["max5_arithmetic"]
    assert entry["corrected"] == 2 and entry["new_errors"] == 0
    assert entry["difference_pp"] == 50
    assert all_rows["occlusion_minus_clean"]["original_arithmetic"]["difference_pp"] == -50
    assert all_rows["occlusion_minus_clean"]["max5_arithmetic"]["difference_pp"] == 25
    assert all_rows["difference_in_differences"]["max5_arithmetic"]["difference_pp"] == 75
    assert np.isclose(result["subsets"]["eligible"]["conditions"]["occlusion"]["max5_arithmetic"]["difference_pp"], 100 / 3)
    assert result["eligibility_per_subject"][0]["eligible"] == 1


def test_endpoint_weights_and_probability_normalization():
    hand = torch.tensor([[0.3, -2.0, 1.4], [-0.2, 0.1, 0.0]])
    body = torch.tensor([[0.5, 0.0, -0.8], [0.9, -1.0, 0.3]])
    arithmetic, mixed = scores({"hand_face_logits": hand, "body_face_logits": body}, torch.tensor([0.0, 0.5, 1.0]))
    assert torch.isfinite(arithmetic).all() and torch.isfinite(mixed).all()
    assert torch.allclose(arithmetic.exp().sum(-1), torch.ones(2))
    assert torch.allclose(mixed.exp().sum(-1), torch.ones(2))
    hp, bp = hand.softmax(-1), body.softmax(-1)
    weights = torch.tensor([0.0, 0.5, 1.0])
    unnormalized = weights * hp + (1 - weights) * bp
    assert torch.allclose(arithmetic.exp(), unnormalized / unnormalized.sum(-1, keepdim=True))


if __name__ == "__main__":
    test_subject_paired_differences_and_eligibility()
    test_endpoint_weights_and_probability_normalization()
    print("2 tests passed")
