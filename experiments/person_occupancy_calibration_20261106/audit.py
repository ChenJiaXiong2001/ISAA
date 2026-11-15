"""Independently verify grouped counts and hashes from the saved sample table."""
import csv
import hashlib
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
with (root / "samples.csv").open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
names = [r["sample"] for r in rows]
assert len(set(names)) == len(names) == summary["samples"]
assert hashlib.sha256("\n".join(names).encode()).hexdigest() == summary["members_sha256"]
assert summary["clean_input_sha256"] == "617e94432a3d5d7d31e67af15d5d38e899f2f600037e1058687671493670ebb2"
assert summary["frame_mask_sha256"] == "23e6ad369a16d2271903c732e0c04ac15cce182eb08d69f3418871714ea4fae2"
for r in rows:
    match = re.search(r"P(\d{3})R\d{3}A(\d{3})_rgb\.npz$", r["sample"], re.I)
    assert match and int(match[1]) == int(r["subject"]) and int(match[2]) == int(r["action"])
    valid, visible = int(r["valid_frames"]), int(r["slot1_visible_frames"])
    assert valid > 0 and 0 <= visible <= valid
    fraction = visible / valid
    assert abs(float(r["slot1_fraction"]) - fraction) < 1e-12
    assert r["occupancy_group"] == ("zero" if visible == 0 else "below_half" if 2 * visible < valid else "half_or_more")
    assert int(r["correct"]) == int(int(r["prediction"]) == int(r["action"]))


def check(subset, recorded):
    assert len(subset) == recorded["samples"]
    assert sum(int(r["correct"]) for r in subset) == recorded["correct"]


for group, recorded in summary["groups"].items():
    check([r for r in rows if r["occupancy_group"] == group], recorded)
for subject, groups in summary["subjects"].items():
    for group, recorded in groups.items():
        check([r for r in rows if int(r["subject"]) == int(subject[1:]) and r["occupancy_group"] == group], recorded)
for action, groups in summary["actions"].items():
    for group, recorded in groups.items():
        check([r for r in rows if int(r["action"]) == int(action[1:]) and r["occupancy_group"] == group], recorded)
assert len([r for r in rows if r["occupancy_group"] == "half_or_more" and int(r["action"]) >= 50]) == 567
print("Verified 3191 rows, ordered SHA, clean hashes, all subject/action/group counts and the 567 interaction-action samples.")
