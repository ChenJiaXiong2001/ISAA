"""Describe paired prediction differences by subject and action."""
import collections
import csv
import json
import re
from pathlib import Path

rows = list(csv.DictReader(Path(__file__).with_name("predictions.csv").open(encoding="utf-8")))
conditions = ("clean", "anchor_pre_05", "anchor_pre_10", "point_05", "point_10", "group_05", "group_10")
result = {}
for condition in conditions:
    result[condition] = {}
    for name, marker in (("subject", "P"), ("action", "A")):
        groups = collections.defaultdict(lambda: [0, 0])
        for row in rows:
            group = re.search(marker + r"(\d{3})", row["sample"]).group(1)
            groups[group][0] += int(row[condition + "_p10"] == row["label"]) - int(row[condition + "_baseline"] == row["label"])
            groups[group][1] += 1
        values = sorted((100 * change / count, group, change, count) for group, (change, count) in groups.items())
        result[condition][name] = {"positive": sum(v[0] > 0 for v in values), "zero": sum(v[0] == 0 for v in values), "negative": sum(v[0] < 0 for v in values), "bottom": values[:3], "top": values[-3:], "equal_weight_mean_pp": sum(v[0] for v in values) / len(values)}
print(json.dumps(result, ensure_ascii=False, indent=2))
