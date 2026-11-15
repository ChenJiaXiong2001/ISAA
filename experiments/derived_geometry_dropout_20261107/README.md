
## 4-epoch matched screen

The mandatory CUDA smoke passed with two finite Inductor SGD updates. A matched
seed-17, 4096-sample, 4-epoch scratch control/candidate screen then produced:

| condition | control | candidate | difference |
|---|---:|---:|---:|
| clean | 36.1646% | 37.6181% | +1.4535pp |
| point 5% | 32.0618% | 33.1718% | +1.1100pp |
| point 10% | 25.8923% | 26.6282% | +0.7359pp |
| group 5% | 35.1343% | 37.0722% | +1.9379pp |
| group 10% | 33.5214% | 36.0603% | +2.5389pp |

This is a short screen, not a comparison with the 95.28% formal checkpoint. The
candidate remains eligible only for an independent short recheck; no long run
has started.
