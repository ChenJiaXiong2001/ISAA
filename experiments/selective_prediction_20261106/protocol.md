# Fixed-coverage selective prediction protocol

Use the existing frozen epoch-83 HF/BF calibration and validation logits and
manifest. Compute classwise gamma=1 weights from calibration cache labels.
Subsequent verification found two classes differ from checkpoint routing
counts. These original outputs are retained, and the checkpoint-weighted
recalculation is in `../disagreement_rejection_20261106/`.
Use the already fixed normalized 50/50 arithmetic/geometric fusion and its
maximum predicted probability as the confidence score. Choose two operating
points before accessing validation labels: the 5th and 10th percentiles of
confidence on the 3,191 calibration clips. Accept scores greater than or equal
to each fixed threshold. Evaluate every validation clip, reporting coverage,
accepted accuracy, rejected correct and incorrect clips, error capture, and
per-subject coverage and accuracy. Use a 20-subject bootstrap for aggregate
coverage and accepted accuracy. Quantiles reflect desired calibration coverage,
not a claimed error guarantee. No training, GPU, or model modification is used.

The calibration subjects also supplied the fusion weights; threshold estimation
is therefore not independent of that calibration. The official validation set
has been repeatedly explored; this is not a blind generalization result. The
previously cached logits can differ slightly from a fresh CUDA run. Conditional
accepted accuracy is not all-sample top-1 accuracy and rejection requires a
real fallback process.
