# Anchor dropout clean annealing

Hypothesis: retaining p=0.025 raw-anchor corruption early builds robustness,
then increasing the fraction of clean samples may reduce the clean accuracy
cost observed for the fixed p025/clean50 candidate. This is not established.
Epochs 1-10 use clean probability 0.5; epochs 11-20 increase it by 0.05 per
epoch, reaching 1.0. Corruption probability on a selected sample stays 0.025.
The shared epoch counter is read inside each persistent data-loader worker.

Predeclared screen: scratch initialization, forced torch.compile/Inductor,
20 epochs, 4096 samples, batch 32, warmup 1, same members and hyperparameters
as existing clean and p025/clean50 controls. Compare epoch-20 `last.pt` with
both controls, using the same input tensors and each checkpoint's frozen
training-only gamma=1 calibration. Evaluate full validation clean, anchor-pre
5%/10%, point 5%/10%, group 5%/10%, and subject-cluster paired bootstrap.

Promotion criterion, fixed before results: clean minus clean control must be at
least -0.25 percentage points and its subject-cluster 95% interval must not
exclude zero; at both anchor-pre severities the candidate must improve by at
least 5 percentage points over clean control, with positive interval lower
bounds. Also report the direct difference against fixed p025/clean50 and all
other conditions, including clean-relative robustness. These are screening
rules, not claims of non-inferiority or independent generalization.

CPU checks: `python experiments/anchor_clean_anneal_20261106/test_cpu.py`.
GPU work is queued until existing evaluations finish. Before training, run the
entry with the same arguments plus `--dry-run` to verify actual CUDA Inductor
forward/backward, without turning off compile or loading weights. Reuse the
original model's documented face LocalSTBlock eager boundary; the added
curriculum operates on data preprocessing outside the model.

The entry imports the original hand-motion preset before validating the fixed
protocol. Server command, after the GPU is free (add `--dry-run` for the
compiled forward/backward/optimizer smoke test):

```bash
python -u experiments/anchor_clean_anneal_20261106/train.py \
  --archive /home/xiong/Desktop/ISAA/data/ntu60_skeletons_rtmw.zip \
  --device cuda --compile --compile-mode default --epochs 20 --warmup-epochs 1 \
  --num-workers 8 --seed 17 --max-samples 4096 --log-interval 20 --pair-calibration-start-epoch 1 \
  --precompression-anchor-dropout-probability 0.025 \
  --precompression-anchor-clean-sample-probability 0.5 \
  --save-dir outputs/anchor_clean_anneal_20ep4096_20261106
```

`evaluate.py --condition-set standard` and `--condition-set anchor-pre` wrap
the established paired evaluators. Invoke each for both existing clean and
fixed p025 controls with `--control`, `--candidate`, `--archive`, and unique
`--output-dir`; they require epoch-20 `last.pt` and matching full manifests.
Run the paired evaluators' existing subject-cluster summarizers on each output.
