# Experiment records

Every number in the top-level README is traceable to a file here. These are the
raw CSVs written by the training and search scripts, unedited.

```
search/
  best_spec.json              winning architecture emitted by the search
  final_search_trials.csv     dense search at lambda=0.05, mu=0.02
                              42 candidates over 6 iterations of local moves
  rerank_top10_full.csv       top 10 candidates retrained at higher fidelity
                              and re-ranked -> row 1 is UNet-AutoBest
  lambda_mu_summary.csv       best spec and score for each of the 25 (lambda, mu)
                              cells of the tuning grid
ee491/
  results_unet_variants_final.csv   the five hand-designed anchors
  results_unet-autobest.csv         the searched architecture, fully trained
ee492/
  results_dsconv.csv                depthwise-separable version of the above
uncertainty/
  summary.csv                 QU-BraTS / ECE / Spearman over 74 validation cases
  per_case_uncertainty.csv    the same metrics, per case (74 rows)
splits/
  train_cases_seed42.txt      the 294 training case IDs
  val_cases_seed42.txt        the 74 validation case IDs
```

## The split is committed on purpose

`split_train_val()` is deterministic given seed 42, but only relative to the
case list that `glob` returns. A data copy with a different number of cases, or
a different sort order, silently produces a different split and therefore
numbers that cannot be compared with the ones here. The two files under
`splits/` remove that ambiguity: they are the exact cases behind every result
in this directory. `evaluate.py` reads the validation list directly:

```bash
BTSEG_VAL_TXT=../results/splits/val_cases_seed42.txt python evaluate.py
```

They contain nothing but public BraTS case identifiers.

## Column notes

- `best_val_mean_class_dice` — mean Dice over the three raw label classes
  (necrotic core, oedema, enhancing tumour). This is the training objective and
  the y-axis of the figures.
- `best_val_wt_dice`, `best_val_tc_dice`, `best_val_et_dice` — Dice over the
  nested BraTS regions. Their mean is the "mean region Dice" reported in the
  README and is the number comparable to published work.
- `gflops_fwd_128` — forward-pass GFLOPs at a 128^3 input with 4 channels,
  measured with fvcore.
- `elapsed_s` — wall-clock training time in seconds on the lab GPU.
- `best_epoch` vs `final_epoch` — early stopping keeps the best-validation
  checkpoint; the two differ whenever the run continued past its best epoch.

Dice values in `search/` are **proxy** scores, used only to rank candidates.
They come from 10 epochs on a fixed 30 % subset of the training cases (2 epochs
for the anchor placement, 6 for the lambda/mu sweep and for the rerank, which
runs on the full split). The subset, more than the epoch budget, is why they sit
at 0.30-0.55. They are not comparable to the values in `ee491/` and `ee492/`.

## A traceability gap worth knowing about

In `ee491/results_unet_variants_final.csv` the `model` column reads
`UNet-Baseline`, `UNet-Lite`, `UNet-UltraLite`, ... while the `ckpt` column
reads `brats2020_unet_baseline_...`, `..._lite_w05_...`,
`..._lite_w05_shallow_...`. Those two cannot come from a single run, because
the checkpoint name is derived from the model name. What happened is that the
anchor experiments were run first under short internal codenames and the
display names were applied afterwards, when the results were collected for the
report. The anchor table is also stitched together from more than one run:
`UNet-Efficient` and `UNet-Tiny` were added in a later batch than the first
three.

The consequence is narrow but real. The hyperparameters are not affected -
the scripts behind those runs and the committed `train_variants.py` were
compared and agree exactly on loss, optimiser, learning rate, weight decay,
scheduler, augmentation probabilities, crop sampling and split fraction, so
the anchors and `UNet-AutoBest` really were trained under the same pipeline.
What is affected is bookkeeping: re-running `train_variants.py` today writes
checkpoints named `brats2020_unet_UNet-Baseline_best_80_20.pth` and friends,
which will not match the `ckpt` strings recorded here. Those strings are kept
as the historical record of the original runs rather than rewritten.

The `ee492/` and `search/` records do not have this problem; their `ckpt`
values match what the committed code produces.

## Not included

The 25 per-cell trial dumps (`tune_lam*_mu*_trials.csv`) produced during the
grid sweep are omitted; `lambda_mu_summary.csv` carries the result of each cell,
which is what the analysis uses. Model checkpoints are omitted as well - see the
repository releases.
