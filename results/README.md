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
```

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

Dice values in `search/` are **proxy** scores from a handful of epochs, used
only to rank candidates. They are far below converged Dice and are not
comparable to the values in `ee491/` and `ee492/`.

## Not included

The 25 per-cell trial dumps (`tune_lam*_mu*_trials.csv`) produced during the
grid sweep are omitted; `lambda_mu_summary.csv` carries the result of each cell,
which is what the analysis uses. Model checkpoints are omitted as well - see the
repository releases.
