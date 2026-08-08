# eval_mc_dropout_uncertainty.py
# -----------------------------------------------------------------------------
# EE492 final phase: uncertainty evaluation for AutoBest-DSConv-MC.
#
# Produces:
#   1. MC-Dropout inference (T stochastic forward passes per validation case)
#   2. Per-region (WT/TC/ET) uncertainty maps  (predictive Bernoulli entropy)
#   3. QU-BraTS uncertainty score   (BraTS 2020 Task 3 metric, reimplemented)
#         score = AUC(Dice vs T) + (1 - AUC(FTP vs T)) + (1 - AUC(FTN vs T))
#   4. Expected Calibration Error (ECE), voxel-wise
#   5. Uncertainty-error Spearman correlation (voxel-level + case-level)
#   6. Qualitative slice figures: FLAIR | GT | Prediction | Uncertainty heatmap
#
# It reuses, UNCHANGED:
#   - base.make_transforms / split / build_datalist / SEED / DATA_ROOT
#   - base.compute_class_region_dice (same region definitions)
#   - the training script's model builder (dropout + DSConv) so the loaded
#     checkpoint maps onto an identical architecture.
#
# Run AFTER training, in the lab:
#   python eval_mc_dropout_uncertainty.py
# -----------------------------------------------------------------------------

import os
import csv
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import spearmanr

from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference

import train_variants as base
# Importing the launcher defines the dropout+DSConv model builder.
# Its training call is guarded by __main__, so this import does NOT train.
import train_autobest_dsconv_mc as mc


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
T_PASSES = 20                       # MC stochastic forward passes
ROI = (128, 128, 128)
THRESHOLDS = np.arange(0, 101, 2)   # uncertainty thresholds for QU-BraTS AUC
ECE_BINS = 15
SPEARMAN_SUBSAMPLE = 100_000        # voxels sampled per case for voxel Spearman
MAX_CASES = None                    # set to a small int for a quick smoke test
NUM_VIS_CASES = 6                   # number of qualitative figures to save
OUT_DIR = "uncertainty_results"
CKPT = f"brats2020_unet_{mc.MODEL_NAME}_best_80_20.pth"

REGIONS = ("WT", "TC", "ET")
LN2 = math.log(2.0)
EPS = 1e-7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# MC-Dropout: eval mode everywhere EXCEPT dropout layers, which stay stochastic
# -----------------------------------------------------------------------------
def enable_mc_dropout(model: nn.Module):
    model.eval()
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()
            n += 1
    if n == 0:
        raise RuntimeError("No dropout layers active -> MC-Dropout is a no-op.")
    return n


@torch.no_grad()
def mc_inference(model, image):
    """
    Runs T_PASSES stochastic forward passes with sliding-window inference.
    Accumulates statistics on the fly to keep memory bounded.

    Returns (all on CPU):
      mean_prob       : (4, H, W, D)  mean softmax probability
      pred            : (H, W, D)     argmax label in {0,1,2,3}
      pred_entropy    : (H, W, D)     H[E[p]]  (total predictive uncertainty)
      mutual_info     : (H, W, D)     H[E[p]] - E[H[p]]  (epistemic)
      class_var       : (4, H, W, D)  per-class variance across passes
    """
    sum_p = None
    sum_p2 = None
    sum_entropy = None  # E[H[p_t]]  accumulated

    for _ in range(T_PASSES):
        logits = sliding_window_inference(
            image, roi_size=ROI, sw_batch_size=base.SW_BATCH_SIZE, predictor=model
        )
        prob = torch.softmax(logits, dim=1)[0]  # (4, H, W, D)

        per_pass_ent = -(prob * torch.log(prob + EPS)).sum(dim=0)  # (H,W,D)

        if sum_p is None:
            sum_p = prob.clone()
            sum_p2 = prob * prob
            sum_entropy = per_pass_ent.clone()
        else:
            sum_p += prob
            sum_p2 += prob * prob
            sum_entropy += per_pass_ent
        del logits, prob, per_pass_ent

    mean_prob = sum_p / T_PASSES
    class_var = (sum_p2 / T_PASSES) - (mean_prob * mean_prob)
    expected_entropy = sum_entropy / T_PASSES
    pred_entropy = -(mean_prob * torch.log(mean_prob + EPS)).sum(dim=0)
    mutual_info = pred_entropy - expected_entropy
    pred = mean_prob.argmax(dim=0)

    return (
        mean_prob.cpu(),
        pred.cpu(),
        pred_entropy.cpu(),
        mutual_info.cpu(),
        class_var.cpu(),
    )


# -----------------------------------------------------------------------------
# Region helpers (same definitions as base.compute_class_region_dice)
#   WT = {1,2,3}, TC = {1,3}, ET = {3}
# -----------------------------------------------------------------------------
def region_prob(mean_prob, region):
    if region == "WT":
        return mean_prob[1] + mean_prob[2] + mean_prob[3]
    if region == "TC":
        return mean_prob[1] + mean_prob[3]
    if region == "ET":
        return mean_prob[3]
    raise ValueError(region)


def region_mask(label_vol, region):
    if region == "WT":
        return (label_vol == 1) | (label_vol == 2) | (label_vol == 3)
    if region == "TC":
        return (label_vol == 1) | (label_vol == 3)
    if region == "ET":
        return label_vol == 3
    raise ValueError(region)


def bernoulli_entropy_pct(p):
    """Per-voxel Bernoulli entropy normalized to [0,100] (max = ln2)."""
    p = p.clamp(EPS, 1 - EPS)
    h = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
    return (h / LN2) * 100.0


def dice_bin(pred_bin, gt_bin):
    inter = float((pred_bin & gt_bin).sum())
    s = float(pred_bin.sum()) + float(gt_bin.sum())
    if s == 0:
        return 1.0
    return 2.0 * inter / s


# -----------------------------------------------------------------------------
# QU-BraTS metric (BraTS 2020 Task 3), faithful reimplementation.
#   Filter voxels with uncertainty > T, compute Dice of remaining.
#   FTP(T) = (TP_100 - TP_T)/TP_100 ; FTN(T) similarly.
#   AUCs integrated over T in [0,100], normalized to [0,1].
#   score = AUC_Dice + (1 - AUC_FTP) + (1 - AUC_FTN)   (higher is better)
# Reference: med.upenn.edu/cbica/brats2020/tasks.html (Task 3 evaluation).
# -----------------------------------------------------------------------------
def qu_brats_region(pred_bin, gt_bin, unc_pct):
    pred_bin = pred_bin.bool()
    gt_bin = gt_bin.bool()
    unc = unc_pct  # already in [0,100]

    tp_full = float((pred_bin & gt_bin).sum())
    tn_full = float((~pred_bin & ~gt_bin).sum())

    dices, ftps, ftns = [], [], []
    for T in THRESHOLDS:
        keep = unc <= T  # voxels confident enough to keep
        p = pred_bin & keep
        g = gt_bin & keep
        dices.append(dice_bin(p, g))

        tp_T = float((pred_bin & gt_bin & keep).sum())
        tn_T = float((~pred_bin & ~gt_bin & keep).sum())
        ftps.append((tp_full - tp_T) / tp_full if tp_full > 0 else 0.0)
        ftns.append((tn_full - tn_T) / tn_full if tn_full > 0 else 0.0)

    x = THRESHOLDS / 100.0
    auc_dice = float(np.trapz(dices, x))
    auc_ftp = float(np.trapz(ftps, x))
    auc_ftn = float(np.trapz(ftns, x))
    score = auc_dice + (1.0 - auc_ftp) + (1.0 - auc_ftn)
    return {
        "auc_dice": auc_dice,
        "auc_ftp": auc_ftp,
        "auc_ftn": auc_ftn,
        "score": score,
        "dice_curve": np.asarray(dices),
        "ftp_curve": np.asarray(ftps),
        "ftn_curve": np.asarray(ftns),
    }


# -----------------------------------------------------------------------------
# Calibration (ECE) over brain voxels, using max-softmax confidence.
# -----------------------------------------------------------------------------
def expected_calibration_error(conf, correct, n_bins=ECE_BINS):
    conf = conf.flatten()
    correct = correct.flatten().astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = conf.size
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc = correct[m].mean()
        avg_conf = conf[m].mean()
        ece += (m.sum() / n) * abs(acc - avg_conf)
    return float(ece)


# -----------------------------------------------------------------------------
# Visualization: FLAIR | GT | Prediction | predictive-entropy heatmap
# -----------------------------------------------------------------------------
def save_case_figure(flair, gt, pred, entropy, case_id, out_path):
    # axial slice = last spatial axis; pick slice with largest tumor area
    tumor = (gt > 0)
    areas = tumor.reshape(-1, gt.shape[-1]).sum(axis=0)
    z = int(np.argmax(areas)) if areas.max() > 0 else gt.shape[-1] // 2

    f = flair[..., z]
    g = gt[..., z]
    p = pred[..., z]
    e = entropy[..., z]

    seg_cmap = ListedColormap(
        [(0, 0, 0, 0), (0.20, 0.55, 1.0, 0.8),
         (0.95, 0.85, 0.1, 0.8), (0.95, 0.25, 0.2, 0.8)]
    )

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    for a in ax:
        a.axis("off")

    ax[0].imshow(f.T, cmap="gray", origin="lower")
    ax[0].set_title("FLAIR")

    ax[1].imshow(f.T, cmap="gray", origin="lower")
    ax[1].imshow(g.T, cmap=seg_cmap, vmin=0, vmax=3, origin="lower")
    ax[1].set_title("Ground truth")

    ax[2].imshow(f.T, cmap="gray", origin="lower")
    ax[2].imshow(p.T, cmap=seg_cmap, vmin=0, vmax=3, origin="lower")
    ax[2].set_title("Prediction")

    ax[3].imshow(f.T, cmap="gray", origin="lower")
    im = ax[3].imshow(e.T, cmap="jet", alpha=0.6, origin="lower")
    ax[3].set_title("Predictive entropy")
    fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)

    fig.suptitle(f"{case_id}  (axial z={z})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)

    # Same data + split + transforms as training/validation
    data_dicts = base.build_datalist(base.DATA_ROOT)
    _, val_files = base.split_train_val(
        data_dicts, val_fraction=0.2, seed=base.SEED, save_split=False
    )
    if MAX_CASES is not None:
        val_files = val_files[:MAX_CASES]

    _, val_transforms = base.make_transforms()
    val_ds = Dataset(val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    # Rebuild identical architecture, load checkpoint
    model = mc.make_model_dsconv_mc(mc.AUTOBEST_SPEC, DEVICE)
    state = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(state)
    n_drop = enable_mc_dropout(model)
    print(f"Loaded {CKPT} | active dropout layers: {n_drop} | T={T_PASSES}")

    rows = []
    vis_done = 0

    # --- accumulators for report figures (saved to curves.npz) ---
    curve_sums = {
        r: {
            "dice": np.zeros(len(THRESHOLDS), dtype=float),
            "ftp": np.zeros(len(THRESHOLDS), dtype=float),
            "ftn": np.zeros(len(THRESHOLDS), dtype=float),
        }
        for r in REGIONS
    }
    rel_count = np.zeros(ECE_BINS)
    rel_corr = np.zeros(ECE_BINS)
    rel_conf = np.zeros(ECE_BINS)
    scatter_ent, scatter_err = [], []
    rng = np.random.RandomState(base.SEED)

    for idx, batch in enumerate(val_loader):
        case_id = batch.get("case_id", [f"case_{idx}"])[0]
        image = batch["image"].to(DEVICE)
        label = batch["label"][0, 0].cpu()  # (H, W, D)

        mean_prob, pred, pred_ent, mut_info, class_var = mc_inference(model, image)

        # --- segmentation Dice (MC mean prediction) ---
        gt = label.long()
        dice = {
            r: dice_bin(region_mask(pred, r), region_mask(gt, r)) for r in REGIONS
        }

        # --- QU-BraTS per region (Bernoulli-entropy uncertainty map, 0-100) ---
        qu = {}
        for r in REGIONS:
            unc_pct = bernoulli_entropy_pct(region_prob(mean_prob, r))
            qu[r] = qu_brats_region(region_mask(pred, r), region_mask(gt, r), unc_pct)
            curve_sums[r]["dice"] += qu[r]["dice_curve"]
            curve_sums[r]["ftp"] += qu[r]["ftp_curve"]
            curve_sums[r]["ftn"] += qu[r]["ftn_curve"]

        # --- ECE over brain voxels (nonzero FLAIR) ---
        flair = image[0, 0].cpu().numpy()
        brain = flair != 0
        conf = mean_prob.max(dim=0).values.numpy()[brain]
        correct = (pred.numpy() == gt.numpy())[brain]
        ece = expected_calibration_error(conf, correct)

        # reliability-diagram accumulation (global bins)
        c_idx = np.clip((conf * ECE_BINS).astype(int), 0, ECE_BINS - 1)
        for b in range(ECE_BINS):
            mb = c_idx == b
            cnt = int(mb.sum())
            if cnt:
                rel_count[b] += cnt
                rel_corr[b] += float(correct[mb].sum())
                rel_conf[b] += float(conf[mb].sum())

        # --- voxel-level uncertainty-error Spearman (subsampled) ---
        ent_flat = pred_ent.numpy()[brain]
        err_flat = (~correct).astype(np.float64)
        if ent_flat.size > SPEARMAN_SUBSAMPLE:
            sel = np.random.RandomState(base.SEED).choice(
                ent_flat.size, SPEARMAN_SUBSAMPLE, replace=False
            )
            rho, _ = spearmanr(ent_flat[sel], err_flat[sel])
        else:
            rho, _ = spearmanr(ent_flat, err_flat)
        rho = float(rho) if not math.isnan(rho) else 0.0

        # pooled voxel-level (entropy, error) sample for the scatter/violin figure
        k = min(2000, ent_flat.size)
        if k > 0:
            si = rng.choice(ent_flat.size, k, replace=False)
            scatter_ent.append(ent_flat[si])
            scatter_err.append(err_flat[si])

        mean_dice = float(np.mean([dice[r] for r in REGIONS]))
        qu_mean = float(np.mean([qu[r]["score"] for r in REGIONS]))

        rows.append({
            "case_id": case_id,
            "dice_WT": dice["WT"], "dice_TC": dice["TC"], "dice_ET": dice["ET"],
            "mean_region_dice": mean_dice,
            "qu_WT": qu["WT"]["score"], "qu_TC": qu["TC"]["score"], "qu_ET": qu["ET"]["score"],
            "qu_mean_score": qu_mean,
            "auc_dice_WT": qu["WT"]["auc_dice"],
            "auc_ftp_WT": qu["WT"]["auc_ftp"],
            "auc_ftn_WT": qu["WT"]["auc_ftn"],
            "ece": ece,
            "voxel_spearman": rho,
            "mean_pred_entropy": float(pred_ent.numpy()[brain].mean()),
            "mean_mutual_info": float(mut_info.numpy()[brain].mean()),
        })
        print(
            f"[{idx+1}/{len(val_loader)}] {case_id} | "
            f"Dice WT/TC/ET {dice['WT']:.3f}/{dice['TC']:.3f}/{dice['ET']:.3f} | "
            f"QU {qu_mean:.3f} | ECE {ece:.4f} | rho {rho:.3f}"
        )

        if vis_done < NUM_VIS_CASES:
            save_case_figure(
                flair, gt.numpy(), pred.numpy(), pred_ent.numpy(), case_id,
                os.path.join(OUT_DIR, "figures", f"{case_id}_uncertainty.png"),
            )
            vis_done += 1

        del image, mean_prob, pred, pred_ent, mut_info, class_var
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # ---- write per-case CSV ----
    csv_path = os.path.join(OUT_DIR, "per_case_uncertainty.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- aggregate ----
    def avg(k):
        return float(np.mean([r[k] for r in rows]))

    # case-level uncertainty-error correlation:
    # mean entropy vs segmentation error (1 - mean region Dice)
    case_ent = np.array([r["mean_pred_entropy"] for r in rows])
    case_err = np.array([1.0 - r["mean_region_dice"] for r in rows])
    case_rho, _ = spearmanr(case_ent, case_err)

    summary = {
        "n_cases": len(rows),
        "T_passes": T_PASSES,
        "mean_region_dice": avg("mean_region_dice"),
        "qu_brats_score_mean": avg("qu_mean_score"),
        "qu_WT": avg("qu_WT"), "qu_TC": avg("qu_TC"), "qu_ET": avg("qu_ET"),
        "ece_mean": avg("ece"),
        "voxel_spearman_mean": avg("voxel_spearman"),
        "case_level_spearman": float(case_rho),
    }
    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    # ---- raw data for report figures ----
    n = len(rows)
    npz_payload = {
        "thresholds": THRESHOLDS,
        "ece_bins": np.array(ECE_BINS),
        "rel_count": rel_count,
        "rel_corr": rel_corr,
        "rel_conf": rel_conf,
        "scatter_entropy": np.concatenate(scatter_ent) if scatter_ent else np.array([]),
        "scatter_error": np.concatenate(scatter_err) if scatter_err else np.array([]),
        "case_entropy": case_ent,
        "case_error": case_err,
    }
    for r in REGIONS:
        npz_payload[f"{r}_dice"] = curve_sums[r]["dice"] / n
        npz_payload[f"{r}_ftp"] = curve_sums[r]["ftp"] / n
        npz_payload[f"{r}_ftn"] = curve_sums[r]["ftn"] / n
    np.savez(os.path.join(OUT_DIR, "curves.npz"), **npz_payload)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:>22}: {v}")
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {os.path.join(OUT_DIR, 'summary.csv')}")
    print(f"Wrote: {os.path.join(OUT_DIR, 'curves.npz')}")
    print(f"Figures: {os.path.join(OUT_DIR, 'figures')}")


if __name__ == "__main__":
    main()
