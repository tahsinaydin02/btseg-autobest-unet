# plot_uncertainty_figures.py
# -----------------------------------------------------------------------------
# Turns eval_mc_dropout_uncertainty.py outputs into report-ready figures.
#
# Reads from results dir (default "uncertainty_results"):
#   - summary.csv
#   - per_case_uncertainty.csv
#   - curves.npz
#
# Produces (PDF + PNG) in <results_dir>/report_figures/:
#   fig1_qubrats_curves   : filtered Dice / FTP / FTN vs uncertainty threshold
#   fig2_reliability       : reliability diagram + ECE
#   fig3_uncertainty_error : case-level scatter + voxel-level distribution
#
# Run (cheap, re-runnable):
#   python plot_uncertainty_figures.py [results_dir]
# -----------------------------------------------------------------------------

import os
import csv
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- report-matching style (serif, like Times New Roman) ----
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.frameon": False,
    "figure.dpi": 130,
})

REGIONS = ("WT", "TC", "ET")
COLORS = {"WT": "#1f6feb", "TC": "#e3a008", "ET": "#d4322c"}


def load_summary(path):
    with open(path) as fh:
        row = next(csv.DictReader(fh))
    out = {}
    for k, v in row.items():
        try:
            out[k] = float(v)
        except (ValueError, TypeError):
            out[k] = v
    return out


def save(fig, out_dir, name):
    fig.savefig(os.path.join(out_dir, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, name + ".png"), bbox_inches="tight")
    plt.close(fig)
    print("Wrote", os.path.join(out_dir, name + ".{pdf,png}"))


# -----------------------------------------------------------------------------
def fig_qubrats_curves(npz, summary, out_dir):
    T = npz["thresholds"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

    for r in REGIONS:
        ax[0].plot(T, npz[f"{r}_dice"], color=COLORS[r], lw=2, label=r)
        ax[1].plot(T, npz[f"{r}_ftp"], color=COLORS[r], lw=2, label=r)
        ax[2].plot(T, npz[f"{r}_ftn"], color=COLORS[r], lw=2, label=r)

    ax[0].set_title("Filtered Dice vs threshold")
    ax[0].set_ylabel("Dice (remaining voxels)")
    ax[1].set_title("Ratio of filtered TP")
    ax[1].set_ylabel("FTP")
    ax[2].set_title("Ratio of filtered TN")
    ax[2].set_ylabel("FTN")
    for a in ax:
        a.set_xlabel("Uncertainty threshold T  (keep unc \u2264 T)")
        a.set_xlim(0, 100)
        a.legend(title=None)

    qu_mean = summary.get("qu_brats_score_mean", float("nan"))
    fig.suptitle(
        f"QU-BraTS uncertainty evaluation  |  mean score = {qu_mean:.3f}  "
        f"(WT {summary.get('qu_WT', float('nan')):.3f}, "
        f"TC {summary.get('qu_TC', float('nan')):.3f}, "
        f"ET {summary.get('qu_ET', float('nan')):.3f})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, out_dir, "fig1_qubrats_curves")


# -----------------------------------------------------------------------------
def fig_reliability(npz, summary, out_dir):
    n_bins = int(npz["ece_bins"])
    count = npz["rel_count"]
    acc = np.divide(npz["rel_corr"], count, out=np.zeros_like(count), where=count > 0)
    conf = np.divide(npz["rel_conf"], count, out=np.zeros_like(count), where=count > 0)
    centers = (np.arange(n_bins) + 0.5) / n_bins
    width = 1.0 / n_bins
    nonempty = count > 0

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6),
                           gridspec_kw={"width_ratios": [3, 2]})

    # reliability bars + gap
    ax[0].plot([0, 1], [0, 1], "--", color="gray", lw=1.2, label="Perfect calibration")
    ax[0].bar(centers[nonempty], acc[nonempty], width=width * 0.9,
              color="#1f6feb", alpha=0.85, edgecolor="white", label="Accuracy")
    # gap (overconfidence) shading
    gap = conf - acc
    ax[0].bar(centers[nonempty], gap[nonempty], width=width * 0.9,
              bottom=acc[nonempty], color="#d4322c", alpha=0.35,
              edgecolor="none", label="Gap (conf \u2212 acc)")
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)
    ax[0].set_xlabel("Confidence (max softmax)")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_title("Reliability diagram")
    ax[0].legend(loc="upper left")
    ax[0].text(0.97, 0.05, f"ECE = {summary.get('ece_mean', float('nan')):.4f}",
               ha="right", va="bottom", transform=ax[0].transAxes,
               fontsize=12, bbox=dict(boxstyle="round", fc="white", ec="gray"))

    # confidence histogram
    frac = count / count.sum() if count.sum() > 0 else count
    ax[1].bar(centers[nonempty], frac[nonempty], width=width * 0.9,
              color="#6e7681", alpha=0.8)
    ax[1].set_xlim(0, 1)
    ax[1].set_xlabel("Confidence (max softmax)")
    ax[1].set_ylabel("Fraction of voxels")
    ax[1].set_title("Confidence histogram")

    fig.tight_layout()
    save(fig, out_dir, "fig2_reliability")


# -----------------------------------------------------------------------------
def fig_uncertainty_error(npz, summary, out_dir):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))

    # (a) case-level: mean predictive entropy vs segmentation error (1 - Dice)
    ce = npz["case_entropy"]
    cerr = npz["case_error"]
    ax[0].scatter(ce, cerr, s=28, color="#1f6feb", alpha=0.75, edgecolor="white")
    if ce.size >= 2 and np.ptp(ce) > 0:
        m, b = np.polyfit(ce, cerr, 1)
        xs = np.linspace(ce.min(), ce.max(), 50)
        ax[0].plot(xs, m * xs + b, "--", color="#d4322c", lw=1.5)
    ax[0].set_xlabel("Mean predictive entropy (per case)")
    ax[0].set_ylabel("Segmentation error  (1 \u2212 mean region Dice)")
    ax[0].set_title("Case-level uncertainty vs error")
    ax[0].text(0.03, 0.95,
               f"Spearman \u03c1 = {summary.get('case_level_spearman', float('nan')):.3f}",
               ha="left", va="top", transform=ax[0].transAxes, fontsize=12,
               bbox=dict(boxstyle="round", fc="white", ec="gray"))

    # (b) voxel-level: entropy distribution for correct vs incorrect voxels
    se = npz["scatter_entropy"]
    serr = npz["scatter_error"]
    if se.size > 0:
        correct = se[serr < 0.5]
        wrong = se[serr >= 0.5]
        data = [correct, wrong]
        parts = ax[1].violinplot(data, showmedians=True, showextrema=False)
        for pc, c in zip(parts["bodies"], ["#1f6feb", "#d4322c"]):
            pc.set_facecolor(c)
            pc.set_alpha(0.6)
        ax[1].set_xticks([1, 2])
        ax[1].set_xticklabels(["Correct\nvoxels", "Incorrect\nvoxels"])
    ax[1].set_ylabel("Predictive entropy")
    ax[1].set_title("Voxel-level entropy by correctness")
    ax[1].text(0.97, 0.95,
               f"voxel \u03c1 = {summary.get('voxel_spearman_mean', float('nan')):.3f}",
               ha="right", va="top", transform=ax[1].transAxes, fontsize=11,
               bbox=dict(boxstyle="round", fc="white", ec="gray"))

    fig.tight_layout()
    save(fig, out_dir, "fig3_uncertainty_error")


# -----------------------------------------------------------------------------
def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "uncertainty_results"
    out_dir = os.path.join(results_dir, "report_figures")
    os.makedirs(out_dir, exist_ok=True)

    npz = np.load(os.path.join(results_dir, "curves.npz"))
    summary = load_summary(os.path.join(results_dir, "summary.csv"))

    fig_qubrats_curves(npz, summary, out_dir)
    fig_reliability(npz, summary, out_dir)
    fig_uncertainty_error(npz, summary, out_dir)

    print("\nAll figures written to", out_dir)


if __name__ == "__main__":
    main()
