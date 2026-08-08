#!/usr/bin/env python3
"""
ch4_figures.py

Generates the 6 Chapter-4 figures (EE491) with a clean, publication-friendly style.

Figures (saved as PNG, 300 dpi):
  1) dice_by_region.png         (WT/TC/ET/MeanClass for anchors + AutoBest)
  2) lambda_mu_heatmap.png         (lambda-mu sensitivity heatmap, default = best_dice)
  3) search_progress.png     (all trials scatter + best-per-iter path + anchor overlay)
  4) rerank_top10.png         (top-10 rerank scatter + line + selected highlight)
  5) pareto_search_view.png     (Params vs MeanDice, colored by GFLOPs, annotated)
  6) dice_per_mparam.png           (Dice per million parameters)

Run:
  python ch4_figures.py
"""

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# optional, for nicer styling + heatmap
try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False


# -----------------------------
# input CSVs (committed under results/) and output directory
# -----------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS   = os.path.join(REPO_ROOT, "results")

variants_csv = os.path.join(RESULTS, "ee491", "results_unet_variants_final.csv")
autobest_csv = os.path.join(RESULTS, "ee491", "results_unet-autobest.csv")
trials_csv   = os.path.join(RESULTS, "search", "final_search_trials.csv")
rerank_csv   = os.path.join(RESULTS, "search", "rerank_top10_full.csv")
lam_mu_csv   = os.path.join(RESULTS, "search", "lambda_mu_summary.csv")

OUT_DIR = os.environ.get("BTSEG_FIG_OUT", os.path.join(REPO_ROOT, "docs", "figures"))


# -----------------------------
# styling
# -----------------------------
def set_style():
    # matplotlib base style
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    if _HAS_SEABORN:
        sns.set_theme(style="whitegrid", context="notebook")
        sns.set_palette("Set2")


# -----------------------------
# helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_eval(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x


def to_num(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_results(variants_path: str, autobest_path: str) -> pd.DataFrame:
    v = pd.read_csv(variants_path)
    a = pd.read_csv(autobest_path)

    keep = [
        "model",
        "params",
        "gflops_fwd_128",
        "best_val_wt_dice",
        "best_val_tc_dice",
        "best_val_et_dice",
        "best_val_mean_class_dice",
        "channels",
        "num_res_units",
    ]

    v = v[[c for c in keep if c in v.columns]].copy()
    a = a[[c for c in keep if c in a.columns]].copy()

    df = pd.concat([v, a], ignore_index=True)

    if "channels" in df.columns:
        df["channels"] = df["channels"].apply(safe_eval)

    df = to_num(df, ["params", "gflops_fwd_128",
                     "best_val_wt_dice", "best_val_tc_dice", "best_val_et_dice",
                     "best_val_mean_class_dice", "num_res_units"])
    return df


def annotate_models(ax, df, xcol, ycol):
    """Annotate points; place AutoBest label to the LEFT to avoid overlap."""
    for _, row in df.iterrows():
        if pd.isna(row[xcol]) or pd.isna(row[ycol]):
            continue
        x = float(row[xcol])
        y = float(row[ycol])
        lab = str(row["model"])

        if "autobest" in lab.lower():
            ax.annotate(
                lab,
                (x, y),
                textcoords="offset points",
                xytext=(-85, 6),
                ha="right",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        else:
            ax.annotate(
                lab,
                (x, y),
                textcoords="offset points",
                xytext=(6, 6),
                ha="left",
                va="bottom",
                fontsize=10,
            )


# -----------------------------
# Figure 1: grouped bars (WT/TC/ET/Mean)
# -----------------------------
def fig_grouped_dice_bars(df: pd.DataFrame, outdir: str):
    need = ["model", "best_val_wt_dice", "best_val_tc_dice", "best_val_et_dice", "best_val_mean_class_dice"]
    if not all(c in df.columns for c in need):
        raise ValueError(f"missing columns for grouped bars: {need}")

    d = df[need].dropna().copy()
    d = d.sort_values("best_val_mean_class_dice", ascending=False)

    models = d["model"].tolist()
    wt = d["best_val_wt_dice"].to_numpy()
    tc = d["best_val_tc_dice"].to_numpy()
    et = d["best_val_et_dice"].to_numpy()
    mean = d["best_val_mean_class_dice"].to_numpy()

    x = np.arange(len(models))
    w = 0.20

    plt.figure(figsize=(13, 5))
    ax = plt.gca()

    # use pleasant colors (Set2 if seaborn exists)
    colors = None
    if _HAS_SEABORN:
        colors = sns.color_palette("Set2", 4)

    ax.bar(x - 1.5*w, wt,   w, label="WT",  color=(colors[0] if colors else None))
    ax.bar(x - 0.5*w, tc,   w, label="TC",  color=(colors[1] if colors else None))
    ax.bar(x + 0.5*w, et,   w, label="ET",  color=(colors[2] if colors else None))
    ax.bar(x + 1.5*w, mean, w, label="Mean Class", color=(colors[3] if colors else None))

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=18, ha="right")
    ax.set_ylabel("Dice")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Validation Dice Comparison (Anchors + AutoBest)")
    ax.legend(ncols=4, frameon=True)
    ax.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "dice_by_region.png"))
    plt.close()


# -----------------------------
# Figure 2: lambda-mu heatmap
# -----------------------------
def fig_lambda_mu_heatmap(lam_mu_path: str, outdir: str, value_col: str = "best_dice"):
    s = pd.read_csv(lam_mu_path)
    if not {"lam", "mu", value_col}.issubset(set(s.columns)):
        raise ValueError(f"lambda_mu_summary.csv must include columns: lam, mu, {value_col}")

    s = to_num(s, ["lam", "mu", value_col])
    piv = s.pivot_table(index="mu", columns="lam", values=value_col, aggfunc="max").sort_index().sort_index(axis=1)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    if _HAS_SEABORN:
        sns.heatmap(
            piv,
            ax=ax,
            cmap="viridis",
            linewidths=0.4,
            linecolor=(0, 0, 0, 0.15),
            cbar_kws={"label": value_col},
        )
    else:
        im = ax.imshow(piv.to_numpy(), aspect="auto", origin="lower")
        plt.colorbar(im, ax=ax, label=value_col)
        ax.set_xticks(np.arange(len(piv.columns)))
        ax.set_xticklabels([f"{x:g}" for x in piv.columns], rotation=0)
        ax.set_yticks(np.arange(len(piv.index)))
        ax.set_yticklabels([f"{y:g}" for y in piv.index])

    ax.set_xlabel("λ (lambda)")
    ax.set_ylabel("μ (mu)")
    ax.set_title(f"λ–μ Sensitivity Heatmap ({value_col})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "lambda_mu_heatmap.png"))
    plt.close()


# -----------------------------
# Figure 3: dense search progress
# -----------------------------
from adjustText import adjust_text

def fig_dense_search_progress(trials_path: str, anchors_df: pd.DataFrame, outdir: str):
    t = pd.read_csv(trials_path)
    need = ["iter", "score", "best_val_mean_class_dice", "params", "gflops_fwd_128"]
    t = to_num(t, need).dropna(subset=need)

    path = (t.sort_values(["iter", "score"], ascending=[True, False])
              .groupby("iter", as_index=False).first().sort_values("iter"))

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.subplots_adjust(right=0.84)  # reserve space for colorbar

    sc = ax.scatter(
        t["params"], t["best_val_mean_class_dice"],
        c=t["gflops_fwd_128"], cmap="viridis",
        s=38, alpha=0.85, edgecolors="none"
    )
    ax.plot(path["params"], path["best_val_mean_class_dice"], linewidth=2.2, alpha=0.9)

    texts = []
    if anchors_df is not None and len(anchors_df) > 0:
        dd = anchors_df.dropna(subset=["params", "best_val_mean_class_dice"]).copy()
        ax.scatter(
            dd["params"], dd["best_val_mean_class_dice"],
            marker="X", s=220,
            linewidths=1.4, edgecolors="black", facecolors="white",
            zorder=4
        )

        for _, row in dd.iterrows():
            x = float(row["params"])
            y = float(row["best_val_mean_class_dice"])
            name = str(row["model"])
            texts.append(ax.text(x, y, name, fontsize=10, fontweight="bold"))

        adjust_text(
            texts,
            ax=ax,
            expand_points=(1.2, 1.4),
            expand_text=(1.2, 1.4),
            force_text=(0.6, 0.8),
            force_points=(0.2, 0.3),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters (log scale)")
    ax.set_ylabel("Best Val Mean Class Dice")
    ax.set_title("Dense Search Progress (All Trials + Best-Per-Iteration Path)")
    ax.set_ylim(0.0, 0.85)

    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label("GFLOPs (forward, 128³)")

    ax.grid(True, linestyle="--", alpha=0.25)
    plt.savefig(os.path.join(outdir, "search_progress.png"), dpi=300)
    plt.close(fig)

from adjustText import adjust_text

def fig_pareto_params_vs_dice(df: pd.DataFrame, outdir: str):
    d = df[["model", "params", "best_val_mean_class_dice", "gflops_fwd_128"]].dropna().copy()

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.subplots_adjust(right=0.84)  # reserve space for colorbar

    sc = ax.scatter(
        d["params"], d["best_val_mean_class_dice"],
        c=d["gflops_fwd_128"], cmap="viridis",
        s=140, alpha=0.95,
        edgecolors="black", linewidths=0.6
    )

    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters (log scale)")
    ax.set_ylabel("Best Val Mean Class Dice")
    ax.set_title("Accuracy–Capacity Trade-off (Anchors + AutoBest)")

    xmin, xmax = d["params"].min(), d["params"].max()
    ax.set_xlim(xmin / 2.0, xmax * 1.25)

    ymin, ymax = d["best_val_mean_class_dice"].min(), d["best_val_mean_class_dice"].max()
    ax.set_ylim(ymin - 0.01, ymax + 0.01)

    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label("GFLOPs (forward, 128³)")

    texts = []
    for _, row in d.iterrows():
        x = float(row["params"])
        y = float(row["best_val_mean_class_dice"])
        name = str(row["model"])
        fw = "bold" if "AutoBest" in name else "normal"
        texts.append(ax.text(x, y, name, fontsize=11, fontweight=fw))

    adjust_text(
        texts,
        ax=ax,
        expand_points=(1.2, 1.4),
        expand_text=(1.2, 1.4),
        force_text=(0.6, 0.8),
        force_points=(0.2, 0.3),
        lim=200,
    )

    ax.grid(True, linestyle="--", alpha=0.25)
    plt.savefig(os.path.join(outdir, "pareto_search_view.png"), dpi=300)
    plt.close(fig)



# -----------------------------
# Figure 4: rerank top-10
# -----------------------------
def fig_rerank_top10(rerank_path: str, outdir: str):
    r = pd.read_csv(rerank_path)
    need = ["rank_in_subset", "best_val_mean_class_dice", "params", "gflops_fwd_128"]
    if not all(c in r.columns for c in need):
        raise ValueError(f"rerank_top10_full.csv missing columns: {need}")

    r = to_num(r, need)
    r = r.dropna(subset=["rank_in_subset", "best_val_mean_class_dice", "params", "gflops_fwd_128"])
    r_sorted = r.sort_values("params")

    plt.figure(figsize=(11, 6))
    ax = plt.gca()

    sc = ax.scatter(
        r_sorted["params"],
        r_sorted["best_val_mean_class_dice"],
        c=r_sorted["gflops_fwd_128"],
        cmap="viridis",
        s=90,
        alpha=0.95,
        edgecolors="black",
        linewidths=0.6,
    )
    ax.plot(r_sorted["params"], r_sorted["best_val_mean_class_dice"], linewidth=2.0, alpha=0.85)

    # highlight rank==1
    sel = r[r["rank_in_subset"] == 1]
    if len(sel) >= 1:
        x = float(sel["params"].iloc[0])
        y = float(sel["best_val_mean_class_dice"].iloc[0])
        ax.scatter([x], [y], s=260, edgecolors="black", linewidths=1.8, facecolors="white", zorder=5)
        ax.annotate(
            "selected",
            (x, y),
            textcoords="offset points",
            xytext=(10, -18),
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters (log scale)")
    ax.set_ylabel("Best Val Mean Class Dice")
    ax.set_title("Re-rank Top-10 on Full Data)")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("GFLOPs (forward, 128³)")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "rerank_top10.png"))
    plt.close()


# -----------------------------
# Figure 5: Pareto scatter (params vs dice, color=gflops)
# -----------------------------


# -----------------------------
# Figure 6: Dice per million params
# -----------------------------
def fig_dice_per_mparam(df: pd.DataFrame, outdir: str):
    need = ["model", "params", "best_val_mean_class_dice"]
    if not all(c in df.columns for c in need):
        raise ValueError(f"missing columns for dice per mparam: {need}")

    d = df[need].dropna().copy()
    d = d[d["params"] > 0].copy()
    d["params_m"] = d["params"] / 1e6
    d["dice_per_mparam"] = d["best_val_mean_class_dice"] / d["params_m"]

    d = d.sort_values("dice_per_mparam", ascending=True)

    plt.figure(figsize=(10, 5))
    ax = plt.gca()

    if _HAS_SEABORN:
        colors = sns.color_palette("Set2", n_colors=len(d))
        ax.barh(d["model"], d["dice_per_mparam"], color=colors)
    else:
        ax.barh(d["model"], d["dice_per_mparam"])

    ax.set_xlabel("Mean Dice / Million Parameters")
    ax.set_title("Parameter Efficiency: Dice per Million Parameters")
    ax.grid(True, axis="x")

    for i, v in enumerate(d["dice_per_mparam"].to_numpy()):
        ax.text(v, i, f"  {v:.2f}", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "dice_per_mparam.png"))
    plt.close()


# -----------------------------
# main
# -----------------------------
def main():
    set_style()
    ensure_dir(OUT_DIR)

    df = load_results(variants_csv, autobest_csv)

    # anchors overlay for dense search (include all UNet models + AutoBest)
    anchors = df[df["model"].astype(str).str.contains("UNet", na=False)].copy()

    # 1) grouped bars
    fig_grouped_dice_bars(df, OUT_DIR)

    # 2) lambda-mu heatmap (use best_dice by default)
    fig_lambda_mu_heatmap(lam_mu_csv, OUT_DIR, value_col="best_dice")

    # 3) dense search progress + anchor overlay
    fig_dense_search_progress(trials_csv, anchors, OUT_DIR)

    # 4) rerank top-10
    fig_rerank_top10(rerank_csv, OUT_DIR)

    # 5) pareto scatter
    fig_pareto_params_vs_dice(df, OUT_DIR)

    # 6) dice per million params
    fig_dice_per_mparam(df, OUT_DIR)

    print("Saved figures into:", OUT_DIR)
    for name in [
        "dice_by_region.png",
        "lambda_mu_heatmap.png",
        "search_progress.png",
        "rerank_top10.png",
        "pareto_search_view.png",
        "dice_per_mparam.png",
    ]:
        print(" -", os.path.join(OUT_DIR, name))


if __name__ == "__main__":
    main()
