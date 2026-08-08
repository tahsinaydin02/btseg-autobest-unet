# auto_search_codegen_lab.py
# note: runs on lab machine with brats data.
# note: anchors are evaluated (short proxy) so their dice is real.
# note: plot includes pareto front and knee point.

import os
import json
import math
import time
import random
import ast
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import train_variants as base


# ------------------------------------------------------------
# performance knobs (lab GPU)
# ------------------------------------------------------------
# main goal: keep GPU busy by avoiding repeated DataLoader construction,
# excessive disk writes (csv/plots), and repeated FLOPs computation.

# how often to flush CSV/plot (in evaluated candidates)
FLUSH_EVERY = 6

# how often to recompute / redraw the progress plot
PLOT_EVERY = 12

# if True, recompute GFLOPs for every spec (slow); otherwise cache per spec
CACHE_MODEL_COSTS = True

# persistent workers reduce CPU overhead; if NUM_WORKERS==0, this has no effect
PERSISTENT_WORKERS = True

# set to False if you want strict determinism (slower). base.STRICT_GPU_DETERMINISM
# controls cudnn.benchmark and deterministic kernels in the backend.
FAST_BUT_NOT_STRICTLY_DETERMINISTIC = True


# ----------------------------
# spec utilities
# ----------------------------
def scale_channels(channels, mult: float, min_ch=8, max_ch=512):
    out = []
    for c in channels:
        v = int(round(c * mult))
        v = max(min_ch, min(max_ch, v))
        v = int(round(v / 8) * 8)  # keep divisible by 8
        v = max(min_ch, v)
        out.append(v)

    # ensure strictly increasing
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + 8
    return tuple(out)


def canonicalize_spec(spec: dict):
    # enforce stride convention for depth
    s = deepcopy(spec)
    s["channels"] = tuple(s["channels"])
    s["strides"] = tuple([2] * (len(s["channels"]) - 1))
    s["num_res_units"] = int(s["num_res_units"])
    return s


def spec_key(spec: dict):
    s = canonicalize_spec(spec)
    return (tuple(s["channels"]), tuple(s["strides"]), int(s["num_res_units"]))


def make_neighbors(spec: dict):
    spec = canonicalize_spec(spec)
    ch = tuple(spec["channels"])
    ru = int(spec["num_res_units"])

    cands = []

    # width scaling (dense exploration)
    for m in (0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40):
        s = deepcopy(spec)
        s["channels"] = scale_channels(ch, m)
        s = canonicalize_spec(s)
        cands.append(s)

    # residual tweak
    for d in (-1, 0, +1):
        s = deepcopy(spec)
        s["num_res_units"] = int(np.clip(ru + d, 0, 2))
        s = canonicalize_spec(s)
        cands.append(s)

    # depth tweak: shallower
    if len(ch) >= 4:
        s = deepcopy(spec)
        s["channels"] = tuple(ch[:-1])
        s["num_res_units"] = int(np.clip(s["num_res_units"], 0, 2))
        s = canonicalize_spec(s)
        cands.append(s)

    # depth tweak: deeper (cap at 5 levels)
    if len(ch) <= 5:
        s = deepcopy(spec)
        last = ch[-1]
        s["channels"] = tuple(ch) + (min(512, last * 2),)
        s = canonicalize_spec(s)
        cands.append(s)

    # deduplicate
    uniq = {}
    for s in cands:
        uniq[spec_key(s)] = s
    return list(uniq.values())


# ----------------------------
# objective + budgets
# ----------------------------
def objective_score(mean_dice: float, params: int, gflops: float, lam: float, mu: float):
    # higher is better
    p_term = math.log(params / 1e6 + 1.0)
    f_term = math.log(max(gflops, 0.0) + 1.0)
    return float(mean_dice) - lam * p_term - mu * f_term


def passes_budget(params: int, gflops: float, max_params=None, max_gflops=None):
    if max_params is not None and params > max_params:
        return False
    if max_gflops is not None and gflops > max_gflops:
        return False
    return True


# ----------------------------
# backend helpers
# ----------------------------
def quick_model_cost(spec: dict):
    """compute (params, gflops) for a spec.

    this is surprisingly expensive if fvcore/ptflops is installed,
    so we cache results per spec_key by default.
    """

    if CACHE_MODEL_COSTS:
        k = spec_key(spec)
        if k in quick_model_cost._cache:
            return quick_model_cost._cache[k]

    device = base.torch.device("cuda" if base.torch.cuda.is_available() else "cpu")
    m = base.make_model(spec, device)
    params = base.count_trainable_params(m)
    gflops = base.compute_gflops(m, device)
    del m
    if base.torch.cuda.is_available():
        base.torch.cuda.empty_cache()

    out = (int(params), float(gflops))
    if CACHE_MODEL_COSTS:
        quick_model_cost._cache[spec_key(spec)] = out
    return out


quick_model_cost._cache = {}


def make_persistent_loaders(train_files, val_files, batch_size: int = 1):
    """create DataLoaders once and reuse across all trials.

    the backend's make_loaders() recreates datasets + loaders each time.
    that is correct but very expensive for auto-search.

    fairness/determinism:
    - we keep an explicit torch.Generator and re-seed it per trial to keep
      the shuffled order comparable.
    - with persistent_workers=True, augmentation RNG inside workers will not
      be perfectly identical across trials. for strict determinism you should
      set PERSISTENT_WORKERS=False (and accept the worker spawn overhead).
    """

    from monai.data import Dataset, DataLoader

    train_transforms, val_transforms = base.make_transforms()
    train_ds = Dataset(train_files, transform=train_transforms)
    val_ds = Dataset(val_files, transform=val_transforms)

    g = base.torch.Generator()
    g.manual_seed(base.SEED)

    pin = bool(base.torch.cuda.is_available())
    num_workers = int(getattr(base, "NUM_WORKERS", 0))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=base.seed_worker,
        generator=g,
        persistent_workers=bool(PERSISTENT_WORKERS) and num_workers > 0,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=base.seed_worker,
        persistent_workers=bool(PERSISTENT_WORKERS) and num_workers > 0,
        pin_memory=pin,
    )
    return train_loader, val_loader, g


def proxy_train_one(train_loader, val_loader, train_gen, spec: dict, model_name: str, proxy_epochs: int):
    # keep comparisons fair without rebuilding the DataLoader every time
    base.set_all_seeds(base.SEED)
    if train_gen is not None:
        train_gen.manual_seed(base.SEED)

    res = base.train_one(
        model_name=model_name,
        spec=spec,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=int(proxy_epochs),
    )
    return res


# ----------------------------
# anchors from backend MODEL_SPECS
# ----------------------------
def get_anchor_specs_from_backend():
    if not hasattr(base, "MODEL_SPECS"):
        raise RuntimeError("backend must define MODEL_SPECS")
    specs = {k: canonicalize_spec(v) for k, v in base.MODEL_SPECS.items()}

    order = ["UNet-Baseline", "UNet-Lite", "UNet-Ultralight", "UNet-Efficient", "UNet-Tiny"]
    anchors = []
    for name in order:
        if name in specs:
            anchors.append((name, specs[name]))
    for name, spec in specs.items():
        if name not in order:
            anchors.append((name, spec))
    return anchors


def choose_start_spec(anchors):
    for name, spec in anchors:
        if name == "UNet-Lite":
            return deepcopy(spec)
    return deepcopy(anchors[0][1])


# ----------------------------
# pareto + knee
# ----------------------------
def pareto_front_indices(params, gflops, dice):
    # maximize dice, minimize params and gflops
    n = len(dice)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            # j dominates i if it's no worse in all objectives and better in at least one
            no_worse = (dice[j] >= dice[i]) and (params[j] <= params[i]) and (gflops[j] <= gflops[i])
            strictly_better = (dice[j] > dice[i]) or (params[j] < params[i]) or (gflops[j] < gflops[i])
            if no_worse and strictly_better:
                keep[i] = False
                break
    return np.where(keep)[0]


def knee_point_on_pareto(df_pf: pd.DataFrame):
    # knee on 2D tradeoff: x=log(params), y=dice
    # method: normalize, then pick max distance to chord between extremes
    if len(df_pf) < 3:
        return None

    x = np.log10(df_pf["params"].astype(float).values)
    y = df_pf["best_val_mean_class_dice"].astype(float).values

    # sort by x
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    df_sorted = df_pf.iloc[order].reset_index(drop=True)

    # endpoints
    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]

    # distance from line (x0,y0)-(x1,y1)
    # line coefficients
    dx = x1 - x0
    dy = y1 - y0
    denom = math.sqrt(dx * dx + dy * dy) + 1e-12

    # perpendicular distance for each point
    dist = np.abs(dy * x - dx * y + x1 * y0 - y1 * x0) / denom

    k = int(np.argmax(dist))
    return df_sorted.iloc[k]


# ----------------------------
# plotting (dense + anchors + pareto + knee)
# ----------------------------
def evaluate_anchors_if_needed(train_loader, val_loader, train_gen, anchors, out_dir, anchor_epochs=2):
    """
    returns a dict keyed by spec_key -> dict with dice, params, gflops, name
    runs a short proxy training for anchors if no cached anchor_eval.json exists
    """
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "anchor_eval.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cached = json.load(f)
        # keys are strings in json; rebuild mapping
        out = {}
        for k_str, v in cached.items():
            out[tuple(json.loads(k_str))] = v
        return out

    anchor_map = {}
    for name, spec in anchors:
        p, f = quick_model_cost(spec)

        # run short proxy so dice is real
        run_name = f"anchor_{name}"
        res = proxy_train_one(train_loader, val_loader, train_gen, spec, run_name, proxy_epochs=int(anchor_epochs))
        dice = float(res.get("best_val_mean_class_dice", -1.0))

        anchor_map[spec_key(spec)] = {
            "name": name,
            "dice": dice,
            "params": int(res.get("params", p)),
            "gflops": float(res.get("gflops_fwd_128", f)),
        }
        print(f"[anchor] {name}: dice={dice:.4f} params={anchor_map[spec_key(spec)]['params']} gflops={anchor_map[spec_key(spec)]['gflops']:.3f}")

    # write cache
    dumpable = {json.dumps(list(k)): v for k, v in anchor_map.items()}
    with open(cache_path, "w") as f:
        json.dump(dumpable, f, indent=2)

    return anchor_map


def plot_dense_tradeoff(df: pd.DataFrame, title: str, out_png: str, anchors_info=None, draw_pareto=True):
    plt.figure(figsize=(7.9, 5.6))

    # main cloud
    x = df["params"].astype(float).values
    y = df["best_val_mean_class_dice"].astype(float).values
    c = df["gflops_fwd_128"].astype(float).values

    sc = plt.scatter(x, y, c=c, s=26, alpha=0.88)
    plt.xscale("log")
    plt.xlabel("Trainable Parameters (log scale)")
    plt.ylabel("Best Val Mean Class Dice")
    plt.title(title)
    cb = plt.colorbar(sc)
    cb.set_label("GFLOPs (forward, 128³)")

    # overlay anchors (real dice)
    if anchors_info is not None and len(anchors_info) > 0:
        ax = plt.gca()
        for k, info in anchors_info.items():
            p = float(info["params"])
            d = float(info["dice"])
            ax.scatter([p], [d], marker="X", s=180, edgecolors="black", linewidths=1.2, zorder=6)
            ax.annotate(
                info["name"],
                (p, d),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
                weight="bold",
                zorder=7,
            )

    # pareto front + knee on (params, gflops, dice) using the df points
    if draw_pareto and len(df) >= 3:
        df2 = df.copy()
        df2 = df2.dropna(subset=["params", "gflops_fwd_128", "best_val_mean_class_dice"])

        p = df2["params"].astype(float).values
        g = df2["gflops_fwd_128"].astype(float).values
        d = df2["best_val_mean_class_dice"].astype(float).values

        pf_idx = pareto_front_indices(p, g, d)
        df_pf = df2.iloc[pf_idx].copy()

        # plot pareto curve in (params, dice) space (sorted by params)
        df_pf = df_pf.sort_values("params").reset_index(drop=True)
        plt.plot(df_pf["params"].values, df_pf["best_val_mean_class_dice"].values, linewidth=2.0, zorder=5)

        # knee point
        knee = knee_point_on_pareto(df_pf)
        if knee is not None:
            kp = float(knee["params"])
            kd = float(knee["best_val_mean_class_dice"])
            plt.scatter([kp], [kd], s=140, marker="o", edgecolors="black", linewidths=1.1, zorder=7)
            plt.annotate(
                "knee",
                (kp, kd),
                textcoords="offset points",
                xytext=(8, -10),
                fontsize=10,
                weight="bold",
                zorder=8,
            )

    plt.grid(True, which="both", ls="--", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=190)
    plt.close()


# ----------------------------
# code generation
# ----------------------------
def write_codegen_py(out_path: str, best_spec: dict, full_epochs: int):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    best_spec = canonicalize_spec(best_spec)

    with open(out_path, "w") as f:
        f.write("# generated_best_model_train.py\n")
        f.write(f"# auto-generated at {ts}\n\n")
        f.write("import train_variants as base\n\n")
        f.write("MODEL_SPECS = {\n")
        f.write(f"    'UNet-AutoBest': {repr(best_spec)},\n")
        f.write("}\n\n")
        f.write("if __name__ == '__main__':\n")
        f.write("    base.MODEL_SPECS = MODEL_SPECS\n")
        f.write(f"    base.train_all(max_epochs={int(full_epochs)}, val_fraction=0.2, batch_size=1)\n")

    print(f"[codegen] wrote: {out_path}")


# ----------------------------
# iterative local search
# ----------------------------
def iterative_search(
    train_loader,
    val_loader,
    train_gen,
    start_spec: dict,
    lam: float,
    mu: float,
    proxy_epochs: int,
    search_iters: int,
    topk_eval: int,
    anchors_info=None,
    max_params=None,
    max_gflops=None,
    run_tag: str = "search",
    out_dir: str = "auto_search_out",
):
    os.makedirs(out_dir, exist_ok=True)

    visited = set()
    rows = []

    current_spec = canonicalize_spec(start_spec)
    best_global = None  # (score, res, spec)

    for it in range(1, search_iters + 1):
        neighbors = make_neighbors(current_spec)

        # pre-filter by cost
        candidates = []
        for s in neighbors:
            k = spec_key(s)
            if k in visited:
                continue
            visited.add(k)

            params, gflops = quick_model_cost(s)
            if not passes_budget(params, gflops, max_params, max_gflops):
                continue
            candidates.append((params, gflops, s))

        # evaluate lighter ones first
        candidates.sort(key=lambda x: (x[0], x[1] if x[1] >= 0 else 1e9))
        candidates = candidates[:topk_eval]

        print("\n" + "=" * 90)
        print(f"[{run_tag}] iter {it}/{search_iters} | eval={len(candidates)} | lam={lam} mu={mu}")
        print(f"[{run_tag}] current_spec={current_spec}")

        best_it = None  # (score, res, spec)

        for j, (params_est, gflops_est, spec) in enumerate(candidates, start=1):
            name = f"{run_tag}_it{it}_c{j}_d{len(spec['channels'])}_ru{spec['num_res_units']}"
            res = proxy_train_one(train_loader, val_loader, train_gen, spec, name, proxy_epochs)

            dice = float(res.get("best_val_mean_class_dice", -1.0))
            params = int(res.get("params", params_est))
            gflops = float(res.get("gflops_fwd_128", gflops_est))
            score = objective_score(dice, params, gflops, lam, mu)

            rows.append({
                "run_tag": run_tag,
                "iter": it,
                "cand": j,
                "lam": lam,
                "mu": mu,
                "score": score,
                "best_val_mean_class_dice": dice,
                "params": params,
                "gflops_fwd_128": gflops,
                "channels": str(tuple(spec["channels"])),
                "strides": str(tuple(spec["strides"])),
                "num_res_units": int(spec["num_res_units"]),
                "ckpt": res.get("ckpt", ""),
                "best_epoch": res.get("best_epoch", ""),
            })

            print(f"[score] {name} dice={dice:.4f} params={params} gflops={gflops:.3f} score={score:.4f}")

            if best_it is None or score > best_it[0]:
                best_it = (score, deepcopy(res), deepcopy(spec))
            if best_global is None or score > best_global[0]:
                best_global = (score, deepcopy(res), deepcopy(spec))

            # avoid excessive disk IO: flush/plot every N evals
            if (len(rows) % FLUSH_EVERY) == 0:
                df = pd.DataFrame(rows)
                df.to_csv(os.path.join(out_dir, f"{run_tag}_trials.csv"), index=False)

            if (len(rows) % PLOT_EVERY) == 0:
                df = pd.DataFrame(rows)
                plot_dense_tradeoff(
                    df,
                    title=f"Dense Search Progress ({run_tag})",
                    out_png=os.path.join(out_dir, f"{run_tag}_progress.png"),
                    anchors_info=anchors_info,
                    draw_pareto=True,
                )

        if best_it is None:
            print(f"[{run_tag}] no candidate evaluated; stopping")
            break

        current_spec = canonicalize_spec(best_it[2])
        print(f"[{run_tag}] move -> current_spec={current_spec} | best_it_score={best_it[0]:.4f}")

    if best_global is None:
        raise RuntimeError("search produced no evaluated candidates")

    best_score, best_res, best_spec = best_global
    best_spec = canonicalize_spec(best_spec)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"{run_tag}_trials.csv"), index=False)
    plot_dense_tradeoff(
        df,
        title=f"Dense Search Progress ({run_tag})",
        out_png=os.path.join(out_dir, f"{run_tag}_progress.png"),
        anchors_info=anchors_info,
        draw_pareto=True,
    )

    return best_score, best_res, best_spec, df


# ----------------------------
# lambda-mu tuning (outer loop)
# ----------------------------
def tune_lambda_mu(
    train_loader,
    val_loader,
    train_gen,
    start_spec: dict,
    anchors_info,
    proxy_epochs: int,
    search_iters: int,
    topk_eval: int,
    out_dir: str,
    max_params=None,
    max_gflops=None,
):
    os.makedirs(out_dir, exist_ok=True)

    lam_grid = [0.05, 0.10, 0.15, 0.20, 0.30]
    mu_grid  = [0.02, 0.05, 0.10, 0.15, 0.20]

    summary = []

    for lam in lam_grid:
        for mu in mu_grid:
            run_tag = f"tune_lam{lam}_mu{mu}".replace(".", "p")
            print("\n" + "#" * 90)
            print(f"[tune] {run_tag}")

            best_score, best_res, best_spec, _ = iterative_search(
                train_loader=train_loader,
                val_loader=val_loader,
                train_gen=train_gen,
                start_spec=start_spec,
                lam=lam,
                mu=mu,
                proxy_epochs=proxy_epochs,
                search_iters=search_iters,
                topk_eval=topk_eval,
                anchors_info=anchors_info,
                max_params=max_params,
                max_gflops=max_gflops,
                run_tag=run_tag,
                out_dir=out_dir,
            )

            summary.append({
                "lam": lam,
                "mu": mu,
                "best_score": float(best_score),
                "best_dice": float(best_res.get("best_val_mean_class_dice", -1.0)),
                "best_params": int(best_res.get("params", -1)),
                "best_gflops": float(best_res.get("gflops_fwd_128", -1.0)),
                "best_spec": json.dumps(best_spec),
                "run_tag": run_tag,
            })
            pd.DataFrame(summary).to_csv(os.path.join(out_dir, "lambda_mu_summary.csv"), index=False)

    sum_df = pd.DataFrame(summary).sort_values("best_score", ascending=False).reset_index(drop=True)
    sum_df.to_csv(os.path.join(out_dir, "lambda_mu_summary.csv"), index=False)
    print("\n[tune] saved lambda_mu_summary.csv")
    print(sum_df.head(10))
    return sum_df



def take_subset(files, frac: float, seed: int):
    """take a fixed random subset for proxy search.

    keeps order reproducible (seeded shuffle).
    """
    files = list(files)
    rng = random.Random(int(seed))
    rng.shuffle(files)
    n = max(1, int(round(len(files) * float(frac))))
    return files[:n]


def parse_spec_row(row: pd.Series):
    """parse spec fields saved in csv rows back into a spec dict."""
    ch = ast.literal_eval(str(row["channels"]))
    st = ast.literal_eval(str(row["strides"]))
    ru = int(row["num_res_units"])
    return canonicalize_spec({"channels": tuple(ch), "strides": tuple(st), "num_res_units": ru})


def rerank_topk_on_full_data(
    df_trials: pd.DataFrame,
    train_loader_full,
    val_loader_full,
    train_gen_full,
    lam: float,
    mu: float,
    k: int,
    proxy_epochs: int,
    out_dir: str,
    tag: str = "rerank_topk",
):
    """re-evaluate top-k specs on full training split to reduce proxy bias.

    selection for re-eval uses the score from df_trials (subset search).
    """
    os.makedirs(out_dir, exist_ok=True)

    df = df_trials.copy()
    if "score" not in df.columns:
        raise RuntimeError("df_trials must contain 'score' column")

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df_k = df.head(int(k)).copy()

    rerows = []
    best = None  # (score, res, spec)

    print("\n" + "=" * 90)
    print(f"[rerank] re-evaluating top-{k} specs on FULL train split | epochs={proxy_epochs}")

    for i, row in df_k.iterrows():
        spec = parse_spec_row(row)
        name = f"{tag}_{i+1:02d}"
        res = proxy_train_one(train_loader_full, val_loader_full, train_gen_full, spec, name, proxy_epochs)

        dice = float(res.get("best_val_mean_class_dice", -1.0))
        params = int(res.get("params", -1))
        gflops = float(res.get("gflops_fwd_128", -1.0))
        score = objective_score(dice, params, gflops, lam, mu)

        rerows.append({
            "rank_in_subset": int(i + 1),
            "lam": float(lam),
            "mu": float(mu),
            "score": float(score),
            "best_val_mean_class_dice": float(dice),
            "params": int(params),
            "gflops_fwd_128": float(gflops),
            "channels": str(tuple(spec["channels"])),
            "strides": str(tuple(spec["strides"])),
            "num_res_units": int(spec["num_res_units"]),
            "ckpt": res.get("ckpt", ""),
            "best_epoch": res.get("best_epoch", ""),
        })

        print(f"[rerank] {name}: dice={dice:.4f} params={params} gflops={gflops:.3f} score={score:.4f}")

        if best is None or score > best[0]:
            best = (score, deepcopy(res), deepcopy(spec))

    df_r = pd.DataFrame(rerows)
    csv_path = os.path.join(out_dir, f"{tag}.csv")
    df_r.to_csv(csv_path, index=False)
    print(f"[rerank] saved: {csv_path}")

    # optional: also plot rerank cloud
    if len(df_r) >= 3:
        plot_dense_tradeoff(
            df_r,
            title=f"Re-rank Top-{k} on Full Data",
            out_png=os.path.join(out_dir, f"{tag}.png"),
            anchors_info=None,
            draw_pareto=True,
        )

    if best is None:
        raise RuntimeError("rerank produced no evaluated models")

    return best[0], best[1], best[2], df_r


def main():
    OUT_DIR = "auto_search_out"

    # optional hard budgets (set if you want guaranteed lightness)
    # example: MAX_PARAMS = 2_000_000 ; MAX_GFLOPS = 15.0
    MAX_PARAMS = None
    MAX_GFLOPS = None

    # use only a subset of training cases for proxy search (speed)
    SEARCH_TRAIN_FRAC = 0.30  # 30% of train split

    # after subset-search, re-evaluate top-k specs on full train split
    TOPK_RERANK = 10
    RERANK_PROXY_EPOCHS = 6  # keep this modest to control total time

    # anchor evaluation epochs (1 or 2 is enough)
    ANCHOR_EPOCHS = 2

    # tuning (cheap)
    TUNE_PROXY_EPOCHS = 6
    TUNE_ITERS = 3
    TUNE_TOPK = 6

    # final search (stronger)
    FINAL_PROXY_EPOCHS = 10
    FINAL_ITERS = 6
    FINAL_TOPK = 8

    # full training epochs for generated best model
    FULL_EPOCHS = getattr(base, "MAX_EPOCHS", 100)

    # prefer throughput on the lab GPU unless you explicitly need strict determinism
    if hasattr(base, "STRICT_GPU_DETERMINISM") and FAST_BUT_NOT_STRICTLY_DETERMINISTIC:
        base.STRICT_GPU_DETERMINISM = False

    random.seed(base.SEED)
    np.random.seed(base.SEED)

    # dataset split once
    data_dicts = base.build_datalist(base.DATA_ROOT)
    train_files, val_files = base.split_train_val(
        data_dicts, val_fraction=0.2, seed=base.SEED, save_split=True, save_dir=OUT_DIR
    )

    # fixed subset for search
    train_files_search = take_subset(train_files, frac=SEARCH_TRAIN_FRAC, seed=base.SEED)
    print(f"\n[setup] proxy search uses {len(train_files_search)}/{len(train_files)} train cases (frac={SEARCH_TRAIN_FRAC})")

    # save subset lists (for reporting / reproducibility)
    with open(os.path.join(OUT_DIR, f"train_cases_search_frac{int(SEARCH_TRAIN_FRAC*100)}_seed{base.SEED}.txt"), "w") as f:
        for d in train_files_search:
            f.write(str(d.get("image", d)) + "\n")

    # build loaders once (huge speedup)
    print("\n[setup] creating persistent DataLoaders (once)...")
    # loaders for proxy search (subset train, full val)
    train_loader_s, val_loader, train_gen_s = make_persistent_loaders(train_files_search, val_files, batch_size=1)
    # loaders for full-data rerank (full train, full val)
    train_loader_full, val_loader_full, train_gen_full = make_persistent_loaders(train_files, val_files, batch_size=1)

    # anchors from backend
    anchors = get_anchor_specs_from_backend()
    start_spec = choose_start_spec(anchors)

    print("\nAnchors loaded from base.MODEL_SPECS:")
    for n, s in anchors:
        print(f"  - {n}: {s}")
    print("\nStart spec:", start_spec)

    # 0) evaluate anchors (cached) on the subset loader (saves time)
    anchors_info = evaluate_anchors_if_needed(
        train_loader=train_loader_s,
        val_loader=val_loader,
        train_gen=train_gen_s,
        anchors=anchors,
        out_dir=OUT_DIR,
        anchor_epochs=ANCHOR_EPOCHS,
    )

    # 1) tune lambda/mu on subset train (speed)
    sum_df = tune_lambda_mu(
        train_loader=train_loader_s,
        val_loader=val_loader,
        train_gen=train_gen_s,
        start_spec=start_spec,
        anchors_info=anchors_info,
        proxy_epochs=TUNE_PROXY_EPOCHS,
        search_iters=TUNE_ITERS,
        topk_eval=TUNE_TOPK,
        out_dir=OUT_DIR,
        max_params=MAX_PARAMS,
        max_gflops=MAX_GFLOPS,
    )

    best_lam = float(sum_df.iloc[0]["lam"])
    best_mu = float(sum_df.iloc[0]["mu"])
    print(f"\n[chosen] lam={best_lam} mu={best_mu}")

    # 2) final search on subset train
    best_score, best_res, best_spec, df_final = iterative_search(
        train_loader=train_loader_s,
        val_loader=val_loader,
        train_gen=train_gen_s,
        start_spec=start_spec,
        lam=best_lam,
        mu=best_mu,
        proxy_epochs=FINAL_PROXY_EPOCHS,
        search_iters=FINAL_ITERS,
        topk_eval=FINAL_TOPK,
        anchors_info=anchors_info,
        max_params=MAX_PARAMS,
        max_gflops=MAX_GFLOPS,
        run_tag="final_search",
        out_dir=OUT_DIR,
    )
    print(f"\n[final_search] best_score={best_score:.4f} best_spec={best_spec}")

    # 3) rerank top-k on full train split (reduces proxy bias)
    r_score, r_res, r_spec, df_r = rerank_topk_on_full_data(
        df_trials=df_final,
        train_loader_full=train_loader_full,
        val_loader_full=val_loader_full,
        train_gen_full=train_gen_full,
        lam=best_lam,
        mu=best_mu,
        k=TOPK_RERANK,
        proxy_epochs=RERANK_PROXY_EPOCHS,
        out_dir=OUT_DIR,
        tag=f"rerank_top{TOPK_RERANK}_full",
    )
    print(f"\n[rerank] best_score={r_score:.4f} best_spec={r_spec}")

    # 4) write final chosen spec + codegen for full training
    best_spec_path = os.path.join(OUT_DIR, "best_spec.json")
    with open(best_spec_path, "w") as f:
        json.dump(r_spec, f, indent=2)
    print(f"[final] saved: {best_spec_path}")

    write_codegen_py(os.path.join(OUT_DIR, "generated_best_model_train.py"), r_spec, full_epochs=FULL_EPOCHS)
    print("\n[done]")


if __name__ == "__main__":
    main()
