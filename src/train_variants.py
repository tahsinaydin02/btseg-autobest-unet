import os
from glob import glob
import csv
import time
import random

import numpy as np
import torch
from monai.data import Dataset, DataLoader, decollate_batch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    MapLabelValued,
    SpatialPadd,
    AsDiscrete,
)
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism


# ------------------------------------------------------------
# config
# ------------------------------------------------------------
# Path to the BraTS 2020 training set. The dataset is NOT redistributed with
# this repository (see README). Override with the BRATS_DATA_ROOT env var:
#     export BRATS_DATA_ROOT=/path/to/MICCAI_BraTS2020_TrainingData
DATA_ROOT = os.environ.get(
    "BRATS_DATA_ROOT",
    os.path.join("data", "MICCAI_BraTS2020_TrainingData"),
)
SEED = 42

# Use >0 to parallelize data loading/augmentation.
NUM_WORKERS = 4

# Sliding window inference batch size (validation speed / VRAM tradeoff).
SW_BATCH_SIZE = 2

# If True: more reproducible but can slow down GPU.
# If False: keep GPU fast; determinism mainly comes from seeds + worker seeding + rebuilding loaders.
STRICT_GPU_DETERMINISM = False

# Training controls
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 20
MIN_DELTA = 0.0  # improvement threshold for early stopping (0.0 = any improvement)

# GFLOPs input (patch) shape: keep constant across models for fair comparison
GFLOPS_INPUT_SHAPE = (1, 4, 128, 128, 128)

MODEL_SPECS = {
    "UNet-Baseline": {
        "channels": (32, 64, 128, 256, 512),
        "strides": (2, 2, 2, 2),
        "num_res_units": 2,
    },
    "UNet-Lite": {
        "channels": (16, 32, 64, 128, 256),
        "strides": (2, 2, 2, 2),
        "num_res_units": 2,
    },
    "UNet-Ultralight": {
        "channels": (16, 32, 64, 128),
        "strides": (2, 2, 2),
        "num_res_units": 1,
    },
    "UNet-Efficient": {
        "channels": (16, 32, 64, 128),
        "strides": (2, 2, 2),
        "num_res_units": 0,
    },
    "UNet-Tiny": {
        "channels": (8, 16, 32, 64),
        "strides": (2, 2, 2),
        "num_res_units": 0,
    },
}


# ------------------------------------------------------------
# data
# ------------------------------------------------------------
def build_datalist(data_root: str):
    pattern = os.path.join(data_root, "BraTS20_Training_*")
    case_dirs = sorted([d for d in glob(pattern) if os.path.isdir(d)])
    data_dicts = []

    for case_dir in case_dirs:
        cid = os.path.basename(case_dir)

        flair = os.path.join(case_dir, cid + "_flair.nii")
        t1 = os.path.join(case_dir, cid + "_t1.nii")
        t1ce = os.path.join(case_dir, cid + "_t1ce.nii")
        t2 = os.path.join(case_dir, cid + "_t2.nii")
        seg = os.path.join(case_dir, cid + "_seg.nii")

        if not (
            os.path.exists(flair)
            and os.path.exists(t1)
            and os.path.exists(t1ce)
            and os.path.exists(t2)
            and os.path.exists(seg)
        ):
            print("Missing modality, skipped:", cid)
            continue

        data_dicts.append({"image": [flair, t1, t1ce, t2], "label": seg, "case_id": cid})

    print("Total usable cases:", len(data_dicts))
    return data_dicts


def split_train_val(data_dicts, val_fraction=0.2, seed=SEED, save_split=True, save_dir="."):
    n = len(data_dicts)
    indices = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    split = int((1.0 - val_fraction) * n)
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_files = [data_dicts[i] for i in train_idx]
    val_files = [data_dicts[i] for i in val_idx]

    print(
        f"Split: {len(train_files)} train, {len(val_files)} val "
        f"(val_fraction={val_fraction}, seed={seed})"
    )

    if save_split:
        os.makedirs(save_dir, exist_ok=True)
        train_txt = os.path.join(save_dir, f"train_cases_seed{seed}.txt")
        val_txt = os.path.join(save_dir, f"val_cases_seed{seed}.txt")

        with open(train_txt, "w") as f_train:
            for i in train_idx:
                f_train.write(data_dicts[i]["case_id"] + "\n")

        with open(val_txt, "w") as f_val:
            for i in val_idx:
                f_val.write(data_dicts[i]["case_id"] + "\n")

        print(f"Saved train split to: {train_txt}")
        print(f"Saved val split to:   {val_txt}")

    return train_files, val_files


def make_transforms():
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            MapLabelValued(keys="label", orig_labels=[0, 1, 2, 4], target_labels=[0, 1, 2, 3]),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(128, 128, 128),
                pos=1,
                neg=1,
                num_samples=2,
                allow_smaller=True,
            ),
            SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 128)),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandAffined(
                keys=["image", "label"],
                prob=0.3,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "nearest"),
            ),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            MapLabelValued(keys="label", orig_labels=[0, 1, 2, 4], target_labels=[0, 1, 2, 3]),
        ]
    )
    return train_transforms, val_transforms


def seed_worker(worker_id: int):
    # make numpy/python/torch rng in this worker deterministic given the DataLoader base seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_loaders(train_files, val_files, batch_size: int):
    train_transforms, val_transforms = make_transforms()
    train_ds = Dataset(train_files, transform=train_transforms)
    val_ds = Dataset(val_files, transform=val_transforms)

    g = torch.Generator()
    g.manual_seed(SEED)

    pin = bool(torch.cuda.is_available())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=False,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        persistent_workers=False,
        pin_memory=pin,
    )
    return train_loader, val_loader


# ------------------------------------------------------------
# model utils
# ------------------------------------------------------------
def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_model(spec, device):
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=spec["channels"],
        strides=spec["strides"],
        num_res_units=spec["num_res_units"],
    ).to(device)
    return model


def compute_gflops(model: torch.nn.Module, device: torch.device, input_shape=GFLOPS_INPUT_SHAPE) -> float:
    """
    Compute approximate GFLOPs for a single forward pass.
    - input_shape should match your training/validation patch.
    - Returns GFLOPs (1e9 FLOPs). If a FLOPs lib isn't available, returns -1.0.

    Notes:
    - Different libs may report MACs vs FLOPs; here we try to report FLOPs.
    - Keep input_shape constant for fair comparison.
    """
    # try fvcore (reports FLOPs)
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore

        model.eval()
        with torch.no_grad():
            x = torch.zeros(*input_shape, device=device)
            flops = FlopCountAnalysis(model, x).total()
        return float(flops) / 1e9
    except Exception:
        pass

    # fallback: ptflops (often reports MACs; we convert MACs->FLOPs by *2)
    try:
        from ptflops import get_model_complexity_info  # type: ignore

        inp = input_shape[1:]  # CHWD
        with torch.no_grad():
            macs, _ = get_model_complexity_info(
                model,
                inp,
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False,
            )
        flops = 2.0 * float(macs)
        return flops / 1e9
    except Exception:
        return -1.0


def set_all_seeds(seed: int):
    set_determinism(seed=seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if STRICT_GPU_DETERMINISM:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print("Warning: torch.use_deterministic_algorithms(True) failed:", e)
    else:
        torch.backends.cudnn.benchmark = True


# ------------------------------------------------------------
# dice helpers
# ------------------------------------------------------------
def dice_binary(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> float:
    """
    pred, gt: binary tensors (0/1) with same shape.
    returns dice in [0,1].
    """
    pred = pred.float()
    gt = gt.float()
    inter = torch.sum(pred * gt)
    denom = torch.sum(pred) + torch.sum(gt)
    if denom.item() == 0.0:
        return 1.0
    return float((2.0 * inter + eps) / (denom + eps))


def compute_class_region_dice(pred_onehot: torch.Tensor, gt_onehot: torch.Tensor):
    """
    pred_onehot, gt_onehot: shape (4, H, W, D) one-hot (classes 0..3).
    returns dict with:
      class_dice_1, class_dice_2, class_dice_3
      wt_dice, tc_dice, et_dice
    """
    # class-wise (labels 1,2,3)
    c1 = dice_binary(pred_onehot[1], gt_onehot[1])
    c2 = dice_binary(pred_onehot[2], gt_onehot[2])
    c3 = dice_binary(pred_onehot[3], gt_onehot[3])

    # region-wise (WT/TC/ET)
    # WT = (1|2|3), TC = (1|3), ET = (3)
    wt_pred = (pred_onehot[1] + pred_onehot[2] + pred_onehot[3]) > 0
    wt_gt = (gt_onehot[1] + gt_onehot[2] + gt_onehot[3]) > 0
    tc_pred = (pred_onehot[1] + pred_onehot[3]) > 0
    tc_gt = (gt_onehot[1] + gt_onehot[3]) > 0
    et_pred = pred_onehot[3] > 0
    et_gt = gt_onehot[3] > 0

    wt = dice_binary(wt_pred, wt_gt)
    tc = dice_binary(tc_pred, tc_gt)
    et = dice_binary(et_pred, et_gt)

    return {
        "class_dice_1": c1,
        "class_dice_2": c2,
        "class_dice_3": c3,
        "wt_dice": wt,
        "tc_dice": tc,
        "et_dice": et,
    }


# ------------------------------------------------------------
# train/val
# ------------------------------------------------------------
def train_one(model_name: str, spec: dict, train_loader, val_loader, max_epochs: int = MAX_EPOCHS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 60)
    print(f"Model: {model_name}")
    print("Device:", device)
    print("Spec:", spec)

    model = make_model(spec, device)
    n_params = count_trainable_params(model)
    gflops = compute_gflops(model, device)
    print("Trainable params:", n_params)
    if gflops >= 0:
        print(f"GFLOPs (forward, {GFLOPS_INPUT_SHAPE}): {gflops:.3f}")
    else:
        print("GFLOPs: unavailable (install fvcore or ptflops to compute)")

    loss_fn = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # cosine LR schedule (epoch-wise)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=1e-6,
    )

    post_pred = AsDiscrete(argmax=True, to_onehot=4)
    post_label = AsDiscrete(to_onehot=4)

    best_val_mean_class_dice = -1.0
    best_epoch = -1
    best_snapshot = None
    bad_epochs = 0

    # final snapshot (last epoch)
    last_snapshot = None

    ckpt_name = f"brats2020_unet_{model_name}_best_80_20.pth"

    t0 = time.time()

    stop_epoch = 0

    for epoch in range(1, max_epochs + 1):
        stop_epoch = epoch
        lr_now = float(optimizer.param_groups[0]["lr"])
        print("-" * 20)
        print(f"Epoch {epoch} / {max_epochs} | lr={lr_now:.6g}")

        # ---------------- train ----------------
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if step % 10 == 0:
                print(f"  Step {step} loss= {loss.item():.4f}")

        epoch_loss /= max(1, step)
        print(f"Epoch {epoch} average loss: {epoch_loss:.4f}")

        # ---------------- validation ----------------
        model.eval()

        cls1_list, cls2_list, cls3_list = [], [], []
        wt_list, tc_list, et_list = [], [], []

        with torch.no_grad():
            for val_batch in val_loader:
                vimg = val_batch["image"].to(device, non_blocking=True)
                vlab = val_batch["label"].to(device, non_blocking=True)

                preds = sliding_window_inference(
                    vimg,
                    roi_size=(128, 128, 128),
                    sw_batch_size=SW_BATCH_SIZE,
                    predictor=model,
                )

                preds_list = decollate_batch(preds)
                vlab_list = decollate_batch(vlab)

                preds_proc = [post_pred(p) for p in preds_list]
                vlab_proc = [post_label(l) for l in vlab_list]

                for p_onehot, g_onehot in zip(preds_proc, vlab_proc):
                    d = compute_class_region_dice(p_onehot, g_onehot)
                    cls1_list.append(d["class_dice_1"])
                    cls2_list.append(d["class_dice_2"])
                    cls3_list.append(d["class_dice_3"])
                    wt_list.append(d["wt_dice"])
                    tc_list.append(d["tc_dice"])
                    et_list.append(d["et_dice"])

        mean_c1 = float(np.mean(cls1_list)) if len(cls1_list) else 0.0
        mean_c2 = float(np.mean(cls2_list)) if len(cls2_list) else 0.0
        mean_c3 = float(np.mean(cls3_list)) if len(cls3_list) else 0.0
        mean_class_dice = float(np.mean([mean_c1, mean_c2, mean_c3]))

        mean_wt = float(np.mean(wt_list)) if len(wt_list) else 0.0
        mean_tc = float(np.mean(tc_list)) if len(tc_list) else 0.0
        mean_et = float(np.mean(et_list)) if len(et_list) else 0.0

        print(
            "  -> Class Dice (1,2,3): "
            f"{mean_c1:.4f}, {mean_c2:.4f}, {mean_c3:.4f} | mean={mean_class_dice:.4f}"
        )
        print(
            "  -> Region Dice (WT,TC,ET): "
            f"{mean_wt:.4f}, {mean_tc:.4f}, {mean_et:.4f}"
        )

        last_snapshot = {
            "final_epoch": epoch,
            "final_lr": lr_now,
            "final_val_mean_class_dice": mean_class_dice,
            "final_val_class_dice_1": mean_c1,
            "final_val_class_dice_2": mean_c2,
            "final_val_class_dice_3": mean_c3,
            "final_val_wt_dice": mean_wt,
            "final_val_tc_dice": mean_tc,
            "final_val_et_dice": mean_et,
        }

        # best checkpoint according to mean class dice
        if mean_class_dice > (best_val_mean_class_dice + MIN_DELTA):
            best_val_mean_class_dice = mean_class_dice
            best_epoch = epoch
            bad_epochs = 0

            torch.save(model.state_dict(), ckpt_name)
            best_snapshot = {
                "best_val_mean_class_dice": best_val_mean_class_dice,
                "best_val_class_dice_1": mean_c1,
                "best_val_class_dice_2": mean_c2,
                "best_val_class_dice_3": mean_c3,
                "best_val_wt_dice": mean_wt,
                "best_val_tc_dice": mean_tc,
                "best_val_et_dice": mean_et,
                "best_lr": lr_now,
            }
            print(f"  -> Best model updated! epoch={epoch}, mean_class_dice={mean_class_dice:.4f}")
        else:
            bad_epochs += 1
            print(f"  -> No improvement. early-stop counter: {bad_epochs}/{EARLY_STOP_PATIENCE}")

        # scheduler step (epoch-wise)
        scheduler.step()

        # early stopping check
        if bad_epochs >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs).")
            break

    elapsed_s = time.time() - t0
    print(
        f"Done: {model_name} | best epoch: {best_epoch} | best mean_class_dice: {best_val_mean_class_dice:.4f}"
    )

    # free vram
    del model, optimizer, loss_fn, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # flatten model spec for reporting
    channels = spec["channels"]
    strides = spec["strides"]
    depth = len(channels)

    result = {
        "model": model_name,
        "params": int(n_params),
        "gflops_fwd_128": float(gflops),
        "channels": str(tuple(channels)),
        "strides": str(tuple(strides)),
        "depth": int(depth),
        "num_res_units": int(spec["num_res_units"]),
        "seed": int(SEED),
        "num_workers": int(NUM_WORKERS),
        "sw_batch_size": int(SW_BATCH_SIZE),
        "max_epochs": int(max_epochs),
        "early_stop_patience": int(EARLY_STOP_PATIENCE),
        "epochs_trained": int(stop_epoch),
        "stop_epoch": int(stop_epoch),
        "best_epoch": int(best_epoch),
        "ckpt": ckpt_name,
        "elapsed_s": float(elapsed_s),
    }

    if best_snapshot is not None:
        result.update(best_snapshot)
    if last_snapshot is not None:
        result.update(last_snapshot)

    # optional derived metrics for easy comparison in Excel
    if gflops > 0:
        result["best_mean_dice_per_gflop"] = float(result.get("best_val_mean_class_dice", 0.0)) / float(gflops)
        result["best_wt_dice_per_gflop"] = float(result.get("best_val_wt_dice", 0.0)) / float(gflops)
        result["best_tc_dice_per_gflop"] = float(result.get("best_val_tc_dice", 0.0)) / float(gflops)
        result["best_et_dice_per_gflop"] = float(result.get("best_val_et_dice", 0.0)) / float(gflops)
    else:
        result["best_mean_dice_per_gflop"] = ""
        result["best_wt_dice_per_gflop"] = ""
        result["best_tc_dice_per_gflop"] = ""
        result["best_et_dice_per_gflop"] = ""

    return result


def train_all(max_epochs: int = MAX_EPOCHS, val_fraction: float = 0.2, batch_size: int = 1):
    data_dicts = build_datalist(DATA_ROOT)
    if len(data_dicts) == 0:
        print("ERROR: No data found. Check DATA_ROOT.")
        return

    train_files, val_files = split_train_val(
        data_dicts, val_fraction=val_fraction, seed=SEED, save_split=True, save_dir="."
    )

    results = []
    for model_name, spec in MODEL_SPECS.items():
        set_all_seeds(SEED)

        # rebuild transforms/datasets/loaders per model to reset random augmentation streams
        train_loader, val_loader = make_loaders(
            train_files=train_files,
            val_files=val_files,
            batch_size=batch_size,
        )

        r = train_one(
            model_name=model_name,
            spec=spec,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=max_epochs,
        )
        results.append(r)

    out_csv = "results_unet_variants.csv"

    fieldnames = [
        # identifiers + config
        "model",
        "seed",
        "num_workers",
        "sw_batch_size",
        "max_epochs",
        "early_stop_patience",
        # architecture
        "channels",
        "strides",
        "depth",
        "num_res_units",
        # efficiency
        "params",
        "gflops_fwd_128",
        "elapsed_s",
        "epochs_trained",
        "stop_epoch",
        # best
        "best_epoch",
        "best_lr",
        "best_val_mean_class_dice",
        "best_val_class_dice_1",
        "best_val_class_dice_2",
        "best_val_class_dice_3",
        "best_val_wt_dice",
        "best_val_tc_dice",
        "best_val_et_dice",
        # final
        "final_epoch",
        "final_lr",
        "final_val_mean_class_dice",
        "final_val_class_dice_1",
        "final_val_class_dice_2",
        "final_val_class_dice_3",
        "final_val_wt_dice",
        "final_val_tc_dice",
        "final_val_et_dice",
        # derived
        "best_mean_dice_per_gflop",
        "best_wt_dice_per_gflop",
        "best_tc_dice_per_gflop",
        "best_et_dice_per_gflop",
        # artifacts
        "ckpt",
    ]

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            w.writerow(row)

    print("\nSaved summary:", out_csv)
    for r in results:
        print(r)


if __name__ == "__main__":
    train_all(max_epochs=MAX_EPOCHS, val_fraction=0.2, batch_size=1)