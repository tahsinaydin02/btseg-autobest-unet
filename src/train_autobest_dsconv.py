"""
train_autobest_dsconv.py
========================
EE492 – Depthwise Separable Convolution (DSConv) applied to UNet-AutoBest.

This script is a self-contained training file that:
  1. Builds a 3D UNet with the AutoBest topology  (24, 32, 80)  /  strides (2,2)
  2. Replaces ALL standard Conv3d layers inside the encoder/decoder residual
     blocks with DepthwiseSeparableConv3d (depthwise + pointwise).
  3. Trains on BraTS2020 with the EXACT same pipeline, hyperparameters,
     augmentations, loss, scheduler, and evaluation as the EE491 experiments
     so that the comparison is perfectly controlled.

Run:
    python train_autobest_dsconv.py

Outputs:
    - brats2020_unet_UNet-AutoBest-DSConv_best_80_20.pth   (best checkpoint)
    - results_dsconv.csv                                     (metrics summary)
"""

import os
from glob import glob
import csv
import time
import random

import numpy as np
import torch
import torch.nn as nn
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


# ============================================================
# CONFIG  –  keep identical to EE491 for fair comparison
# ============================================================
# Path to the BraTS 2020 training set. The dataset is NOT redistributed with
# this repository (see README). Override with the BRATS_DATA_ROOT env var:
#     export BRATS_DATA_ROOT=/path/to/MICCAI_BraTS2020_TrainingData
DATA_ROOT = os.environ.get(
    "BRATS_DATA_ROOT",
    os.path.join("data", "MICCAI_BraTS2020_TrainingData"),
)
SEED = 42
NUM_WORKERS = 4
SW_BATCH_SIZE = 2
STRICT_GPU_DETERMINISM = False
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 20
MIN_DELTA = 0.0
GFLOPS_INPUT_SHAPE = (1, 4, 128, 128, 128)

# AutoBest spec from EE491
AUTOBEST_SPEC = {
    "channels": (24, 32, 80),
    "strides": (2, 2),
    "num_res_units": 2,
}


# ============================================================
# DEPTHWISE SEPARABLE CONV3D MODULE
# ============================================================
class DepthwiseSeparableConv3d(nn.Module):
    """
    Depthwise Separable 3D Convolution.

    Factorises a standard Conv3d(C_in, C_out, k, s, p) into:
      1. Depthwise  : Conv3d(C_in, C_in, k, s, p, groups=C_in)
                      – one filter per input channel (spatial filtering)
      2. Pointwise  : Conv3d(C_in, C_out, 1)
                      – 1×1×1 convolution   (channel mixing)

    FLOPs ratio vs standard conv  ≈  1/C_out  +  1/k³
    For k=3, C_out=80 :  ≈  1/80 + 1/27  ≈  0.050  →  ~20× cheaper
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,  # bias only after pointwise
        )
        self.pointwise = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


# ============================================================
# MONKEY-PATCH: replace Conv3d inside a MONAI UNet
# ============================================================
def _replace_conv3d_with_dsconv(module: nn.Module) -> nn.Module:
    """
    Recursively walk the module tree.  For every nn.Conv3d whose
    kernel_size > 1  (i.e. spatial convolutions, NOT 1×1 projections
    or the final segmentation head), replace it with DSConv.

    We intentionally keep 1×1 convolutions as-is because:
      - They are already pointwise (no spatial component to factorise).
      - The residual-unit skip projections and the final head are 1×1.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv3d):
            k = child.kernel_size
            # only replace spatial convolutions (kernel > 1 in any dim)
            if any(ki > 1 for ki in k):
                new_conv = DepthwiseSeparableConv3d(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=k[0],       # MONAI uses isotropic kernels
                    stride=child.stride[0],
                    padding=child.padding[0],
                    bias=(child.bias is not None),
                )
                setattr(module, name, new_conv)
        else:
            # recurse into sub-modules
            _replace_conv3d_with_dsconv(child)
    return module


def make_model_dsconv(spec: dict, device: torch.device) -> nn.Module:
    """
    1. Build a standard MONAI UNet with the given spec.
    2. Replace all spatial Conv3d with DepthwiseSeparableConv3d.
    3. Move to device.
    """
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=spec["channels"],
        strides=spec["strides"],
        num_res_units=spec["num_res_units"],
    )
    model = _replace_conv3d_with_dsconv(model)
    return model.to(device)


# ============================================================
# DATA  –  identical to EE491
# ============================================================
def build_datalist(data_root: str):
    pattern = os.path.join(data_root, "BraTS20_Training_*")
    case_dirs = sorted([d for d in glob(pattern) if os.path.isdir(d)])
    data_dicts = []
    for case_dir in case_dirs:
        cid = os.path.basename(case_dir)
        flair = os.path.join(case_dir, cid + "_flair.nii")
        t1    = os.path.join(case_dir, cid + "_t1.nii")
        t1ce  = os.path.join(case_dir, cid + "_t1ce.nii")
        t2    = os.path.join(case_dir, cid + "_t2.nii")
        seg   = os.path.join(case_dir, cid + "_seg.nii")
        if not all(os.path.exists(f) for f in [flair, t1, t1ce, t2, seg]):
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
    train_idx, val_idx = indices[:split], indices[split:]
    train_files = [data_dicts[i] for i in train_idx]
    val_files   = [data_dicts[i] for i in val_idx]
    print(f"Split: {len(train_files)} train, {len(val_files)} val (val_fraction={val_fraction}, seed={seed})")
    if save_split:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"train_cases_seed{seed}.txt"), "w") as f:
            for i in train_idx:
                f.write(data_dicts[i]["case_id"] + "\n")
        with open(os.path.join(save_dir, f"val_cases_seed{seed}.txt"), "w") as f:
            for i in val_idx:
                f.write(data_dicts[i]["case_id"] + "\n")
    return train_files, val_files


def make_transforms():
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        MapLabelValued(keys="label", orig_labels=[0, 1, 2, 4], target_labels=[0, 1, 2, 3]),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label",
            spatial_size=(128, 128, 128), pos=1, neg=1,
            num_samples=2, allow_smaller=True,
        ),
        SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 128)),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandAffined(
            keys=["image", "label"], prob=0.3,
            rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1),
            mode=("bilinear", "nearest"),
        ),
    ])
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        MapLabelValued(keys="label", orig_labels=[0, 1, 2, 4], target_labels=[0, 1, 2, 3]),
    ])
    return train_transforms, val_transforms


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_loaders(train_files, val_files, batch_size: int):
    train_transforms, val_transforms = make_transforms()
    train_ds = Dataset(train_files, transform=train_transforms)
    val_ds   = Dataset(val_files, transform=val_transforms)
    g = torch.Generator()
    g.manual_seed(SEED)
    pin = bool(torch.cuda.is_available())
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, worker_init_fn=seed_worker,
        generator=g, persistent_workers=False, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=NUM_WORKERS, worker_init_fn=seed_worker,
        persistent_workers=False, pin_memory=pin,
    )
    return train_loader, val_loader


# ============================================================
# UTILS  –  identical to EE491
# ============================================================
def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_gflops(model: nn.Module, device: torch.device, input_shape=GFLOPS_INPUT_SHAPE) -> float:
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        with torch.no_grad():
            x = torch.zeros(*input_shape, device=device)
            flops = FlopCountAnalysis(model, x).total()
        return float(flops) / 1e9
    except Exception:
        pass
    try:
        from ptflops import get_model_complexity_info
        inp = input_shape[1:]
        with torch.no_grad():
            macs, _ = get_model_complexity_info(
                model, inp, as_strings=False,
                print_per_layer_stat=False, verbose=False,
            )
        return 2.0 * float(macs) / 1e9
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
            print("Warning:", e)
    else:
        torch.backends.cudnn.benchmark = True


# ============================================================
# DICE HELPERS  –  identical to EE491
# ============================================================
def dice_binary(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> float:
    pred, gt = pred.float(), gt.float()
    inter = torch.sum(pred * gt)
    denom = torch.sum(pred) + torch.sum(gt)
    if denom.item() == 0.0:
        return 1.0
    return float((2.0 * inter + eps) / (denom + eps))


def compute_class_region_dice(pred_onehot: torch.Tensor, gt_onehot: torch.Tensor):
    c1 = dice_binary(pred_onehot[1], gt_onehot[1])
    c2 = dice_binary(pred_onehot[2], gt_onehot[2])
    c3 = dice_binary(pred_onehot[3], gt_onehot[3])
    wt_pred = (pred_onehot[1] + pred_onehot[2] + pred_onehot[3]) > 0
    wt_gt   = (gt_onehot[1]  + gt_onehot[2]  + gt_onehot[3])  > 0
    tc_pred = (pred_onehot[1] + pred_onehot[3]) > 0
    tc_gt   = (gt_onehot[1]  + gt_onehot[3])  > 0
    et_pred = pred_onehot[3] > 0
    et_gt   = gt_onehot[3]   > 0
    return {
        "class_dice_1": c1, "class_dice_2": c2, "class_dice_3": c3,
        "wt_dice": dice_binary(wt_pred, wt_gt),
        "tc_dice": dice_binary(tc_pred, tc_gt),
        "et_dice": dice_binary(et_pred, et_gt),
    }


# ============================================================
# DIAGNOSTIC: print conv replacement summary
# ============================================================
def print_conv_summary(model: nn.Module):
    """Count and print how many DSConv vs Conv3d remain in the model."""
    n_dsconv = 0
    n_conv3d = 0
    n_conv1x1 = 0
    for name, m in model.named_modules():
        if isinstance(m, DepthwiseSeparableConv3d):
            n_dsconv += 1
        elif isinstance(m, nn.Conv3d):
            k = m.kernel_size
            if all(ki == 1 for ki in k):
                n_conv1x1 += 1
            else:
                n_conv3d += 1
    print(f"\n[Conv Summary]  DSConv3d: {n_dsconv}  |  Conv3d(1x1, kept): {n_conv1x1}  |  Conv3d(spatial, NOT replaced): {n_conv3d}")
    if n_conv3d > 0:
        print("  WARNING: some spatial Conv3d were not replaced. Check module tree.")
    print()


# ============================================================
# TRAINING LOOP  –  identical to EE491
# ============================================================
def train_one(model_name: str, model: nn.Module, n_params: int, gflops: float,
              train_loader, val_loader, max_epochs: int = MAX_EPOCHS):
    device = next(model.parameters()).device

    loss_fn = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    post_pred  = AsDiscrete(argmax=True, to_onehot=4)
    post_label = AsDiscrete(to_onehot=4)

    best_val_mean_class_dice = -1.0
    best_epoch = -1
    best_snapshot = None
    bad_epochs = 0
    last_snapshot = None

    ckpt_name = f"brats2020_unet_{model_name}_best_80_20.pth"
    t0 = time.time()
    stop_epoch = 0

    for epoch in range(1, max_epochs + 1):
        stop_epoch = epoch
        lr_now = float(optimizer.param_groups[0]["lr"])
        print("-" * 20)
        print(f"Epoch {epoch} / {max_epochs} | lr={lr_now:.6g}")

        # ---- train ----
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

        # ---- validation ----
        model.eval()
        cls1_list, cls2_list, cls3_list = [], [], []
        wt_list, tc_list, et_list = [], [], []
        with torch.no_grad():
            for val_batch in val_loader:
                vimg = val_batch["image"].to(device, non_blocking=True)
                vlab = val_batch["label"].to(device, non_blocking=True)
                preds = sliding_window_inference(
                    vimg, roi_size=(128, 128, 128),
                    sw_batch_size=SW_BATCH_SIZE, predictor=model,
                )
                preds_list = decollate_batch(preds)
                vlab_list  = decollate_batch(vlab)
                preds_proc = [post_pred(p) for p in preds_list]
                vlab_proc  = [post_label(l) for l in vlab_list]
                for p_onehot, g_onehot in zip(preds_proc, vlab_proc):
                    d = compute_class_region_dice(p_onehot, g_onehot)
                    cls1_list.append(d["class_dice_1"])
                    cls2_list.append(d["class_dice_2"])
                    cls3_list.append(d["class_dice_3"])
                    wt_list.append(d["wt_dice"])
                    tc_list.append(d["tc_dice"])
                    et_list.append(d["et_dice"])

        mean_c1 = float(np.mean(cls1_list)) if cls1_list else 0.0
        mean_c2 = float(np.mean(cls2_list)) if cls2_list else 0.0
        mean_c3 = float(np.mean(cls3_list)) if cls3_list else 0.0
        mean_class_dice = float(np.mean([mean_c1, mean_c2, mean_c3]))
        mean_wt = float(np.mean(wt_list)) if wt_list else 0.0
        mean_tc = float(np.mean(tc_list)) if tc_list else 0.0
        mean_et = float(np.mean(et_list)) if et_list else 0.0

        print(f"  -> Class Dice (1,2,3): {mean_c1:.4f}, {mean_c2:.4f}, {mean_c3:.4f} | mean={mean_class_dice:.4f}")
        print(f"  -> Region Dice (WT,TC,ET): {mean_wt:.4f}, {mean_tc:.4f}, {mean_et:.4f}")

        last_snapshot = {
            "final_epoch": epoch, "final_lr": lr_now,
            "final_val_mean_class_dice": mean_class_dice,
            "final_val_class_dice_1": mean_c1, "final_val_class_dice_2": mean_c2, "final_val_class_dice_3": mean_c3,
            "final_val_wt_dice": mean_wt, "final_val_tc_dice": mean_tc, "final_val_et_dice": mean_et,
        }

        if mean_class_dice > (best_val_mean_class_dice + MIN_DELTA):
            best_val_mean_class_dice = mean_class_dice
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), ckpt_name)
            best_snapshot = {
                "best_val_mean_class_dice": best_val_mean_class_dice,
                "best_val_class_dice_1": mean_c1, "best_val_class_dice_2": mean_c2, "best_val_class_dice_3": mean_c3,
                "best_val_wt_dice": mean_wt, "best_val_tc_dice": mean_tc, "best_val_et_dice": mean_et,
                "best_lr": lr_now,
            }
            print(f"  -> Best model updated! epoch={epoch}, mean_class_dice={mean_class_dice:.4f}")
        else:
            bad_epochs += 1
            print(f"  -> No improvement. early-stop counter: {bad_epochs}/{EARLY_STOP_PATIENCE}")

        scheduler.step()

        if bad_epochs >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs).")
            break

    elapsed_s = time.time() - t0
    print(f"Done: {model_name} | best epoch: {best_epoch} | best mean_class_dice: {best_val_mean_class_dice:.4f}")

    del optimizer, loss_fn, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    spec = AUTOBEST_SPEC
    result = {
        "model": model_name,
        "params": int(n_params),
        "gflops_fwd_128": float(gflops),
        "channels": str(tuple(spec["channels"])),
        "strides": str(tuple(spec["strides"])),
        "depth": len(spec["channels"]),
        "num_res_units": spec["num_res_units"],
        "seed": SEED,
        "num_workers": NUM_WORKERS,
        "sw_batch_size": SW_BATCH_SIZE,
        "max_epochs": max_epochs,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "epochs_trained": stop_epoch,
        "stop_epoch": stop_epoch,
        "best_epoch": best_epoch,
        "ckpt": ckpt_name,
        "elapsed_s": float(elapsed_s),
    }
    if best_snapshot:
        result.update(best_snapshot)
    if last_snapshot:
        result.update(last_snapshot)
    if gflops > 0:
        result["best_mean_dice_per_gflop"] = result.get("best_val_mean_class_dice", 0.0) / gflops
        result["best_wt_dice_per_gflop"]   = result.get("best_val_wt_dice", 0.0) / gflops
        result["best_tc_dice_per_gflop"]   = result.get("best_val_tc_dice", 0.0) / gflops
        result["best_et_dice_per_gflop"]   = result.get("best_val_et_dice", 0.0) / gflops
    return result


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("  EE492 – UNet-AutoBest with Depthwise Separable Convolutions")
    print("=" * 70)

    # ---- data ----
    data_dicts = build_datalist(DATA_ROOT)
    if not data_dicts:
        print("ERROR: No data found. Check DATA_ROOT.")
        return
    train_files, val_files = split_train_val(data_dicts, val_fraction=0.2, seed=SEED)

    # ---- seeds ----
    set_all_seeds(SEED)

    # ---- build DSConv model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model_dsconv(AUTOBEST_SPEC, device)

    n_params = count_trainable_params(model)
    gflops   = compute_gflops(model, device)

    print(f"\nModel: UNet-AutoBest-DSConv")
    print(f"  Topology  : channels={AUTOBEST_SPEC['channels']}, strides={AUTOBEST_SPEC['strides']}, res_units={AUTOBEST_SPEC['num_res_units']}")
    print(f"  Params    : {n_params:,}  ({n_params/1e6:.3f} M)")
    if gflops >= 0:
        print(f"  GFLOPs    : {gflops:.3f}")
    else:
        print(f"  GFLOPs    : unavailable (install fvcore or ptflops)")

    # ---- diagnostic: verify replacement ----
    print_conv_summary(model)

    # ---- print full architecture for report ----
    print("=" * 40 + "  FULL MODEL  " + "=" * 40)
    print(model)
    print("=" * 94)

    # ---- loaders ----
    train_loader, val_loader = make_loaders(train_files, val_files, batch_size=1)

    # ---- train ----
    result = train_one(
        model_name="UNet-AutoBest-DSConv",
        model=model,
        n_params=n_params,
        gflops=gflops,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=MAX_EPOCHS,
    )

    # ---- save CSV ----
    out_csv = "results_dsconv.csv"
    fieldnames = [
        "model", "seed", "num_workers", "sw_batch_size", "max_epochs", "early_stop_patience",
        "channels", "strides", "depth", "num_res_units",
        "params", "gflops_fwd_128", "elapsed_s", "epochs_trained", "stop_epoch",
        "best_epoch", "best_lr",
        "best_val_mean_class_dice", "best_val_class_dice_1", "best_val_class_dice_2", "best_val_class_dice_3",
        "best_val_wt_dice", "best_val_tc_dice", "best_val_et_dice",
        "final_epoch", "final_lr",
        "final_val_mean_class_dice", "final_val_class_dice_1", "final_val_class_dice_2", "final_val_class_dice_3",
        "final_val_wt_dice", "final_val_tc_dice", "final_val_et_dice",
        "best_mean_dice_per_gflop", "best_wt_dice_per_gflop", "best_tc_dice_per_gflop", "best_et_dice_per_gflop",
        "ckpt",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow({k: result.get(k, "") for k in fieldnames})

    print(f"\nResults saved to: {out_csv}")
    print("\n--- SUMMARY ---")
    for k in ["model", "params", "gflops_fwd_128", "best_epoch",
              "best_val_mean_class_dice", "best_val_wt_dice", "best_val_tc_dice", "best_val_et_dice",
              "best_mean_dice_per_gflop"]:
        print(f"  {k:30s} : {result.get(k, 'N/A')}")
    print()


if __name__ == "__main__":
    main()
