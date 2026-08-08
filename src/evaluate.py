import os
import json
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
    MapLabelValued,
    AsDiscrete,
)
from monai.networks.nets import UNet
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference


# ----------------- PATHS -----------------
# All paths below can be overridden with environment variables so that no
# machine-specific path is hard-coded in the repository.
# Path to the BraTS 2020 training set. The dataset is NOT redistributed with
# this repository (see README). Override with the BRATS_DATA_ROOT env var:
#     export BRATS_DATA_ROOT=/path/to/MICCAI_BraTS2020_TrainingData
DATA_ROOT = os.environ.get(
    "BRATS_DATA_ROOT",
    os.path.join("data", "MICCAI_BraTS2020_TrainingData"),
)

# Validation split written by the training script (val_cases_seed42.txt).
VAL_TXT = os.environ.get("BTSEG_VAL_TXT", "val_cases_seed42.txt")

# Checkpoint to evaluate. Trained checkpoints are published under the v1.0
# release; the default below matches the filename written by train_variants.py
# for the baseline anchor.
MODEL_PATH = os.environ.get("BTSEG_CKPT", "brats2020_unet_UNet-Baseline_best_80_20.pth")

# Architecture of that checkpoint. A state_dict carries no architecture, so it
# has to be supplied. Defaults to the baseline anchor; override for any other
# model, e.g. for UNet-AutoBest:
#     export BTSEG_SPEC='{"channels":[24,32,80],"strides":[2,2],"num_res_units":2}'
# and additionally set BTSEG_DSCONV=1 for the depthwise-separable variants.
SPEC = json.loads(
    os.environ.get(
        "BTSEG_SPEC",
        '{"channels": [32, 64, 128, 256, 512], "strides": [2, 2, 2, 2], "num_res_units": 2}',
    )
)
DSCONV = os.environ.get("BTSEG_DSCONV", "0") == "1"

NUM_CLASSES = 4  # {0,1,2,3}


# ----------------- HELPERS -----------------
def load_val_ids(txt_path):
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"{txt_path} not found")
    with open(txt_path, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    print(f"Validation cases ({len(ids)}):")
    for cid in ids:
        print("  ", cid)
    return ids


def build_val_datalist(data_root, val_ids):
    data_dicts = []
    for cid in val_ids:
        case_dir = os.path.join(data_root, cid)
        flair = os.path.join(case_dir, cid + "_flair.nii")
        t1    = os.path.join(case_dir, cid + "_t1.nii")
        t1ce  = os.path.join(case_dir, cid + "_t1ce.nii")
        t2    = os.path.join(case_dir, cid + "_t2.nii")
        seg   = os.path.join(case_dir, cid + "_seg.nii")

        # fall back to .nii.gz
        def pick_nii_or_gz(p):
            if os.path.exists(p):
                return p
            if os.path.exists(p + ".gz"):
                return p + ".gz"
            return None

        flair = pick_nii_or_gz(flair)
        t1    = pick_nii_or_gz(t1)
        t1ce  = pick_nii_or_gz(t1ce)
        t2    = pick_nii_or_gz(t2)
        seg   = pick_nii_or_gz(seg)

        if not all([flair, t1, t1ce, t2, seg]):
            raise FileNotFoundError(f"Missing modality for {cid}")

        data_dicts.append(
            {"image": [flair, t1, t1ce, t2], "label": seg, "case_id": cid}
        )
    return data_dicts


def get_val_transform():
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            MapLabelValued(
                keys="label",
                orig_labels=[0, 1, 2, 4],
                target_labels=[0, 1, 2, 3],
            ),
        ]
    )


def load_model(device):
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=NUM_CLASSES,
        channels=tuple(SPEC["channels"]),
        strides=tuple(SPEC["strides"]),
        num_res_units=int(SPEC["num_res_units"]),
    )

    if DSCONV:
        # The depthwise-separable models store different parameter names
        # (conv.depthwise.* / conv.pointwise.*), so the substitution has to
        # happen before the state_dict is loaded.
        from train_autobest_dsconv import _replace_conv3d_with_dsconv

        model = _replace_conv3d_with_dsconv(model)

    model = model.to(device)

    state_dict = torch.load(MODEL_PATH, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as err:
        raise RuntimeError(
            f"Could not load {MODEL_PATH} into the architecture given by "
            f"BTSEG_SPEC={SPEC} (BTSEG_DSCONV={int(DSCONV)}).\n"
            "A checkpoint only stores weights, so the architecture must match "
            "exactly. See the release notes for the spec of each published "
            f"checkpoint.\n\nOriginal error:\n{err}"
        ) from err

    model.eval()
    return model


# ----------------- MAIN EVAL -----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Eval device:", device)

    val_ids = load_val_ids(VAL_TXT)
    val_dicts = build_val_datalist(DATA_ROOT, val_ids)
    print(f"Total val dicts: {len(val_dicts)}")

    val_ds = Dataset(val_dicts, transform=get_val_transform())
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    model = load_model(device)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
    post_label = AsDiscrete(to_onehot=NUM_CLASSES)

    dice_metric.reset()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = sliding_window_inference(
                images,
                roi_size=(128, 128, 128),
                sw_batch_size=1,
                predictor=model,
            )  # (1,4,H,W,D)

            preds_list = decollate_batch(logits)
            labs_list  = decollate_batch(labels)

            preds_proc = [post_pred(p) for p in preds_list]
            labs_proc  = [post_label(l) for l in labs_list]

            dice_metric(y_pred=preds_proc, y=labs_proc)

    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    print(f"\nValidation mean Dice over {len(val_dicts)} cases: {mean_dice:.4f}")


if __name__ == "__main__":
    main()
