# export_preds_15cases.py
# exports predictions (.nii) for a small subset (default 15 cases) for 3D Slicer visualization.

import os
import argparse
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
    EnsureTyped,
    Invertd,
    AsDiscreted,
    SaveImaged,
    MapLabelValued,
)
from monai.inferers import sliding_window_inference

import train_variants as base


# same label convention as training script:
# braTS labels: {0, 1, 2, 4}
# training internal labels: {0, 1, 2, 3} where 3->4
_TO_TRAIN_LABELS = MapLabelValued(keys="label", orig_labels=[0, 1, 2, 4], target_labels=[0, 1, 2, 3])
_TO_BRATS_LABELS = MapLabelValued(keys="pred", orig_labels=[0, 1, 2, 3], target_labels=[0, 1, 2, 4])


def make_infer_transforms():
    # keeps track of spatial transforms so we can invert back to original space before saving.
    return Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=False),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            _TO_TRAIN_LABELS,
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def make_post_transforms(output_dir: str):
    # inverts back to the original image space and saves nifti volumes.
    return Compose(
        [
            AsDiscreted(keys="pred", argmax=True),
            Invertd(
                keys="pred",
                transform=make_infer_transforms(),
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                nearest_interp=True,
                to_tensor=True,
            ),
            _TO_BRATS_LABELS,
            SaveImaged(
                keys="pred",
                meta_keys="pred_meta_dict",
                output_dir=output_dir,
                output_postfix="pred",
                resample=False,
                separate_folder=False,
                output_ext=".nii",
            ),
        ]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=base.DATA_ROOT, help="BraTS training root folder")
    ap.add_argument("--ckpt", type=str, required=True, help="path to .pth checkpoint")
    ap.add_argument("--out_dir", type=str, default="preds/UNet-AutoBest", help="output folder for predictions")
    ap.add_argument("--num_cases", type=int, default=15, help="how many cases to export")
    ap.add_argument("--roi", type=int, nargs=3, default=[128, 128, 128], help="sliding window roi size")
    ap.add_argument("--overlap", type=float, default=0.25, help="sliding window overlap")
    ap.add_argument("--sw_batch", type=int, default=2, help="sliding window batch size")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # model spec from your generated_best_model_train.py
    spec = {"channels": (24, 32, 80), "strides": (2, 2), "num_res_units": 2}
    model = base.make_model(spec, device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    data_dicts = base.build_datalist(args.data_root)
    data_dicts = data_dicts[: args.num_cases]

    infer_tf = make_infer_transforms()
    post_tf = make_post_transforms(args.out_dir)

    ds = Dataset(data=data_dicts, transform=infer_tf)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            case_id = batch.get("case_id", ["unknown"])[0]

            pred_logits = sliding_window_inference(
                images,
                roi_size=tuple(args.roi),
                sw_batch_size=int(args.sw_batch),
                predictor=model,
                overlap=float(args.overlap),
            )

            batch["pred"] = pred_logits

            # apply post transforms item-wise and save
            items = decollate_batch(batch)
            for it in items:
                _ = post_tf(it)

            print(f"saved: {case_id}")

    print(f"done. predictions are in: {args.out_dir}")


if __name__ == "__main__":
    main()
