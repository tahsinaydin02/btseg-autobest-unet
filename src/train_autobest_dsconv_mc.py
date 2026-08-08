# train_autobest_dsconv_mc.py
# -----------------------------------------------------------------------------
# EE492 final phase: AutoBest-DSConv trained WITH dropout, for MC-Dropout
# uncertainty estimation at inference time.
#
# Design principle (matches the midterm's controlled-comparison methodology):
#   The training loop, transforms, loss, optimizer, scheduler, data split,
#   and Dice computation are reused UNCHANGED from the existing base module.
#   The ONLY change vs. the midterm DSConv model is:
#       (1) the MONAI UNet is built with dropout=0.2, and
#       (2) the DSConv replacement preserves those dropout layers.
#
#   We achieve this by monkeypatching `base.make_model` and reusing
#   `base.train_all`, exactly like generated_best_model_train.py reuses base.
#
# Why dropout MUST be added at TRAINING time:
#   MC-Dropout = keeping dropout active during inference. If the network has no
#   dropout layers, there is nothing to sample over. So the model must be
#   (re)trained with dropout enabled. This script produces that model.
#
# Run in the lab:
#   python train_autobest_dsconv_mc.py
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
from monai.networks.nets import UNet

import train_variants as base


# -----------------------------------------------------------------------------
# Hyperparameters for the uncertainty variant
# -----------------------------------------------------------------------------
DROPOUT_P = 0.2          # literature-standard for MC-Dropout on BraTS
MODEL_NAME = "UNet-AutoBest-DSConv-MC"
AUTOBEST_SPEC = {"channels": (24, 32, 80), "strides": (2, 2), "num_res_units": 2}


# -----------------------------------------------------------------------------
# Depthwise separable convolution (identical factorization to the midterm)
#   (i)  depthwise Conv3d: groups=C_in, bias=False  -> spatial filtering
#   (ii) pointwise Conv3d: kernel 1x1x1, bias=True  -> channel mixing
# Stride/padding/dilation are carried by the depthwise stage so that strided
# (downsampling) convolutions keep their behavior.
# -----------------------------------------------------------------------------
class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, conv: nn.Conv3d):
        super().__init__()
        self.depthwise = nn.Conv3d(
            conv.in_channels,
            conv.in_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.in_channels,   # depthwise
            bias=False,
        )
        self.pointwise = nn.Conv3d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=(conv.bias is not None),
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


def convert_to_dsconv(module: nn.Module) -> nn.Module:
    """
    Recursively replace spatial Conv3d (kernel > 1, groups == 1) with DSConv.
    Rules (same scope as the midterm):
      - Spatial Conv3d (k>1) -> DepthwiseSeparableConv3d
      - 1x1x1 Conv3d         -> kept (channel-mixing projections / seg head)
      - ConvTranspose3d      -> kept (learned upsampling)
      - Dropout modules      -> untouched (this is what makes MC-Dropout work)
    The `groups == 1` guard prevents touching the depthwise conv inside any
    DSConv we just created; we also do not recurse into a replaced module.
    """
    for name, child in module.named_children():
        is_spatial_conv = (
            isinstance(child, nn.Conv3d)
            and child.kernel_size[0] > 1
            and child.groups == 1
        )
        if is_spatial_conv:
            setattr(module, name, DepthwiseSeparableConv3d(child))
        else:
            convert_to_dsconv(child)
    return module


# -----------------------------------------------------------------------------
# Diagnostic: confirm DSConv scope AND that dropout layers survived the walk.
# If "Dropout modules" is 0, MC-Dropout will silently do nothing -> abort.
# -----------------------------------------------------------------------------
def conv_summary(model: nn.Module):
    dsconv = sum(isinstance(m, DepthwiseSeparableConv3d) for m in model.modules())
    transp = sum(isinstance(m, nn.ConvTranspose3d) for m in model.modules())
    drop = sum(
        isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d))
        for m in model.modules()
    )
    # Count standalone Conv3d that are NOT inside a DSConv we built.
    conv1x1 = 0
    conv_spatial_kept = 0
    for parent in model.modules():
        if isinstance(parent, DepthwiseSeparableConv3d):
            continue
        for child in parent.children():
            if isinstance(child, nn.Conv3d) and not isinstance(
                parent, DepthwiseSeparableConv3d
            ):
                # skip convs that live inside a DSConv
                pass
    for m in model.modules():
        if isinstance(m, DepthwiseSeparableConv3d):
            continue
    # Simpler robust count over all Conv3d, classifying by kernel, excluding
    # those whose parent is a DSConv.
    dsconv_internal = set()
    for m in model.modules():
        if isinstance(m, DepthwiseSeparableConv3d):
            dsconv_internal.add(id(m.depthwise))
            dsconv_internal.add(id(m.pointwise))
    for m in model.modules():
        if isinstance(m, nn.Conv3d) and id(m) not in dsconv_internal:
            if m.kernel_size[0] == 1:
                conv1x1 += 1
            else:
                conv_spatial_kept += 1

    print(
        f"[Conv Summary] DSConv3d: {dsconv} | "
        f"Conv3d(1x1, kept): {conv1x1} | "
        f"Conv3d(spatial, NOT replaced): {conv_spatial_kept} | "
        f"ConvTranspose3d: {transp} | "
        f"Dropout modules: {drop}"
    )
    if drop == 0:
        raise RuntimeError(
            "No Dropout modules found. MC-Dropout cannot work. "
            "Check that UNet(..., dropout=DROPOUT_P) was passed and that "
            "num_res_units > 0."
        )
    return drop


# -----------------------------------------------------------------------------
# Patched model builder: dropout-enabled UNet -> DSConv conversion.
# Signature matches base.make_model(spec, device) so the base training loop
# uses it transparently.
# -----------------------------------------------------------------------------
def make_model_dsconv_mc(spec, device):
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=spec["channels"],
        strides=spec["strides"],
        num_res_units=spec["num_res_units"],
        dropout=DROPOUT_P,          # <-- the change that enables MC-Dropout
    )
    convert_to_dsconv(model)        # DSConv replacement; dropout preserved
    conv_summary(model)             # verify dropout survived
    return model.to(device)


# Monkeypatch + restrict to the single AutoBest spec.
base.make_model = make_model_dsconv_mc
base.MODEL_SPECS = {MODEL_NAME: AUTOBEST_SPEC}


if __name__ == "__main__":
    # Same schedule as the midterm DSConv run for a clean comparison.
    base.train_all(max_epochs=100, val_fraction=0.2, batch_size=1)
