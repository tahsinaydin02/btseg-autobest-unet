import os
import glob
import numpy as np
import nibabel as nib
import napari
import datetime

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QComboBox, QGroupBox,
    QSlider, QFormLayout, QFileDialog
)
from qtpy.QtCore import Qt

from PIL import Image, ImageDraw, ImageFont
from napari.utils.colormaps import DirectLabelColormap


# -----------------------------------------------------------------------------
# 1) FILE HANDLING
# -----------------------------------------------------------------------------
def find_case_directories(data_root: str):
    pattern = os.path.join(data_root, "BraTS20_Training_*")
    return sorted([d for d in glob.glob(pattern) if os.path.isdir(d)])


def extract_case_id(case_dir: str):
    return os.path.basename(case_dir)


def get_brats_file_paths(case_dir: str):
    cid = extract_case_id(case_dir)

    def pick(path_no_gz: str):
        if os.path.exists(path_no_gz):
            return path_no_gz
        if os.path.exists(path_no_gz + ".gz"):
            return path_no_gz + ".gz"
        return None

    flair = pick(os.path.join(case_dir, cid + "_flair.nii"))
    seg = pick(os.path.join(case_dir, cid + "_seg.nii"))
    return flair, seg


def get_prediction_path(pred_root: str, model_name: str, case_id: str):
    """
    Strictly locates a single prediction file.
    If multiple candidates exist, raises an error (prevents "dice changes" across runs).
    """
    model_dir = os.path.join(pred_root, model_name)
    if not os.path.isdir(model_dir):
        model_dir = pred_root

    # first: exact filenames (stable)
    candidates = [
        os.path.join(model_dir, f"{case_id}_pred.nii.gz"),
        os.path.join(model_dir, f"{case_id}_pred.nii"),
        os.path.join(model_dir, f"{case_id}_flair_pred.nii.gz"),
        os.path.join(model_dir, f"{case_id}_flair_pred.nii"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    # fallback: wildcard (must be unique)
    hits = sorted(glob.glob(os.path.join(model_dir, f"{case_id}*_pred.nii*")))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise RuntimeError(
            f"Multiple prediction files found for {case_id} in {model_dir}:\n" + "\n".join(hits)
        )
    return None


# -----------------------------------------------------------------------------
# 2) LABEL NORMALIZATION + METRICS
# -----------------------------------------------------------------------------
def normalize_brats_labels(arr: np.ndarray) -> np.ndarray:
    """
    Normalizes to BraTS canonical {0,1,2,4}.
    Handles {0,1,2,3} etc by mapping 3 -> 4.
    Always returns an int16 copy (no in-place side effects on inputs).
    """
    x = np.rint(arr).astype(np.int16, copy=True)
    x[x == 3] = 4
    return x


def dice_binary(a: np.ndarray, b: np.ndarray, empty_value=np.nan) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return (2.0 * inter / denom) if denom > 0 else empty_value


def compute_all_dice(gt_arr: np.ndarray, pred_arr: np.ndarray,
                     empty_tc_et=np.nan, empty_wt=np.nan) -> dict:
    """
    WT/TC/ET + class dice (1/2/4).
    - WT empty -> empty_wt (default NaN)
    - TC/ET empty -> empty_tc_et (default NaN)
    """
    gt = normalize_brats_labels(gt_arr)
    pr = normalize_brats_labels(pred_arr)

    wt_gt = np.isin(gt, [1, 2, 4])
    wt_pr = np.isin(pr, [1, 2, 4])

    tc_gt = np.isin(gt, [1, 4])
    tc_pr = np.isin(pr, [1, 4])

    et_gt = (gt == 4)
    et_pr = (pr == 4)

    return {
        "WT": dice_binary(wt_gt, wt_pr, empty_value=empty_wt),
        "TC": dice_binary(tc_gt, tc_pr, empty_value=empty_tc_et),
        "ET": dice_binary(et_gt, et_pr, empty_value=empty_tc_et),
        "Dice(1)": dice_binary(gt == 1, pr == 1, empty_value=np.nan),
        "Dice(2)": dice_binary(gt == 2, pr == 2, empty_value=np.nan),
        "Dice(4)": dice_binary(gt == 4, pr == 4, empty_value=np.nan),
    }


def fmt_metric(x: float) -> str:
    if isinstance(x, (float, np.floating)) and np.isnan(x):
        return "N/A"
    return f"{float(x):.4f}"


# -----------------------------------------------------------------------------
# 3) GUI (single napari viewer, split-view via grid)
# -----------------------------------------------------------------------------
class SegmentationAnalysisInterface(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer

        # exact requested colors: 1=red, 2=green, 4=yellow
        self.color_dict = {
            0: (0, 0, 0, 0),
            1: (1, 0, 0, 1),   # red
            2: (0, 1, 0, 1),   # green
            4: (1, 1, 0, 1),   # yellow
        }
        self.direct_cmap = DirectLabelColormap(color_dict=self.color_dict)

        self.current_case_id = None
        self.current_model_name = None
        self.current_metrics = None
        self.current_gt_path = None
        self.current_pred_path = None

        self.initialize_user_interface()

    def initialize_user_interface(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        title = QLabel("Glioma Segmentation Analysis Tool")
        title.setStyleSheet("font-weight: bold; font-size: 16px; color: #2E86C1; margin: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # dataset config
        file_group = QGroupBox("Dataset Configuration")
        file_layout = QFormLayout()

        self.data_edit = QLineEdit()
        self.data_edit.setPlaceholderText("Select BraTS Training Data Directory...")
        btn_data = QPushButton("Browse")
        btn_data.setFixedWidth(60)
        btn_data.clicked.connect(self.browse_data_directory)
        r1 = QHBoxLayout()
        r1.addWidget(self.data_edit)
        r1.addWidget(btn_data)
        file_layout.addRow("Dataset Root:", r1)

        self.pred_edit = QLineEdit()
        self.pred_edit.setPlaceholderText("Select Prediction Output Directory...")
        btn_pred = QPushButton("Browse")
        btn_pred.setFixedWidth(60)
        btn_pred.clicked.connect(self.browse_predictions_directory)
        r2 = QHBoxLayout()
        r2.addWidget(self.pred_edit)
        r2.addWidget(btn_pred)
        file_layout.addRow("Predictions:", r2)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # selection
        self.case_combo = QComboBox()
        self.model_combo = QComboBox()
        sel_layout = QFormLayout()
        sel_layout.addRow("Subject ID:", self.case_combo)
        sel_layout.addRow("Model Variant:", self.model_combo)

        btn_refresh = QPushButton("Refresh Directories")
        btn_refresh.clicked.connect(self.refresh_lists)

        layout.addLayout(sel_layout)
        layout.addWidget(btn_refresh)

        # load
        self.btn_load = QPushButton("Load Subject Data")
        self.btn_load.setStyleSheet(
            "font-weight: bold; background-color: #28B463; color: white; height: 40px; margin-top: 10px;"
        )
        self.btn_load.clicked.connect(self.load_selected_case)
        layout.addWidget(self.btn_load)

        # visualization controls
        ctrl_group = QGroupBox("Visualization Controls")
        ctrl_layout = QVBoxLayout()

        self.btn_grid = QPushButton("Toggle Split-View Comparison")
        self.btn_grid.setCheckable(True)
        self.btn_grid.setChecked(True)
        self.btn_grid.setStyleSheet("font-weight: bold; padding: 5px;")
        self.btn_grid.clicked.connect(self.toggle_view_mode)
        ctrl_layout.addWidget(self.btn_grid)

        op_layout = QFormLayout()
        self.slider_gt = QSlider(Qt.Horizontal)
        self.slider_gt.setRange(0, 100)
        self.slider_gt.setValue(80)
        self.slider_gt.valueChanged.connect(self.update_layer_opacity)
        op_layout.addRow("Ground Truth Opacity:", self.slider_gt)

        self.slider_pred = QSlider(Qt.Horizontal)
        self.slider_pred.setRange(0, 100)
        self.slider_pred.setValue(60)
        self.slider_pred.valueChanged.connect(self.update_layer_opacity)
        op_layout.addRow("Prediction Opacity:", self.slider_pred)

        ctrl_layout.addLayout(op_layout)
        ctrl_group.setLayout(ctrl_layout)
        layout.addWidget(ctrl_group)

        # metrics
        res_group = QGroupBox("Quantitative Metrics (DICE)")
        res_layout = QVBoxLayout()
        self.lbl_dice = QLabel("Metrics not calculated.")
        self.lbl_dice.setStyleSheet("font-family: monospace; font-weight: bold; color: #117864; font-size: 13px;")
        res_layout.addWidget(self.lbl_dice)
        res_group.setLayout(res_layout)
        layout.addWidget(res_group)

        # export
        self.btn_export = QPushButton("Export PNG")
        self.btn_export.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_export.clicked.connect(self.export_report_png)
        layout.addWidget(self.btn_export)

        layout.addStretch()

    # --- UI actions ---
    def browse_data_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select BraTS Data Directory")
        if directory:
            self.data_edit.setText(directory)

    def browse_predictions_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Predictions Directory")
        if directory:
            self.pred_edit.setText(directory)

    def refresh_lists(self):
        data_root = self.data_edit.text()
        if os.path.isdir(data_root):
            self.case_combo.clear()
            for c in find_case_directories(data_root):
                self.case_combo.addItem(extract_case_id(c))

        pred_root = self.pred_edit.text()
        if os.path.isdir(pred_root):
            self.model_combo.clear()
            valid_items = sorted([x for x in os.listdir(pred_root) if not x.startswith('.')])
            for item in valid_items:
                self.model_combo.addItem(item)

    def load_selected_case(self):
        case_id = self.case_combo.currentText()
        if not case_id:
            return

        data_root = self.data_edit.text()
        pred_root = self.pred_edit.text()
        model_name = self.model_combo.currentText()

        try:
            flair_path, gt_path = get_brats_file_paths(os.path.join(data_root, case_id))
            pred_path = get_prediction_path(pred_root, model_name, case_id)
        except Exception as e:
            self.lbl_dice.setText(f"Error selecting prediction:\n{e}")
            return

        if not (flair_path and gt_path and pred_path):
            self.lbl_dice.setText("Error: Required files not found.")
            return

        # load nifti
        flair_img = nib.load(flair_path)
        gt_img = nib.load(gt_path)
        pr_img = nib.load(pred_path)

        flair_data = flair_img.get_fdata().transpose(2, 1, 0)
        gt_raw = gt_img.get_fdata().transpose(2, 1, 0)
        pr_raw = pr_img.get_fdata().transpose(2, 1, 0)

        gt_data = normalize_brats_labels(gt_raw)
        pred_data = normalize_brats_labels(pr_raw)

        # hard safety checks
        if gt_data.shape != pred_data.shape:
            self.lbl_dice.setText(f"Error: shape mismatch GT {gt_data.shape} vs Pred {pred_data.shape}")
            return

        # debugging (stability + correctness)
        same_ratio = float(np.mean(gt_data == pred_data))
        gt_wt = int(np.isin(gt_data, [1, 2, 4]).sum())
        pr_wt = int(np.isin(pred_data, [1, 2, 4]).sum())
        gt_et = int((gt_data == 4).sum())
        pr_et = int((pred_data == 4).sum())
        affine_equal = bool(np.allclose(gt_img.affine, pr_img.affine))

        print("GT path:", gt_path)
        print("Pred path:", pred_path)
        print("unique GT:", np.unique(gt_data), "unique Pred:", np.unique(pred_data))
        print("same_ratio(gt==pred):", same_ratio)
        print("WT vox gt/pr:", gt_wt, pr_wt, "ET vox gt/pr:", gt_et, pr_et)
        print("affine equal (GT vs Pred):", affine_equal)

        # clear + add layers
        self.viewer.layers.clear()

        self.layer_flair_L = self.viewer.add_image(flair_data, name="MRI (Reference)", colormap="gray")

        self.layer_gt = self.viewer.add_labels(
            gt_data, name="Ground Truth", opacity=self.slider_gt.value() / 100.0
        )
        self.layer_gt.colormap = self.direct_cmap
        self.layer_gt.refresh()

        self.layer_flair_R = self.viewer.add_image(flair_data, name="MRI (Copy)", colormap="gray")

        self.layer_pred = self.viewer.add_labels(
            pred_data, name="Prediction", opacity=self.slider_pred.value() / 100.0
        )
        self.layer_pred.colormap = self.direct_cmap
        self.layer_pred.refresh()

        # metrics
        metrics = compute_all_dice(gt_data, pred_data, empty_tc_et=np.nan, empty_wt=np.nan)
        self.current_metrics = metrics
        self.current_case_id = case_id
        self.current_model_name = model_name
        self.current_gt_path = gt_path
        self.current_pred_path = pred_path

        self.lbl_dice.setText(
            f"WT: {fmt_metric(metrics['WT'])}   TC: {fmt_metric(metrics['TC'])}   ET: {fmt_metric(metrics['ET'])}\n"
            f"Dice(1): {fmt_metric(metrics['Dice(1)'])}   Dice(2): {fmt_metric(metrics['Dice(2)'])}   Dice(4): {fmt_metric(metrics['Dice(4)'])}\n"
            f"GT: {os.path.basename(gt_path)}\nPred: {os.path.basename(pred_path)}"
        )
        
        # view setup
        self.viewer.dims.ndisplay = 2
        mid_slice = flair_data.shape[0] // 2
        self.viewer.dims.set_current_step(0, mid_slice)

        self.btn_grid.setChecked(True)
        self.btn_grid.setText("Switch to Overlay Mode")
        self.apply_view_configuration(True)

        self.viewer.reset_view()

    def update_layer_opacity(self):
        if hasattr(self, 'layer_gt'):
            self.layer_gt.opacity = self.slider_gt.value() / 100.0
        if hasattr(self, 'layer_pred'):
            self.layer_pred.opacity = self.slider_pred.value() / 100.0

    def toggle_view_mode(self):
        is_split_view = self.btn_grid.isChecked()
        if is_split_view:
            self.btn_grid.setText("Switch to Overlay Mode")
        else:
            self.btn_grid.setText("Switch to Split-View Comparison")
        self.apply_view_configuration(is_split_view)

    def apply_view_configuration(self, enable_split):
        if not hasattr(self, 'layer_flair_R'):
            return

        if enable_split:
            self.layer_flair_R.visible = True
            self.viewer.grid.enabled = True
            self.viewer.grid.stride = 2
            self.viewer.grid.shape = (1, 2)

            self.viewer.text_overlay.visible = True
            self.viewer.text_overlay.text = "LEFT: Ground Truth Segmentation  |  RIGHT: Model Prediction"
            self.viewer.text_overlay.color = "cyan"
            self.viewer.text_overlay.font_size = 10
        else:
            self.viewer.grid.enabled = False
            self.layer_flair_R.visible = False
            self.viewer.text_overlay.visible = False

    def export_report_png(self):
        if not hasattr(self, "layer_gt") or not hasattr(self, "layer_pred"):
            self.lbl_dice.setText("Load a subject first, then export.")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Report PNG", "report_GT_vs_Pred.png", "PNG (*.png)"
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".png"):
            out_path += ".png"

        # save current view state
        prev_grid = bool(self.viewer.grid.enabled)
        prev_stride = getattr(self.viewer.grid, "stride", 1)
        prev_shape = getattr(self.viewer.grid, "shape", (1, 1))
        prev_text_vis = bool(self.viewer.text_overlay.visible)
        prev_text = self.viewer.text_overlay.text
        prev_flairR_vis = self.layer_flair_R.visible if hasattr(self, "layer_flair_R") else False

        # force split-view for export
        self.apply_view_configuration(True)

        z = int(self.viewer.dims.current_step[0])
        img_np = self.viewer.screenshot(canvas_only=True)

        # restore previous state
        self.viewer.grid.enabled = prev_grid
        self.viewer.grid.stride = prev_stride
        self.viewer.grid.shape = prev_shape
        self.viewer.text_overlay.visible = prev_text_vis
        self.viewer.text_overlay.text = prev_text
        if hasattr(self, "layer_flair_R"):
            self.layer_flair_R.visible = prev_flairR_vis

        # build report image with header
        im = Image.fromarray(img_np)
        pad = 24
        header_h = 140

        W, H = im.size
        canvas = Image.new("RGB", (W + 2 * pad, H + header_h + 2 * pad), (255, 255, 255))
        canvas.paste(im, (pad, pad + header_h))

        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()

        case_id = self.current_case_id or "UnknownCase"
        model_name = self.current_model_name or "UnknownModel"
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        m = self.current_metrics or {}

        met_line = "WT/TC/ET: n/a"
        cls_line = "Dice(1/2/4): n/a"
        if m:
            met_line = f"WT: {fmt_metric(m.get('WT', np.nan))}   TC: {fmt_metric(m.get('TC', np.nan))}   ET: {fmt_metric(m.get('ET', np.nan))}"
            cls_line = f"Dice(1): {fmt_metric(m.get('Dice(1)', np.nan))}   Dice(2): {fmt_metric(m.get('Dice(2)', np.nan))}   Dice(4): {fmt_metric(m.get('Dice(4)', np.nan))}"

        title1 = "BraTS Segmentation Report"
        title2 = f"Case: {case_id}   |   Model: {model_name}   |   Slice(z): {z}   |   {ts}"
        title3 = met_line
        title4 = cls_line
        title5 = f"GT: {os.path.basename(self.current_gt_path) if self.current_gt_path else 'n/a'}"
        title6 = f"Pred: {os.path.basename(self.current_pred_path) if self.current_pred_path else 'n/a'}"
        title7 = "LEFT: Ground Truth    RIGHT: Prediction    Labels: 1=Red, 2=Green, 4=Yellow (3→4)"

        y0 = pad + 8
        draw.text((pad, y0), title1, fill=(0, 0, 0), font=font)
        draw.text((pad, y0 + 22), title2, fill=(0, 0, 0), font=font)
        draw.text((pad, y0 + 44), title3, fill=(0, 0, 0), font=font)
        draw.text((pad, y0 + 66), title4, fill=(0, 0, 0), font=font)
        draw.text((pad, y0 + 88), title5, fill=(0, 0, 0), font=font)
        draw.text((pad, y0 + 108), title6, fill=(0, 0, 0), font=font)
        draw.text((pad, y0 + 128), title7, fill=(0, 0, 0), font=font)

        canvas.save(out_path, dpi=(300, 300))
        self.lbl_dice.setText(self.lbl_dice.text() + f"\nSaved report: {out_path}")


# -----------------------------------------------------------------------------
# 4) MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    viewer = napari.Viewer(title="Glioma Segmentation Analysis Tool")

    interface = SegmentationAnalysisInterface(viewer)
    viewer.window.add_dock_widget(interface, area='right', name="Analysis Toolbox")

    napari.run()
