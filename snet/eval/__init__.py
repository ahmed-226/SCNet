"""Evaluation metrics for all three stages."""

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion


# ─── Stage 1 metrics ──────────────────────────────────────────────────────────

def compute_localisation_error(pred_x_mm, pred_y_mm, gt_x_mm, gt_y_mm):
    """2-D spine-center localisation error in mm."""
    return float(np.sqrt((pred_x_mm - gt_x_mm) ** 2 + (pred_y_mm - gt_y_mm) ** 2))


def compute_heatmap_mse(pred_hm, gt_hm):
    """MSE between predicted and GT heatmap arrays."""
    return float(np.mean((pred_hm - gt_hm) ** 2))


# ─── Stage 2 metric ───────────────────────────────────────────────────────────

def compute_centroid_localisation_error(pred_vox, gt_vox, spacing=2.0):
    """Euclidean distance between predicted and GT centroid in mm."""
    return float(np.linalg.norm(np.array(pred_vox) - np.array(gt_vox)) * spacing)


# ─── Stage 3 metrics ──────────────────────────────────────────────────────────

def compute_dice(pred_bin, gt_bin, eps=1e-6):
    """Dice coefficient for two binary arrays."""
    intersection = (pred_bin * gt_bin).sum()
    return float((2.0 * intersection + eps) / (pred_bin.sum() + gt_bin.sum() + eps))


def compute_surface(binary_vol):
    """Extract surface voxels via erosion-based boundary."""
    eroded = binary_erosion(binary_vol)
    return binary_vol.astype(bool) & ~eroded


def compute_hausdorff(pred_bin, gt_bin):
    """95th-percentile Hausdorff distance (mm) using distance transforms."""
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float("nan")

    surf_pred = compute_surface(pred_bin.astype(bool))
    surf_gt = compute_surface(gt_bin.astype(bool))

    dt_pred = distance_transform_edt(~surf_pred)
    dt_gt = distance_transform_edt(~surf_gt)

    d_pred_to_gt = dt_gt[surf_pred]
    d_gt_to_pred = dt_pred[surf_gt]

    all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_distances, 95))
