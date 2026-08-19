"""Stage 2 dataset and augmentation."""

import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset

from snet.data.io import is_valid_path, parse_centroid_json
from snet.data.preprocessing import (
    reorient_to_ras,
    world_to_voxel,
    resample_volume,
    full_preprocess,
    centroids_to_voxel,
)


# ─── Stage-1 crop simulation ─────────────────────────────────────────────────

def simulate_stage1_crop(volume, centroids_vox, crop_size=(96, 96, 128), rng=None):
    """Simulate Stage-1 output for training: centre X,Y on spine mean; random Z.

    Returns
    -------
    crop : np.ndarray, shape crop_size
    z_start : int
    spine_cx, spine_cy : int
    """
    CX, CY, CZ = crop_size
    vol_x, vol_y, vol_z = volume.shape

    if rng is None:
        rng = np.random.default_rng()

    all_ci = [v[0] for v in centroids_vox.values()]
    all_cj = [v[1] for v in centroids_vox.values()]
    spine_cx = int(round(np.mean(all_ci)))
    spine_cy = int(round(np.mean(all_cj)))

    hx, hy = CX // 2, CY // 2
    x0 = spine_cx - hx
    y0 = spine_cy - hy

    max_z_start = max(vol_z - CZ, 0)
    z_start = int(rng.integers(0, max_z_start + 1)) if max_z_start > 0 else 0

    crop = np.zeros(crop_size, dtype=volume.dtype)

    xs_vol = max(x0, 0); xe_vol = min(x0 + CX, vol_x)
    xs_cr = xs_vol - x0; xe_cr = xs_cr + (xe_vol - xs_vol)
    ys_vol = max(y0, 0); ye_vol = min(y0 + CY, vol_y)
    ys_cr = ys_vol - y0; ye_cr = ys_cr + (ye_vol - ys_vol)
    zs_vol = z_start; ze_vol = min(z_start + CZ, vol_z)
    zs_cr = 0; ze_cr = ze_vol - zs_vol

    crop[xs_cr:xe_cr, ys_cr:ye_cr, zs_cr:ze_cr] = \
        volume[xs_vol:xe_vol, ys_vol:ye_vol, zs_vol:ze_vol]

    return crop, z_start, spine_cx, spine_cy


def remap_centroids_to_crop(centroids_vox, spine_cx, spine_cy, z_start,
                             crop_size=(96, 96, 128)):
    """Convert global voxel centroids to crop coordinate space."""
    CX, CY, CZ = crop_size
    hx, hy = CX // 2, CY // 2
    x0, y0 = spine_cx - hx, spine_cy - hy
    return {label: (ci - x0, cj - y0, ck - z_start)
            for label, (ci, cj, ck) in centroids_vox.items()}


# ─── Augmentation ─────────────────────────────────────────────────────────────

class IntensityAugment:
    """Random scale + shift on CT only."""

    def __init__(self, scale=(0.75, 1.25), shift=(-0.25, 0.25)):
        self.scale = scale
        self.shift = shift

    def __call__(self, ct):
        s = np.random.uniform(*self.scale)
        b = np.random.uniform(*self.shift)
        return np.clip(ct * s + b, -1.0, 1.0).astype(np.float32)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class SpineDatasetStage2(Dataset):
    """One item per subject (full-spine crop, NOT per-vertebra)."""

    def __init__(self, csv_path, split, crop_size=(96, 96, 128),
                 target_spacing=2.0, sigma=0.75, augment=None):
        self.crop_size = crop_size
        self.target_spacing = target_spacing
        self.sigma = sigma
        self.augment = augment
        self.rng = np.random.default_rng()

        df = pd.read_csv(csv_path)
        df_split = df[df["type"] == split].reset_index(drop=True)

        self.rows = []
        for _, row in df_split.iterrows():
            if is_valid_path(row.get("image_path")) and is_valid_path(row.get("centroid_json_path")):
                self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        subj_name = row["name"]

        vol_nib = nib.load(str(row["image_path"]))
        mask_nib = nib.load(str(row["mask_path"])) if is_valid_path(row.get("mask_path")) else None
        _, raw_centroids = parse_centroid_json(str(row["centroid_json_path"]))

        vol_pp, mask_pp, ras_affine, zooms, orig_affine = full_preprocess(
            vol_nib, mask_nib,
            target_spacing=self.target_spacing,
            sigma=self.sigma,
        )

        centroids_2mm = centroids_to_voxel(
            raw_centroids, orig_affine, ras_affine, zooms, self.target_spacing,
        )
        for label in list(centroids_2mm.keys()):
            if np.any(mask_pp == label):
                coords = np.array(np.where(mask_pp == label))
                centroids_2mm[label] = tuple(np.round(coords.mean(axis=1)).astype(int))

        ct_crop, z_start, scx, scy = simulate_stage1_crop(
            vol_pp, centroids_2mm, self.crop_size, rng=self.rng,
        )
        gt_centroids_crop = remap_centroids_to_crop(
            centroids_2mm, scx, scy, z_start, self.crop_size,
        )

        if self.augment is not None:
            ct_crop = self.augment(ct_crop)

        return {
            "ct_crop": torch.from_numpy(ct_crop[None].astype(np.float32)),
            "gt_centroids_crop": gt_centroids_crop,
            "subject": subj_name,
        }


def stage2_collate(batch):
    """Custom collate: stack tensors, keep gt_centroids_crop as list of dicts."""
    return {
        "ct_crop": torch.stack([b["ct_crop"] for b in batch], dim=0),
        "gt_centroids_crop": [b["gt_centroids_crop"] for b in batch],
        "subject": [b["subject"] for b in batch],
    }
