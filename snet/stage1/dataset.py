"""Stage 1 dataset and augmentation."""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import scipy.ndimage as ndi
import torch
from torch.utils.data import Dataset

from snet.data.io import is_valid_path, parse_centroid_json
from snet.data.preprocessing import (
    reorient_to_ras,
    world_to_voxel,
    resample_volume,
    center_pad_crop,
)
from snet.stage1.heatmap import generate_spine_centerline_heatmap


# ─── Preprocessing (Stage-1 specific: center-pad/crop) ────────────────────────

def preprocess_stage1(vol_nib, raw_centroids, target_spacing=8.0, smooth_sigma=0.75,
                      divisor=2048.0, input_size=(64, 64, 128)):
    """Full Stage-1 preprocessing pipeline.

    Returns
    -------
    vol_out : np.ndarray float32, shape == input_size
    centroids_out : dict {label: (i,j,k)} in the final grid
    ras_affine : (4,4)
    orig_affine : (4,4)
    """
    orig_affine = vol_nib.affine.copy()

    vol_ras = reorient_to_ras(vol_nib)
    ras_affine = vol_ras.affine.copy()
    zooms_ras = np.array(vol_ras.header.get_zooms()[:3], dtype=np.float64)
    vol_data = vol_ras.get_fdata(dtype=np.float32)

    centroids_ras = {}
    for label, (x, y, z) in raw_centroids.items():
        world_mm = orig_affine @ np.array([x, y, z, 1.0])
        vox_ras = world_to_voxel(ras_affine, world_mm[:3])
        vox_ras = np.clip(vox_ras, 0, np.array(vol_data.shape) - 1)
        centroids_ras[label] = tuple(vox_ras)

    vol_8mm = resample_volume(vol_data, zooms_ras, target_spacing, order=1)
    scale = zooms_ras / target_spacing

    centroids_8mm = {}
    for label, (vi, vj, vk) in centroids_ras.items():
        new_idx = np.round(np.array([vi, vj, vk]) * scale).astype(int)
        new_idx = np.clip(new_idx, 0, np.array(vol_8mm.shape) - 1)
        centroids_8mm[label] = tuple(new_idx)

    from scipy.ndimage import gaussian_filter

    vol_smooth = gaussian_filter(vol_8mm, sigma=smooth_sigma)
    vol_norm = np.clip(vol_smooth / divisor, -1.0, 1.0).astype(np.float32)
    vol_out, offset = center_pad_crop(vol_norm, input_size)

    centroids_out = {}
    for label, vox in centroids_8mm.items():
        new_vox = np.array(vox) + offset
        new_vox = np.clip(new_vox, 0, np.array(input_size) - 1)
        centroids_out[label] = tuple(new_vox.astype(int))

    return vol_out, centroids_out, ras_affine, orig_affine


# ─── Dataset ──────────────────────────────────────────────────────────────────

class SpineCenterlineDataset(Dataset):
    """One item per subject: full 8 mm CT volume + spine centerline heatmap.

    Preprocessed data is cached as ``.npz`` on disk.
    """

    def __init__(self, csv_path, split, input_size=(64, 64, 128), sigma_spine=3.0,
                 augment=None, cache_dir="cache/stage1"):
        self.input_size = input_size
        self.sigma_spine = sigma_spine
        self.augment = augment

        self.cache_dir = os.path.join(cache_dir, split)
        os.makedirs(self.cache_dir, exist_ok=True)

        df = pd.read_csv(csv_path)
        df_split = df[df["type"] == split].reset_index(drop=True)

        self.samples = []
        for _, row in df_split.iterrows():
            subj_name = row.get("name", "?")
            if not (is_valid_path(row.get("image_path"))
                    and is_valid_path(row.get("centroid_json_path"))):
                continue

            npz_path = os.path.join(self.cache_dir, f"{subj_name}.npz")

            if not os.path.exists(npz_path):
                try:
                    vol_nib = nib.load(str(row["image_path"]))
                    _, raw_ctr = parse_centroid_json(str(row["centroid_json_path"]))
                    vol_pp, centroids_pp, _, _ = preprocess_stage1(
                        vol_nib, raw_ctr, input_size=input_size,
                    )
                    heatmap = generate_spine_centerline_heatmap(
                        centroids_pp, shape=input_size, sigma=sigma_spine,
                    )
                    np.savez_compressed(npz_path, ct_vol=vol_pp, heatmap=heatmap)
                except Exception:
                    continue

            self.samples.append({"patch_path": npz_path, "subject": subj_name})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        data = np.load(s["patch_path"])
        ct = data["ct_vol"].astype(np.float32)
        hm = data["heatmap"].astype(np.float32)

        if self.augment is not None:
            ct, hm = self.augment(ct, hm)

        return {
            "ct_vol": torch.from_numpy(ct[None]),
            "heatmap": torch.from_numpy(hm[None]),
            "subject": s["subject"],
        }


# ─── Augmentation ─────────────────────────────────────────────────────────────

class Augment3D_Stage1:
    """Intensity jitter (CT only) + small rotation (CT & heatmap jointly)."""

    def __init__(self, cfg):
        self.int_scale = cfg["aug_intensity_scale"]
        self.int_shift = cfg["aug_intensity_shift"]
        self.rot = cfg.get("aug_rotation", (-15, 15))

    @staticmethod
    def _build_rot_matrix(shape, rx, ry, rz):
        cx, cy, cz = [s / 2.0 for s in shape]
        cos_x, sin_x = np.cos(rx), np.sin(rx)
        cos_y, sin_y = np.cos(ry), np.sin(ry)
        cos_z, sin_z = np.cos(rz), np.sin(rz)
        Rx = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        Ry = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        Rz = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        offset = np.array([cx, cy, cz]) - R @ np.array([cx, cy, cz])
        return R, offset

    def __call__(self, ct, heatmap):
        scale = np.random.uniform(*self.int_scale)
        shift = np.random.uniform(*self.int_shift)
        ct = np.clip(ct * scale + shift, -1.0, 1.0)

        r = [np.radians(np.random.uniform(*self.rot)) for _ in range(3)]
        R, off = self._build_rot_matrix(ct.shape, *r)
        ct = ndi.affine_transform(ct, R, offset=off, order=1, cval=0.0)
        heatmap = ndi.affine_transform(heatmap, R, offset=off, order=1, cval=0.0)
        heatmap = np.clip(heatmap, 0.0, 1.0)

        return ct.astype(np.float32), heatmap.astype(np.float32)
