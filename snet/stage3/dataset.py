"""Stage 3 dataset, crop utilities, and augmentation."""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import scipy.ndimage as ndi
import torch
from torch.utils.data import Dataset

from snet.data.io import is_valid_path, parse_centroid_json
from snet.data.preprocessing import full_preprocess, centroids_to_voxel


# ─── Crop utilities ───────────────────────────────────────────────────────────

def crop_volume_around_centroid(volume, centroid_vox, crop_size=(128, 128, 96)):
    """Crop a 3-D volume centred on *centroid_vox*, zero-padding near borders.

    Returns
    -------
    crop : np.ndarray, shape crop_size
    offset : tuple (i0, j0, k0)
    """
    ci, cj, ck = [int(round(c)) for c in centroid_vox]
    si, sj, sk = crop_size
    hi, hj, hk = si // 2, sj // 2, sk // 2

    i0, j0, k0 = ci - hi, cj - hj, ck - hk
    i1, j1, k1 = i0 + si, j0 + sj, k0 + sk

    vol_i, vol_j, vol_k = volume.shape

    is_ = max(i0, 0); ie = min(i1, vol_i)
    js = max(j0, 0); je = min(j1, vol_j)
    ks = max(k0, 0); ke = min(k1, vol_k)

    pi0 = is_ - i0; pi1 = pi0 + (ie - is_)
    pj0 = js - j0; pj1 = pj0 + (je - js)
    pk0 = ks - k0; pk1 = pk0 + (ke - ks)

    crop = np.zeros(crop_size, dtype=volume.dtype)
    crop[pi0:pi1, pj0:pj1, pk0:pk1] = volume[is_:ie, js:je, ks:ke]
    return crop, (i0, j0, k0)


def generate_gaussian_heatmap(crop_size=(128, 128, 96), sigma=20.0):
    """3-D Gaussian heatmap always centred at the exact crop centre."""
    si, sj, sk = crop_size
    hi, hj, hk = si // 2, sj // 2, sk // 2
    ii = np.arange(si) - hi
    jj = np.arange(sj) - hj
    kk = np.arange(sk) - hk
    grid_i, grid_j, grid_k = np.meshgrid(ii, jj, kk, indexing="ij")
    heatmap = np.exp(-(grid_i**2 + grid_j**2 + grid_k**2) / (2.0 * sigma**2))
    return heatmap.astype(np.float32)


def extract_binary_mask(mask_crop, target_label):
    """Binarise a multi-label mask crop for *target_label*."""
    return (mask_crop == target_label).astype(np.float32)


# ─── Augmentation ─────────────────────────────────────────────────────────────

class Augment3D_Stage3:
    """Intensity jitter + small rotation, applied jointly to CT and mask."""

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

    def __call__(self, ct, mask):
        scale = np.random.uniform(*self.int_scale)
        shift = np.random.uniform(*self.int_shift)
        ct = np.clip(ct * scale + shift, -1.0, 1.0)

        r = [np.radians(np.random.uniform(*self.rot)) for _ in range(3)]
        R, offset = self._build_rot_matrix(ct.shape, *r)
        ct = ndi.affine_transform(ct, R, offset=offset, order=1, cval=0.0)
        mask = ndi.affine_transform(mask, R, offset=offset, order=0, cval=0.0)

        return ct.astype(np.float32), mask.astype(np.float32)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class VertebraeDataset(Dataset):
    """One item per (subject, vertebra) pair, with disk-cached patches."""

    def __init__(self, csv_path, split, crop_size=(128, 128, 96),
                 heatmap_sigma=20.0, augment=None, cache_dir="cache/stage3"):
        self.crop_size = crop_size
        self.heatmap_sigma = heatmap_sigma
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

            mask_nib = nib.load(str(row["mask_path"])) if is_valid_path(row.get("mask_path")) else None
            _, raw_centroids = parse_centroid_json(str(row["centroid_json_path"]))

            needs_processing = False
            subject_patch_paths = {}

            for label in raw_centroids.keys():
                npz_path = os.path.join(self.cache_dir, f"{subj_name}_lbl{label}.npz")
                if os.path.exists(npz_path):
                    subject_patch_paths[label] = npz_path
                else:
                    needs_processing = True
                    subject_patch_paths[label] = npz_path

            if needs_processing:
                vol_nib = nib.load(str(row["image_path"]))
                vol_pp, mask_pp, ras_affine, zooms, orig_affine = full_preprocess(vol_nib, mask_nib)
                centroids_1mm = centroids_to_voxel(raw_centroids, orig_affine, ras_affine, zooms)
                vol_shape = np.array(vol_pp.shape)

                for label, vox in centroids_1mm.items():
                    if mask_nib is not None and np.any(mask_pp == label):
                        coords = np.array(np.where(mask_pp == label))
                        vox = tuple(np.round(coords.mean(axis=1)).astype(int))

                    vox_clamped = tuple(int(np.clip(v, 0, vol_shape[ax] - 1))
                                        for ax, v in enumerate(vox))
                    ct_crop, _ = crop_volume_around_centroid(vol_pp, vox_clamped, self.crop_size)
                    mk_crop, _ = crop_volume_around_centroid(mask_pp, vox_clamped, self.crop_size)
                    np.savez_compressed(subject_patch_paths[label], ct_crop=ct_crop, mk_crop=mk_crop)
                del vol_pp, mask_pp, vol_nib

            for label, patch_path in subject_patch_paths.items():
                if os.path.exists(patch_path):
                    self.samples.append({
                        "patch_path": patch_path,
                        "label": label,
                        "subject": subj_name,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        data = np.load(s["patch_path"])
        ct_crop = data["ct_crop"]
        mk_crop = data["mk_crop"]

        heatmap = generate_gaussian_heatmap(self.crop_size, self.heatmap_sigma)
        binary_mask = extract_binary_mask(mk_crop, s["label"])

        if self.augment is not None:
            ct_crop, binary_mask = self.augment(ct_crop, binary_mask)

        ct_t = torch.from_numpy(ct_crop[None])
        hm_t = torch.from_numpy(heatmap[None])
        bm_t = torch.from_numpy(binary_mask[None])
        inp_t = torch.cat([ct_t, hm_t], dim=0)

        return {
            "input": inp_t,
            "ct_crop": ct_t,
            "heatmap": hm_t,
            "binary_mask": bm_t,
            "label": s["label"],
            "subject": s["subject"],
        }
