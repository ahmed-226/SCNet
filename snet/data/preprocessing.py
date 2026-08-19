"""Shared preprocessing utilities used across all three stages."""

import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter, zoom


# ─── Orientation helpers ──────────────────────────────────────────────────────

def reorient_to_ras(nib_img):
    """Reorient a NIfTI image to standard RAS+ orientation."""
    orig_ornt = nib.io_orientation(nib_img.affine)
    ras_ornt = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(orig_ornt, ras_ornt)
    return nib_img.as_reoriented(transform)


def world_to_voxel(affine, world_xyz):
    """Convert a world-space (mm) coordinate to voxel indices using the affine."""
    inv_affine = np.linalg.inv(affine)
    vox = inv_affine[:3, :3] @ np.array(world_xyz) + inv_affine[:3, 3]
    return np.round(vox).astype(int)


def resample_volume(vol, current_spacing, target_spacing=1.0, order=1):
    """Resample a volume to target isotropic spacing via ``scipy.ndimage.zoom``."""
    zoom_factors = np.array(current_spacing) / target_spacing
    return zoom(vol, zoom_factors, order=order)


def center_pad_crop(vol, target_shape):
    """Center-pad or crop *vol* to *target_shape*.

    Returns
    -------
    out : np.ndarray
        The padded/cropped volume with shape ``target_shape``.
    offset : np.ndarray, shape (3,)
        Shift to apply to voxel indices when mapping from the original
        (pre-pad/crop) space into the final grid:
        ``final_idx = original_idx + offset``.
    """
    in_shape = np.array(vol.shape)
    out_shape = np.array(target_shape)
    insert = (out_shape - in_shape) // 2

    out = np.zeros(out_shape, dtype=vol.dtype)

    src_slices, dst_slices = [], []
    for d in range(3):
        src_start = max(0, -insert[d])
        src_end = src_start + min(in_shape[d], out_shape[d] - max(0, insert[d]))
        dst_start = max(0, insert[d])
        dst_end = dst_start + (src_end - src_start)
        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))

    out[tuple(dst_slices)] = vol[tuple(src_slices)]
    offset = np.array([dst_slices[d].start - src_slices[d].start for d in range(3)])
    return out, offset


# ─── Full preprocessing pipelines ─────────────────────────────────────────────

def full_preprocess(vol_nib, mask_nib=None, target_spacing=1.0, sigma=0.75,
                    divisor=2048.0):
    """Preprocessing pipeline shared by Stages 1-3.

    Steps: RAS reorientation → resample → Gaussian smooth → intensity normalise.

    Parameters
    ----------
    vol_nib : nibabel.Nifti1Image
    mask_nib : nibabel.Nifti1Image or None
    target_spacing : float
        Target isotropic spacing in mm.
    sigma : float
        Gaussian smoothing sigma.
    divisor : float
        Intensity normalisation divisor.

    Returns
    -------
    vol_n : np.ndarray, float32
    mask_out : np.ndarray, int16
    ras_affine : np.ndarray, (4, 4)
    zooms : np.ndarray, (3,)
    orig_affine : np.ndarray, (4, 4)
    """
    orig_affine = vol_nib.affine.copy()

    vol_ras = reorient_to_ras(vol_nib)
    ras_affine = vol_ras.affine.copy()
    zooms = np.array(vol_ras.header.get_zooms()[:3], dtype=np.float64)
    vol_data = vol_ras.get_fdata(dtype=np.float32)

    vol_r = resample_volume(vol_data, zooms, target_spacing, order=1)
    vol_s = gaussian_filter(vol_r, sigma=sigma)
    vol_n = np.clip(vol_s / divisor, -1.0, 1.0).astype(np.float32)

    if mask_nib is not None:
        mask_ras = reorient_to_ras(mask_nib)
        mask_data = mask_ras.get_fdata().astype(np.int16)
        mask_out = resample_volume(mask_data, zooms, target_spacing, order=0).astype(np.int16)
    else:
        mask_out = np.zeros(vol_n.shape, dtype=np.int16)

    return vol_n, mask_out, ras_affine, zooms, orig_affine


def centroids_to_voxel(raw_centroids, original_affine, ras_affine,
                       original_zooms, target_spacing=1.0):
    """Convert JSON centroids to integer voxel indices in the resampled RAS volume.

    Works identically for any target spacing (used with 1 mm, 2 mm, or 8 mm).
    When a mask is available, callers should override each centroid with the
    mask's center-of-mass (same pattern as the original notebooks).
    """
    scale = original_zooms / target_spacing
    inv_affine = np.linalg.inv(ras_affine)
    result = {}
    for label, (vx, vy, vz) in raw_centroids.items():
        world_pt = original_affine @ np.array([vx, vy, vz, 1.0])
        ras_vox = inv_affine[:3, :3] @ world_pt[:3] + inv_affine[:3, 3]
        vox_new = np.round(ras_vox * scale).astype(int)
        result[label] = tuple(vox_new)
    return result
