"""Stage 1 ground-truth heatmap generation."""

import numpy as np
from scipy.ndimage import center_of_mass


def generate_3d_gaussian(center, sigma, shape):
    """Generate a single 3-D Gaussian blob at *center* with given *sigma*."""
    ci, cj, ck = center
    si, sj, sk = shape
    ii = np.arange(si) - ci
    jj = np.arange(sj) - cj
    kk = np.arange(sk) - ck
    grid_i, grid_j, grid_k = np.meshgrid(ii, jj, kk, indexing="ij")
    heatmap = np.exp(-(grid_i**2 + grid_j**2 + grid_k**2) / (2.0 * sigma**2))
    return heatmap.astype(np.float32)


def generate_spine_centerline_heatmap(centroids, shape, sigma=3.0):
    """Build the spinal centerline heatmap by merging per-vertebra Gaussians.

    Parameters
    ----------
    centroids : dict[int, tuple] or list[tuple]
        ``{label: (i, j, k)}`` voxel positions.
    shape : tuple
        Output shape, e.g. ``(64, 64, 128)``.
    sigma : float
        Gaussian standard deviation in voxels.

    Returns
    -------
    heatmap : np.ndarray, float32
    """
    heatmap = np.zeros(shape, dtype=np.float32)
    coords = centroids.values() if isinstance(centroids, dict) else centroids
    for coord in coords:
        heatmap = np.maximum(heatmap, generate_3d_gaussian(coord, sigma, shape))
    return heatmap


def extract_spine_coordinate(predicted_heatmap, target_spacing=8.0, offset=None):
    """Extract the (X, Y) spine-center from a predicted centerline heatmap.

    Returns
    -------
    com_x_mm, com_y_mm : float
        Physical (mm) coordinates in the 8 mm RAS space.
    com_vox : tuple[float, float, float]
        Voxel indices in the final grid (for diagnostics).
    """
    hm_relu = np.maximum(predicted_heatmap, 0.0)

    if hm_relu.sum() == 0:
        ci, cj, ck = [s // 2 for s in predicted_heatmap.shape]
    else:
        ci, cj, ck = center_of_mass(hm_relu)

    if offset is not None:
        ci -= offset[0]
        cj -= offset[1]

    com_x_mm = ci * target_spacing
    com_y_mm = cj * target_spacing
    return com_x_mm, com_y_mm, (float(ci + (offset[0] if offset is not None else 0)),
                                 float(cj + (offset[1] if offset is not None else 0)),
                                 float(ck + (offset[2] if offset is not None else 0)))
