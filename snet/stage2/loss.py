"""Stage 2 loss: modified L2 with dynamic GT heatmap generation."""

import torch
import torch.nn as nn


def generate_gt_heatmap_3d(centroid_crop, sigma, crop_size):
    """Generate a 3-D Gaussian GT heatmap for a single vertebra.

    Parameters
    ----------
    centroid_crop : (ci, cj, ck) in crop coordinates
    sigma : float or Tensor scalar
    crop_size : (CX, CY, CZ)
    """
    CX, CY, CZ = crop_size
    ci, cj, ck = centroid_crop

    ii = torch.arange(CX, dtype=torch.float32) - ci
    jj = torch.arange(CY, dtype=torch.float32) - cj
    kk = torch.arange(CZ, dtype=torch.float32) - ck

    gi, gj, gk = torch.meshgrid(ii, jj, kk, indexing="ij")
    dist_sq = gi**2 + gj**2 + gk**2
    sigma_f = sigma.float() if isinstance(sigma, torch.Tensor) else torch.tensor(float(sigma))
    dist_sq = dist_sq.to(sigma_f.device)

    return torch.exp(-dist_sq / (2.0 * sigma_f**2))


def modified_l2_loss(predicted_heatmaps, gt_centroids_crop, learned_sigmas,
                     crop_size, max_vertebrae=25, alpha=100.0, pos_weight=100.0):
    """Modified L2 loss with sigma penalty and dynamic GT generation.

    Parameters
    ----------
    predicted_heatmaps : Tensor [B, 25, CX, CY, CZ]
    gt_centroids_crop : list of dicts (length B), each ``{label: (ci, cj, ck)}``
    learned_sigmas : Tensor [25] (nn.Parameter)
    crop_size : (CX, CY, CZ)
    alpha : sigma penalty weight
    """
    B = predicted_heatmaps.shape[0]
    CX, CY, CZ = crop_size
    device = predicted_heatmaps.device

    loss_acc = predicted_heatmaps.sum() * 0.0

    for b in range(B):
        for vert_idx in range(max_vertebrae):
            label = vert_idx + 1
            pred_h = predicted_heatmaps[b, vert_idx]
            sigma_i = learned_sigmas[vert_idx]

            centroid = gt_centroids_crop[b].get(label, None)
            in_window = centroid is not None and 0 <= centroid[2] < CZ

            if in_window:
                gt_h = generate_gt_heatmap_3d(centroid, sigma_i, crop_size).to(device)
            else:
                gt_h = torch.zeros((CX, CY, CZ), device=device)

            weight_mask = torch.where(gt_h > 0.05, pos_weight, 1.0)
            heatmap_loss = ((pred_h - gt_h) ** 2 * weight_mask).sum()
            sigma_penalty = alpha * (sigma_i ** 2)

            loss_acc = loss_acc + heatmap_loss + sigma_penalty

    return loss_acc / B
