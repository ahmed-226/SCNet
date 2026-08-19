"""Stage 3 inference: per-vertebra segmentation and volume reconstruction."""

import os
import gc
import numpy as np
import pandas as pd
import nibabel as nib
import torch

from snet.config import STAGE3_CFG as CFG
from snet.data.io import is_valid_path, parse_centroid_json
from snet.data.preprocessing import full_preprocess, centroids_to_voxel
from snet.stage3.model import UNet3D
from snet.stage3.dataset import (
    crop_volume_around_centroid,
    generate_gaussian_heatmap,
)


def merge_crop_into_volume(pred_prob, vol_shape, offset, crop_size, label,
                           pred_label_vol, max_prob_vol, sigmoid_thr=0.5):
    """Merge a per-vertebra prediction into the master volume in-place."""
    ox, oy, oz = offset
    sx, sy, sz = crop_size
    vx, vy, vz = vol_shape

    xs = max(ox, 0); ys = max(oy, 0); zs = max(oz, 0)
    xe = min(ox + sx, vx); ye = min(oy + sy, vy); ze = min(oz + sz, vz)

    px0 = xs - ox; py0 = ys - oy; pz0 = zs - oz
    px1 = px0 + (xe - xs); py1 = py0 + (ye - ys); pz1 = pz0 + (ze - zs)

    crop_valid = pred_prob[px0:px1, py0:py1, pz0:pz1]
    current_max = max_prob_vol[xs:xe, ys:ye, zs:ze]

    update_mask = (crop_valid > sigmoid_thr) & (crop_valid > current_max)
    max_prob_vol[xs:xe, ys:ye, zs:ze][update_mask] = crop_valid[update_mask]
    pred_label_vol[xs:xe, ys:ye, zs:ze][update_mask] = label


def run_inference(csv_path=None, checkpoint_path=None, device=None):
    """Run Stage-3 inference on all test subjects.

    Returns list of per-vertebra result dicts with pred/gt binary masks.
    """
    csv_path = csv_path or CFG["csv_path"]
    checkpoint_path = checkpoint_path or CFG["checkpoint_path"]
    device = torch.device(device or CFG["device"])

    model = UNet3D(in_channels=2, num_filters=64).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    model.eval()

    df_test = pd.read_csv(csv_path)
    df_test = df_test[df_test["type"] == "test"].reset_index(drop=True)

    test_results = []
    for _, row in df_test.iterrows():
        subj_name = row["name"]
        try:
            vol_nib = nib.load(str(row["image_path"]))
            mask_nib = nib.load(str(row["mask_path"])) if is_valid_path(row.get("mask_path")) else None

            vol_pp, mask_pp, ras_affine, zooms, orig_affine = full_preprocess(vol_nib, mask_nib)
            vol_shape = vol_pp.shape

            _, raw_centroids = parse_centroid_json(str(row["centroid_json_path"]))
            centroids_1mm = centroids_to_voxel(raw_centroids, orig_affine, ras_affine, zooms)

            pred_label_vol = np.zeros(vol_shape, dtype=np.int16)
            max_prob_vol = np.zeros(vol_shape, dtype=np.float32)

            for label, vox_idx in centroids_1mm.items():
                ct_crop, offset = crop_volume_around_centroid(vol_pp, vox_idx, CFG["crop_size"])
                heatmap = generate_gaussian_heatmap(CFG["crop_size"], CFG["heatmap_sigma"])

                inp_t = torch.from_numpy(
                    np.stack([ct_crop, heatmap], axis=0)[None]
                ).to(device)

                with torch.no_grad():
                    pred_sigmoid = model(inp_t)[0, 0].cpu().numpy()

                merge_crop_into_volume(pred_sigmoid, vol_shape, offset, CFG["crop_size"],
                                       label, pred_label_vol, max_prob_vol, CFG["sigmoid_threshold"])

            # Save as NIfTI
            save_path = f"pred_{subj_name}_seg.nii.gz"
            out_nib = nib.Nifti1Image(pred_label_vol, np.eye(4))
            nib.save(out_nib, save_path)

            for label in sorted(centroids_1mm.keys()):
                pred_bin = (pred_label_vol == label)
                gt_bin = (mask_pp == label) if mask_pp is not None else np.zeros_like(pred_bin, dtype=bool)
                test_results.append({
                    "subject": subj_name,
                    "label": label,
                    "pred_bin": pred_bin,
                    "gt_bin": gt_bin,
                })

            del vol_pp, mask_pp, pred_label_vol, max_prob_vol, vol_nib
            gc.collect()

        except Exception as e:
            print(f"  Error on {subj_name}: {e}")

    return test_results


if __name__ == "__main__":
    results = run_inference()
    print(f"Inference complete. {len(results)} vertebra results.")
