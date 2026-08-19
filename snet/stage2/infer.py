"""Stage 2 inference: predict vertebra centroids from SC-Net heatmaps."""

import os
import gc
import numpy as np
import pandas as pd
import nibabel as nib
import torch

from snet.config import STAGE2_CFG as CFG
from snet.data.io import is_valid_path, parse_centroid_json
from snet.data.preprocessing import full_preprocess, centroids_to_voxel
from snet.stage2.model import SCNet
from snet.stage2.dataset import simulate_stage1_crop


def predict_centroids(model, vol_pp, centroids_2mm, crop_size, device):
    """Run inference on a single preprocessed volume.

    Returns
    -------
    pred_centroids_crop : dict {label: (ci, cj, ck)} in crop space
    pred_centroids_vol : dict {label: (ci, cj, ck)} in full-volume space
    heatmaps_np : np.ndarray [25, CX, CY, CZ]
    """
    CX, CY, CZ = crop_size

    if centroids_2mm:
        scx = int(round(np.mean([v[0] for v in centroids_2mm.values()])))
        scy = int(round(np.mean([v[1] for v in centroids_2mm.values()])))
    else:
        scx, scy = vol_pp.shape[0] // 2, vol_pp.shape[1] // 2

    rng_fixed = np.random.default_rng(0)
    ct_crop, z_start, scx_used, scy_used = simulate_stage1_crop(
        vol_pp, centroids_2mm if centroids_2mm else {0: (scx, scy, 0)},
        crop_size, rng=rng_fixed,
    )

    inp_t = torch.from_numpy(ct_crop[None, None].astype(np.float32)).to(device)
    with torch.no_grad():
        pred_hm, _ = model(inp_t)
    heatmaps_np = pred_hm[0].cpu().numpy()

    pred_centroids_crop = {}
    pred_centroids_vol = {}
    for vert_idx in range(CFG["max_vertebrae"]):
        label = vert_idx + 1
        hm = heatmaps_np[vert_idx]
        flat_idx = np.argmax(hm)
        ci_c, cj_c, ck_c = np.unravel_index(flat_idx, hm.shape)
        pred_centroids_crop[label] = (int(ci_c), int(cj_c), int(ck_c))

        hx, hy = CX // 2, CY // 2
        x0 = scx_used - hx
        y0 = scy_used - hy
        pred_centroids_vol[label] = (int(ci_c + x0), int(cj_c + y0), int(ck_c + z_start))

    return pred_centroids_crop, pred_centroids_vol, heatmaps_np


def run_inference(csv_path=None, checkpoint_path=None, device=None):
    """Run Stage-2 inference on all test subjects.

    Returns list of per-vertebra result dicts.
    """
    csv_path = csv_path or CFG["csv_path"]
    checkpoint_path = checkpoint_path or CFG["checkpoint_path"]
    device = torch.device(device or CFG["device"])

    model = SCNet(in_channels=1, num_classes=CFG["max_vertebrae"],
                  sigma_init=CFG["sigma_init"]).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    model.eval()

    df_test = pd.read_csv(csv_path)
    df_test = df_test[df_test["type"] == "test"].reset_index(drop=True)

    all_results = []
    for _, row in df_test.iterrows():
        subj_name = row["name"]
        try:
            vol_nib = nib.load(str(row["image_path"]))
            mask_nib = nib.load(str(row["mask_path"])) if is_valid_path(row.get("mask_path")) else None
            _, raw_centroids = parse_centroid_json(str(row["centroid_json_path"]))

            vol_pp, mask_pp, ras_affine, zooms, orig_affine = full_preprocess(
                vol_nib, mask_nib,
                target_spacing=CFG["target_spacing"],
                sigma=CFG["smooth_sigma"],
            )
            centroids_2mm = centroids_to_voxel(
                raw_centroids, orig_affine, ras_affine, zooms, CFG["target_spacing"],
            )
            for label in list(centroids_2mm.keys()):
                if mask_pp is not None and np.any(mask_pp == label):
                    coords = np.array(np.where(mask_pp == label))
                    centroids_2mm[label] = tuple(np.round(coords.mean(axis=1)).astype(int))

            _, pred_vol, _ = predict_centroids(
                model, vol_pp, centroids_2mm, CFG["crop_size"], device,
            )

            for label in sorted(centroids_2mm.keys()):
                gt_v = np.array(centroids_2mm[label])
                if label in pred_vol:
                    pr_v = np.array(pred_vol[label])
                    err = np.linalg.norm(gt_v - pr_v)
                    all_results.append({
                        "subject": subj_name, "label": label,
                        "error_vox": err, "error_mm": err * CFG["target_spacing"],
                    })

            del vol_pp, mask_pp, vol_nib
            gc.collect()

        except Exception as e:
            print(f"  Error on {subj_name}: {e}")

    return all_results


if __name__ == "__main__":
    results = run_inference()
    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
