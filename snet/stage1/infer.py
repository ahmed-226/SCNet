"""Stage 1 inference: extract spine (X, Y) center-of-mass for Stage 2."""

import gc
import numpy as np
import pandas as pd
import nibabel as nib
import torch

from snet.config import STAGE1_CFG as CFG
from snet.data.io import is_valid_path, parse_centroid_json
from snet.data.preprocessing import reorient_to_ras, center_pad_crop
from snet.stage1.model import UNet3D_Stage1
from snet.stage1.dataset import preprocess_stage1
from snet.stage1.heatmap import extract_spine_coordinate


def run_inference(csv_path=None, checkpoint_path=None, device=None):
    """Run Stage-1 inference on all test subjects.

    Returns a list of dicts with predicted and GT spine centers (mm).
    """
    csv_path = csv_path or CFG["csv_path"]
    checkpoint_path = checkpoint_path or CFG["checkpoint_path"]
    device = torch.device(device or CFG["device"])

    model = UNet3D_Stage1(in_channels=1, num_filters=64).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    model.eval()

    df_test = pd.read_csv(csv_path)
    df_test = df_test[df_test["type"] == "test"].reset_index(drop=True)

    results = []
    for _, row in df_test.iterrows():
        subj_name = row["name"]
        try:
            vol_nib = nib.load(str(row["image_path"]))
            _, raw_ctr = parse_centroid_json(str(row["centroid_json_path"]))

            vol_pp, centroids_pp, _, _ = preprocess_stage1(
                vol_nib, raw_ctr,
                target_spacing=CFG["target_spacing"],
                smooth_sigma=CFG["smooth_sigma"],
                divisor=CFG["intensity_divisor"],
                input_size=CFG["input_size"],
            )

            # Recompute offset
            vol_ras_tmp = reorient_to_ras(vol_nib)
            zooms_tmp = np.array(vol_ras_tmp.header.get_zooms()[:3], dtype=np.float64)
            shape_8mm = tuple(int(round(s * z / CFG["target_spacing"]))
                              for s, z in zip(vol_ras_tmp.shape, zooms_tmp))
            _, offset = center_pad_crop(np.zeros(shape_8mm), CFG["input_size"])
            del vol_ras_tmp

            inp_t = torch.from_numpy(vol_pp[None, None]).to(device)
            with torch.no_grad():
                pred_hm = model(inp_t)[0, 0].cpu().numpy()

            com_x_mm, com_y_mm, _ = extract_spine_coordinate(
                pred_hm, CFG["target_spacing"], offset)

            # GT spine center
            if centroids_pp:
                gt_xs = [v[0] for v in centroids_pp.values()]
                gt_ys = [v[1] for v in centroids_pp.values()]
                gt_x_mm = (np.mean(gt_xs) - offset[0]) * CFG["target_spacing"]
                gt_y_mm = (np.mean(gt_ys) - offset[1]) * CFG["target_spacing"]
            else:
                gt_x_mm = gt_y_mm = float("nan")

            results.append({
                "subject": subj_name,
                "com_x_mm": round(com_x_mm, 2),
                "com_y_mm": round(com_y_mm, 2),
                "gt_x_mm": round(gt_x_mm, 2),
                "gt_y_mm": round(gt_y_mm, 2),
                "error_mm": round(np.sqrt((com_x_mm - gt_x_mm) ** 2
                                          + (com_y_mm - gt_y_mm) ** 2), 2),
            })
            del vol_pp, pred_hm, vol_nib
            gc.collect()

        except Exception as e:
            print(f"  Error on {subj_name}: {e}")

    return results


import os

if __name__ == "__main__":
    results = run_inference()
    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
