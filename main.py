#!/usr/bin/env python
"""SNET pipeline entry point.

Usage
-----
    python main.py --stage 1 --mode train
    python main.py --stage 2 --mode infer
    python main.py --stage 3 --mode train
    python main.py --stage 123 --mode infer  # runs all 3 stages sequentially
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="SNET 3-Stage Spinal Segmentation Pipeline")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 123], default=123,
                        help="Which stage to run (1, 2, 3, or 123 for the full pipeline)")
    parser.add_argument("--mode", type=str, choices=["train", "infer"], default="train",
                        help="Run training or inference")
    parser.add_argument("--csv", type=str, default=None, help="Override CSV path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda/cpu)")
    args = parser.parse_args()

    stages = [1, 2, 3] if args.stage == 123 else [args.stage]

    for stage in stages:
        print(f"\n{'='*60}")
        print(f"  Stage {stage} — {args.mode.upper()}")
        print(f"{'='*60}\n")

        if stage == 1:
            if args.mode == "train":
                from snet.stage1.train import train
                if args.csv:
                    from snet.config import STAGE1_CFG
                    STAGE1_CFG["csv_path"] = args.csv
                if args.checkpoint:
                    from snet.config import STAGE1_CFG
                    STAGE1_CFG["checkpoint_path"] = args.checkpoint
                if args.device:
                    from snet.config import STAGE1_CFG
                    STAGE1_CFG["device"] = args.device
                train()
            else:
                from snet.stage1.infer import run_inference
                results = run_inference(csv_path=args.csv, checkpoint_path=args.checkpoint,
                                        device=args.device)
                if results:
                    import pandas as pd
                    print(pd.DataFrame(results).to_string(index=False))

        elif stage == 2:
            if args.mode == "train":
                from snet.stage2.train import train
                if args.csv:
                    from snet.config import STAGE2_CFG
                    STAGE2_CFG["csv_path"] = args.csv
                if args.checkpoint:
                    from snet.config import STAGE2_CFG
                    STAGE2_CFG["checkpoint_path"] = args.checkpoint
                if args.device:
                    from snet.config import STAGE2_CFG
                    STAGE2_CFG["device"] = args.device
                train()
            else:
                from snet.stage2.infer import run_inference
                results = run_inference(csv_path=args.csv, checkpoint_path=args.checkpoint,
                                        device=args.device)
                if results:
                    import pandas as pd
                    print(pd.DataFrame(results).to_string(index=False))

        elif stage == 3:
            if args.mode == "train":
                from snet.stage3.train import train
                if args.csv:
                    from snet.config import STAGE3_CFG
                    STAGE3_CFG["csv_path"] = args.csv
                if args.checkpoint:
                    from snet.config import STAGE3_CFG
                    STAGE3_CFG["checkpoint_path"] = args.checkpoint
                if args.device:
                    from snet.config import STAGE3_CFG
                    STAGE3_CFG["device"] = args.device
                train()
            else:
                from snet.stage3.infer import run_inference
                results = run_inference(csv_path=args.csv, checkpoint_path=args.checkpoint,
                                        device=args.device)
                if results:
                    from snet.eval import compute_dice, compute_hausdorff
                    for r in results:
                        if r["gt_bin"].sum() > 0:
                            dice = compute_dice(r["pred_bin"], r["gt_bin"])
                            hd = compute_hausdorff(r["pred_bin"], r["gt_bin"])
                            print(f"  {r['subject']} | label {r['label']:3d} | "
                                  f"Dice={dice:.4f} | HD95={hd:.2f} mm")

    print(f"\nPipeline finished.")


if __name__ == "__main__":
    main()
