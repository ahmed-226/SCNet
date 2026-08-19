"""Stage 2 training loop."""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from snet.config import STAGE2_CFG as CFG
from snet.stage2.model import SCNet
from snet.stage2.loss import modified_l2_loss
from snet.stage2.dataset import SpineDatasetStage2, IntensityAugment, stage2_collate


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def validate(model, val_loader, crop_size, alpha, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_loader:
            inp = batch["ct_crop"].to(device)
            gt_list = batch["gt_centroids_crop"]
            pred_hm, sigmas = model(inp)
            losses.append(modified_l2_loss(pred_hm, gt_list, sigmas, crop_size,
                                           alpha=alpha, pos_weight=100.0).item())
    model.train()
    return float(np.mean(losses)) if losses else float("inf")


def train():
    device = torch.device(CFG["device"])

    train_ds = SpineDatasetStage2(CFG["csv_path"], "train",
                                  crop_size=CFG["crop_size"],
                                  target_spacing=CFG["target_spacing"],
                                  sigma=CFG["smooth_sigma"])
    val_ds = SpineDatasetStage2(CFG["csv_path"], "val",
                                crop_size=CFG["crop_size"],
                                target_spacing=CFG["target_spacing"],
                                sigma=CFG["smooth_sigma"])

    if len(train_ds) == 0:
        print("No training samples — aborting.")
        return

    train_ds.augment = IntensityAugment()

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                              num_workers=CFG["num_workers"],
                              pin_memory=(device.type == "cuda"),
                              drop_last=True, collate_fn=stage2_collate)
    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False,
                            num_workers=CFG["num_workers"],
                            pin_memory=(device.type == "cuda"),
                            collate_fn=stage2_collate) if len(val_ds) > 0 else None

    model = SCNet(in_channels=1, num_classes=CFG["max_vertebrae"],
                  sigma_init=CFG["sigma_init"]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=CFG["lr"],
                                momentum=CFG["momentum"], nesterov=CFG["nesterov"],
                                weight_decay=CFG["weight_decay"])

    best_val_loss = float("inf")
    start_iteration = 1

    if CFG["resume_path"] and os.path.exists(CFG["resume_path"]):
        ckpt = torch.load(CFG["resume_path"], map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        best_val_loss = ckpt.get("val_loss", float("inf"))
        start_iteration = ckpt.get("iteration", 0) + 1
        print(f"Resumed from iteration {start_iteration}")

    os.makedirs(os.path.dirname(CFG["checkpoint_path"]) or ".", exist_ok=True)

    model.train()
    running_loss = 0.0
    data_iter = infinite_loader(train_loader)

    print(f"Stage 2 training: {CFG['iterations']:,} iterations")
    for iteration in range(start_iteration, CFG["iterations"] + 1):
        batch = next(data_iter)
        inp = batch["ct_crop"].to(device)
        gt_list = batch["gt_centroids_crop"]

        pred_hm, sigmas = model(inp)
        loss = modified_l2_loss(pred_hm, gt_list, sigmas, CFG["crop_size"],
                                alpha=CFG["alpha"], pos_weight=100.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if iteration % CFG["log_every"] == 0:
            avg = running_loss / CFG["log_every"]
            sig_min = model.learned_sigmas.min().item()
            sig_max = model.learned_sigmas.max().item()
            print(f"  Iter {iteration:6d}/{CFG['iterations']}  |  "
                  f"Train Loss: {avg:.5f}  |  sigma [{sig_min:.3f}, {sig_max:.3f}]")
            running_loss = 0.0

        if iteration % CFG["val_every"] == 0 and val_loader is not None:
            val_loss = validate(model, val_loader, CFG["crop_size"], CFG["alpha"], device)
            print(f"  -> Val Loss: {val_loss:.5f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "iteration": iteration,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                }, CFG["checkpoint_path"])
                print(f"    * New best val loss={val_loss:.5f} — saved.")

    print(f"Training complete. Best Val Loss: {best_val_loss:.5f}")


if __name__ == "__main__":
    train()
