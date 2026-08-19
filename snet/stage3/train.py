"""Stage 3 training loop."""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from snet.config import STAGE3_CFG as CFG
from snet.stage3.model import UNet3D
from snet.stage3.loss import BCEDiceLoss
from snet.stage3.dataset import VertebraeDataset, Augment3D_Stage3


def dice_coefficient(pred_bin, target_bin, eps=1e-6):
    pred_bin = pred_bin.bool().float()
    target_bin = target_bin.bool().float()
    intersection = (pred_bin * target_bin).sum()
    return (2.0 * intersection + eps) / (pred_bin.sum() + target_bin.sum() + eps)


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def validate(model, val_loader, criterion, device, threshold=0.5):
    model.eval()
    losses, dices = [], []
    with torch.no_grad():
        for batch in val_loader:
            inp = batch["input"].to(device)
            gt = batch["binary_mask"].to(device)
            pred = model(inp)
            losses.append(criterion(pred, gt).item())
            pred_bin = (pred > threshold).float()
            dices.append(dice_coefficient(pred_bin, gt).item())
    model.train()
    return float(np.mean(losses)), float(np.mean(dices))


def train():
    device = torch.device(CFG["device"])

    train_ds = VertebraeDataset(CFG["csv_path"], "train",
                                crop_size=CFG["crop_size"],
                                heatmap_sigma=CFG["heatmap_sigma"])
    val_ds = VertebraeDataset(CFG["csv_path"], "val",
                              crop_size=CFG["crop_size"],
                              heatmap_sigma=CFG["heatmap_sigma"])

    if len(train_ds) == 0:
        print("No training samples — aborting.")
        return

    train_ds.augment = Augment3D_Stage3(CFG)

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                              num_workers=CFG["num_workers"],
                              pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False,
                            num_workers=CFG["num_workers"],
                            pin_memory=(device.type == "cuda")) if len(val_ds) > 0 else None

    model = UNet3D(in_channels=2, num_filters=64).to(device)
    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"],
                                 weight_decay=CFG["weight_decay"])

    best_val_dice = -1.0
    start_iteration = 1

    if CFG["resume_path"] and os.path.exists(CFG["resume_path"]):
        ckpt = torch.load(CFG["resume_path"], map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        best_val_dice = ckpt.get("val_dice", -1.0)
        start_iteration = ckpt.get("iteration", 0) + 1
        print(f"Resumed from iteration {start_iteration}")

    os.makedirs(os.path.dirname(CFG["checkpoint_path"]) or ".", exist_ok=True)

    model.train()
    running_loss = 0.0
    data_iter = infinite_loader(train_loader)

    print(f"Stage 3 training: {CFG['iterations']:,} iterations")
    for iteration in range(start_iteration, CFG["iterations"] + 1):
        batch = next(data_iter)
        inp = batch["input"].to(device)
        gt = batch["binary_mask"].to(device)

        optimizer.zero_grad()
        loss = criterion(model(inp), gt)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if iteration % CFG["log_every"] == 0:
            avg = running_loss / CFG["log_every"]
            print(f"  Iter {iteration:6d}/{CFG['iterations']}  |  Train Loss: {avg:.5f}")
            running_loss = 0.0

        if iteration % CFG["val_every"] == 0 and val_loader is not None:
            val_loss, val_dice = validate(model, val_loader, criterion, device,
                                          CFG["sigmoid_threshold"])
            print(f"  -> Val Loss: {val_loss:.5f}  |  Val Dice: {val_dice:.4f}")
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save({
                    "iteration": iteration,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_dice": val_dice,
                }, CFG["checkpoint_path"])
                print(f"    * New best Dice={val_dice:.4f} — saved.")

    print(f"Training complete. Best Val Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    train()
