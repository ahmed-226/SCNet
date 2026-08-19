"""Stage 1 training loop."""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from snet.config import STAGE1_CFG as CFG
from snet.stage1.model import UNet3D_Stage1
from snet.stage1.loss import WeightedMSELoss
from snet.stage1.dataset import SpineCenterlineDataset, Augment3D_Stage1


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def validate(model, val_loader, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_loader:
            inp = batch["ct_vol"].to(device)
            gt = batch["heatmap"].to(device)
            losses.append(criterion(model(inp), gt).item())
    model.train()
    return float(np.mean(losses)) if losses else float("inf")


def train():
    device = torch.device(CFG["device"])

    # Datasets
    train_ds = SpineCenterlineDataset(CFG["csv_path"], "train",
                                      input_size=CFG["input_size"],
                                      sigma_spine=CFG["sigma_spine"])
    val_ds = SpineCenterlineDataset(CFG["csv_path"], "val",
                                    input_size=CFG["input_size"],
                                    sigma_spine=CFG["sigma_spine"])

    if len(train_ds) == 0:
        print("No training samples — aborting.")
        return

    train_ds.augment = Augment3D_Stage1(CFG)

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                              num_workers=CFG["num_workers"],
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False,
                            num_workers=CFG["num_workers"],
                            pin_memory=(device.type == "cuda")) if len(val_ds) > 0 else None

    # Model
    model = UNet3D_Stage1(in_channels=1, num_filters=64).to(device)
    criterion = WeightedMSELoss(pos_weight=10.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"],
                                 weight_decay=CFG["weight_decay"])

    best_val_loss = float("inf")
    start_iteration = 1

    # Resume
    if CFG["resume_path"] and os.path.exists(CFG["resume_path"]):
        ckpt = torch.load(CFG["resume_path"], map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        best_val_loss = ckpt.get("val_loss", float("inf"))
        start_iteration = ckpt.get("iteration", 0) + 1
        print(f"Resumed from iteration {start_iteration}")

    os.makedirs(os.path.dirname(CFG["checkpoint_path"]) or ".", exist_ok=True)

    # Training
    model.train()
    running_loss = 0.0
    data_iter = infinite_loader(train_loader)

    print(f"Stage 1 training: {CFG['iterations']:,} iterations")
    for iteration in range(start_iteration, CFG["iterations"] + 1):
        batch = next(data_iter)
        inp = batch["ct_vol"].to(device)
        gt = batch["heatmap"].to(device)

        optimizer.zero_grad()
        loss = criterion(model(inp), gt)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if iteration % CFG["log_every"] == 0:
            avg = running_loss / CFG["log_every"]
            print(f"  Iter {iteration:6d}/{CFG['iterations']}  |  Train MSE: {avg:.6f}")
            running_loss = 0.0

        if iteration % CFG["val_every"] == 0 and val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device)
            print(f"  -> Val  MSE : {val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "iteration": iteration,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                }, CFG["checkpoint_path"])
                print(f"    * New best val MSE={val_loss:.6f} — saved.")

    print(f"Training complete. Best Val MSE: {best_val_loss:.6f}")


if __name__ == "__main__":
    train()
