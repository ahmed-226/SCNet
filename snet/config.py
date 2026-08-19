"""Centralised configuration for all three pipeline stages."""

import torch

CUDA_AVAILABLE = torch.cuda.is_available()

# ─── Stage 1: Spinal Centerline Localisation ──────────────────────────────────
STAGE1_CFG = {
    # Paths
    "csv_path": "data/verse19_dataset.csv",
    "checkpoint_path": "checkpoints/best_model_stage1.pth",
    "resume_path": None,

    # Preprocessing — coarse 8 mm grid
    "target_spacing": 8.0,
    "smooth_sigma": 0.75,
    "intensity_divisor": 2048.0,
    "intensity_clamp": 1.0,

    # Input grid [X, Y, Z]
    "input_size": (64, 64, 128),

    # Ground-truth heatmap
    "sigma_spine": 3.0,

    # Augmentation
    "aug_intensity_scale": (0.75, 1.25),
    "aug_intensity_shift": (-0.25, 0.25),
    "aug_rotation": (-15, 15),

    # Training
    "lr": 1e-4,
    "weight_decay": 5e-4,
    "batch_size": 1,
    "iterations": 20_000,
    "log_every": 100,
    "val_every": 500,
    "num_workers": 2,

    # Device
    "device": "cuda" if CUDA_AVAILABLE else "cpu",
}

# ─── Stage 2: Vertebra Centroid Detection (SC-Net) ────────────────────────────
STAGE2_CFG = {
    # Paths
    "csv_path": "data/verse19_dataset.csv",
    "checkpoint_path": "checkpoints/best_model_stage2.pth",
    "resume_path": None,

    # Preprocessing — 2 mm isotropic
    "target_spacing": 2.0,
    "smooth_sigma": 0.75,
    "intensity_divisor": 2048.0,
    "intensity_clamp": 1.0,

    # Crop (X, Y, Z)
    "crop_size": (96, 96, 128),

    # SC-Net / Heatmap
    "max_vertebrae": 25,
    "sigma_init": 5.0,
    "alpha": 100.0,

    # Training
    "lr": 1e-8,
    "momentum": 0.9,
    "nesterov": True,
    "weight_decay": 5e-4,
    "batch_size": 1,
    "iterations": 50_000,
    "log_every": 200,
    "val_every": 1000,
    "num_workers": 2,

    # Device
    "device": "cuda" if CUDA_AVAILABLE else "cpu",
}

# ─── Stage 3: Per-Vertebra Binary Segmentation ────────────────────────────────
STAGE3_CFG = {
    # Paths
    "csv_path": "data/verse19_dataset.csv",
    "checkpoint_path": "checkpoints/best_model_stage3.pth",
    "resume_path": None,

    # Preprocessing — 1 mm isotropic
    "target_spacing": 1.0,
    "smooth_sigma": 0.75,
    "intensity_divisor": 2048.0,
    "intensity_clamp": 1.0,

    # Crop / Heatmap
    "crop_size": (128, 128, 96),
    "heatmap_sigma": 20.0,

    # Augmentation
    "aug_intensity_scale": (0.75, 1.25),
    "aug_intensity_shift": (-0.25, 0.25),
    "aug_rotation": (-15, 15),

    # Training
    "lr": 1e-4,
    "weight_decay": 1e-7,
    "batch_size": 1,
    "iterations": 10_000,
    "log_every": 50,
    "val_every": 100,
    "num_workers": 2,

    # Inference
    "sigmoid_threshold": 0.5,

    # Device
    "device": "cuda" if CUDA_AVAILABLE else "cpu",
}
