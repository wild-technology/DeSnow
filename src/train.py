"""
Training script for marine snow removal models.

Supports:
- L1 (pixel) loss
- Perceptual loss (VGG feature matching)
- Cosine annealing LR schedule
- Checkpoint saving/resuming
- torch.compile() for Blackwell (sm_120) optimization
- Automatic mixed precision (bfloat16) for RTX 5090 Tensor Cores

Target hardware: RTX 5090 (32GB VRAM) + CUDA 13.2 + Threadripper
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from src.dataset import MarineSnowDataset, SyntheticSnowDataset
from src.models.unet import UNet, LightweightUNet


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for preserving texture and structure."""

    def __init__(self, layers: tuple = (3, 8, 15)):
        super().__init__()
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        except (ImportError, Exception):
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True).features

        self.blocks = nn.ModuleList()
        prev = 0
        for layer_idx in layers:
            self.blocks.append(vgg[prev:layer_idx + 1])
            prev = layer_idx + 1

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=pred.device)
        x_pred = pred
        x_target = target
        for block in self.blocks:
            x_pred = block(x_pred)
            x_target = block(x_target)
            loss += nn.functional.l1_loss(x_pred, x_target)
        return loss


def build_model(config: dict) -> nn.Module:
    """Build model from config."""
    arch = config["model"]["architecture"]
    if arch == "unet":
        return UNet(
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
            base_filters=config["model"]["base_filters"],
            depth=config["model"]["depth"],
        )
    elif arch == "lightweight_unet":
        return LightweightUNet(
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
            base_filters=config["model"]["base_filters"],
            depth=config["model"]["depth"],
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def train(config_path: str, resume: str = None):
    """Run training."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Print GPU info if available
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_mem:.0f}GB)")
        print(f"CUDA version: {torch.version.cuda}")

    # Build model
    model = build_model(config).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    # torch.compile() for Blackwell optimization
    use_compile = config["model"].get("compile", False) and device.type == "cuda"
    if use_compile:
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # Automatic mixed precision setup (bfloat16 for Blackwell Tensor Cores)
    use_amp = config["model"].get("amp", False) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp and amp_dtype == torch.float16)
    if use_amp:
        print(f"AMP enabled with dtype={amp_dtype}")

    # Build datasets
    train_dataset = MarineSnowDataset(
        root_dir=config["data"]["train_dir"],
        image_size=config["data"]["image_size"],
        augment=True,
    )
    val_dataset = MarineSnowDataset(
        root_dir=config["data"]["val_dir"],
        image_size=config["data"]["image_size"],
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )

    # Loss functions
    l1_loss = nn.L1Loss()
    perceptual_loss = PerceptualLoss().to(device) if config["training"]["perceptual_weight"] > 0 else None

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )

    # Resume from checkpoint
    start_epoch = 0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(start_epoch, config["training"]["epochs"]):
        # Train
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            snow = batch["snow"].to(device)
            clean = batch["clean"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                pred = model(snow)
                loss = config["training"]["l1_weight"] * l1_loss(pred, clean)
                if perceptual_loss:
                    loss += config["training"]["perceptual_weight"] * perceptual_loss(pred, clean)

            optimizer.zero_grad()
            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)
        elapsed = time.time() - t0

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                snow = batch["snow"].to(device)
                clean = batch["clean"].to(device)
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    pred = model(snow)
                    val_loss += l1_loss(pred, clean).item()

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch+1}/{config['training']['epochs']} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | Time: {elapsed:.1f}s"
        )

        # Save checkpoint (save uncompiled model state_dict)
        if (epoch + 1) % config["training"]["save_every"] == 0:
            state_dict = model._orig_mod.state_dict() if use_compile else model.state_dict()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                checkpoint_dir / f"checkpoint_epoch{epoch+1}.pth",
            )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state_dict = model._orig_mod.state_dict() if use_compile else model.state_dict()
            torch.save(state_dict, checkpoint_dir / "best_model.pth")
            print(f"  -> New best model (val_loss={val_loss:.4f})")

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train marine snow removal model")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    args = parser.parse_args()
    train(args.config, args.resume)
