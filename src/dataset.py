"""
Dataset classes for marine snow removal training.

Supports the MSRB (Marine Snow Removal Benchmark) dataset format:
    train/snow/  - images with marine snow
    train/clean/ - corresponding clean images
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


class MarineSnowDataset(Dataset):
    """Paired dataset for marine snow removal training.

    Expects directory structure:
        root_dir/snow/   - input images (with marine snow)
        root_dir/clean/  - target images (clean)

    Filenames must match between snow/ and clean/ directories.
    """

    def __init__(
        self,
        root_dir: str,
        image_size: int = 384,
        augment: bool = True,
        extensions: tuple = (".png", ".jpg", ".jpeg", ".bmp", ".tif"),
    ):
        self.root_dir = Path(root_dir)
        self.snow_dir = self.root_dir / "snow"
        self.clean_dir = self.root_dir / "clean"
        self.image_size = image_size
        self.augment = augment

        if not self.snow_dir.exists():
            raise FileNotFoundError(f"Snow directory not found: {self.snow_dir}")
        if not self.clean_dir.exists():
            raise FileNotFoundError(f"Clean directory not found: {self.clean_dir}")

        # Find matching pairs
        snow_files = {f.stem: f for f in self.snow_dir.iterdir() if f.suffix.lower() in extensions}
        clean_files = {f.stem: f for f in self.clean_dir.iterdir() if f.suffix.lower() in extensions}

        common_stems = sorted(set(snow_files.keys()) & set(clean_files.keys()))
        if not common_stems:
            raise ValueError(f"No matching image pairs found in {root_dir}")

        self.pairs = [(snow_files[s], clean_files[s]) for s in common_stems]
        self.transform = self._build_transform() if augment else self._build_val_transform()

    def _build_transform(self):
        if not HAS_ALBUMENTATIONS:
            return None
        return A.Compose(
            [
                A.RandomCrop(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ],
            additional_targets={"clean": "image"},
        )

    def _build_val_transform(self):
        if not HAS_ALBUMENTATIONS:
            return None
        return A.Compose(
            [
                A.CenterCrop(self.image_size, self.image_size),
            ],
            additional_targets={"clean": "image"},
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        snow_path, clean_path = self.pairs[idx]

        # Load images (BGR → RGB)
        snow = cv2.imread(str(snow_path))
        clean = cv2.imread(str(clean_path))
        snow = cv2.cvtColor(snow, cv2.COLOR_BGR2RGB)
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transform:
            transformed = self.transform(image=snow, clean=clean)
            snow = transformed["image"]
            clean = transformed["clean"]
        else:
            # Resize if no augmentation pipeline
            snow = cv2.resize(snow, (self.image_size, self.image_size))
            clean = cv2.resize(clean, (self.image_size, self.image_size))

        # Convert to float tensors [0, 1]
        snow_tensor = torch.from_numpy(snow).permute(2, 0, 1).float() / 255.0
        clean_tensor = torch.from_numpy(clean).permute(2, 0, 1).float() / 255.0

        return {
            "snow": snow_tensor,
            "clean": clean_tensor,
            "filename": snow_path.stem,
        }


class SyntheticSnowDataset(Dataset):
    """Generate synthetic marine snow on clean underwater images.

    Useful when paired data is unavailable — takes clean underwater images
    and adds synthetic marine snow particles for self-supervised training.
    """

    def __init__(
        self,
        image_dir: str,
        image_size: int = 384,
        num_particles_range: tuple = (100, 600),
        particle_size_range: tuple = (1, 8),
        augment: bool = True,
        extensions: tuple = (".png", ".jpg", ".jpeg", ".bmp", ".tif"),
    ):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.num_particles_range = num_particles_range
        self.particle_size_range = particle_size_range

        self.images = sorted(
            [f for f in self.image_dir.iterdir() if f.suffix.lower() in extensions]
        )
        if not self.images:
            raise ValueError(f"No images found in {image_dir}")

        if HAS_ALBUMENTATIONS and augment:
            self.transform = A.Compose(
                [
                    A.RandomCrop(image_size, image_size),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                ]
            )
        else:
            self.transform = None

    def _add_synthetic_snow(self, image: np.ndarray) -> np.ndarray:
        """Add synthetic marine snow particles to an image."""
        h, w = image.shape[:2]
        snow_image = image.copy()

        num_particles = np.random.randint(*self.num_particles_range)
        for _ in range(num_particles):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            radius = np.random.randint(*self.particle_size_range)

            # Particles are bright and slightly translucent
            brightness = np.random.randint(180, 255)
            alpha = np.random.uniform(0.3, 0.9)

            # Draw particle with Gaussian blur for soft edges
            overlay = snow_image.copy()
            cv2.circle(overlay, (x, y), radius, (brightness, brightness, brightness), -1)
            snow_image = cv2.addWeighted(overlay, alpha, snow_image, 1 - alpha, 0)

        return snow_image

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.images[idx]
        clean = cv2.imread(str(img_path))
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=clean)
            clean = transformed["image"]
        else:
            clean = cv2.resize(clean, (self.image_size, self.image_size))

        # Generate synthetic snow version
        snow = self._add_synthetic_snow(clean)

        clean_tensor = torch.from_numpy(clean).permute(2, 0, 1).float() / 255.0
        snow_tensor = torch.from_numpy(snow).permute(2, 0, 1).float() / 255.0

        return {
            "snow": snow_tensor,
            "clean": clean_tensor,
            "filename": img_path.stem,
        }
