"""
Hybrid inference pipeline for marine snow removal.

Supports three modes:
1. Single image: U-Net only (or morphological fallback)
2. Video: Temporal median → U-Net refinement
3. Hybrid: Automatic mode selection based on input type

The pipeline processes ROV footage to remove marine snow before
feeding images into photogrammetry/SfM pipelines.
"""

import cv2
import numpy as np
import time
import torch
from pathlib import Path
from typing import Optional, Union

from src.models.unet import UNet, LightweightUNet
from src.traditional.temporal_median import TemporalMedianFilter, AdaptiveTemporalFilter
from src.traditional.morphological import MorphologicalSnowRemover


class DeSnowPipeline:
    """Hybrid marine snow removal pipeline."""

    def __init__(
        self,
        mode: str = "hybrid",
        model_path: Optional[str] = None,
        model_type: str = "unet",
        temporal_window: int = 7,
        device: str = "cuda",
        tile_size: int = 512,
        tile_overlap: int = 64,
        use_adaptive_temporal: bool = False,
        compile_model: bool = False,
        use_amp: bool = False,
    ):
        """
        Args:
            mode: "temporal", "unet", "morphological", or "hybrid".
            model_path: Path to trained U-Net weights. Required for "unet"/"hybrid".
            model_type: "unet" or "lightweight_unet".
            temporal_window: Number of frames for temporal median filter.
            device: "cuda" or "cpu".
            tile_size: Tile size for processing large images.
            tile_overlap: Overlap between tiles.
            use_adaptive_temporal: Use motion-compensated temporal filter.
            compile_model: Use torch.compile() for Blackwell sm_120 optimization.
            use_amp: Use automatic mixed precision (bfloat16 on RTX 5090).
        """
        self.mode = mode
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.compile_model = compile_model and self.device.type == "cuda"
        self.use_amp = use_amp and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported() else torch.float16

        # Initialize temporal filter
        if mode in ("temporal", "hybrid"):
            if use_adaptive_temporal:
                self.temporal_filter = AdaptiveTemporalFilter(window_size=temporal_window)
            else:
                self.temporal_filter = TemporalMedianFilter(window_size=temporal_window)
        else:
            self.temporal_filter = None

        # Initialize morphological filter (fallback for single images)
        self.morph_filter = MorphologicalSnowRemover()

        # Initialize U-Net model
        self.model = None
        if mode in ("unet", "hybrid") and model_path:
            self._load_model(model_path, model_type)

    def _load_model(self, model_path: str, model_type: str):
        """Load trained U-Net model."""
        if model_type == "lightweight_unet":
            self.model = LightweightUNet()
        else:
            self.model = UNet()

        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # torch.compile() for Blackwell sm_120 optimization
        if self.compile_model:
            self.model = torch.compile(self.model)

    def _unet_inference(self, image: np.ndarray) -> np.ndarray:
        """Run U-Net inference on a single image, with tiling for large images."""
        h, w = image.shape[:2]

        if h <= self.tile_size and w <= self.tile_size:
            return self._unet_single_tile(image)

        # Tiled inference for large images
        return self._unet_tiled(image)

    def _unet_single_tile(self, image: np.ndarray) -> np.ndarray:
        """Run U-Net on a single image/tile."""
        # Pad to multiple of 16 for U-Net
        h, w = image.shape[:2]
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

        # BGR → RGB, normalize, to tensor
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            output = self.model(tensor)

        # Back to numpy
        result = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
        result = (result * 255).clip(0, 255).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            result = result[:h, :w]

        return result

    def _unet_tiled(self, image: np.ndarray) -> np.ndarray:
        """Run U-Net with overlapping tiles for large images."""
        h, w = image.shape[:2]
        step = self.tile_size - self.tile_overlap
        result = np.zeros_like(image, dtype=np.float32)
        weight = np.zeros((h, w, 1), dtype=np.float32)

        for y in range(0, h, step):
            for x in range(0, w, step):
                # Extract tile
                y_end = min(y + self.tile_size, h)
                x_end = min(x + self.tile_size, w)
                y_start = max(0, y_end - self.tile_size)
                x_start = max(0, x_end - self.tile_size)

                tile = image[y_start:y_end, x_start:x_end]
                processed = self._unet_single_tile(tile).astype(np.float32)

                # Accumulate with blending weights
                tile_h, tile_w = processed.shape[:2]
                w_mask = np.ones((tile_h, tile_w, 1), dtype=np.float32)
                result[y_start:y_end, x_start:x_end] += processed * w_mask
                weight[y_start:y_end, x_start:x_end] += w_mask

        # Normalize by weight
        weight = np.maximum(weight, 1e-6)
        result = (result / weight).clip(0, 255).astype(np.uint8)
        return result

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """Process a single image (no temporal information available).

        Args:
            image: BGR image as uint8 numpy array.

        Returns:
            Cleaned image.
        """
        if self.model is not None:
            return self._unet_inference(image)
        else:
            # Fallback to morphological filter
            return self.morph_filter.remove_snow(image)

    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Process a video frame through the hybrid pipeline.

        Call this for each frame in sequence. Returns None until the temporal
        buffer is full, then returns filtered frames.

        Args:
            frame: BGR image as uint8 numpy array.

        Returns:
            Filtered frame, or None if buffer not yet full.
        """
        if self.mode == "unet":
            return self.process_image(frame)

        if self.mode == "morphological":
            return self.morph_filter.remove_snow(frame)

        # Temporal + optional U-Net refinement
        if self.temporal_filter is not None:
            temporal_result = self.temporal_filter.add_frame(frame)
            if temporal_result is None:
                return None

            # Apply U-Net refinement if available
            if self.model is not None:
                return self._unet_inference(temporal_result)
            return temporal_result

        return self.process_image(frame)

    def process_video(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        progress_callback=None,
    ) -> dict:
        """Process an entire video file.

        Args:
            input_path: Path to input video.
            output_path: Optional path to save output video.
            progress_callback: Optional callable(frame_idx, total_frames).

        Returns:
            Dict with processing statistics.
        """
        if self.temporal_filter is not None:
            self.temporal_filter.reset()

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_times = []
        output_count = 0

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()
            result = self.process_frame(frame)
            elapsed = time.time() - t0
            frame_times.append(elapsed)

            if result is not None:
                output_count += 1
                if writer:
                    writer.write(result)

            if progress_callback:
                progress_callback(i + 1, total_frames)

        cap.release()
        if writer:
            writer.release()

        avg_time = np.mean(frame_times) if frame_times else 0
        return {
            "input_frames": total_frames,
            "output_frames": output_count,
            "avg_frame_time_ms": avg_time * 1000,
            "fps": 1.0 / avg_time if avg_time > 0 else 0,
            "resolution": f"{width}x{height}",
        }

    def process_image_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        extensions: tuple = (".png", ".jpg", ".jpeg", ".bmp", ".tif"),
    ) -> dict:
        """Process all images in a directory (single-image mode).

        Args:
            input_dir: Directory of input images.
            output_dir: Directory to save output images.
            extensions: File extensions to process.

        Returns:
            Dict with processing statistics.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(
            [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]
        )

        frame_times = []
        for img_path in images:
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            t0 = time.time()
            result = self.process_image(image)
            elapsed = time.time() - t0
            frame_times.append(elapsed)

            cv2.imwrite(str(output_dir / img_path.name), result)

        avg_time = np.mean(frame_times) if frame_times else 0
        return {
            "processed_images": len(frame_times),
            "avg_time_ms": avg_time * 1000,
            "fps": 1.0 / avg_time if avg_time > 0 else 0,
        }
