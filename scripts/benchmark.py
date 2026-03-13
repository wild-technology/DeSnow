"""
Benchmark different marine snow removal approaches.

Compares:
- Morphological filtering (CPU baseline)
- Frequency domain filtering (CPU)
- Temporal median filtering (CPU, requires video)
- U-Net inference (GPU)
- Hybrid pipeline (GPU + CPU)

Reports PSNR, SSIM, and inference time for each approach.

Usage:
    python -m scripts.benchmark --test-dir data/msrb/test --model checkpoints/best_model.pth
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.traditional.morphological import MorphologicalSnowRemover, FrequencyDomainFilter
from src.traditional.temporal_median import TemporalMedianFilter
from src.pipeline import DeSnowPipeline


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Compute PSNR and SSIM between prediction and target."""
    psnr = peak_signal_noise_ratio(target, pred)
    ssim = structural_similarity(target, pred, channel_axis=2)
    return {"psnr": psnr, "ssim": ssim}


def benchmark_method(name, process_fn, snow_images, clean_images):
    """Benchmark a single method across all test images."""
    psnr_vals, ssim_vals, times = [], [], []

    for snow, clean in zip(snow_images, clean_images):
        t0 = time.time()
        result = process_fn(snow)
        elapsed = time.time() - t0
        times.append(elapsed)

        metrics = compute_metrics(result, clean)
        psnr_vals.append(metrics["psnr"])
        ssim_vals.append(metrics["ssim"])

    return {
        "method": name,
        "psnr_mean": np.mean(psnr_vals),
        "psnr_std": np.std(psnr_vals),
        "ssim_mean": np.mean(ssim_vals),
        "ssim_std": np.std(ssim_vals),
        "time_mean_ms": np.mean(times) * 1000,
        "time_std_ms": np.std(times) * 1000,
        "fps": 1.0 / np.mean(times) if np.mean(times) > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark marine snow removal methods")
    parser.add_argument("--test-dir", required=True, help="Test directory with snow/ and clean/ subdirs")
    parser.add_argument("--model", default=None, help="Path to trained U-Net model")
    parser.add_argument("--model-type", default="unet", choices=["unet", "lightweight_unet"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=100, help="Max images to evaluate")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    snow_dir = test_dir / "snow"
    clean_dir = test_dir / "clean"

    # Load test images
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
    snow_files = sorted([f for f in snow_dir.iterdir() if f.suffix.lower() in extensions])
    snow_files = snow_files[: args.max_images]

    snow_images = []
    clean_images = []
    for sf in snow_files:
        cf = clean_dir / sf.name
        if not cf.exists():
            # Try different extension
            cf = None
            for ext in extensions:
                candidate = clean_dir / (sf.stem + ext)
                if candidate.exists():
                    cf = candidate
                    break
            if cf is None:
                continue

        snow_img = cv2.imread(str(sf))
        clean_img = cv2.imread(str(cf))
        if snow_img is not None and clean_img is not None:
            snow_images.append(snow_img)
            clean_images.append(clean_img)

    print(f"Loaded {len(snow_images)} test image pairs")
    if not snow_images:
        print("No test images found!")
        return

    # Also compute baseline (no processing)
    results = []

    # Baseline: no processing
    results.append(benchmark_method(
        "No processing (baseline)",
        lambda x: x,
        snow_images, clean_images,
    ))

    # Morphological
    morph = MorphologicalSnowRemover()
    results.append(benchmark_method(
        "Morphological",
        morph.remove_snow,
        snow_images, clean_images,
    ))

    # Frequency domain
    freq = FrequencyDomainFilter()
    results.append(benchmark_method(
        "Frequency domain",
        freq.selective_remove,
        snow_images, clean_images,
    ))

    # U-Net (if model provided)
    if args.model:
        unet_pipeline = DeSnowPipeline(
            mode="unet",
            model_path=args.model,
            model_type=args.model_type,
            device=args.device,
        )
        results.append(benchmark_method(
            f"U-Net ({args.model_type})",
            unet_pipeline.process_image,
            snow_images, clean_images,
        ))

    # Print results table
    print("\n" + "=" * 90)
    print(f"{'Method':<30} {'PSNR (dB)':<15} {'SSIM':<15} {'Time (ms)':<15} {'FPS':<10}")
    print("=" * 90)
    for r in results:
        print(
            f"{r['method']:<30} "
            f"{r['psnr_mean']:.2f}±{r['psnr_std']:.2f}    "
            f"{r['ssim_mean']:.4f}±{r['ssim_std']:.4f}  "
            f"{r['time_mean_ms']:.1f}±{r['time_std_ms']:.1f}     "
            f"{r['fps']:.1f}"
        )
    print("=" * 90)


if __name__ == "__main__":
    main()
