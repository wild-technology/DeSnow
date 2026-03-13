# DeSnow

Marine snow removal from underwater ROV imagery for photogrammetry pipelines.

## Problem

Marine snow (suspended organic particles) in underwater ROV footage wastes feature detector tie points during Structure-from-Motion (SfM) workflows. Instead of matching on actual scene geometry, feature matchers lock onto floating particles, degrading 3D reconstruction quality.

## Approach

A hybrid pipeline combining traditional computer vision with deep learning:

| Stage | Method | Compute Cost | Purpose |
|-------|--------|-------------|---------|
| 1 | Temporal median filter | ~2-5ms/frame CPU | Remove transient particles across video frames |
| 2 | U-Net refinement | ~20-50ms/frame GPU | Clean residual/static particles |

**Single image mode**: U-Net only (or morphological fallback without a trained model).

**Video mode**: Temporal median → U-Net for best results.

## Project Structure

```
DeSnow/
├── configs/default.yaml          # Training and inference configuration
├── data/download_msrb.sh         # Dataset download instructions
├── src/
│   ├── traditional/
│   │   ├── temporal_median.py    # Temporal median + adaptive (flow-aligned) filter
│   │   └── morphological.py     # Morphological detection + inpainting, frequency filtering
│   ├── models/
│   │   └── unet.py              # Standard U-Net + lightweight depthwise-separable variant
│   ├── dataset.py               # MSRB dataset loader + synthetic snow generator
│   ├── pipeline.py              # Hybrid inference pipeline (temporal + U-Net)
│   ├── train.py                 # Training with L1 + perceptual loss
│   └── inference.py             # CLI for batch inference
├── scripts/
│   ├── benchmark.py             # PSNR/SSIM benchmarking across methods
│   └── evaluate_tiepoints.py    # Tie point quality evaluation (SIFT/ORB)
└── tests/
    └── test_pipeline.py         # Unit tests
```

## Hardware Requirements

Developed and tested for:

| Component | Target |
|-----------|--------|
| **GPU** | NVIDIA RTX 5090 (32GB GDDR7, sm_120 Blackwell) |
| **CPU** | AMD Threadripper |
| **CUDA** | 13.2 (via NVIDIA minor version compatibility with PyTorch cu130) |
| **PyTorch** | >= 2.10.0 |

The pipeline also works on older GPUs (RTX 30/40 series) and CPU-only systems (traditional filters only).

### Blackwell Optimizations

- **torch.compile()**: Fuses operations for sm_120, reducing kernel launch overhead
- **bfloat16 AMP**: RTX 5090 Tensor Cores deliver ~2x throughput with bfloat16
- **Large tile inference**: 32GB VRAM allows 1024px tiles (vs 512px on 8-12GB cards)
- **Threadripper data loading**: 8+ workers for parallel data loading

## Quick Start

### Install

```bash
# Install PyTorch with CUDA 13.0 support (compatible with CUDA 13.2)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Install remaining dependencies
pip install -r requirements.txt
```

### Inference (no training needed)

```bash
# Morphological filter — no model required
python -m src.inference --input image.jpg --output clean.jpg --mode morphological

# Process a directory of images
python -m src.inference --input images/ --output output/ --mode morphological

# Process video with temporal filtering
python -m src.inference --input dive.mp4 --output clean.mp4 --mode temporal
```

### Training

```bash
# 1. Set up the MSRB dataset (follow instructions in script)
bash data/download_msrb.sh

# 2. Train U-Net
python -m src.train --config configs/default.yaml

# 3. Inference with trained model
python -m src.inference --input image.jpg --output clean.jpg --mode unet --model checkpoints/best_model.pth

# 4. Hybrid mode (temporal + U-Net)
python -m src.inference --input dive.mp4 --output clean.mp4 --mode hybrid --model checkpoints/best_model.pth

# 5. With Blackwell optimizations (RTX 5090)
python -m src.inference --input dive.mp4 --output clean.mp4 --mode hybrid \
    --model checkpoints/best_model.pth --compile --amp --tile-size 1024
```

### Benchmarking

```bash
# Compare methods on test set
python -m scripts.benchmark --test-dir data/msrb/test --model checkpoints/best_model.pth

# Evaluate tie point quality improvement
python -m scripts.evaluate_tiepoints --test-dir data/msrb/test --model checkpoints/best_model.pth
```

### Tests

```bash
pytest tests/
```

## Methods Comparison

| Method | PSNR | Speed | GPU Required | Notes |
|--------|------|-------|-------------|-------|
| Morphological | Fair | ~5ms/frame | No | Good baseline, no training needed |
| Frequency domain | Fair | ~8ms/frame | No | Conservative, preserves detail |
| Temporal median | Good | ~3ms/frame | No | Best for video, exploits temporal info |
| U-Net | Good | ~30ms/frame | Yes | Trained on MSRB dataset |
| Hybrid (temporal+U-Net) | Best | ~35ms/frame | Yes | Recommended for video processing |

## Training Data

- **MSRB Dataset**: 2,300 train + 400 test paired images with synthetic marine snow ([GitHub](https://github.com/ychtanaka/marine-snow))
- **Synthetic mode**: The `SyntheticSnowDataset` class can generate training pairs from any clean underwater images
