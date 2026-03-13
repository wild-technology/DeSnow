"""
Command-line inference script for marine snow removal.

Usage:
    # Single image
    python -m src.inference --input image.jpg --output clean.jpg

    # Directory of images
    python -m src.inference --input images/ --output output/ --mode unet

    # Video file
    python -m src.inference --input dive.mp4 --output clean.mp4 --mode hybrid

    # Morphological only (no model needed)
    python -m src.inference --input image.jpg --output clean.jpg --mode morphological
"""

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from src.pipeline import DeSnowPipeline


def main():
    parser = argparse.ArgumentParser(description="Marine snow removal inference")
    parser.add_argument("--input", required=True, help="Input image, directory, or video")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument(
        "--mode",
        choices=["temporal", "unet", "morphological", "hybrid"],
        default="hybrid",
        help="Processing mode",
    )
    parser.add_argument("--model", default=None, help="Path to trained model weights")
    parser.add_argument(
        "--model-type",
        choices=["unet", "lightweight_unet"],
        default="unet",
        help="Model architecture",
    )
    parser.add_argument("--temporal-window", type=int, default=7, help="Temporal filter window size")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size for large images")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive temporal filter")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile() (Blackwell optimization)")
    parser.add_argument("--amp", action="store_true", help="Use AMP with bfloat16 (RTX 5090 Tensor Cores)")
    args = parser.parse_args()

    input_path = Path(args.input)

    # Validate mode + model combination
    if args.mode in ("unet", "hybrid") and args.model is None:
        if args.mode == "hybrid":
            print("Warning: No model provided, hybrid mode will use temporal + morphological only")
        else:
            parser.error("--model is required for unet mode")

    pipeline = DeSnowPipeline(
        mode=args.mode,
        model_path=args.model,
        model_type=args.model_type,
        temporal_window=args.temporal_window,
        device=args.device,
        tile_size=args.tile_size,
        use_adaptive_temporal=args.adaptive,
        compile_model=args.compile,
        use_amp=args.amp,
    )

    if input_path.is_dir():
        # Process directory of images
        print(f"Processing directory: {input_path}")
        stats = pipeline.process_image_directory(input_path, args.output)
        print(f"Processed {stats['processed_images']} images")
        print(f"Average time: {stats['avg_time_ms']:.1f}ms ({stats['fps']:.1f} FPS)")

    elif input_path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        # Process video
        print(f"Processing video: {input_path}")
        pbar = tqdm(total=100, desc="Processing")

        def progress(current, total):
            pbar.n = int(100 * current / total)
            pbar.refresh()

        stats = pipeline.process_video(input_path, args.output, progress_callback=progress)
        pbar.close()
        print(f"Processed {stats['input_frames']} frames → {stats['output_frames']} output frames")
        print(f"Average time: {stats['avg_frame_time_ms']:.1f}ms ({stats['fps']:.1f} FPS)")
        print(f"Resolution: {stats['resolution']}")

    else:
        # Single image
        print(f"Processing image: {input_path}")
        image = cv2.imread(str(input_path))
        if image is None:
            print(f"Error: Cannot read image: {input_path}")
            return

        result = pipeline.process_image(image)
        cv2.imwrite(str(args.output), result)
        print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
