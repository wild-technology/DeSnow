"""
Evaluate tie point quality improvement after marine snow removal.

This is the key metric for photogrammetry workflows: how many feature
detector tie points are "wasted" on marine snow particles vs scene geometry.

Strategy:
1. Run feature detection (SIFT/ORB) on original and cleaned images
2. For paired images: classify keypoints as "on snow" or "on scene" using
   the known snow mask
3. Measure the ratio of scene-relevant keypoints before/after cleaning
4. Run feature matching between image pairs to assess match quality

Usage:
    python -m scripts.evaluate_tiepoints --test-dir data/msrb/test \
        --model checkpoints/best_model.pth
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.traditional.morphological import MorphologicalSnowRemover
from src.pipeline import DeSnowPipeline


def detect_keypoints(image: np.ndarray, method: str = "sift") -> tuple:
    """Detect keypoints and compute descriptors.

    Args:
        image: BGR image.
        method: "sift" or "orb".

    Returns:
        (keypoints, descriptors)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == "sift":
        detector = cv2.SIFT_create(nfeatures=2000)
    else:
        detector = cv2.ORB_create(nFeatures=2000)
    return detector.detectAndCompute(gray, None)


def classify_keypoints(keypoints, snow_mask: np.ndarray) -> dict:
    """Classify keypoints as on-snow or on-scene.

    Args:
        keypoints: List of cv2.KeyPoint.
        snow_mask: Binary mask where 255 = snow particle.

    Returns:
        Dict with counts and ratio.
    """
    on_snow = 0
    on_scene = 0

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if 0 <= y < snow_mask.shape[0] and 0 <= x < snow_mask.shape[1]:
            if snow_mask[y, x] > 127:
                on_snow += 1
            else:
                on_scene += 1
        else:
            on_scene += 1

    total = on_snow + on_scene
    return {
        "total": total,
        "on_snow": on_snow,
        "on_scene": on_scene,
        "scene_ratio": on_scene / total if total > 0 else 0,
        "snow_ratio": on_snow / total if total > 0 else 0,
    }


def match_keypoints(desc1, desc2, method: str = "sift") -> int:
    """Match descriptors between two images and return good match count."""
    if desc1 is None or desc2 is None:
        return 0

    if method == "sift":
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = matcher.knnMatch(desc1, desc2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    return len(good_matches)


def main():
    parser = argparse.ArgumentParser(description="Evaluate tie point quality")
    parser.add_argument("--test-dir", required=True, help="Test directory with snow/ and clean/ subdirs")
    parser.add_argument("--model", default=None, help="Path to trained model")
    parser.add_argument("--model-type", default="unet", choices=["unet", "lightweight_unet"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--method", default="sift", choices=["sift", "orb"])
    parser.add_argument("--max-images", type=int, default=50)
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    snow_dir = test_dir / "snow"
    clean_dir = test_dir / "clean"

    # Load test images
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
    snow_files = sorted([f for f in snow_dir.iterdir() if f.suffix.lower() in extensions])
    snow_files = snow_files[: args.max_images]

    # Initialize snow detector for mask generation
    snow_detector = MorphologicalSnowRemover()

    # Initialize pipeline
    pipeline = None
    if args.model:
        pipeline = DeSnowPipeline(
            mode="unet",
            model_path=args.model,
            model_type=args.model_type,
            device=args.device,
        )

    print(f"Evaluating tie point quality using {args.method.upper()}")
    print(f"{'='*70}")

    before_stats = {"total": [], "on_snow": [], "on_scene": [], "scene_ratio": []}
    after_stats = {"total": [], "on_snow": [], "on_scene": [], "scene_ratio": []}

    for sf in snow_files:
        cf = clean_dir / sf.name
        if not cf.exists():
            for ext in extensions:
                candidate = clean_dir / (sf.stem + ext)
                if candidate.exists():
                    cf = candidate
                    break
            else:
                continue

        snow_img = cv2.imread(str(sf))
        clean_img = cv2.imread(str(cf))
        if snow_img is None or clean_img is None:
            continue

        # Detect snow mask from difference between snow and clean
        diff = cv2.absdiff(snow_img, clean_img)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, snow_mask = cv2.threshold(diff_gray, 20, 255, cv2.THRESH_BINARY)

        # Keypoints on original snowy image
        kps_before, _ = detect_keypoints(snow_img, args.method)
        stats_before = classify_keypoints(kps_before, snow_mask)
        for k in before_stats:
            before_stats[k].append(stats_before[k])

        # Process image (remove snow)
        if pipeline:
            cleaned = pipeline.process_image(snow_img)
        else:
            cleaned = snow_detector.remove_snow(snow_img)

        # Keypoints on cleaned image
        kps_after, _ = detect_keypoints(cleaned, args.method)
        stats_after = classify_keypoints(kps_after, snow_mask)
        for k in after_stats:
            after_stats[k].append(stats_after[k])

    print(f"\nResults over {len(before_stats['total'])} images:")
    print(f"{'='*70}")
    print(f"{'Metric':<35} {'Before':<15} {'After':<15}")
    print(f"{'-'*70}")
    print(
        f"{'Avg keypoints total':<35} "
        f"{np.mean(before_stats['total']):.0f}       "
        f"{np.mean(after_stats['total']):.0f}"
    )
    print(
        f"{'Avg keypoints on snow':<35} "
        f"{np.mean(before_stats['on_snow']):.0f}       "
        f"{np.mean(after_stats['on_snow']):.0f}"
    )
    print(
        f"{'Avg keypoints on scene':<35} "
        f"{np.mean(before_stats['on_scene']):.0f}       "
        f"{np.mean(after_stats['on_scene']):.0f}"
    )
    print(
        f"{'Scene keypoint ratio':<35} "
        f"{np.mean(before_stats['scene_ratio']):.1%}       "
        f"{np.mean(after_stats['scene_ratio']):.1%}"
    )
    improvement = np.mean(after_stats["scene_ratio"]) - np.mean(before_stats["scene_ratio"])
    print(f"\nScene keypoint ratio improvement: {improvement:+.1%}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
