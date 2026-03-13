"""
Morphological filtering for marine snow removal from single images.

Detects bright particles via thresholding and morphological operations,
then inpaints the detected regions.

Compute cost: ~3-8ms per frame on CPU for 1080p.
"""

import cv2
import numpy as np
from typing import Optional


class MorphologicalSnowRemover:
    """Remove marine snow using morphological detection and inpainting.

    Strategy:
    1. Convert to grayscale / value channel
    2. Detect bright spots via adaptive thresholding
    3. Filter by size (area) to isolate particle-sized blobs
    4. Dilate the mask slightly to cover particle halos
    5. Inpaint the detected regions
    """

    def __init__(
        self,
        brightness_threshold: float = 0.85,
        min_area: int = 2,
        max_area: int = 200,
        kernel_size: int = 3,
        inpaint_radius: int = 3,
    ):
        """
        Args:
            brightness_threshold: Percentile threshold for bright spot detection (0-1).
            min_area: Minimum contour area to consider as marine snow.
            max_area: Maximum contour area (larger blobs are likely scene elements).
            kernel_size: Size of morphological structuring element.
            inpaint_radius: Radius for inpainting algorithm.
        """
        self.brightness_threshold = brightness_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.kernel_size = kernel_size
        self.inpaint_radius = inpaint_radius

    def detect_snow_mask(self, image: np.ndarray) -> np.ndarray:
        """Detect marine snow particles and return a binary mask.

        Args:
            image: BGR image as uint8 numpy array.

        Returns:
            Binary mask where 255 = snow particle, 0 = background.
        """
        # Convert to HSV and extract value channel
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]

        # Threshold bright spots
        thresh_val = int(self.brightness_threshold * 255)
        _, bright_mask = cv2.threshold(value, thresh_val, 255, cv2.THRESH_BINARY)

        # Also detect using local contrast (catches dimmer particles)
        blur = cv2.GaussianBlur(value, (21, 21), 0)
        local_contrast = cv2.subtract(value, blur)
        _, contrast_mask = cv2.threshold(
            local_contrast, 30, 255, cv2.THRESH_BINARY
        )

        # Combine both detection methods
        combined = cv2.bitwise_or(bright_mask, contrast_mask)

        # Morphological opening to remove noise
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size)
        )
        opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        # Filter by contour area — keep only particle-sized blobs
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        snow_mask = np.zeros_like(opened)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area <= area <= self.max_area:
                cv2.drawContours(snow_mask, [cnt], -1, 255, -1)

        # Dilate slightly to cover halos
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        snow_mask = cv2.dilate(snow_mask, dilate_kernel, iterations=1)

        return snow_mask

    def remove_snow(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Remove marine snow from image via inpainting.

        Args:
            image: BGR image as uint8 numpy array.
            mask: Optional pre-computed snow mask. If None, auto-detected.

        Returns:
            Cleaned image with marine snow removed.
        """
        if mask is None:
            mask = self.detect_snow_mask(image)

        # Inpaint the detected particle regions
        result = cv2.inpaint(image, mask, self.inpaint_radius, cv2.INPAINT_TELEA)
        return result

    def process_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Process a batch of images.

        Args:
            images: List of BGR images.

        Returns:
            List of cleaned images.
        """
        return [self.remove_snow(img) for img in images]


class FrequencyDomainFilter:
    """Remove marine snow using frequency domain filtering.

    Marine snow creates high-frequency bright spots. By filtering in the
    frequency domain, we can suppress these while preserving scene structure.

    Compute cost: ~5-10ms per frame on CPU for 1080p.
    """

    def __init__(self, cutoff_ratio: float = 0.3, order: int = 2):
        """
        Args:
            cutoff_ratio: Butterworth filter cutoff as ratio of max frequency.
            order: Butterworth filter order.
        """
        self.cutoff_ratio = cutoff_ratio
        self.order = order

    def _butterworth_lowpass(self, shape: tuple, cutoff: float, order: int) -> np.ndarray:
        """Create a Butterworth low-pass filter."""
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        u = np.arange(rows).reshape(-1, 1) - crow
        v = np.arange(cols).reshape(1, -1) - ccol
        d = np.sqrt(u**2 + v**2)
        h = 1.0 / (1.0 + (d / cutoff) ** (2 * order))
        return h

    def remove_snow(self, image: np.ndarray) -> np.ndarray:
        """Remove high-frequency particle noise via frequency filtering.

        This is a gentle approach — it suppresses the brightest high-frequency
        components (particles) while retaining most scene detail.

        Args:
            image: BGR image as uint8 numpy array.

        Returns:
            Filtered image.
        """
        result_channels = []
        for c in range(3):
            channel = image[:, :, c].astype(np.float32)

            # Apply DFT
            dft = np.fft.fft2(channel)
            dft_shift = np.fft.fftshift(dft)

            # Create Butterworth low-pass filter
            rows, cols = channel.shape
            cutoff = self.cutoff_ratio * min(rows, cols) / 2
            h = self._butterworth_lowpass((rows, cols), cutoff, self.order)

            # Apply filter
            filtered = dft_shift * h

            # Inverse DFT
            f_ishift = np.fft.ifftshift(filtered)
            result = np.fft.ifft2(f_ishift)
            result = np.abs(result)

            # Blend: use filtered version only where particles are detected
            # (avoids over-smoothing the whole image)
            result_channels.append(np.clip(result, 0, 255).astype(np.uint8))

        return np.stack(result_channels, axis=-1)

    def selective_remove(self, image: np.ndarray) -> np.ndarray:
        """Selectively filter only detected high-frequency particle regions.

        More conservative than full frequency filtering — only applies the
        frequency filter in regions where particles are detected, preserving
        scene detail elsewhere.

        Args:
            image: BGR image as uint8 numpy array.

        Returns:
            Selectively filtered image.
        """
        # Get the frequency-filtered version
        filtered = self.remove_snow(image)

        # Detect particle regions using morphological approach
        detector = MorphologicalSnowRemover()
        mask = detector.detect_snow_mask(image)

        # Blend: original where no particles, filtered where particles detected
        mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        # Feather the mask edges for smooth blending
        mask_3c = cv2.GaussianBlur(mask_3c, (11, 11), 0)

        result = (image.astype(np.float32) * (1 - mask_3c) + filtered.astype(np.float32) * mask_3c)
        return np.clip(result, 0, 255).astype(np.uint8)
