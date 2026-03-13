"""
Temporal median filtering for marine snow removal from video sequences.

Exploits the fact that marine snow particles are transient (they move between
frames) while the scene background is relatively static. Computing the median
across a sliding window of frames effectively removes particles.

Compute cost: ~2-5ms per frame on CPU for 1080p.
"""

import cv2
import numpy as np
from collections import deque
from pathlib import Path
from typing import Optional, Union


class TemporalMedianFilter:
    """Remove marine snow by computing temporal median across video frames.

    Works best when:
    - Camera motion is slow relative to frame rate
    - Marine snow is the primary transient element
    - Video has sufficient frame rate (>10 FPS)
    """

    def __init__(self, window_size: int = 7):
        """
        Args:
            window_size: Number of frames in the sliding window. Must be odd.
                Larger values remove more particles but increase latency and
                may blur moving scene elements. 5-11 is typical.
        """
        if window_size % 2 == 0:
            window_size += 1
        self.window_size = window_size
        self.buffer: deque = deque(maxlen=window_size)

    def reset(self):
        """Clear the frame buffer."""
        self.buffer.clear()

    def add_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Add a frame and return the filtered result if buffer is full.

        Args:
            frame: BGR image as uint8 numpy array, shape (H, W, 3).

        Returns:
            Filtered frame if buffer is full, None otherwise.
        """
        self.buffer.append(frame.copy())

        if len(self.buffer) < self.window_size:
            return None

        # Stack frames and compute median along temporal axis
        stacked = np.stack(list(self.buffer), axis=0)
        median = np.median(stacked, axis=0).astype(np.uint8)
        return median

    def process_video(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> list[np.ndarray]:
        """Process an entire video file.

        Args:
            input_path: Path to input video.
            output_path: Optional path to save output video.

        Returns:
            List of filtered frames.
        """
        self.reset()
        cap = cv2.VideoCapture(str(input_path))

        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        filtered_frames = []
        # First pass: fill the buffer
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        # Process all frames
        for frame in frames:
            result = self.add_frame(frame)
            if result is not None:
                filtered_frames.append(result)
                if writer:
                    writer.write(result)

        if writer:
            writer.release()

        return filtered_frames

    def process_frame_sequence(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Process a list of frames in memory.

        Args:
            frames: List of BGR images as uint8 numpy arrays.

        Returns:
            List of filtered frames (shorter than input by window_size - 1).
        """
        self.reset()
        results = []
        for frame in frames:
            result = self.add_frame(frame)
            if result is not None:
                results.append(result)
        return results


class AdaptiveTemporalFilter:
    """Adaptive temporal median that accounts for camera motion.

    Uses optical flow to align frames before computing the temporal median,
    making it more robust to ROV motion.

    Compute cost: ~15-30ms per frame on CPU for 1080p (flow estimation dominates).
    """

    def __init__(self, window_size: int = 5):
        if window_size % 2 == 0:
            window_size += 1
        self.window_size = window_size
        self.buffer: deque = deque(maxlen=window_size)
        self.gray_buffer: deque = deque(maxlen=window_size)

    def reset(self):
        self.buffer.clear()
        self.gray_buffer.clear()

    def _align_to_reference(
        self, frame: np.ndarray, ref_gray: np.ndarray, frame_gray: np.ndarray
    ) -> np.ndarray:
        """Align frame to reference using optical flow."""
        flow = cv2.calcOpticalFlowFarneback(
            frame_gray, ref_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        h, w = frame.shape[:2]
        coords = np.mgrid[0:h, 0:w].astype(np.float32)
        coords[0] += flow[:, :, 1]
        coords[1] += flow[:, :, 0]
        aligned = cv2.remap(
            frame,
            coords[1],
            coords[0],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return aligned

    def add_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Add frame and return filtered result when buffer is full."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.buffer.append(frame.copy())
        self.gray_buffer.append(gray)

        if len(self.buffer) < self.window_size:
            return None

        # Reference is the center frame
        center = self.window_size // 2
        ref_gray = self.gray_buffer[center]

        # Align all frames to center frame
        aligned = []
        for i, (f, g) in enumerate(zip(self.buffer, self.gray_buffer)):
            if i == center:
                aligned.append(f)
            else:
                aligned.append(self._align_to_reference(f, ref_gray, g))

        stacked = np.stack(aligned, axis=0)
        median = np.median(stacked, axis=0).astype(np.uint8)
        return median
