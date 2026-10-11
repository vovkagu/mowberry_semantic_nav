"""Visual Language Model (VLM) interface and a deterministic stub.

The real VLM will receive the JPEG bytes of the latest camera frame plus the
user prompt and return labelled bounding boxes. Until it is wired in, the
stub decodes the frame (to report the true image size) and returns a fixed
scene that matches the example in the project spec: two trees and a shed
door.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """One labelled bounding box in pixel coordinates (origin top-left)."""

    label: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    score: float

    @property
    def u_center(self):
        """Horizontal centre column of the box."""
        return (self.x_min + self.x_max) / 2.0


@dataclass(frozen=True)
class VlmResult:
    """VLM output: the analysed image size and its detections."""

    image_width: int
    image_height: int
    detections: list = field(default_factory=list)


class VlmStub:
    """Stand-in for the real VLM client. Same ``detect`` signature."""

    def detect(self, jpeg_bytes, prompt):
        """Return fixed detections scaled to the decoded frame size.

        Args:
            jpeg_bytes: compressed image payload (``CompressedImage.data``).
            prompt: user command; ignored by the stub, but a real VLM will
                use it for open-vocabulary detection.

        Raises:
            ValueError: if the payload cannot be decoded as an image.
        """
        image = cv2.imdecode(
            np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('compressed frame could not be decoded')
        height, width = image.shape[:2]

        # Boxes are given as fractions of the frame so the stub works for
        # both the 320x240 network stream and full-resolution frames.
        scene = [
            ('tree', 0.05, 0.20, 0.25, 0.95, 0.91),
            ('tree', 0.72, 0.15, 0.92, 0.95, 0.88),
            ('shed_door', 0.42, 0.30, 0.58, 0.80, 0.83),
        ]
        detections = [
            Detection(label, x0 * width, y0 * height,
                      x1 * width, y1 * height, score)
            for label, x0, y0, x1, y1, score in scene
        ]
        return VlmResult(width, height, detections)
