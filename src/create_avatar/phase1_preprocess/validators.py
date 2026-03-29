from __future__ import annotations

"""Image quality validators for face detection input.

Checks that input images meet the minimum requirements for
reliable 3D face reconstruction.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ValidationResult:
    """Result of image quality validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    resolution: tuple[int, int]  # (width, height)
    face_resolution: int | None  # Approximate face size in pixels


MIN_IMAGE_SIZE = 256
MIN_FACE_SIZE = 128
MAX_YAW_DEGREES = 35


def validate_image(
    image_path: Path | str,
    landmarks_3d: np.ndarray | None = None,
) -> ValidationResult:
    """Validate that an image meets quality requirements for avatar generation.

    Args:
        image_path: Path to the image file.
        landmarks_3d: Optional MediaPipe 3D landmarks (468, 3).
            If provided, also validates pose and face size.

    Returns:
        ValidationResult with errors and warnings.
    """
    image_path = Path(image_path)
    errors = []
    warnings = []
    face_resolution = None

    # Check file exists
    if not image_path.exists():
        return ValidationResult(
            is_valid=False,
            errors=[f"File not found: {image_path}"],
            warnings=[],
            resolution=(0, 0),
            face_resolution=None,
        )

    # Load image to check resolution
    suffix = image_path.suffix.lower()
    if suffix in (".heic", ".heif"):
        from pillow_heif import register_heif_opener
        from PIL import Image, ImageOps

        register_heif_opener()
        pil_img = ImageOps.exif_transpose(Image.open(image_path))
        width, height = pil_img.size
    else:
        img = cv2.imread(str(image_path))
        if img is None:
            return ValidationResult(
                is_valid=False,
                errors=[f"Failed to read image: {image_path}"],
                warnings=[],
                resolution=(0, 0),
                face_resolution=None,
            )
        height, width = img.shape[:2]

    resolution = (width, height)

    # Check minimum resolution
    if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
        errors.append(
            f"Image too small: {width}x{height}. Minimum: {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE}"
        )

    # Landmark-based checks
    if landmarks_3d is not None:
        # Check face size
        landmarks_2d = landmarks_3d[:, :2].copy()
        landmarks_2d[:, 0] *= width
        landmarks_2d[:, 1] *= height

        face_width = landmarks_2d[:, 0].max() - landmarks_2d[:, 0].min()
        face_height = landmarks_2d[:, 1].max() - landmarks_2d[:, 1].min()
        face_resolution = int(max(face_width, face_height))

        if face_resolution < MIN_FACE_SIZE:
            errors.append(
                f"Face too small in image: {face_resolution}px. Minimum: {MIN_FACE_SIZE}px"
            )

        # Check head pose (yaw estimation from landmarks)
        nose = landmarks_3d[1]  # Nose tip
        left_ear = landmarks_3d[234]
        right_ear = landmarks_3d[454]

        # Estimate yaw from nose position relative to ears
        ear_center_x = (left_ear[0] + right_ear[0]) / 2
        nose_offset = nose[0] - ear_center_x
        ear_span = abs(right_ear[0] - left_ear[0])

        if ear_span > 0.01:
            yaw_ratio = abs(nose_offset) / ear_span
            estimated_yaw = yaw_ratio * 90  # Rough approximation

            if estimated_yaw > MAX_YAW_DEGREES:
                errors.append(
                    f"Face not frontal enough (estimated yaw: {estimated_yaw:.0f}°). "
                    f"Maximum: {MAX_YAW_DEGREES}°"
                )
            elif estimated_yaw > 20:
                warnings.append(
                    f"Face slightly rotated (estimated yaw: {estimated_yaw:.0f}°). "
                    "Frontal photos produce better results."
                )

    is_valid = len(errors) == 0

    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        resolution=resolution,
        face_resolution=face_resolution,
    )
