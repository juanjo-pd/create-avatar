from __future__ import annotations

"""Face detection and preprocessing using MediaPipe FaceLandmarker.

Detects the face in an input image, aligns it (horizontal eye-line),
crops to a square, and resizes for downstream processing.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from create_avatar.config import config

# Default model path
_MODEL_PATH = Path(__file__).parent.parent.parent.parent / "data" / "face_landmarker_v2_with_blendshapes.task"


@dataclass
class AlignedFace:
    """Result of face detection and alignment."""

    image: np.ndarray  # (H, W, 3) aligned and cropped face image (BGR)
    image_rgb: np.ndarray  # (H, W, 3) same in RGB
    landmarks_2d: np.ndarray  # (478, 2) 2D landmarks in cropped image coords
    landmarks_3d: np.ndarray  # (478, 3) 3D landmarks (normalized)
    blendshape_scores: dict  # MediaPipe blendshape scores (52 ARKit-compatible)
    bbox: tuple  # (x, y, w, h) face bbox in original image
    rotation_angle: float  # Rotation applied for alignment (degrees)
    original_image_path: Path  # Path to original input image
    original_image_size: tuple  # (width, height) of original image


# MediaPipe landmark indices for key facial points
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
NOSE_TIP = 1
CHIN = 152
FOREHEAD = 10


def _load_image(image_path: Path) -> np.ndarray:
    """Load an image, supporting HEIC format. Returns RGB numpy array."""
    image_path = Path(image_path)
    suffix = image_path.suffix.lower()

    if suffix in (".heic", ".heif"):
        from pillow_heif import register_heif_opener
        from PIL import Image, ImageOps

        register_heif_opener()
        pil_img = ImageOps.exif_transpose(Image.open(image_path))
        return np.array(pil_img.convert("RGB"))
    else:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _compute_eye_angle(landmarks_px: np.ndarray) -> float:
    """Compute rotation angle to make eye-line horizontal."""
    left_eye = landmarks_px[LEFT_EYE_OUTER][:2]
    right_eye = landmarks_px[RIGHT_EYE_OUTER][:2]
    delta = right_eye - left_eye
    return np.degrees(np.arctan2(delta[1], delta[0]))


def _get_face_bbox(
    landmarks: np.ndarray,
    image_shape: tuple,
    margin: float = 0.3,
) -> tuple:
    """Compute a square bounding box around the face with margin."""
    h, w = image_shape[:2]

    x_min, y_min = landmarks.min(axis=0)
    x_max, y_max = landmarks.max(axis=0)

    face_w = x_max - x_min
    face_h = y_max - y_min
    face_size = max(face_w, face_h)

    margin_px = face_size * margin
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    half_size = (face_size + 2 * margin_px) / 2

    x1 = int(max(0, cx - half_size))
    y1 = int(max(0, cy - half_size))
    x2 = int(min(w, cx + half_size))
    y2 = int(min(h, cy + half_size))

    side = min(x2 - x1, y2 - y1)
    return (x1, y1, side, side)


def detect_and_align(
    image_path: Path,
    output_size: int = None,
    margin: float = None,
    model_path: Path = None,
) -> AlignedFace:
    """Detect face, align, crop, and resize.

    Args:
        image_path: Path to input image (JPEG, PNG, HEIC).
        output_size: Output image size in pixels. Defaults to config.input_image_size.
        margin: Extra margin around face. Defaults to config.face_crop_margin.
        model_path: Path to MediaPipe FaceLandmarker model file.

    Returns:
        AlignedFace with processed image and landmarks.

    Raises:
        ValueError: If no face is detected or image cannot be loaded.
    """
    image_path = Path(image_path)
    if output_size is None:
        output_size = config.input_image_size
    if margin is None:
        margin = config.face_crop_margin
    if model_path is None:
        model_path = _MODEL_PATH

    # Load image as RGB
    img_rgb = _load_image(image_path)
    orig_h, orig_w = img_rgb.shape[:2]

    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    # Detect face with FaceLandmarker
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise ValueError(f"No face detected in {image_path}")

    face_landmarks = result.face_landmarks[0]

    # Extract landmarks as numpy arrays
    landmarks_3d = np.array(
        [(lm.x, lm.y, lm.z) for lm in face_landmarks]
    )

    # Extract blendshape scores
    blendshape_scores = {}
    if result.face_blendshapes:
        for bs in result.face_blendshapes[0]:
            blendshape_scores[bs.category_name] = bs.score

    # Convert normalized coords to pixel coords
    landmarks_2d_px = landmarks_3d[:, :2].copy()
    landmarks_2d_px[:, 0] *= orig_w
    landmarks_2d_px[:, 1] *= orig_h

    # Compute rotation for eye alignment
    angle = _compute_eye_angle(landmarks_2d_px)

    # Rotate image and landmarks
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    center = (orig_w / 2, orig_h / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_bgr = cv2.warpAffine(img_bgr, rot_matrix, (orig_w, orig_h))

    # Rotate landmarks
    ones = np.ones((len(landmarks_2d_px), 1))
    landmarks_hom = np.hstack([landmarks_2d_px, ones])
    rotated_landmarks = (rot_matrix @ landmarks_hom.T).T

    # Compute bounding box on rotated image
    bbox = _get_face_bbox(rotated_landmarks, rotated_bgr.shape, margin)
    x, y, w, h = bbox

    # Crop
    cropped = rotated_bgr[y : y + h, x : x + w]

    # Adjust landmarks to crop coordinates
    crop_landmarks = rotated_landmarks.copy()
    crop_landmarks[:, 0] -= x
    crop_landmarks[:, 1] -= y

    # Resize
    resized = cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4)

    # Scale landmarks to resized coordinates
    scale_x = output_size / w
    scale_y = output_size / h
    crop_landmarks[:, 0] *= scale_x
    crop_landmarks[:, 1] *= scale_y

    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    return AlignedFace(
        image=resized,
        image_rgb=resized_rgb,
        landmarks_2d=crop_landmarks,
        landmarks_3d=landmarks_3d,
        blendshape_scores=blendshape_scores,
        bbox=bbox,
        rotation_angle=angle,
        original_image_path=image_path,
        original_image_size=(orig_w, orig_h),
    )
