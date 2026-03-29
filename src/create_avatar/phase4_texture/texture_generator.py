from __future__ import annotations

"""Procedural texture generation for no-photo avatar mode.

Generates a basic skin-tone texture map for avatars created
from parametric FLAME presets without a source photo.
"""

import numpy as np
import cv2


# Base skin tone RGB values
SKIN_TONES = {
    "light": (230, 200, 180),
    "medium": (200, 165, 135),
    "dark": (140, 100, 75),
    "olive": (195, 170, 130),
}


def generate_skin_texture(
    texture_size: int = 1024,
    skin_tone: str = "medium",
    noise_strength: float = 0.03,
    seed: int = None,
) -> np.ndarray:
    """Generate a procedural skin texture map.

    Creates a base skin color with subtle noise for realism.

    Args:
        texture_size: Output texture resolution.
        skin_tone: Skin tone name from SKIN_TONES.
        noise_strength: Intensity of color noise (0.0 to 0.1).
        seed: Random seed for reproducibility.

    Returns:
        (texture_size, texture_size, 3) RGB texture map.
    """
    rng = np.random.default_rng(seed)

    if skin_tone not in SKIN_TONES:
        skin_tone = "medium"

    base_color = np.array(SKIN_TONES[skin_tone], dtype=np.float32)

    # Create base texture
    texture = np.full((texture_size, texture_size, 3), base_color, dtype=np.float32)

    # Add Perlin-like noise for natural skin variation
    noise = rng.normal(0, noise_strength * 255, (texture_size, texture_size, 3)).astype(np.float32)

    # Smooth the noise for more natural look
    noise = cv2.GaussianBlur(noise, (31, 31), 10)

    texture += noise
    texture = np.clip(texture, 0, 255).astype(np.uint8)

    return texture
