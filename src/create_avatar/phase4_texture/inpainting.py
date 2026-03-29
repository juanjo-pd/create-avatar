from __future__ import annotations

"""Inpainting for filling occluded or missing UV texture regions.

Uses OpenCV's Telea inpainting algorithm to fill holes in the
UV texture map where the original photo had no visible data.
"""

import cv2
import numpy as np


def inpaint_texture(
    texture: np.ndarray,
    radius: int = 5,
) -> np.ndarray:
    """Fill black (empty) regions in a UV texture using inpainting.

    Args:
        texture: (H, W, 3) RGB texture with black regions to fill.
        radius: Inpainting neighborhood radius.

    Returns:
        (H, W, 3) inpainted texture.
    """
    # Create mask: black pixels (all channels near zero) are holes
    gray = cv2.cvtColor(texture, cv2.COLOR_RGB2GRAY)
    mask = (gray < 5).astype(np.uint8) * 255

    if mask.sum() == 0:
        return texture

    # Convert to BGR for OpenCV inpainting
    texture_bgr = cv2.cvtColor(texture, cv2.COLOR_RGB2BGR)

    # Inpaint using Telea method
    inpainted = cv2.inpaint(texture_bgr, mask, radius, cv2.INPAINT_TELEA)

    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)


def blend_seams(
    texture: np.ndarray,
    blur_radius: int = 15,
) -> np.ndarray:
    """Soften seam boundaries in the texture map.

    Applies Gaussian blur at the boundary between original and inpainted
    regions for smoother transitions.

    Args:
        texture: (H, W, 3) texture with potential hard seams.
        blur_radius: Gaussian blur kernel size.

    Returns:
        (H, W, 3) texture with softened seams.
    """
    # Detect edges (seam boundaries)
    gray = cv2.cvtColor(texture, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 100)

    # Dilate edges to create seam region
    kernel = np.ones((blur_radius, blur_radius), np.uint8)
    seam_mask = cv2.dilate(edges, kernel, iterations=1)

    # Blur the seam regions
    blurred = cv2.GaussianBlur(texture, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)

    # Blend: use blurred version only in seam regions
    seam_mask_3ch = seam_mask[:, :, np.newaxis].astype(np.float32) / 255.0
    result = texture.astype(np.float32) * (1 - seam_mask_3ch) + blurred.astype(np.float32) * seam_mask_3ch

    return result.astype(np.uint8)
