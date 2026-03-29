from __future__ import annotations

"""Generate synthetic blendshapes for missing ARKit targets.

Creates approximations for blendshapes not available in the reference set,
using FLAME mesh topology and region masks.
"""

import numpy as np


def generate_tongue_out(
    neutral_vertices: np.ndarray,
    faces: np.ndarray,
    masks: dict = None,
) -> np.ndarray:
    """Generate a synthetic tongueOut blendshape.

    Pushes lower lip/chin vertices forward and down to simulate
    tongue protrusion. Since FLAME doesn't model the tongue explicitly,
    this is an approximation using lip/jaw region deformation.

    Args:
        neutral_vertices: (5023, 3) neutral vertex positions.
        faces: (9976, 3) face indices.
        masks: FLAME vertex masks dict with 'lips' key.

    Returns:
        (5023, 3) tongueOut vertex positions.
    """
    result = neutral_vertices.copy()

    if masks is not None and "lips" in masks:
        lip_indices = np.array(masks["lips"])
    else:
        # Fallback: estimate lip region from geometry
        # Lips are typically in the lower-center front of the face
        center_x = neutral_vertices[:, 0].mean()
        center_z = neutral_vertices[:, 2].mean()

        # Find vertices near mouth area (front, center, lower-middle)
        y_range = neutral_vertices[:, 1].max() - neutral_vertices[:, 1].min()
        y_center = neutral_vertices[:, 1].mean()
        mouth_y = y_center - y_range * 0.15  # Below center

        dist_to_mouth = np.sqrt(
            (neutral_vertices[:, 0] - center_x) ** 2 +
            (neutral_vertices[:, 1] - mouth_y) ** 2
        )
        lip_indices = np.where(dist_to_mouth < y_range * 0.1)[0]

    if len(lip_indices) == 0:
        return result

    # Compute mouth center
    lip_verts = neutral_vertices[lip_indices]
    mouth_center = lip_verts.mean(axis=0)

    # Find lower lip vertices (below mouth center Y)
    lower_mask = lip_verts[:, 1] < mouth_center[1]
    lower_lip_indices = lip_indices[lower_mask]

    if len(lower_lip_indices) == 0:
        lower_lip_indices = lip_indices

    # Compute forward direction (face normal direction at mouth)
    # In FLAME, Z is typically the depth axis (forward)
    z_min = neutral_vertices[:, 2].min()
    face_depth = neutral_vertices[:, 2].max() - z_min

    # Push lower lip vertices forward (negative Z in FLAME) and slightly down
    tongue_displacement = face_depth * 0.08  # 8% of face depth

    for idx in lower_lip_indices:
        v = neutral_vertices[idx]
        # Distance from mouth center determines displacement falloff
        dist = np.linalg.norm(v[:2] - mouth_center[:2])
        max_dist = np.linalg.norm(lip_verts[:, :2] - mouth_center[:2], axis=1).max()
        falloff = max(0, 1.0 - dist / (max_dist + 1e-8))
        falloff = falloff ** 2  # Quadratic falloff

        # Push forward and down
        result[idx, 2] -= tongue_displacement * falloff  # forward
        result[idx, 1] -= tongue_displacement * 0.5 * falloff  # down

    return result
