"""Generate viseme morph targets from ARKit blendshapes.

Visemes are computed as linear combinations of the 52 ARKit blendshape
vertex displacements, weighted according to the definitions in viseme_definitions.py.
"""

import numpy as np

from create_avatar.phase3_blendshapes.arkit_names import (
    ARKIT_BLENDSHAPE_NAMES,
    VISEME_NAMES,
)
from create_avatar.phase3_blendshapes.viseme_definitions import VISEME_DEFINITIONS


def generate_viseme_vertices(
    neutral_vertices: np.ndarray,
    blendshape_vertices: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Generate viseme morph target vertices from ARKit blendshapes.

    Each viseme is a linear combination of ARKit blendshape displacements:
        viseme_vertices = neutral + sum(weight_i * (blendshape_i - neutral))

    Args:
        neutral_vertices: (N, 3) neutral pose vertex positions.
        blendshape_vertices: Dict mapping ARKit blendshape name to (N, 3) vertices.
            Must contain all 52 ARKit blendshape names.

    Returns:
        Dict mapping viseme name to (N, 3) vertex positions.
    """
    # Validate inputs
    missing = set(ARKIT_BLENDSHAPE_NAMES) - set(blendshape_vertices.keys())
    if missing:
        raise ValueError(f"Missing ARKit blendshapes: {missing}")

    # Precompute deltas (displacement from neutral) for each blendshape
    deltas: dict[str, np.ndarray] = {}
    for name in ARKIT_BLENDSHAPE_NAMES:
        deltas[name] = blendshape_vertices[name] - neutral_vertices

    # Generate each viseme
    viseme_vertices: dict[str, np.ndarray] = {}

    for viseme_name in VISEME_NAMES:
        definition = VISEME_DEFINITIONS[viseme_name]

        # Start from neutral
        combined_delta = np.zeros_like(neutral_vertices)

        # Add weighted blendshape deltas
        for bs_name, weight in definition.items():
            if bs_name in deltas:
                combined_delta += weight * deltas[bs_name]

        viseme_vertices[viseme_name] = neutral_vertices + combined_delta

    return viseme_vertices


def generate_viseme_deltas(
    blendshape_deltas: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Generate viseme deltas (displacements) from ARKit blendshape deltas.

    Similar to generate_viseme_vertices but works with deltas directly,
    which is more efficient when you already have the displacements.

    Args:
        blendshape_deltas: Dict mapping ARKit name to (N, 3) displacement from neutral.

    Returns:
        Dict mapping viseme name to (N, 3) displacement from neutral.
    """
    viseme_deltas: dict[str, np.ndarray] = {}

    for viseme_name in VISEME_NAMES:
        definition = VISEME_DEFINITIONS[viseme_name]

        # Get vertex count from any blendshape
        sample_key = next(iter(blendshape_deltas))
        combined_delta = np.zeros_like(blendshape_deltas[sample_key])

        for bs_name, weight in definition.items():
            if bs_name in blendshape_deltas:
                combined_delta += weight * blendshape_deltas[bs_name]

        viseme_deltas[viseme_name] = combined_delta

    return viseme_deltas
