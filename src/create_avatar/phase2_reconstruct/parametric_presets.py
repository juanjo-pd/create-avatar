from __future__ import annotations

"""Parametric FLAME shape presets for generating avatars without a photo.

Each preset defines shape parameters (100-dim PCA coefficients) that
produce a distinctive face shape. The first ~10 components capture
most of the variation: overall face width, jaw shape, forehead height, etc.

Presets are empirically tuned starting from the FLAME mean face.
"""

from dataclasses import dataclass

import numpy as np

from create_avatar.config import config


@dataclass
class ShapePreset:
    """A named FLAME shape parameter preset."""

    name: str
    description: str
    shape_params: np.ndarray  # (100,) FLAME shape coefficients
    skin_tone: str  # Suggested skin tone for texture generation


# Component semantics (approximate, based on FLAME PCA analysis):
# PC0: overall face width (+ = wider)
# PC1: face height / jaw prominence (+ = longer face, prominent jaw)
# PC2: nose size / mid-face projection (+ = larger nose, more projected)
# PC3: forehead height (+ = taller forehead)
# PC4: face roundness (+ = rounder)
# PC5: cheekbone prominence (+ = more prominent)
# PC6: chin shape (+ = pointed chin)
# PC7: eye area width (+ = wider set eyes)
# PC8: lip fullness / lower face (+ = fuller lips)
# PC9: face asymmetry (keep near 0 for symmetric faces)


def _make_shape(components: dict[int, float]) -> np.ndarray:
    """Create a 100-dim shape vector from sparse component values."""
    params = np.zeros(config.flame_num_shape_params, dtype=np.float32)
    for idx, val in components.items():
        params[idx] = val
    return params


PRESETS: dict[str, ShapePreset] = {
    "male_average": ShapePreset(
        name="male_average",
        description="Average male face with balanced proportions",
        shape_params=_make_shape({0: 0.5, 1: 0.6, 2: 0.3, 5: 0.2}),
        skin_tone="medium",
    ),
    "female_average": ShapePreset(
        name="female_average",
        description="Average female face with balanced proportions",
        shape_params=_make_shape({0: -0.4, 1: -0.3, 2: -0.2, 4: 0.4, 6: 0.3, 8: 0.3}),
        skin_tone="medium",
    ),
    "male_angular": ShapePreset(
        name="male_angular",
        description="Angular male face with strong jawline",
        shape_params=_make_shape({0: 0.3, 1: 1.2, 2: 0.4, 5: 0.8, 6: -0.5}),
        skin_tone="light",
    ),
    "female_round": ShapePreset(
        name="female_round",
        description="Round female face with soft features",
        shape_params=_make_shape({0: 0.2, 1: -0.5, 4: 1.0, 5: -0.3, 6: 0.6, 8: 0.5}),
        skin_tone="medium",
    ),
    "male_slim": ShapePreset(
        name="male_slim",
        description="Slim male face with narrow features",
        shape_params=_make_shape({0: -0.8, 1: 0.8, 2: -0.3, 3: 0.5, 4: -0.5, 6: 0.4}),
        skin_tone="light",
    ),
    "female_angular": ShapePreset(
        name="female_angular",
        description="Angular female face with defined cheekbones",
        shape_params=_make_shape({0: -0.3, 1: 0.4, 2: -0.2, 5: 1.0, 6: -0.3, 8: 0.2}),
        skin_tone="dark",
    ),
    "male_round": ShapePreset(
        name="male_round",
        description="Round male face with broad features",
        shape_params=_make_shape({0: 1.0, 1: -0.3, 2: 0.5, 4: 0.8, 5: -0.2, 8: 0.4}),
        skin_tone="dark",
    ),
    "neutral": ShapePreset(
        name="neutral",
        description="Gender-neutral mean face",
        shape_params=_make_shape({}),
        skin_tone="medium",
    ),
}


def get_preset(name: str) -> ShapePreset:
    """Get a named shape preset.

    Args:
        name: Preset name (e.g., 'male_average', 'female_round').

    Returns:
        ShapePreset with shape parameters and metadata.

    Raises:
        KeyError: If preset name is not found.
    """
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[name]


def random_preset(rng: np.random.Generator | None = None) -> ShapePreset:
    """Generate a random but plausible shape preset.

    Samples from a normal distribution on the first 10 PCA components
    with sigma=1.5, which covers most of the natural variation.

    Args:
        rng: Optional numpy random generator for reproducibility.

    Returns:
        ShapePreset with randomly sampled shape parameters.
    """
    if rng is None:
        rng = np.random.default_rng()

    params = np.zeros(config.flame_num_shape_params, dtype=np.float32)
    # Sample first 10 components (capture ~95% of shape variation)
    params[:10] = rng.normal(0, 1.5, size=10).astype(np.float32)
    # Clamp to avoid extreme deformations
    params = np.clip(params, -3.0, 3.0)

    skin_tones = ["light", "medium", "dark"]
    skin_tone = rng.choice(skin_tones)

    return ShapePreset(
        name=f"random_{rng.integers(0, 99999):05d}",
        description="Randomly generated face shape",
        shape_params=params,
        skin_tone=skin_tone,
    )


def random_variation(
    preset: ShapePreset,
    sigma: float = 0.3,
    rng: np.random.Generator | None = None,
) -> ShapePreset:
    """Create a slight variation of an existing preset.

    Adds Gaussian noise to the shape parameters while preserving
    the overall character of the preset.

    Args:
        preset: Base preset to vary.
        sigma: Standard deviation of the noise. 0.3 = subtle, 0.8 = noticeable.
        rng: Optional numpy random generator.

    Returns:
        New ShapePreset with slightly varied shape parameters.
    """
    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(0, sigma, size=preset.shape_params.shape).astype(np.float32)
    varied_params = np.clip(preset.shape_params + noise, -3.0, 3.0)

    return ShapePreset(
        name=f"{preset.name}_var_{rng.integers(0, 99999):05d}",
        description=f"Variation of {preset.name}",
        shape_params=varied_params,
        skin_tone=preset.skin_tone,
    )


def list_presets() -> list[str]:
    """Return all available preset names."""
    return list(PRESETS.keys())
