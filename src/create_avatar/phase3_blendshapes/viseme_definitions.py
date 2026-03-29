"""Viseme definitions as weighted combinations of ARKit blendshapes.

Each viseme is defined as a dict mapping ARKit blendshape names to weights [0.0, 1.0].
These weights are used to generate viseme morph targets by linearly combining
the ARKit blendshape vertex displacements.

References:
- Oculus Lipsync viseme set
- Standard phoneme-to-viseme mappings for English/Spanish
"""

# Type alias for readability
VisemeDefinition = dict[str, float]

# Each viseme is a weighted combination of ARKit blendshapes.
# Weights represent the intensity of each blendshape contribution.
VISEME_DEFINITIONS: dict[str, VisemeDefinition] = {
    "viseme_sil": {
        # Silence: relaxed, mouth closed
        "mouthClose": 0.1,
    },
    "viseme_PP": {
        # p, b, m: lips pressed together
        "mouthClose": 0.9,
        "mouthPressLeft": 0.6,
        "mouthPressRight": 0.6,
        "mouthPucker": 0.2,
    },
    "viseme_FF": {
        # f, v: lower lip to upper teeth
        "mouthFunnel": 0.3,
        "mouthLowerDownLeft": 0.4,
        "mouthLowerDownRight": 0.4,
        "mouthRollLower": 0.5,
        "jawOpen": 0.1,
    },
    "viseme_TH": {
        # th: tongue between teeth
        "jawOpen": 0.15,
        "tongueOut": 0.6,
        "mouthLowerDownLeft": 0.2,
        "mouthLowerDownRight": 0.2,
    },
    "viseme_DD": {
        # t, d, n: tongue to palate
        "jawOpen": 0.2,
        "mouthLowerDownLeft": 0.3,
        "mouthLowerDownRight": 0.3,
        "mouthStretchLeft": 0.1,
        "mouthStretchRight": 0.1,
    },
    "viseme_kk": {
        # k, g: back of tongue
        "jawOpen": 0.25,
        "mouthLowerDownLeft": 0.2,
        "mouthLowerDownRight": 0.2,
        "mouthStretchLeft": 0.15,
        "mouthStretchRight": 0.15,
    },
    "viseme_CH": {
        # ch, j, sh: tense smile-like shape
        "jawOpen": 0.15,
        "mouthSmileLeft": 0.3,
        "mouthSmileRight": 0.3,
        "mouthFunnel": 0.4,
        "mouthStretchLeft": 0.2,
        "mouthStretchRight": 0.2,
    },
    "viseme_SS": {
        # s, z: teeth nearly closed
        "jawOpen": 0.05,
        "mouthSmileLeft": 0.2,
        "mouthSmileRight": 0.2,
        "mouthStretchLeft": 0.3,
        "mouthStretchRight": 0.3,
    },
    "viseme_nn": {
        # n, l: mouth slightly open
        "jawOpen": 0.15,
        "mouthLowerDownLeft": 0.2,
        "mouthLowerDownRight": 0.2,
        "mouthSmileLeft": 0.1,
        "mouthSmileRight": 0.1,
    },
    "viseme_RR": {
        # r: lips slightly rounded
        "jawOpen": 0.2,
        "mouthPucker": 0.4,
        "mouthFunnel": 0.3,
        "mouthRollLower": 0.1,
        "mouthRollUpper": 0.1,
    },
    "viseme_aa": {
        # a: mouth wide open
        "jawOpen": 0.7,
        "mouthFunnel": 0.2,
        "mouthLowerDownLeft": 0.3,
        "mouthLowerDownRight": 0.3,
        "mouthUpperUpLeft": 0.2,
        "mouthUpperUpRight": 0.2,
    },
    "viseme_E": {
        # e: mouth semi-open, stretched
        "jawOpen": 0.3,
        "mouthSmileLeft": 0.4,
        "mouthSmileRight": 0.4,
        "mouthStretchLeft": 0.3,
        "mouthStretchRight": 0.3,
    },
    "viseme_I": {
        # i: mouth nearly closed, stretched
        "jawOpen": 0.1,
        "mouthSmileLeft": 0.5,
        "mouthSmileRight": 0.5,
        "mouthStretchLeft": 0.4,
        "mouthStretchRight": 0.4,
    },
    "viseme_O": {
        # o: lips rounded
        "jawOpen": 0.4,
        "mouthFunnel": 0.6,
        "mouthPucker": 0.3,
        "mouthRollLower": 0.1,
        "mouthRollUpper": 0.1,
    },
    "viseme_U": {
        # u: lips very rounded (tight)
        "jawOpen": 0.2,
        "mouthFunnel": 0.5,
        "mouthPucker": 0.7,
        "mouthRollLower": 0.15,
        "mouthRollUpper": 0.15,
    },
}


def get_viseme_definition(viseme_name: str) -> VisemeDefinition:
    """Get the ARKit blendshape weights for a viseme.

    Args:
        viseme_name: Viseme name (e.g., 'viseme_aa').

    Returns:
        Dict mapping ARKit blendshape names to weights.

    Raises:
        KeyError: If viseme name is not found.
    """
    if viseme_name not in VISEME_DEFINITIONS:
        available = ", ".join(VISEME_DEFINITIONS.keys())
        raise KeyError(f"Unknown viseme '{viseme_name}'. Available: {available}")
    return VISEME_DEFINITIONS[viseme_name]


def validate_definitions():
    """Validate that all viseme definitions reference valid ARKit blendshape names."""
    from create_avatar.phase3_blendshapes.arkit_names import ARKIT_BLENDSHAPE_NAMES

    valid_names = set(ARKIT_BLENDSHAPE_NAMES)
    errors = []

    for viseme_name, definition in VISEME_DEFINITIONS.items():
        for bs_name, weight in definition.items():
            if bs_name not in valid_names:
                errors.append(f"{viseme_name}: invalid blendshape '{bs_name}'")
            if not 0.0 <= weight <= 1.0:
                errors.append(f"{viseme_name}: weight {weight} for '{bs_name}' out of [0,1]")

    if errors:
        raise ValueError("Viseme definition errors:\n" + "\n".join(errors))
