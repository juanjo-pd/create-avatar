"""Canonical ARKit blendshape and viseme names.

These names MUST match exactly what Three.js expects in morphTargetDictionary.
They are stored in camelCase in the GLB's mesh.extras.targetNames array.

Reference: https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation
"""

# The 52 ARKit Face Blendshape names in canonical order.
# This order determines the morph target index in the exported GLB.
ARKIT_BLENDSHAPE_NAMES: list[str] = [
    # Brows (5)
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    # Cheeks (3)
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    # Eyes (14)
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    # Jaw (4)
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    # Mouth (24)
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    # Nose (2)
    "noseSneerLeft",
    "noseSneerRight",
    # Tongue (1)
    "tongueOut",
]

# The 15 Oculus/standard viseme names for lip-sync.
# Prefixed with "viseme_" to avoid collision with ARKit blendshape names.
VISEME_NAMES: list[str] = [
    "viseme_sil",  # Silence / mouth closed
    "viseme_PP",   # p, b, m — lips together
    "viseme_FF",   # f, v — lower lip to upper teeth
    "viseme_TH",   # th — tongue between teeth
    "viseme_DD",   # t, d, n — tongue to palate
    "viseme_kk",   # k, g — back of tongue
    "viseme_CH",   # ch, j, sh — tense smile
    "viseme_SS",   # s, z — teeth nearly closed
    "viseme_nn",   # n, l — mouth slightly open
    "viseme_RR",   # r — lips slightly rounded
    "viseme_aa",   # a — mouth open
    "viseme_E",    # e — mouth semi-open, stretched
    "viseme_I",    # i — mouth nearly closed, stretched
    "viseme_O",    # o — lips rounded
    "viseme_U",    # u — lips very rounded
]

# Combined list: all morph target names in the order they appear in the GLB
ALL_MORPH_TARGET_NAMES: list[str] = ARKIT_BLENDSHAPE_NAMES + VISEME_NAMES

# Quick validation
assert len(ARKIT_BLENDSHAPE_NAMES) == 52, f"Expected 52 ARKit blendshapes, got {len(ARKIT_BLENDSHAPE_NAMES)}"
assert len(VISEME_NAMES) == 15, f"Expected 15 visemes, got {len(VISEME_NAMES)}"
assert len(ALL_MORPH_TARGET_NAMES) == 67, f"Expected 67 total morph targets, got {len(ALL_MORPH_TARGET_NAMES)}"
assert len(set(ALL_MORPH_TARGET_NAMES)) == 67, "Duplicate morph target names found!"
