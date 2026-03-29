from __future__ import annotations

"""GLB avatar validation for Three.js compatibility.

Validates that exported GLB files contain the correct morph targets,
skeleton, and materials for use with Three.js GLTFLoader.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pygltflib


@dataclass
class GLBValidationResult:
    """Result of GLB validation."""

    is_valid: bool
    file_path: Path
    file_size_mb: float
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    morph_target_count: int = 0
    morph_target_names: list = field(default_factory=list)
    has_skeleton: bool = False
    has_material: bool = False
    has_texture: bool = False
    joint_count: int = 0
    vertex_count: int = 0


# Expected morph target names (52 ARKit + 15 visemes)
EXPECTED_ARKIT_NAMES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
    "tongueOut",
]

EXPECTED_VISEME_NAMES = [
    "viseme_sil", "viseme_PP", "viseme_FF", "viseme_TH", "viseme_DD",
    "viseme_kk", "viseme_CH", "viseme_SS", "viseme_nn", "viseme_RR",
    "viseme_aa", "viseme_E", "viseme_I", "viseme_O", "viseme_U",
]

EXPECTED_NAMES = EXPECTED_ARKIT_NAMES + EXPECTED_VISEME_NAMES


def validate_glb(glb_path: Path) -> GLBValidationResult:
    """Validate a GLB file for Three.js avatar compatibility.

    Checks:
    - File exists and can be parsed
    - Contains mesh with morph targets
    - Morph target count matches expected (67)
    - Morph target names match ARKit + viseme conventions
    - Contains skeleton/skin
    - Contains material with texture

    Args:
        glb_path: Path to the GLB file.

    Returns:
        GLBValidationResult with detailed validation info.
    """
    glb_path = Path(glb_path)

    result = GLBValidationResult(
        is_valid=False,
        file_path=glb_path,
        file_size_mb=0,
    )

    # Check file exists
    if not glb_path.exists():
        result.errors.append(f"File not found: {glb_path}")
        return result

    result.file_size_mb = glb_path.stat().st_size / (1024 * 1024)

    # Parse GLTF
    try:
        gltf = pygltflib.GLTF2().load(str(glb_path))
    except Exception as e:
        result.errors.append(f"Failed to parse GLB: {e}")
        return result

    # Check meshes
    if not gltf.meshes:
        result.errors.append("No meshes found in GLB")
        return result

    mesh = gltf.meshes[0]

    # Check morph targets
    if mesh.primitives:
        prim = mesh.primitives[0]
        if prim.targets:
            result.morph_target_count = len(prim.targets)
        else:
            result.warnings.append("No morph targets found on primary primitive")

    # Check morph target names
    if mesh.extras and isinstance(mesh.extras, dict):
        target_names = mesh.extras.get("targetNames", [])
        result.morph_target_names = target_names
    elif hasattr(mesh, "extras") and mesh.extras:
        result.warnings.append("mesh.extras exists but targetNames not found")

    # Validate morph target count
    if result.morph_target_count == 0:
        result.errors.append("No morph targets (blendshapes) found")
    elif result.morph_target_count < 52:
        result.errors.append(
            f"Only {result.morph_target_count} morph targets found. "
            f"Expected at least 52 ARKit blendshapes"
        )
    elif result.morph_target_count < 67:
        result.warnings.append(
            f"{result.morph_target_count} morph targets found. "
            f"Missing visemes (expected 52 ARKit + 15 visemes = 67)"
        )

    # Validate morph target names
    if result.morph_target_names:
        expected_set = set(EXPECTED_NAMES)
        actual_set = set(result.morph_target_names)

        missing = expected_set - actual_set
        if missing:
            result.warnings.append(f"Missing morph targets: {sorted(missing)}")

        extra = actual_set - expected_set - {"_neutral"}
        if extra:
            result.warnings.append(f"Extra morph targets: {sorted(extra)}")

    # Check skeleton/skin
    if gltf.skins:
        result.has_skeleton = True
        result.joint_count = len(gltf.skins[0].joints)
    else:
        result.warnings.append("No skeleton/skin found")

    # Check materials
    if gltf.materials:
        result.has_material = True
        mat = gltf.materials[0]
        if mat.pbrMetallicRoughness and mat.pbrMetallicRoughness.baseColorTexture:
            result.has_texture = True
        else:
            result.warnings.append("Material has no base color texture")
    else:
        result.warnings.append("No materials found")

    # Check vertex count
    if mesh.primitives and mesh.primitives[0].attributes:
        pos_accessor_idx = mesh.primitives[0].attributes.POSITION
        if pos_accessor_idx is not None and pos_accessor_idx < len(gltf.accessors):
            result.vertex_count = gltf.accessors[pos_accessor_idx].count

    # File size sanity check
    if result.file_size_mb < 0.01:
        result.warnings.append(f"File suspiciously small: {result.file_size_mb:.3f} MB")
    elif result.file_size_mb > 50:
        result.warnings.append(f"File very large: {result.file_size_mb:.1f} MB")

    # Determine overall validity
    result.is_valid = len(result.errors) == 0

    return result


def print_validation_report(result: GLBValidationResult) -> None:
    """Print a human-readable validation report."""
    status = "PASS" if result.is_valid else "FAIL"
    print(f"\n=== GLB Validation: {status} ===")
    print(f"File: {result.file_path}")
    print(f"Size: {result.file_size_mb:.2f} MB")
    print(f"Vertices: {result.vertex_count}")
    print(f"Morph targets: {result.morph_target_count}")
    print(f"Skeleton: {'Yes' if result.has_skeleton else 'No'} ({result.joint_count} joints)")
    print(f"Material: {'Yes' if result.has_material else 'No'}")
    print(f"Texture: {'Yes' if result.has_texture else 'No'}")

    if result.errors:
        print(f"\nErrors:")
        for e in result.errors:
            print(f"  - {e}")

    if result.warnings:
        print(f"\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")

    if result.morph_target_names:
        print(f"\nMorph target names ({len(result.morph_target_names)}):")
        for name in result.morph_target_names:
            print(f"  - {name}")
