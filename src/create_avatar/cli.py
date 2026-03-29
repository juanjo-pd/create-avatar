from __future__ import annotations

"""CLI entry point for the avatar generation pipeline."""

import click
from pathlib import Path
import subprocess

from create_avatar.config import config

BLENDER_PATH = "/Applications/Blender.app/Contents/MacOS/Blender"
SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"


@click.group()
@click.option("--output-dir", type=click.Path(), default=None, help="Output directory")
@click.option("--device", type=click.Choice(["cpu", "mps"]), default="cpu", help="Compute device")
@click.pass_context
def main(ctx, output_dir, device):
    """Create Avatar: Generate GLB bust avatars with ARKit blendshapes + visemes."""
    ctx.ensure_object(dict)
    if output_dir:
        config.output_dir = Path(output_dir)
    config.device = device


def _run_pipeline(
    avatar_id: str,
    flame_output,
    flame_model=None,
    aligned_face=None,
    skin_tone: str = "medium",
):
    """Run phases 3-6 of the pipeline (shared between from-photo and generate).

    Args:
        avatar_id: Avatar identifier.
        flame_output: FlameOutput from Phase 2.
        flame_model: FlameModel instance (for masks/bust generation).
        aligned_face: Optional AlignedFace for photo texture projection.
        skin_tone: Skin tone for procedural texture (when no photo).
    """
    import numpy as np
    import cv2
    from create_avatar.phase3_blendshapes.blendshape_transfer import BlendshapeTransfer
    from create_avatar.phase3_blendshapes.viseme_generator import generate_viseme_deltas
    from create_avatar.phase3_blendshapes.arkit_names import ARKIT_BLENDSHAPE_NAMES, VISEME_NAMES
    from create_avatar.phase3_blendshapes.synthetic_blendshapes import generate_tongue_out
    from create_avatar.phase5_assembly.bust_generator import generate_bust_mesh
    from create_avatar.utils.mesh_io import save_vertices_as_obj

    avatar_dir = config.output_dir / avatar_id
    avatar_dir.mkdir(parents=True, exist_ok=True)

    # Generate bust (neck + shoulders) from FLAME neck boundary
    click.echo("\n[Phase 2b] Generating bust (neck + shoulders)...")
    bust_result = generate_bust_mesh(
        vertices=flame_output.vertices,
        faces=flame_output.faces,
    )
    bust_vertices = bust_result["vertices"]
    bust_faces = bust_result["faces"]
    head_vcount = bust_result["head_vertex_count"]
    click.echo(f"  Head: {head_vcount} + Bust: {bust_result['bust_vertex_count']} = {len(bust_vertices)} verts")

    # Save neutral mesh (with bust)
    save_vertices_as_obj(avatar_dir / "neutral.obj", bust_vertices, bust_faces)
    np.savez(
        avatar_dir / "flame_params.npz",
        shape=flame_output.shape_params,
        expression=flame_output.expression_params,
        pose=flame_output.pose_params,
    )

    # Phase 3: Blendshape Transfer
    click.echo("\n[Phase 3] Transferring ARKit blendshapes...")
    transfer = BlendshapeTransfer(
        source_neutral_path=config.data_dir / "arkit_reference" / "Neutral.obj",
        source_blendshapes_dir=config.data_dir / "arkit_reference",
        target_neutral_vertices=flame_output.vertices,
        target_faces=flame_output.faces,
        method="rbf",
    )
    transfer_result = transfer.transfer()
    deltas = transfer_result["blendshape_deltas"]
    bs_verts = transfer_result["blendshape_vertices"]

    # Generate synthetic tongueOut
    click.echo("[Phase 3a] Generating tongueOut...")
    masks = flame_model.masks if flame_model else None
    tongue_verts = generate_tongue_out(flame_output.vertices, flame_output.faces, masks)
    bs_verts["tongueOut"] = tongue_verts
    deltas["tongueOut"] = tongue_verts - flame_output.vertices

    # Save blendshape OBJs (with bust vertices appended)
    bs_dir = avatar_dir / "blendshapes"
    bs_dir.mkdir(parents=True, exist_ok=True)

    bust_extra = bust_vertices[head_vcount:]  # Bust-only vertices (unchanged by blendshapes)

    for name in ARKIT_BLENDSHAPE_NAMES:
        if name in bs_verts:
            # Append bust vertices (static) to head blendshape
            full_verts = np.vstack([bs_verts[name], bust_extra]) if len(bust_extra) > 0 else bs_verts[name]
            save_vertices_as_obj(bs_dir / f"{name}.obj", full_verts, bust_faces)

    # Generate visemes
    click.echo("[Phase 3b] Generating visemes...")
    viseme_deltas = generate_viseme_deltas(deltas)
    for name in VISEME_NAMES:
        viseme_verts = flame_output.vertices + viseme_deltas[name]
        full_verts = np.vstack([viseme_verts, bust_extra]) if len(bust_extra) > 0 else viseme_verts
        save_vertices_as_obj(bs_dir / f"{name}.obj", full_verts, bust_faces)

    # Save names list for Blender
    with open(bs_dir / "_names.txt", "w") as f:
        for name in ARKIT_BLENDSHAPE_NAMES:
            if name in bs_verts:
                f.write(name + "\n")
        for name in VISEME_NAMES:
            f.write(name + "\n")

    n_arkit = len([n for n in ARKIT_BLENDSHAPE_NAMES if n in bs_verts])
    click.echo(f"  {n_arkit} ARKit + {len(VISEME_NAMES)} visemes = {n_arkit + len(VISEME_NAMES)} morph targets")

    # Phase 4: Texture
    click.echo("\n[Phase 4] Generating texture...")
    texture_path = avatar_dir / "texture.png"

    if aligned_face is not None and hasattr(flame_output, 'camera_scale'):
        try:
            from create_avatar.phase4_texture.uv_projection import project_photo_texture

            click.echo("  Projecting photo onto UV (camera-aligned)...")
            texture = project_photo_texture(
                photo=aligned_face.image_rgb,
                vertices_3d=flame_output.vertices,
                camera_scale=flame_output.camera_scale,
                camera_tx=flame_output.camera_tx,
                camera_ty=flame_output.camera_ty,
                landmarks_2d=aligned_face.landmarks_2d,
                texture_size=config.texture_resolution,
            )
            click.echo(f"  Camera: scale={flame_output.camera_scale:.2f}")
        except Exception as e:
            click.echo(f"  Projection failed ({e}), using procedural")
            from create_avatar.phase4_texture.texture_generator import generate_skin_texture
            texture = generate_skin_texture(texture_size=config.texture_resolution, skin_tone=skin_tone, seed=42)
    else:
        from create_avatar.phase4_texture.texture_generator import generate_skin_texture
        texture = generate_skin_texture(texture_size=config.texture_resolution, skin_tone=skin_tone, seed=42)
        click.echo("  Procedural texture generated")

    cv2.imwrite(str(texture_path), cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))
    click.echo(f"  Saved: {texture_path}")

    # Phase 5-6: Blender Assembly + GLB Export
    click.echo("\n[Phase 5-6] Assembling avatar and exporting GLB...")
    glb_path = avatar_dir / "avatar.glb"

    cmd = [
        BLENDER_PATH, "-b",
        "--python", str(SCRIPTS_DIR / "blender_assembly.py"),
        "--",
        "--head-obj", str(avatar_dir / "neutral.obj"),
        "--blendshapes-dir", str(bs_dir),
        "--texture-png", str(texture_path),
        "--output-glb", str(glb_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        click.echo(f"  Blender error: {result.stderr[-500:]}")
        return None

    # Extract key lines from Blender output
    for line in result.stdout.splitlines():
        if any(k in line for k in ["Added shape key", "Exported", "Assembly complete"]):
            if "shape key" not in line:  # Don't spam all 66 shape keys
                click.echo(f"  {line.strip()}")

    # Validate
    click.echo("\n[Validation] Checking GLB...")
    from create_avatar.phase6_export.validator import validate_glb, print_validation_report
    validation = validate_glb(glb_path)
    print_validation_report(validation)

    return glb_path


@main.command()
@click.argument("photo_path", type=click.Path(exists=True))
@click.option("--avatar-id", default=None, help="Custom avatar ID (default: derived from filename)")
@click.option("--output-size", default=512, help="Face crop resolution for processing")
def from_photo(photo_path, avatar_id, output_size):
    """Generate a personalized avatar from a selfie/photo."""
    from create_avatar.phase1_preprocess.face_detect import detect_and_align
    from create_avatar.phase1_preprocess.validators import validate_image
    from create_avatar.phase2_reconstruct.flame_model import FlameModel
    from create_avatar.phase2_reconstruct.mediapipe_fitting import MediaPipeFLAMEFitter

    photo_path = Path(photo_path)
    if avatar_id is None:
        avatar_id = photo_path.stem

    avatar_dir = config.output_dir / avatar_id
    avatar_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"=== Creating avatar: {avatar_id} ===")
    click.echo(f"Device: {config.device}")

    # Phase 1: Face detection
    click.echo("\n[Phase 1] Detecting and aligning face...")
    aligned = detect_and_align(photo_path, output_size=output_size)

    validation = validate_image(photo_path, aligned.landmarks_3d)
    if not validation.is_valid:
        click.echo("Validation FAILED:")
        for err in validation.errors:
            click.echo(f"  - {err}")
        return

    for warn in validation.warnings:
        click.echo(f"  Warning: {warn}")

    click.echo(f"  {aligned.landmarks_2d.shape[0]} landmarks, {len(aligned.blendshape_scores)} blendshapes")

    import cv2
    cv2.imwrite(str(avatar_dir / "aligned_face.jpg"), aligned.image)

    # Phase 2: FLAME fitting
    click.echo("\n[Phase 2] Fitting FLAME model to face...")
    flame = FlameModel()
    fitter = MediaPipeFLAMEFitter(flame, num_shape_params=100, num_expression_params=20)

    # Use landmarks from CROPPED image, normalized to 0-1
    # This ensures the camera params match the cropped photo for texture projection
    lm2d_cropped = aligned.landmarks_2d.copy()
    lm2d_cropped[:, 0] /= output_size
    lm2d_cropped[:, 1] /= output_size
    flame_output = fitter.fit(lm2d_cropped, num_iterations=3000, verbose=True)

    # Phases 3-6
    glb_path = _run_pipeline(avatar_id, flame_output, flame_model=flame, aligned_face=aligned)

    if glb_path:
        click.echo(f"\n=== Avatar complete: {glb_path} ===")


@main.command()
@click.option("--preset", type=str, default="random", help="FLAME shape preset name or 'random'")
@click.option("--count", default=1, help="Number of avatars to generate")
@click.option("--skin-tone", type=str, default="medium", help="Skin tone for texture")
def generate(preset, count, skin_tone):
    """Generate avatar(s) from parametric FLAME presets (no photo needed)."""
    from create_avatar.phase2_reconstruct.flame_model import FlameModel
    from create_avatar.phase2_reconstruct.parametric_presets import (
        get_preset, random_preset, random_variation, list_presets,
    )

    click.echo(f"Generating {count} avatar(s) with preset: {preset}")
    click.echo(f"Available: {', '.join(list_presets())}")

    flame = FlameModel()

    for i in range(count):
        if preset == "random":
            shape_preset = random_preset()
        else:
            shape_preset = get_preset(preset)
            if count > 1:
                shape_preset = random_variation(shape_preset, sigma=0.3)

        click.echo(f"\n--- Avatar {i+1}/{count}: {shape_preset.name} ---")

        # Phase 2: Generate from FLAME params directly
        click.echo("[Phase 2] Generating FLAME mesh from preset...")
        flame_output = flame.generate(shape_params=shape_preset.shape_params)

        # Phases 3-6
        glb_path = _run_pipeline(
            shape_preset.name, flame_output,
            flame_model=flame, skin_tone=skin_tone or shape_preset.skin_tone,
        )

        if glb_path:
            click.echo(f"=== Avatar complete: {glb_path} ===")


@main.command()
@click.argument("input_dir", type=click.Path(exists=True))
def batch(input_dir):
    """Process all photos in a directory."""
    input_dir = Path(input_dir)
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.heic", "*.heif")
    photos = []
    for ext in extensions:
        photos.extend(input_dir.glob(ext))

    click.echo(f"Found {len(photos)} photos in {input_dir}")

    for i, photo in enumerate(sorted(photos)):
        click.echo(f"\n{'='*50}")
        click.echo(f"[{i+1}/{len(photos)}] Processing: {photo.name}")
        ctx = click.get_current_context()
        ctx.invoke(from_photo, photo_path=str(photo), avatar_id=photo.stem)


@main.command()
@click.argument("glb_path", type=click.Path(exists=True))
def validate(glb_path):
    """Validate a GLB file for Three.js compatibility."""
    from create_avatar.phase6_export.validator import validate_glb, print_validation_report
    result = validate_glb(Path(glb_path))
    print_validation_report(result)


@main.command(name="list-presets")
def list_presets_cmd():
    """List available FLAME shape presets."""
    from create_avatar.phase2_reconstruct.parametric_presets import PRESETS
    click.echo("Available FLAME shape presets:\n")
    for name, preset in PRESETS.items():
        click.echo(f"  {name:20s} - {preset.description} (skin: {preset.skin_tone})")


if __name__ == "__main__":
    main()
