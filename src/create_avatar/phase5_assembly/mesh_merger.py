from __future__ import annotations

"""Orchestrator for Blender headless avatar assembly.

Calls the Blender assembly script as a subprocess.
"""

import subprocess
from pathlib import Path


def assemble_avatar(
    head_neutral_obj: Path,
    blendshapes_dir: Path,
    texture_png: Path,
    output_glb: Path,
    output_blend: Path = None,
    blender_path: str = "blender",
) -> Path:
    """Assemble avatar from components using Blender headless.

    Args:
        head_neutral_obj: Path to neutral head mesh OBJ.
        blendshapes_dir: Directory with blendshape OBJ files.
        texture_png: Path to UV texture PNG.
        output_glb: Output GLB file path.
        output_blend: Optional output .blend file path.
        blender_path: Path to Blender executable.

    Returns:
        Path to the exported GLB file.

    Raises:
        FileNotFoundError: If Blender or input files are missing.
        subprocess.CalledProcessError: If Blender script fails.
    """
    script_path = Path(__file__).parent.parent.parent.parent / "scripts" / "blender_assembly.py"

    if not script_path.exists():
        raise FileNotFoundError(f"Blender assembly script not found: {script_path}")

    cmd = [
        blender_path, "-b", "--python", str(script_path),
        "--",
        "--head-obj", str(head_neutral_obj),
        "--blendshapes-dir", str(blendshapes_dir),
        "--texture-png", str(texture_png),
        "--output-glb", str(output_glb),
    ]

    if output_blend:
        cmd.extend(["--output-blend", str(output_blend)])

    output_glb.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        print(f"Blender stderr:\n{result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

    print(result.stdout)
    return output_glb
