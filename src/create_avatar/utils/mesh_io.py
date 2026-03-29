from __future__ import annotations

"""Mesh I/O using DECA head_template.obj as the topology reference.

The key insight: DECA's head_template.obj has correct face ordering + UV mapping.
We NEVER change the faces or UVs — only replace vertex positions.
"""

from pathlib import Path
import numpy as np

from create_avatar.config import config

# Cached template lines
_template_lines = None


def _get_template_lines():
    """Load and cache DECA head_template.obj lines."""
    global _template_lines
    if _template_lines is not None:
        return _template_lines

    path = config.vendor_dir / "deca" / "data" / "head_template.obj"
    if not path.exists():
        return None

    with open(path) as f:
        _template_lines = f.readlines()
    return _template_lines


def save_flame_obj(
    path: Path,
    vertices: np.ndarray,
    extra_vertices: np.ndarray = None,
    extra_faces: np.ndarray = None,
) -> None:
    """Save FLAME mesh as OBJ using DECA's template for faces + UVs.

    Replaces only the vertex positions (v lines) in head_template.obj,
    keeping all vt, f, and other lines exactly as-is. This ensures
    the UV mapping is always correct.

    Args:
        path: Output OBJ path.
        vertices: (5023, 3) FLAME vertex positions.
        extra_vertices: Optional additional vertices (bust extension).
        extra_faces: Optional additional faces (bust, using vertex indices >= 5023).
    """
    template = _get_template_lines()
    if template is None:
        # Fallback: write plain OBJ without UVs
        _save_plain_obj(path, vertices, extra_faces)
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    v_idx = 0  # Counter for vertex lines in template

    with open(path, "w") as f:
        for line in template:
            if line.startswith("v ") and not line.startswith("vt") and not line.startswith("vn"):
                # Replace vertex position with our fitted vertex
                if v_idx < len(vertices):
                    v = vertices[v_idx]
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                else:
                    f.write(line)  # Keep original if we run out
                v_idx += 1
            else:
                # Keep vt, f, vn, comments, etc. exactly as-is
                f.write(line)

        # Append bust extension if provided
        if extra_vertices is not None and len(extra_vertices) > 0:
            # Get the last UV index used in template
            last_vt = sum(1 for l in template if l.startswith("vt "))
            bust_vt_idx = last_vt + 1  # 1-indexed

            # Write bust vertices
            for v in extra_vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Write a neutral UV for bust
            f.write("vt 0.5 0.02\n")

            # Write bust faces
            if extra_faces is not None:
                for face in extra_faces:
                    v0, v1, v2 = face + 1  # 1-indexed
                    f.write(f"f {v0}/{bust_vt_idx} {v1}/{bust_vt_idx} {v2}/{bust_vt_idx}\n")


def _save_plain_obj(path: Path, vertices: np.ndarray, faces: np.ndarray = None):
    """Fallback: save plain OBJ without UVs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if faces is not None:
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


# Keep old name as alias for compatibility
def save_vertices_as_obj(path, vertices, faces, include_uvs=True):
    """Compatibility wrapper — uses save_flame_obj when possible."""
    if include_uvs and len(vertices) >= 5023:
        extra_v = vertices[5023:] if len(vertices) > 5023 else None
        # Extract bust faces (faces with any vertex >= 5023)
        extra_f = None
        if faces is not None and len(faces) > 9976:
            extra_f = faces[9976:]
        save_flame_obj(path, vertices[:5023], extra_v, extra_f)
    else:
        _save_plain_obj(path, vertices, faces)


def load_mesh(path: Path):
    """Load a mesh from OBJ file."""
    import trimesh
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")
    mesh = trimesh.load(path, process=False, force="mesh")
    return mesh
