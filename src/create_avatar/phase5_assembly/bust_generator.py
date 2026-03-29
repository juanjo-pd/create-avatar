from __future__ import annotations

"""Bust (neck + shoulders) mesh generator.

Generates a smooth neck and shoulders by extruding the FLAME neck
boundary edge loop downward and outward.
"""

import numpy as np
from pathlib import Path

from create_avatar.config import config
from create_avatar.utils.mesh_io import save_vertices_as_obj


def _find_neck_loop(faces: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Find the neck boundary edge loop from FLAME mesh topology.

    Returns ordered vertex indices forming the neck opening.
    """
    # Try cached loop first
    cache_path = config.data_dir / "flame" / "neck_loop.npy"
    if cache_path.exists():
        return np.load(cache_path)

    # Compute from topology
    from collections import Counter

    edge_count = Counter()
    for face in faces:
        for i in range(3):
            edge = tuple(sorted([int(face[i]), int(face[(i + 1) % 3])]))
            edge_count[edge] += 1

    boundary_edges = [e for e, c in edge_count.items() if c == 1]

    # Build adjacency
    adj = {}
    for a, b in boundary_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    # Find connected loops
    visited = set()
    loops = []
    for start in adj:
        if start in visited:
            continue
        loop = []
        current = start
        while current not in visited:
            visited.add(current)
            loop.append(current)
            neighbors = [n for n in adj[current] if n not in visited]
            if neighbors:
                current = neighbors[0]
            else:
                break
        loops.append(np.array(loop))

    # Neck loop = the one with lowest average Y
    neck_idx = min(range(len(loops)), key=lambda i: vertices[loops[i]][:, 1].mean())
    return loops[neck_idx]


def generate_bust_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    shoulder_width: float = 1.8,
    bust_depth: float = 0.06,
    num_rings: int = 6,
) -> dict:
    """Generate bust mesh extending from FLAME neck boundary.

    Args:
        vertices: (5023, 3) FLAME vertices.
        faces: (9976, 3) FLAME faces.
        shoulder_width: Width multiplier for shoulders vs neck.
        bust_depth: Vertical extent of bust.
        num_rings: Number of extrusion rings.

    Returns:
        Dict with combined vertices, faces, and metadata.
    """
    neck_loop = _find_neck_loop(faces, vertices)
    n_loop = len(neck_loop)
    head_vcount = len(vertices)

    neck_verts = vertices[neck_loop]
    neck_center = neck_verts.mean(axis=0)

    # Determine winding direction: check which side faces point
    # Sample a head face adjacent to the neck to determine outward direction
    neck_set = set(neck_loop)
    outward_check = 0.0
    for face in faces:
        face_verts_in_neck = [v for v in face if v in neck_set]
        if len(face_verts_in_neck) >= 1:
            v0, v1, v2 = vertices[face]
            normal = np.cross(v1 - v0, v2 - v0)
            # Outward normal should point away from center
            to_center = neck_center - (v0 + v1 + v2) / 3
            outward_check += np.dot(normal, to_center)
            if abs(outward_check) > 0.001:
                break

    reverse_winding = outward_check > 0

    # Generate rings
    ring_vertices = []
    for ring_i in range(1, num_rings + 1):
        t = ring_i / num_rings
        t_smooth = t * t * (3 - 2 * t)  # Smoothstep

        y_drop = -bust_depth * t
        scale = 1.0 + (shoulder_width - 1.0) * t_smooth
        z_flat = 1.0 - 0.2 * t_smooth  # Flatten front-back slightly

        ring = []
        for v in neck_verts:
            offset = v - neck_center
            new_v = neck_center.copy()
            new_v[0] += offset[0] * scale
            new_v[1] += offset[1] + y_drop
            new_v[2] += offset[2] * z_flat
            ring.append(new_v)
        ring_vertices.append(np.array(ring))

    # Combine vertices
    bust_verts = np.vstack(ring_vertices)
    all_verts = np.vstack([vertices, bust_verts])

    # Generate faces
    new_faces = []

    def add_quad(a, b, c, d):
        if reverse_winding:
            new_faces.append([a, c, b])
            new_faces.append([b, c, d])
        else:
            new_faces.append([a, b, c])
            new_faces.append([c, b, d])

    # Neck to first ring
    for i in range(n_loop):
        i_next = (i + 1) % n_loop
        add_quad(
            neck_loop[i], neck_loop[i_next],
            head_vcount + i, head_vcount + i_next,
        )

    # Ring to ring
    for ring_i in range(num_rings - 1):
        base = head_vcount + ring_i * n_loop
        next_base = head_vcount + (ring_i + 1) * n_loop
        for i in range(n_loop):
            i_next = (i + 1) % n_loop
            add_quad(
                base + i, base + i_next,
                next_base + i, next_base + i_next,
            )

    # Close bottom
    last_base = head_vcount + (num_rings - 1) * n_loop
    bottom_center = all_verts[last_base:last_base + n_loop].mean(axis=0)
    bc_idx = len(all_verts)
    all_verts = np.vstack([all_verts, bottom_center.reshape(1, 3)])

    for i in range(n_loop):
        i_next = (i + 1) % n_loop
        if reverse_winding:
            new_faces.append([last_base + i_next, last_base + i, bc_idx])
        else:
            new_faces.append([last_base + i, last_base + i_next, bc_idx])

    all_faces = np.vstack([faces, np.array(new_faces, dtype=np.int64)])

    return {
        "vertices": all_verts,
        "faces": all_faces,
        "head_vertex_count": head_vcount,
        "bust_vertex_count": len(bust_verts) + 1,
        "neck_loop": neck_loop,
    }
