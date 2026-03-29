from __future__ import annotations

"""Correspondence computation between ARKit and FLAME mesh topologies.

Computes per-face nearest-face mapping between source (ARKit 3084 verts)
and target (FLAME 5023 verts) meshes using nearest-centroid matching.
This enables Deformation Transfer between different topologies.
"""

from pathlib import Path

import numpy as np
from scipy.spatial import KDTree
import trimesh


def compute_face_correspondence(
    source_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
) -> dict:
    """Compute per-face correspondence from source to target mesh.

    For each target face, finds the nearest source face by centroid distance.

    Args:
        source_mesh: Source mesh (ARKit neutral, 3084 verts).
        target_mesh: Target mesh (FLAME mean face, 5023 verts).

    Returns:
        Dict with:
        - source_to_target: (Fs,) for each source face, nearest target face index
        - target_to_source: (Ft,) for each target face, nearest source face index
    """
    # Compute face centroids
    source_centroids = source_mesh.vertices[source_mesh.faces].mean(axis=1)  # (Fs, 3)
    target_centroids = target_mesh.vertices[target_mesh.faces].mean(axis=1)  # (Ft, 3)

    # Build KD-trees
    source_tree = KDTree(source_centroids)
    target_tree = KDTree(target_centroids)

    # For each target face, find nearest source face
    _, target_to_source = source_tree.query(target_centroids)

    # For each source face, find nearest target face
    _, source_to_target = target_tree.query(source_centroids)

    return {
        "source_to_target": source_to_target.astype(np.int64),
        "target_to_source": target_to_source.astype(np.int64),
    }


def compute_vertex_correspondence(
    source_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
) -> dict:
    """Compute per-vertex correspondence between meshes.

    For each target vertex, finds the nearest source vertex.

    Args:
        source_mesh: Source mesh.
        target_mesh: Target mesh.

    Returns:
        Dict with:
        - target_to_source_verts: (Vt,) for each target vert, nearest source vert
        - source_to_target_verts: (Vs,) for each source vert, nearest target vert
        - target_to_source_dists: (Vt,) distances
    """
    source_tree = KDTree(source_mesh.vertices)
    target_tree = KDTree(target_mesh.vertices)

    dists_t2s, target_to_source = source_tree.query(target_mesh.vertices)
    dists_s2t, source_to_target = target_tree.query(source_mesh.vertices)

    return {
        "target_to_source_verts": target_to_source.astype(np.int64),
        "source_to_target_verts": source_to_target.astype(np.int64),
        "target_to_source_dists": dists_t2s,
        "source_to_target_dists": dists_s2t,
    }


def compute_and_save_correspondence(
    source_obj: Path,
    target_obj: Path,
    output_path: Path,
) -> dict:
    """Compute and save face + vertex correspondence to NPZ.

    Args:
        source_obj: Path to source mesh OBJ (ARKit neutral).
        target_obj: Path to target mesh OBJ (FLAME mean face).
        output_path: Path to save correspondence NPZ.

    Returns:
        Combined correspondence dict.
    """
    source = trimesh.load(source_obj, process=False, force="mesh")
    target = trimesh.load(target_obj, process=False, force="mesh")

    print(f"Source: {source.vertices.shape[0]} verts, {source.faces.shape[0]} faces")
    print(f"Target: {target.vertices.shape[0]} verts, {target.faces.shape[0]} faces")

    face_corr = compute_face_correspondence(source, target)
    vert_corr = compute_vertex_correspondence(source, target)

    result = {**face_corr, **vert_corr}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **result)
    print(f"Saved correspondence: {output_path}")

    return result
