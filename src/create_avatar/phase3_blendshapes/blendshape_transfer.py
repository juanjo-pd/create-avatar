from __future__ import annotations

"""Transfer ARKit blendshapes from reference mesh to FLAME mesh.

Uses vertex correspondence + RBF interpolation to transfer blendshape
deformations from the ARKit reference topology (3084 verts) to the
FLAME topology (5023 verts).

Two methods:
1. Nearest-vertex delta transfer (fast, good quality for similar regions)
2. RBF-interpolated transfer (slower, smoother, better for mismatched regions)
"""

from pathlib import Path

import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import RBFInterpolator
import trimesh

from create_avatar.phase3_blendshapes.arkit_names import ARKIT_BLENDSHAPE_NAMES


class BlendshapeTransfer:
    """Transfer blendshapes between mesh topologies.

    Given a source mesh (ARKit reference) with known blendshapes and a
    target mesh (FLAME), transfers the deformations to the target topology.
    """

    def __init__(
        self,
        source_neutral_path: Path,
        source_blendshapes_dir: Path,
        target_neutral_vertices: np.ndarray,
        target_faces: np.ndarray,
        method: str = "rbf",
    ):
        """Initialize blendshape transfer.

        Args:
            source_neutral_path: Path to ARKit Neutral.obj.
            source_blendshapes_dir: Directory with 52 ARKit blendshape OBJs.
            target_neutral_vertices: (5023, 3) FLAME neutral vertices.
            target_faces: (9976, 3) FLAME face indices.
            method: Transfer method - "nearest" or "rbf".
        """
        self.method = method

        # Load source neutral
        src_mesh = trimesh.load(source_neutral_path, process=False, force="mesh")
        self.source_neutral = src_mesh.vertices  # (3084, 3)
        self.source_faces = src_mesh.faces

        self.target_neutral = target_neutral_vertices  # (5023, 3)
        self.target_faces = target_faces

        # Align source to target (scale + translate to match)
        self._align_meshes()

        # Build correspondence
        self.source_tree = KDTree(self.source_neutral_aligned)

        # Load source blendshapes
        self.source_blendshapes = {}
        self.source_deltas = {}
        self._load_source_blendshapes(source_blendshapes_dir)

    def _align_meshes(self):
        """Align source mesh to target mesh (rigid alignment by bounding box)."""
        # Compute bounding boxes
        src_min = self.source_neutral.min(axis=0)
        src_max = self.source_neutral.max(axis=0)
        src_center = (src_min + src_max) / 2
        src_scale = (src_max - src_min).max()

        tgt_min = self.target_neutral.min(axis=0)
        tgt_max = self.target_neutral.max(axis=0)
        tgt_center = (tgt_min + tgt_max) / 2
        tgt_scale = (tgt_max - tgt_min).max()

        # Normalize source to match target scale and position
        scale_factor = tgt_scale / src_scale if src_scale > 0 else 1.0
        self.source_neutral_aligned = (self.source_neutral - src_center) * scale_factor + tgt_center
        self.scale_factor = scale_factor
        self.src_center = src_center
        self.tgt_center = tgt_center

    def _load_source_blendshapes(self, blendshapes_dir: Path):
        """Load and compute deltas for all source blendshapes."""
        blendshapes_dir = Path(blendshapes_dir)

        for name in ARKIT_BLENDSHAPE_NAMES:
            obj_path = blendshapes_dir / f"{name}.obj"
            if not obj_path.exists():
                continue

            mesh = trimesh.load(obj_path, process=False, force="mesh")
            aligned_verts = (mesh.vertices - self.src_center) * self.scale_factor + self.tgt_center

            self.source_blendshapes[name] = aligned_verts
            self.source_deltas[name] = aligned_verts - self.source_neutral_aligned

    def transfer_nearest(self) -> dict:
        """Transfer blendshapes using nearest-vertex correspondence.

        For each target vertex, finds the nearest source vertex and
        copies its blendshape delta.

        Returns:
            Dict mapping blendshape name to (5023, 3) target deltas.
        """
        # Find nearest source vertex for each target vertex
        distances, indices = self.source_tree.query(self.target_neutral)

        # Distance-based weight: vertices far from source get reduced deltas
        max_dist = np.percentile(distances, 95)
        weights = np.clip(1.0 - distances / (max_dist * 2), 0, 1)
        weights = weights[:, np.newaxis]  # (5023, 1)

        target_deltas = {}
        for name, src_delta in self.source_deltas.items():
            # Transfer delta from nearest source vertex, weighted by distance
            target_delta = src_delta[indices] * weights
            target_deltas[name] = target_delta

        return target_deltas

    def transfer_rbf(self, smoothing: float = 0.001) -> dict:
        """Transfer blendshapes using RBF interpolation.

        Fits an RBF function to the source deltas and evaluates it at
        target vertex positions. Produces smoother results.

        Args:
            smoothing: RBF smoothing parameter.

        Returns:
            Dict mapping blendshape name to (5023, 3) target deltas.
        """
        target_deltas = {}

        # Only use source vertices that have non-negligible deltas
        for name, src_delta in self.source_deltas.items():
            delta_magnitude = np.linalg.norm(src_delta, axis=1)
            max_delta = delta_magnitude.max()

            if max_delta < 1e-7:
                # No deformation for this blendshape
                target_deltas[name] = np.zeros_like(self.target_neutral)
                continue

            # Use vertices with any significant deformation as RBF centers
            # Plus some neutral vertices for stability
            active_mask = delta_magnitude > max_delta * 0.01
            n_active = active_mask.sum()

            if n_active < 10:
                # Too few active vertices, use nearest
                distances, indices = self.source_tree.query(self.target_neutral)
                weights = np.clip(1.0 - distances / (distances.max() * 2), 0, 1)[:, np.newaxis]
                target_deltas[name] = src_delta[indices] * weights
                continue

            # Sample some neutral vertices for RBF stability
            neutral_mask = ~active_mask
            n_neutral_sample = min(neutral_mask.sum(), n_active)
            if n_neutral_sample > 0:
                rng = np.random.default_rng(42)
                neutral_indices = np.where(neutral_mask)[0]
                sampled = rng.choice(neutral_indices, size=n_neutral_sample, replace=False)
                combined_mask = active_mask.copy()
                combined_mask[sampled] = True
            else:
                combined_mask = active_mask

            src_points = self.source_neutral_aligned[combined_mask]
            src_values = src_delta[combined_mask]

            # Fit RBF for each coordinate
            try:
                rbf = RBFInterpolator(
                    src_points, src_values,
                    kernel="thin_plate_spline",
                    smoothing=smoothing,
                )
                target_delta = rbf(self.target_neutral)
            except Exception:
                # Fallback to nearest
                distances, indices = self.source_tree.query(self.target_neutral)
                weights = np.clip(1.0 - distances / (distances.max() * 2), 0, 1)[:, np.newaxis]
                target_delta = src_delta[indices] * weights

            target_deltas[name] = target_delta

        return target_deltas

    def transfer(self) -> dict:
        """Transfer all blendshapes using the configured method.

        Returns:
            Dict with:
            - blendshape_deltas: {name: (5023, 3) deltas}
            - blendshape_vertices: {name: (5023, 3) absolute positions}
        """
        if self.method == "rbf":
            print("Transferring blendshapes via RBF interpolation...")
            deltas = self.transfer_rbf()
        else:
            print("Transferring blendshapes via nearest-vertex...")
            deltas = self.transfer_nearest()

        vertices = {}
        for name, delta in deltas.items():
            vertices[name] = self.target_neutral + delta

        print(f"Transferred {len(deltas)} blendshapes to target mesh")

        return {
            "blendshape_deltas": deltas,
            "blendshape_vertices": vertices,
        }
