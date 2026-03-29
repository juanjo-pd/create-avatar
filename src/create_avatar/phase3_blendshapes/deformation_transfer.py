from __future__ import annotations

"""Deformation Transfer for generating ARKit blendshapes on arbitrary face meshes.

Implements the Deformation Transfer algorithm (Sumner & Popović, 2004) to transfer
blendshape deformations from a source mesh (ARKit reference) to a target mesh (FLAME).

This requires:
1. A source neutral mesh + 52 source blendshape meshes (ARKit reference)
2. A correspondence mapping between source and target topologies
3. The target neutral mesh (reconstructed FLAME face)

The correspondence mesh is computed once via Non-Rigid ICP and stored in data/.
"""

from pathlib import Path
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

from create_avatar.phase3_blendshapes.arkit_names import ARKIT_BLENDSHAPE_NAMES
from create_avatar.utils.mesh_io import load_mesh


@dataclass
class DeformationTransferResult:
    """Result of deformation transfer."""

    blendshape_vertices: dict[str, np.ndarray]  # name -> (N, 3) vertices
    blendshape_deltas: dict[str, np.ndarray]  # name -> (N, 3) displacements from neutral
    neutral_vertices: np.ndarray  # (N, 3) target neutral vertices
    faces: np.ndarray  # (F, 3) face indices


class DeformationTransfer:
    """Transfer blendshape deformations from source to target mesh topology.

    The algorithm works by:
    1. Computing per-triangle deformation gradients from source neutral to each blendshape
    2. Solving for target vertex positions that best reproduce those gradients
       on the target mesh topology
    """

    def __init__(
        self,
        source_neutral_path: Path,
        source_blendshapes_dir: Path,
        correspondence_path: Path | None = None,
    ):
        """Initialize with source mesh data.

        Args:
            source_neutral_path: Path to source neutral OBJ (ARKit reference neutral).
            source_blendshapes_dir: Directory containing 52 OBJ files named by ARKit blendshape names.
            correspondence_path: Path to correspondence mapping file (NPZ).
                If None, assumes source and target have compatible topologies.
        """
        self.source_neutral = load_mesh(source_neutral_path)
        self.source_blendshapes_dir = source_blendshapes_dir
        self.correspondence_path = correspondence_path

        # Load correspondence if provided
        self.correspondence = None
        if correspondence_path and correspondence_path.exists():
            data = np.load(correspondence_path)
            self.correspondence = {
                "source_to_target": data["source_to_target"],
                "target_to_source": data["target_to_source"],
            }

        # Load source blendshapes
        self.source_blendshapes: dict[str, np.ndarray] = {}
        self._load_source_blendshapes()

    def _load_source_blendshapes(self):
        """Load the 52 ARKit blendshape meshes from OBJ files."""
        for name in ARKIT_BLENDSHAPE_NAMES:
            obj_path = self.source_blendshapes_dir / f"{name}.obj"
            if obj_path.exists():
                mesh = load_mesh(obj_path)
                self.source_blendshapes[name] = mesh.vertices
            else:
                print(f"Warning: Missing source blendshape: {obj_path}")

    def _compute_triangle_deformation_gradient(
        self,
        source_verts: np.ndarray,
        deformed_verts: np.ndarray,
        faces: np.ndarray,
    ) -> np.ndarray:
        """Compute per-triangle deformation gradients.

        For each triangle, compute the 3x3 matrix T such that
        the deformed edge vectors = T @ source edge vectors.

        Args:
            source_verts: (N, 3) source neutral vertices.
            deformed_verts: (N, 3) deformed (blendshape) vertices.
            faces: (F, 3) triangle indices.

        Returns:
            (F, 3, 3) per-triangle deformation gradients.
        """
        num_faces = len(faces)
        gradients = np.zeros((num_faces, 3, 3))

        for i, face in enumerate(faces):
            v0_src, v1_src, v2_src = source_verts[face]
            v0_def, v1_def, v2_def = deformed_verts[face]

            # Edge vectors
            e1_src = v1_src - v0_src
            e2_src = v2_src - v0_src
            n_src = np.cross(e1_src, e2_src)
            n_src_norm = np.linalg.norm(n_src)
            if n_src_norm > 1e-10:
                n_src = n_src / n_src_norm

            e1_def = v1_def - v0_def
            e2_def = v2_def - v0_def
            n_def = np.cross(e1_def, e2_def)
            n_def_norm = np.linalg.norm(n_def)
            if n_def_norm > 1e-10:
                n_def = n_def / n_def_norm

            # Build source matrix [e1 | e2 | n]^T
            V_src = np.column_stack([e1_src, e2_src, n_src])
            V_def = np.column_stack([e1_def, e2_def, n_def])

            # Deformation gradient: T = V_def @ V_src^{-1}
            try:
                gradients[i] = V_def @ np.linalg.inv(V_src)
            except np.linalg.LinAlgError:
                gradients[i] = np.eye(3)

        return gradients

    def _build_transfer_system(
        self,
        target_neutral_verts: np.ndarray,
        target_faces: np.ndarray,
        deformation_gradients: np.ndarray,
        regularization: float = 0.001,
    ) -> np.ndarray:
        """Solve for target deformed vertices given deformation gradients.

        Builds and solves the linear system that finds vertex positions
        on the target mesh whose per-triangle deformation gradients
        best match the given gradients.

        Args:
            target_neutral_verts: (M, 3) target neutral vertices.
            target_faces: (Ft, 3) target triangle indices.
            deformation_gradients: (Ft, 3, 3) desired per-triangle gradients.
            regularization: Regularization weight for smoothness.

        Returns:
            (M, 3) deformed target vertex positions.
        """
        num_verts = len(target_neutral_verts)
        num_faces = len(target_faces)

        # We solve for each coordinate (x, y, z) independently
        result = np.zeros_like(target_neutral_verts)

        for coord in range(3):
            # Build sparse system Ax = b
            rows = []
            cols = []
            vals = []
            rhs = np.zeros(num_faces * 3 + num_verts)

            for fi, face in enumerate(target_faces):
                v0, v1, v2 = face

                # Source edge vectors for this triangle
                e1 = target_neutral_verts[v1] - target_neutral_verts[v0]
                e2 = target_neutral_verts[v2] - target_neutral_verts[v0]
                n = np.cross(e1, e2)
                n_norm = np.linalg.norm(n)
                if n_norm > 1e-10:
                    n = n / n_norm

                T = deformation_gradients[fi]

                # Target edge1: x[v1] - x[v0] = T @ e1 for the coord dimension
                row_idx = fi * 3
                rows.extend([row_idx, row_idx])
                cols.extend([v0, v1])
                vals.extend([-1.0, 1.0])
                rhs[row_idx] = T[coord] @ e1

                # Target edge2: x[v2] - x[v0] = T @ e2
                row_idx = fi * 3 + 1
                rows.extend([row_idx, row_idx])
                cols.extend([v0, v2])
                vals.extend([-1.0, 1.0])
                rhs[row_idx] = T[coord] @ e2

                # Normal constraint
                row_idx = fi * 3 + 2
                rows.extend([row_idx, row_idx, row_idx])
                cols.extend([v0, v1, v2])
                n_contrib = T[coord] @ n
                vals.extend([-n_contrib, n_contrib * 0.5, n_contrib * 0.5])
                rhs[row_idx] = T[coord] @ n

            # Regularization: x_i ≈ target_neutral[i]
            for vi in range(num_verts):
                row_idx = num_faces * 3 + vi
                rows.append(row_idx)
                cols.append(vi)
                vals.append(regularization)
                rhs[row_idx] = regularization * target_neutral_verts[vi, coord]

            A = sparse.csr_matrix(
                (vals, (rows, cols)),
                shape=(num_faces * 3 + num_verts, num_verts),
            )

            solution = lsqr(A, rhs)[0]
            result[:, coord] = solution

        return result

    def transfer(
        self,
        target_neutral_vertices: np.ndarray,
        target_faces: np.ndarray,
        regularization: float = 0.001,
    ) -> DeformationTransferResult:
        """Transfer all 52 ARKit blendshapes to the target mesh.

        Args:
            target_neutral_vertices: (M, 3) target neutral vertex positions.
            target_faces: (Ft, 3) target face indices.
            regularization: Smoothness regularization weight.

        Returns:
            DeformationTransferResult with blendshape vertices and deltas.
        """
        source_neutral_verts = self.source_neutral.vertices
        source_faces = self.source_neutral.faces

        blendshape_vertices = {}
        blendshape_deltas = {}

        for name in ARKIT_BLENDSHAPE_NAMES:
            if name not in self.source_blendshapes:
                print(f"Skipping {name}: source blendshape not loaded")
                continue

            print(f"Transferring blendshape: {name}")

            # 1. Compute deformation gradients on source mesh
            gradients = self._compute_triangle_deformation_gradient(
                source_neutral_verts,
                self.source_blendshapes[name],
                source_faces,
            )

            # 2. If we have correspondence, remap gradients to target topology
            if self.correspondence is not None:
                target_gradients = self._remap_gradients(
                    gradients, target_faces, self.correspondence
                )
            else:
                target_gradients = gradients[:len(target_faces)]

            # 3. Solve for target deformed vertices
            deformed = self._build_transfer_system(
                target_neutral_vertices,
                target_faces,
                target_gradients,
                regularization,
            )

            blendshape_vertices[name] = deformed
            blendshape_deltas[name] = deformed - target_neutral_vertices

        return DeformationTransferResult(
            blendshape_vertices=blendshape_vertices,
            blendshape_deltas=blendshape_deltas,
            neutral_vertices=target_neutral_vertices,
            faces=target_faces,
        )

    def _remap_gradients(
        self,
        source_gradients: np.ndarray,
        target_faces: np.ndarray,
        correspondence: dict,
    ) -> np.ndarray:
        """Remap deformation gradients from source to target topology using correspondence.

        Uses nearest-face mapping when topologies differ.
        """
        num_target_faces = len(target_faces)
        target_to_source = correspondence["target_to_source"]

        remapped = np.zeros((num_target_faces, 3, 3))
        for i in range(num_target_faces):
            if i < len(target_to_source):
                src_idx = target_to_source[i]
                if src_idx < len(source_gradients):
                    remapped[i] = source_gradients[src_idx]
                else:
                    remapped[i] = np.eye(3)
            else:
                remapped[i] = np.eye(3)

        return remapped
