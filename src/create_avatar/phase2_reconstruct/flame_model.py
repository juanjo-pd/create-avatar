from __future__ import annotations

"""FLAME 2023 Open parametric head model wrapper.

Generates 3D face meshes from shape, expression, and pose parameters.
Includes MediaPipe landmark embedding for direct landmark projection.

FLAME 2023 Open (CC-BY-4.0): 5023 vertices, 9976 faces
- shapedirs: (5023, 3, 400) = 300 identity + 100 expression components
- 5 joints: neck, head, jaw, left_eye, right_eye
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from create_avatar.config import config
from create_avatar.utils.device import cpu_map_location


# FLAME 2023 Open component counts
NUM_SHAPE_PARAMS = 300
NUM_EXPRESSION_PARAMS = 100


@dataclass
class FlameOutput:
    """Output from FLAME model forward pass."""

    vertices: np.ndarray  # (5023, 3) vertex positions
    faces: np.ndarray  # (9976, 3) face indices
    landmarks: np.ndarray  # (5, 3) joint landmarks OR (105, 3) MediaPipe landmarks
    shape_params: np.ndarray  # identity shape parameters
    expression_params: np.ndarray  # expression parameters
    pose_params: np.ndarray  # (6,) pose parameters


class FlameModel:
    """Wrapper for the FLAME 2023 Open parametric head model.

    FLAME (Faces Learned with an Articulated Model and Expressions) is a
    statistical head model that separates identity shape, expression,
    and pose into independent parameter spaces.
    """

    NUM_VERTICES = 5023
    NUM_FACES = 9976

    def __init__(
        self,
        flame_model_path: Path = None,
        mediapipe_embedding_path: Path = None,
        masks_path: Path = None,
        device: str = None,
        num_shape_params: int = 300,
        num_expression_params: int = 100,
    ):
        if flame_model_path is None:
            flame_model_path = config.flame_dir / config.flame_model_path

        if not flame_model_path.exists():
            raise FileNotFoundError(
                f"FLAME model not found at {flame_model_path}. "
                "Download FLAME 2023 Open from https://flame.is.tue.mpg.de/"
            )

        self.device = torch.device(device or config.device)
        self.num_shape_params = min(num_shape_params, NUM_SHAPE_PARAMS)
        self.num_expression_params = min(num_expression_params, NUM_EXPRESSION_PARAMS)

        self._load_model(flame_model_path)

        # Load MediaPipe landmark embedding
        if mediapipe_embedding_path is None:
            mediapipe_embedding_path = config.flame_dir / "mediapipe_landmark_embedding.npz"
        self.mp_embedding = None
        if mediapipe_embedding_path.exists():
            self._load_mediapipe_embedding(mediapipe_embedding_path)

        # Load masks
        if masks_path is None:
            masks_path = config.flame_dir / "FLAME_masks.pkl"
        self.masks = None
        if masks_path.exists():
            self._load_masks(masks_path)

    def _load_model(self, model_path: Path):
        """Load FLAME model parameters from pickle file."""
        import pickle

        with open(model_path, "rb") as f:
            flame_data = pickle.load(f, encoding="latin1")

        # Mean template vertices (5023, 3)
        self.v_template = torch.tensor(
            np.array(flame_data["v_template"]), dtype=torch.float32, device=self.device
        )

        # Shape basis: (5023, 3, 400) = first 300 identity + last 100 expression
        shapedirs = np.array(flame_data["shapedirs"])
        total_dims = shapedirs.shape[2]

        # FLAME 2023: 300 shape + 100 expression = 400 total
        self.shapedirs = torch.tensor(
            shapedirs[:, :, :self.num_shape_params],
            dtype=torch.float32, device=self.device,
        )
        self.expressiondirs = torch.tensor(
            shapedirs[:, :, NUM_SHAPE_PARAMS:NUM_SHAPE_PARAMS + self.num_expression_params],
            dtype=torch.float32, device=self.device,
        )

        # Joint regressor (5, 5023) sparse
        j_regressor = flame_data["J_regressor"]
        if hasattr(j_regressor, "toarray"):
            j_regressor = j_regressor.toarray()
        self.j_regressor = torch.tensor(
            np.array(j_regressor), dtype=torch.float32, device=self.device
        )

        # Skinning weights (5023, 5)
        self.weights = torch.tensor(
            np.array(flame_data["weights"]), dtype=torch.float32, device=self.device
        )

        # Kinematic tree (2, 5)
        self.kintree_table = flame_data["kintree_table"]

        # Pose blend shapes (5023, 3, 36)
        if "posedirs" in flame_data:
            self.posedirs = torch.tensor(
                np.array(flame_data["posedirs"]), dtype=torch.float32, device=self.device
            )
        else:
            self.posedirs = None

        # Face topology (9976, 3)
        self.faces = np.array(flame_data["f"], dtype=np.int64)

    def _load_mediapipe_embedding(self, path: Path):
        """Load MediaPipe landmark embedding.

        Contains mapping from 105 MediaPipe landmarks to FLAME mesh:
        - lmk_face_idx: (105,) triangle indices on FLAME mesh
        - lmk_b_coords: (105, 3) barycentric coordinates within each triangle
        - landmark_indices: (105,) which MediaPipe landmark indices these correspond to
        """
        data = np.load(path, allow_pickle=True)
        self.mp_embedding = {
            "lmk_face_idx": data["lmk_face_idx"],  # (105,) face/triangle indices
            "lmk_b_coords": data["lmk_b_coords"],  # (105, 3) barycentric coords
            "landmark_indices": data["landmark_indices"],  # (105,) MediaPipe indices
        }

    def _load_masks(self, path: Path):
        """Load FLAME vertex masks for mesh regions."""
        import pickle
        with open(path, "rb") as f:
            self.masks = pickle.load(f, encoding="latin1")

    def get_mediapipe_landmarks(self, vertices: np.ndarray) -> np.ndarray:
        """Compute 3D positions of MediaPipe landmarks on FLAME mesh.

        Uses barycentric interpolation on FLAME triangles.

        Args:
            vertices: (5023, 3) FLAME mesh vertex positions.

        Returns:
            (105, 3) 3D landmark positions.
        """
        if self.mp_embedding is None:
            raise RuntimeError("MediaPipe embedding not loaded")

        face_idx = self.mp_embedding["lmk_face_idx"]
        b_coords = self.mp_embedding["lmk_b_coords"]

        # Get triangle vertices for each landmark
        tri_verts = vertices[self.faces[face_idx]]  # (105, 3, 3)

        # Barycentric interpolation
        landmarks = np.einsum("lbc,lb->lc", tri_verts, b_coords)

        return landmarks

    def generate(
        self,
        shape_params: np.ndarray = None,
        expression_params: np.ndarray = None,
        pose_params: np.ndarray = None,
    ) -> FlameOutput:
        """Generate a 3D face mesh from FLAME parameters.

        Args:
            shape_params: Identity shape coefficients. None = mean face.
            expression_params: Expression coefficients. None = neutral.
            pose_params: (6,) pose [neck_rot(3), jaw_rot(3)]. None = frontal.

        Returns:
            FlameOutput with vertices, faces, landmarks, and parameters.
        """
        if shape_params is None:
            shape_params = np.zeros(self.num_shape_params, dtype=np.float32)
        if expression_params is None:
            expression_params = np.zeros(self.num_expression_params, dtype=np.float32)
        if pose_params is None:
            pose_params = np.zeros(6, dtype=np.float32)

        # Ensure correct sizes
        shape_params = np.pad(shape_params, (0, max(0, self.num_shape_params - len(shape_params))))[:self.num_shape_params]
        expression_params = np.pad(expression_params, (0, max(0, self.num_expression_params - len(expression_params))))[:self.num_expression_params]

        shape_t = torch.tensor(shape_params, dtype=torch.float32, device=self.device)
        expr_t = torch.tensor(expression_params, dtype=torch.float32, device=self.device)

        # Apply deformations: v = v_template + shapedirs @ shape + exprdirs @ expr
        shape_offsets = torch.einsum("vcs,s->vc", self.shapedirs, shape_t)
        expr_offsets = torch.einsum("vcs,s->vc", self.expressiondirs, expr_t)

        vertices = self.v_template + shape_offsets + expr_offsets
        vertices_np = vertices.detach().cpu().numpy()

        # Compute landmarks
        if self.mp_embedding is not None:
            landmarks_np = self.get_mediapipe_landmarks(vertices_np)
        else:
            landmarks_np = torch.matmul(self.j_regressor, vertices).detach().cpu().numpy()

        return FlameOutput(
            vertices=vertices_np,
            faces=self.faces,
            landmarks=landmarks_np,
            shape_params=shape_params.astype(np.float32),
            expression_params=expression_params.astype(np.float32),
            pose_params=pose_params.astype(np.float32),
        )

    def get_mean_face(self) -> FlameOutput:
        """Generate the mean (average) face with neutral expression."""
        return self.generate()

    def get_neck_boundary_vertices(self) -> np.ndarray:
        """Get vertex indices at the neck boundary (for bust merge).

        Returns:
            Array of vertex indices forming the neck boundary.
        """
        if self.masks is not None and "boundary" in self.masks:
            return np.array(self.masks["boundary"])
        if self.masks is not None and "neck" in self.masks:
            return np.array(self.masks["neck"])
        raise RuntimeError("FLAME masks not loaded")

    def get_face_region_vertices(self) -> np.ndarray:
        """Get vertex indices of the face region only."""
        if self.masks is not None and "face" in self.masks:
            return np.array(self.masks["face"])
        raise RuntimeError("FLAME masks not loaded")
