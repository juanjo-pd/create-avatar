from __future__ import annotations

"""FLAME shape fitting from MediaPipe landmarks.

Optimizes FLAME shape parameters so that the projected MediaPipe landmarks
on the FLAME mesh match the 2D landmarks detected by MediaPipe in the photo.

This eliminates the need for DECA and PyTorch3D entirely.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from create_avatar.phase2_reconstruct.flame_model import FlameModel, FlameOutput


class MediaPipeFLAMEFitter:
    """Fit FLAME shape parameters from MediaPipe 2D landmarks.

    Uses gradient descent to find FLAME shape + expression parameters
    that produce 3D landmarks matching the detected 2D landmarks.
    """

    def __init__(
        self,
        flame_model: FlameModel = None,
        num_shape_params: int = 100,
        num_expression_params: int = 50,
    ):
        """Initialize fitter.

        Args:
            flame_model: Pre-loaded FlameModel instance. Created if None.
            num_shape_params: Number of shape params to optimize (first N PCA dims).
            num_expression_params: Number of expression params to optimize.
        """
        if flame_model is None:
            flame_model = FlameModel()

        self.flame = flame_model
        self.device = self.flame.device
        self.num_shape = num_shape_params
        self.num_expr = num_expression_params

        if self.flame.mp_embedding is None:
            raise RuntimeError(
                "MediaPipe landmark embedding not loaded. "
                "Place mediapipe_landmark_embedding.npz in data/flame/"
            )

        # Pre-compute embedding tensors
        self.lmk_face_idx = torch.tensor(
            self.flame.mp_embedding["lmk_face_idx"], dtype=torch.long, device=self.device
        )
        self.lmk_b_coords = torch.tensor(
            self.flame.mp_embedding["lmk_b_coords"], dtype=torch.float32, device=self.device
        )
        self.mp_indices = self.flame.mp_embedding["landmark_indices"]  # (105,) numpy

        # Faces as tensor for indexing
        self.faces_t = torch.tensor(
            self.flame.faces, dtype=torch.long, device=self.device
        )

    def _compute_landmarks_from_vertices(self, vertices: torch.Tensor) -> torch.Tensor:
        """Compute 105 MediaPipe landmark positions from FLAME vertices.

        Uses barycentric interpolation on FLAME triangles.

        Args:
            vertices: (5023, 3) vertex positions.

        Returns:
            (105, 3) landmark positions.
        """
        # Get triangle vertices for each landmark
        tri_verts = vertices[self.faces_t[self.lmk_face_idx]]  # (105, 3, 3)
        # Barycentric interpolation
        landmarks = torch.einsum("lvc,lv->lc", tri_verts, self.lmk_b_coords)
        return landmarks

    def _project_to_2d(
        self,
        landmarks_3d: torch.Tensor,
        scale: torch.Tensor,
        tx: torch.Tensor,
        ty: torch.Tensor,
    ) -> torch.Tensor:
        """Weak perspective projection (orthographic + scale + translation).

        Args:
            landmarks_3d: (105, 3) 3D landmarks.
            scale: Scalar scale factor.
            tx, ty: 2D translation.

        Returns:
            (105, 2) projected 2D landmarks.
        """
        # FLAME: Y points up. MediaPipe/image: Y points down. Flip Y.
        proj_x = landmarks_3d[:, 0] * scale + tx
        proj_y = -landmarks_3d[:, 1] * scale + ty
        return torch.stack([proj_x, proj_y], dim=1)

    def fit(
        self,
        target_landmarks_2d: np.ndarray,
        target_mp_indices: np.ndarray = None,
        num_iterations: int = 3000,
        lr_shape: float = 0.05,
        lr_camera: float = 0.1,
        shape_reg: float = 0.0001,
        expr_reg: float = 0.001,
        verbose: bool = True,
    ) -> FlameOutput:
        """Fit FLAME parameters to match detected MediaPipe 2D landmarks.

        Args:
            target_landmarks_2d: (N, 2) detected 2D landmarks (normalized 0-1).
                N can be 478 (all MediaPipe) or 105 (pre-filtered to FLAME subset).
            target_mp_indices: (N,) MediaPipe landmark indices corresponding to
                target_landmarks_2d rows. If None and N==478, uses all available.
            num_iterations: Number of optimization steps.
            lr_shape: Learning rate for shape parameters.
            lr_camera: Learning rate for camera parameters.
            shape_reg: L2 regularization weight on shape params.
            expr_reg: L2 regularization weight on expression params.
            verbose: Print progress.

        Returns:
            FlameOutput with optimized parameters and mesh.
        """
        # Match target landmarks to FLAME embedding indices
        if target_landmarks_2d.shape[0] >= 478:
            # Full MediaPipe output: extract only the 105 landmarks we have embedding for
            target_2d = target_landmarks_2d[self.mp_indices]
        elif target_landmarks_2d.shape[0] == 105:
            target_2d = target_landmarks_2d
        else:
            raise ValueError(
                f"Expected 478 (full MediaPipe) or 105 (FLAME subset) landmarks, "
                f"got {target_landmarks_2d.shape[0]}"
            )

        target_t = torch.tensor(target_2d, dtype=torch.float32, device=self.device)

        # Compute initial camera params from mean face landmarks and target
        with torch.no_grad():
            mean_lm3d = self._compute_landmarks_from_vertices(self.flame.v_template)

            # Project mean landmarks the same way as _project_to_2d (with Y flip)
            mean_proj_x = mean_lm3d[:, 0]  # X stays
            mean_proj_y = -mean_lm3d[:, 1]  # Y flipped

            # Estimate initial scale: match bounding box ranges
            src_range_x = mean_proj_x.max() - mean_proj_x.min()
            src_range_y = mean_proj_y.max() - mean_proj_y.min()
            src_range = max(src_range_x, src_range_y)

            dst_range = (target_t.max(dim=0).values - target_t.min(dim=0).values).max()
            init_scale = (dst_range / src_range).item() if src_range > 1e-8 else 5.0

            # Estimate initial translation: match centers (after scale)
            src_cx = mean_proj_x.mean() * init_scale
            src_cy = mean_proj_y.mean() * init_scale
            dst_cx = target_t[:, 0].mean()
            dst_cy = target_t[:, 1].mean()
            init_tx = (dst_cx - src_cx).item()
            init_ty = (dst_cy - src_cy).item()

        # Initialize optimizable parameters
        shape_params = torch.zeros(self.num_shape, device=self.device, requires_grad=True)
        expr_params = torch.zeros(self.num_expr, device=self.device, requires_grad=True)
        scale = torch.tensor(init_scale, device=self.device, requires_grad=True)
        tx = torch.tensor(init_tx, device=self.device, requires_grad=True)
        ty = torch.tensor(init_ty, device=self.device, requires_grad=True)

        if verbose:
            print(f"  Initial camera: scale={init_scale:.2f}, tx={init_tx:.3f}, ty={init_ty:.3f}")

        # Optimizer
        optimizer = torch.optim.Adam([
            {"params": [shape_params], "lr": lr_shape},
            {"params": [expr_params], "lr": lr_shape * 0.5},
            {"params": [scale, tx, ty], "lr": lr_camera},
        ])

        best_loss = float("inf")
        best_shape = None
        best_expr = None

        for i in range(num_iterations):
            optimizer.zero_grad()

            # Pad shape/expr to full size
            full_shape = torch.zeros(self.flame.num_shape_params, device=self.device)
            full_shape[:self.num_shape] = shape_params
            full_expr = torch.zeros(self.flame.num_expression_params, device=self.device)
            full_expr[:self.num_expr] = expr_params

            # Compute vertices
            shape_offsets = torch.einsum("vcs,s->vc", self.flame.shapedirs, full_shape)
            expr_offsets = torch.einsum("vcs,s->vc", self.flame.expressiondirs, full_expr)
            vertices = self.flame.v_template + shape_offsets + expr_offsets

            # Compute 3D landmarks
            landmarks_3d = self._compute_landmarks_from_vertices(vertices)

            # Project to 2D
            landmarks_2d = self._project_to_2d(landmarks_3d, scale, tx, ty)

            # Landmark loss
            lmk_loss = F.mse_loss(landmarks_2d, target_t)

            # Regularization
            reg_loss = shape_reg * torch.sum(shape_params ** 2) + expr_reg * torch.sum(expr_params ** 2)

            loss = lmk_loss + reg_loss

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_shape = shape_params.detach().clone()
                best_expr = expr_params.detach().clone()
                best_scale = scale.detach().clone()
                best_tx = tx.detach().clone()
                best_ty = ty.detach().clone()

            if verbose and (i + 1) % 500 == 0:
                print(f"  Iter {i+1}/{num_iterations}: loss={loss.item():.6f} "
                      f"(lmk={lmk_loss.item():.6f}, reg={reg_loss.item():.6f})")

        # Generate final mesh with best parameters
        final_shape = np.zeros(self.flame.num_shape_params, dtype=np.float32)
        final_shape[:self.num_shape] = best_shape.cpu().numpy()
        final_expr = np.zeros(self.flame.num_expression_params, dtype=np.float32)
        final_expr[:self.num_expr] = best_expr.cpu().numpy()

        result = self.flame.generate(
            shape_params=final_shape,
            expression_params=final_expr,
        )

        # Store camera parameters on the result for texture projection
        result.camera_scale = best_scale.item()
        result.camera_tx = best_tx.item()
        result.camera_ty = best_ty.item()

        if verbose:
            print(f"  Final loss: {best_loss:.6f}")
            print(f"  Shape params magnitude: {np.linalg.norm(final_shape[:self.num_shape]):.3f}")
            print(f"  Camera: scale={result.camera_scale:.3f}, tx={result.camera_tx:.3f}, ty={result.camera_ty:.3f}")

        return result
