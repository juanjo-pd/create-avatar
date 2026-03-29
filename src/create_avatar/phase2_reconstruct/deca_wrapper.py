from __future__ import annotations

"""DECA wrapper with CPU/MPS device patching.

Wraps the DECA face reconstruction model to run on CPU or MPS
instead of the default CUDA. Handles loading pretrained models
and patching device references.

DECA must be cloned into vendor/deca/ and pretrained models downloaded.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from create_avatar.config import config
from create_avatar.utils.device import cpu_map_location


@dataclass
class ReconstructionResult:
    """Output from DECA face reconstruction."""

    shape_params: np.ndarray  # (100,) FLAME shape parameters
    expression_params: np.ndarray  # (50,) expression parameters
    pose_params: np.ndarray  # (6,) pose (neck + jaw rotation)
    vertices: np.ndarray  # (5023, 3) reconstructed face vertices
    faces: np.ndarray  # (9976, 3) face indices
    landmarks_3d: np.ndarray  # (68, 3) 3D facial landmarks
    albedo_texture: np.ndarray  # (256, 256, 3) UV albedo texture
    uv_coords: np.ndarray  # (5023, 2) UV coordinates


class DECAReconstructor:
    """DECA face reconstruction with CPU/MPS support.

    Patches DECA's CUDA assumptions to run on CPU.
    """

    def __init__(
        self,
        deca_dir: Path = None,
        device: str = None,
    ):
        """Initialize DECA reconstructor.

        Args:
            deca_dir: Path to cloned DECA repository. Defaults to vendor/deca.
            device: Compute device ('cpu' or 'mps'). Defaults to config.device.

        Raises:
            FileNotFoundError: If DECA directory or pretrained models are missing.
        """
        if deca_dir is None:
            deca_dir = config.vendor_dir / "deca"

        self.deca_dir = Path(deca_dir)
        self.device = device or config.device
        self.deca = None

        if not self.deca_dir.exists():
            raise FileNotFoundError(
                f"DECA not found at {self.deca_dir}. "
                "Clone it with: git clone https://github.com/yfeng95/DECA vendor/deca"
            )

    def _load(self):
        """Lazy-load DECA with CPU patching."""
        import sys

        # Add DECA to Python path
        deca_path = str(self.deca_dir)
        if deca_path not in sys.path:
            sys.path.insert(0, deca_path)

        # Patch torch.load to force CPU
        original_torch_load = torch.load

        def patched_load(*args, **kwargs):
            kwargs["map_location"] = cpu_map_location
            return original_torch_load(*args, **kwargs)

        torch.load = patched_load

        try:
            from decalib.deca import DECA
            from decalib.utils.config import cfg as deca_cfg

            # Configure for CPU
            deca_cfg.model.use_tex = True
            deca_cfg.rasterizer_type = "pytorch3d"
            deca_cfg.model.topology_path = str(self.deca_dir / "data" / "head_template.obj")
            deca_cfg.model.flame_model_path = str(config.flame_dir / "generic_model.pkl")

            self.deca = DECA(config=deca_cfg, device=self.device)

        finally:
            torch.load = original_torch_load

    def reconstruct(self, image: np.ndarray) -> ReconstructionResult:
        """Reconstruct 3D face from an aligned image.

        Args:
            image: (H, W, 3) aligned face image (RGB, values 0-255).

        Returns:
            ReconstructionResult with FLAME parameters and mesh.
        """
        if self.deca is None:
            self._load()

        # Preprocess image for DECA
        img_tensor = torch.tensor(
            image.transpose(2, 0, 1), dtype=torch.float32
        ).unsqueeze(0) / 255.0

        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            codedict = self.deca.encode(img_tensor)
            opdict = self.deca.decode(codedict)

        # Extract results
        shape_params = codedict["shape"].cpu().numpy().squeeze()
        expression_params = codedict["exp"].cpu().numpy().squeeze()
        pose_params = codedict["pose"].cpu().numpy().squeeze()

        vertices = opdict["verts"].cpu().numpy().squeeze()
        landmarks = opdict["landmarks3d"].cpu().numpy().squeeze()

        # Get albedo texture if available
        albedo = np.zeros((256, 256, 3), dtype=np.uint8)
        if "albedo_images" in opdict:
            albedo_tensor = opdict["albedo_images"]
            albedo = (albedo_tensor.cpu().numpy().squeeze().transpose(1, 2, 0) * 255).astype(np.uint8)

        # Get UV coordinates from FLAME topology
        uv_coords = np.zeros((vertices.shape[0], 2), dtype=np.float32)
        if hasattr(self.deca, "uv_face_eye_mask"):
            pass  # TODO: Extract UV coords from DECA's renderer

        faces = self.deca.flame.faces_tensor.cpu().numpy() if hasattr(self.deca, "flame") else np.zeros((0, 3), dtype=np.int64)

        return ReconstructionResult(
            shape_params=shape_params,
            expression_params=expression_params,
            pose_params=pose_params,
            vertices=vertices,
            faces=faces,
            landmarks_3d=landmarks,
            albedo_texture=albedo,
            uv_coords=uv_coords,
        )
