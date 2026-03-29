from __future__ import annotations

"""High-resolution UV texture projection using DECA's UV layout directly.

Rasterizes UV triangles at any target resolution (2048x2048+).
For each UV pixel, computes barycentric coords within the UV triangle,
uses those to interpolate the 3D vertex positions, projects to 2D
photo coords using the fitter's camera, and samples the photo.

No dependency on texture_data_256.npy — uses vt/ft from head_template.obj.
"""

from pathlib import Path

import cv2
import numpy as np

from create_avatar.config import config


def project_photo_texture(
    photo: np.ndarray,
    vertices_3d: np.ndarray,
    camera_scale: float,
    camera_tx: float,
    camera_ty: float,
    landmarks_2d: np.ndarray = None,
    texture_size: int = 2048,
) -> np.ndarray:
    """Project photo onto FLAME UV texture at high resolution.

    Uses the OBJ's own UV triangles (vt + ft) for rasterization,
    bypassing DECA's precomputed 256x256 rasterizer entirely.
    """
    # Load UV mapping from head_template
    uv_data = np.load(config.data_dir / "flame" / "deca_uv_mapping.npz")
    geo_faces = uv_data["geo_faces"]  # (9976, 3) vertex indices
    uv_faces = uv_data["uv_faces"]   # (9976, 3) UV coord indices
    uv_coords = uv_data["uv_coords"] # (5118, 2) UV positions in [0, 1]

    ph, pw = photo.shape[:2]

    # Project 3D vertices to 2D photo pixel coords
    px = (vertices_3d[:, 0] * camera_scale + camera_tx) * pw
    py = (-vertices_3d[:, 1] * camera_scale + camera_ty) * ph

    # Face mask to filter background
    face_mask = _create_face_mask(ph, pw, landmarks_2d)

    # Load mean texture as base
    mean_path = config.vendor_dir / "deca" / "data" / "mean_texture.jpg"
    if mean_path.exists():
        mean_tex = cv2.cvtColor(cv2.imread(str(mean_path)), cv2.COLOR_BGR2RGB)
        base = cv2.resize(mean_tex, (texture_size, texture_size))
    else:
        base = np.full((texture_size, texture_size, 3), 180, dtype=np.uint8)

    texture = base.copy()
    blend_mask = np.zeros((texture_size, texture_size), dtype=np.float32)

    # Rasterize each UV triangle
    for fi in range(len(geo_faces)):
        gv = geo_faces[fi]   # 3 vertex indices
        uvi = uv_faces[fi]   # 3 UV coord indices

        # UV coords for this triangle (in [0, 1])
        uv0 = uv_coords[uvi[0]]
        uv1 = uv_coords[uvi[1]]
        uv2 = uv_coords[uvi[2]]

        # Convert to pixel coords in texture image
        # UV: (0,0) = bottom-left, (1,1) = top-right
        # Image: (0,0) = top-left, so flip Y
        t0 = np.array([uv0[0] * texture_size, (1 - uv0[1]) * texture_size])
        t1 = np.array([uv1[0] * texture_size, (1 - uv1[1]) * texture_size])
        t2 = np.array([uv2[0] * texture_size, (1 - uv2[1]) * texture_size])

        # Bounding box
        tmin = np.floor(np.minimum(t0, np.minimum(t1, t2))).astype(int)
        tmax = np.ceil(np.maximum(t0, np.maximum(t1, t2))).astype(int)
        tmin = np.clip(tmin, 0, texture_size - 1)
        tmax = np.clip(tmax, 0, texture_size - 1)

        if tmax[0] <= tmin[0] or tmax[1] <= tmin[1]:
            continue

        # 3D projected positions for this triangle's vertices
        p0 = np.array([px[gv[0]], py[gv[0]]])
        p1 = np.array([px[gv[1]], py[gv[1]]])
        p2 = np.array([px[gv[2]], py[gv[2]]])

        # Precompute barycentric denominator
        denom = (t1[1] - t2[1]) * (t0[0] - t2[0]) + (t2[0] - t1[0]) * (t0[1] - t2[1])
        if abs(denom) < 1e-10:
            continue

        # Rasterize pixels in bounding box
        for ty in range(tmin[1], tmax[1] + 1):
            for tx in range(tmin[0], tmax[0] + 1):
                # Barycentric coords
                w0 = ((t1[1] - t2[1]) * (tx - t2[0]) + (t2[0] - t1[0]) * (ty - t2[1])) / denom
                w1 = ((t2[1] - t0[1]) * (tx - t2[0]) + (t0[0] - t2[0]) * (ty - t2[1])) / denom
                w2 = 1 - w0 - w1

                if w0 < -0.001 or w1 < -0.001 or w2 < -0.001:
                    continue

                # Interpolate 2D photo position
                sx = int(w0 * p0[0] + w1 * p1[0] + w2 * p2[0])
                sy = int(w0 * p0[1] + w1 * p1[1] + w2 * p2[1])

                if 0 <= sx < pw and 0 <= sy < ph and face_mask[sy, sx] > 128:
                    texture[ty, tx] = photo[sy, sx]
                    blend_mask[ty, tx] = 1.0

    # Smooth blend
    blend_smooth = cv2.GaussianBlur(blend_mask, (15, 15), 4)
    m = blend_smooth[:, :, np.newaxis]
    texture = (texture.astype(float) * m + base.astype(float) * (1 - m))
    texture = np.clip(texture, 0, 255).astype(np.uint8)

    return texture


def _create_face_mask(ph: int, pw: int, landmarks_2d: np.ndarray) -> np.ndarray:
    """Create face region mask from MediaPipe landmarks."""
    mask = np.zeros((ph, pw), dtype=np.uint8)
    if landmarks_2d is None or len(landmarks_2d) < 100:
        mask[:] = 255
        return mask

    lm = landmarks_2d.copy()
    if lm.max() <= 1.5:
        lm[:, 0] *= pw
        lm[:, 1] *= ph

    oval = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
    ]
    oval = [i for i in oval if i < len(lm)]
    if len(oval) < 3:
        mask[:] = 255
        return mask

    points = lm[oval].astype(np.int32)
    cv2.fillPoly(mask, [points], 255)
    kernel = np.ones((25, 25), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    return mask
