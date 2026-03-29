"""Add ARKit blendshapes + visemes to an Avaturn T1 GLB.

Blender headless script that:
1. Imports the Avaturn GLB
2. Loads ARKit reference blendshapes
3. Aligns ARKit reference to Avaturn face geometry
4. Transfers blendshapes via nearest-vertex correspondence
5. Adds them as shape keys on the body mesh
6. Generates visemes as weighted combinations
7. Exports as GLB with morph targets

Usage:
    blender -b --python scripts/add_blendshapes_to_avaturn.py -- \
        --input capturas/juanjo.glb \
        --arkit-dir data/arkit_reference \
        --output output/avaturn/juanjo_with_bs.glb
"""

import sys
import os
import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix
from pathlib import Path

# Parse args
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []
args = {}
i = 0
while i < len(argv):
    if argv[i].startswith("--"):
        args[argv[i][2:].replace("-", "_")] = argv[i + 1]
        i += 2
    else:
        i += 1

input_glb = args.get("input")
arkit_dir = args.get("arkit_dir")
output_glb = args.get("output")

print(f"=== Add Blendshapes to Avaturn ===")
print(f"Input:  {input_glb}")
print(f"ARKit:  {arkit_dir}")
print(f"Output: {output_glb}")

# Clear scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

# Import Avaturn GLB
bpy.ops.import_scene.gltf(filepath=input_glb)
print(f"Imported GLB")

# Find the body mesh
body_obj = None
for obj in bpy.data.objects:
    if obj.type == "MESH" and "body" in obj.name.lower():
        body_obj = obj
        break

if not body_obj:
    print("ERROR: No body mesh found!")
    sys.exit(1)

print(f"Body mesh: {body_obj.name}, {len(body_obj.data.vertices)} vertices")

# Apply all transforms to get world-space coordinates
bpy.context.view_layer.objects.active = body_obj
body_obj.select_set(True)

# Get LOCAL vertex positions (GLB stores in local space with armature transform)
body_verts = np.array([(v.co.x, v.co.y, v.co.z) for v in body_obj.data.vertices])

# Apply object transform to get real positions
import mathutils
mat = body_obj.matrix_world
body_verts_world = np.array([(mat @ v.co).to_tuple() for v in body_obj.data.vertices])
print(f"Body verts world Y: [{body_verts_world[:,1].min():.3f}, {body_verts_world[:,1].max():.3f}]")
print(f"Body verts local Y: [{body_verts[:,1].min():.3f}, {body_verts[:,1].max():.3f}]")

# Identify face region - find the HEIGHT axis (largest range)
ranges = body_verts.max(axis=0) - body_verts.min(axis=0)
height_axis = np.argmax(ranges)
print(f"Height axis: {'XYZ'[height_axis]} (range={ranges[height_axis]:.3f})")

h_max = body_verts[:, height_axis].max()
y_threshold = h_max - 0.25  # Top 25cm = head region
face_mask = body_verts[:, height_axis] > y_threshold
face_indices = np.where(face_mask)[0]
face_verts_local = body_verts[face_indices]
print(f"Face vertices: {len(face_indices)} ({'XYZ'[height_axis]} > {y_threshold:.3f})")

# Load ARKit reference neutral
def read_obj_verts(path):
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith("v ") and not line.startswith("vt") and not line.startswith("vn"):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts)

arkit_path = Path(arkit_dir)
neutral_verts = read_obj_verts(arkit_path / "Neutral.obj")
print(f"ARKit neutral: {len(neutral_verts)} verts")

# Remap ARKit axes to Avaturn: ARKit(X,Y,Z) -> Avaturn(X,Z,Y)
def remap_axes(verts):
    """Convert ARKit coordinate system to Avaturn/Blender."""
    return np.column_stack([verts[:, 0], verts[:, 2], verts[:, 1]])

neutral_remapped = remap_axes(neutral_verts)

arkit_center = neutral_remapped.mean(axis=0)
arkit_extent = neutral_remapped.max(axis=0) - neutral_remapped.min(axis=0)

face_center_local = face_verts_local.mean(axis=0)
face_extent_local = face_verts_local.max(axis=0) - face_verts_local.min(axis=0)

# Scale ARKit to match Avaturn face size
scale = face_extent_local.max() / arkit_extent.max()
print(f"Scale factor: {scale:.6f}")

# Align in local space
neutral_aligned = (neutral_remapped - arkit_center) * scale + face_center_local

# Find nearest ARKit vertex for each Avaturn face vertex (brute force, no scipy needed)
def find_nearest(source, targets):
    """For each target point, find nearest source point index and distance."""
    indices = np.zeros(len(targets), dtype=int)
    dists = np.zeros(len(targets))
    for i, t in enumerate(targets):
        d = np.linalg.norm(source - t, axis=1)
        indices[i] = np.argmin(d)
        dists[i] = d[indices[i]]
    return dists, indices

distances, arkit_indices = find_nearest(neutral_aligned, face_verts_local)
print(f"Correspondence: avg dist = {distances.mean():.4f}, max = {distances.max():.4f}")

# ARKit blendshape names
ARKIT_NAMES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut",
]

VISEME_DEFS = {
    "viseme_sil": {"mouthClose": 0.1},
    "viseme_PP": {"mouthClose": 0.9, "mouthPressLeft": 0.6, "mouthPressRight": 0.6, "mouthPucker": 0.2},
    "viseme_FF": {"mouthFunnel": 0.3, "mouthLowerDownLeft": 0.4, "mouthLowerDownRight": 0.4, "mouthRollLower": 0.5, "jawOpen": 0.1},
    "viseme_TH": {"jawOpen": 0.15, "tongueOut": 0.6, "mouthLowerDownLeft": 0.2, "mouthLowerDownRight": 0.2},
    "viseme_DD": {"jawOpen": 0.2, "mouthLowerDownLeft": 0.3, "mouthLowerDownRight": 0.3, "mouthStretchLeft": 0.1, "mouthStretchRight": 0.1},
    "viseme_kk": {"jawOpen": 0.25, "mouthLowerDownLeft": 0.2, "mouthLowerDownRight": 0.2, "mouthStretchLeft": 0.15, "mouthStretchRight": 0.15},
    "viseme_CH": {"jawOpen": 0.15, "mouthSmileLeft": 0.3, "mouthSmileRight": 0.3, "mouthFunnel": 0.4, "mouthStretchLeft": 0.2, "mouthStretchRight": 0.2},
    "viseme_SS": {"jawOpen": 0.05, "mouthSmileLeft": 0.2, "mouthSmileRight": 0.2, "mouthStretchLeft": 0.3, "mouthStretchRight": 0.3},
    "viseme_nn": {"jawOpen": 0.15, "mouthLowerDownLeft": 0.2, "mouthLowerDownRight": 0.2, "mouthSmileLeft": 0.1, "mouthSmileRight": 0.1},
    "viseme_RR": {"jawOpen": 0.2, "mouthPucker": 0.4, "mouthFunnel": 0.3, "mouthRollLower": 0.1, "mouthRollUpper": 0.1},
    "viseme_aa": {"jawOpen": 0.7, "mouthFunnel": 0.2, "mouthLowerDownLeft": 0.3, "mouthLowerDownRight": 0.3, "mouthUpperUpLeft": 0.2, "mouthUpperUpRight": 0.2},
    "viseme_E": {"jawOpen": 0.3, "mouthSmileLeft": 0.4, "mouthSmileRight": 0.4, "mouthStretchLeft": 0.3, "mouthStretchRight": 0.3},
    "viseme_I": {"jawOpen": 0.1, "mouthSmileLeft": 0.5, "mouthSmileRight": 0.5, "mouthStretchLeft": 0.4, "mouthStretchRight": 0.4},
    "viseme_O": {"jawOpen": 0.4, "mouthFunnel": 0.6, "mouthPucker": 0.3, "mouthRollLower": 0.1, "mouthRollUpper": 0.1},
    "viseme_U": {"jawOpen": 0.2, "mouthFunnel": 0.5, "mouthPucker": 0.7, "mouthRollLower": 0.15, "mouthRollUpper": 0.15},
}

# Add Basis shape key
body_obj.shape_key_add(name="Basis", from_mix=False)

# Distance-based weight falloff for smoother transitions
max_dist = np.percentile(distances, 90)
weights = np.clip(1.0 - distances / (max_dist * 2.5), 0.05, 1.0)

# Transfer each ARKit blendshape
arkit_deltas = {}
n_added = 0
for name in ARKIT_NAMES:
    bs_path = arkit_path / f"{name}.obj"
    if not bs_path.exists():
        print(f"  Skip (missing): {name}")
        continue

    bs_verts = read_obj_verts(bs_path)
    bs_remapped = remap_axes(bs_verts)
    bs_aligned = (bs_remapped - arkit_center) * scale + face_center_local

    # Compute delta in aligned space
    delta = bs_aligned - neutral_aligned  # (3084, 3)

    # Transfer delta to Avaturn face vertices via nearest-vertex correspondence
    face_delta = delta[arkit_indices] * weights[:, np.newaxis]  # (n_face, 3)

    # Store for viseme generation
    arkit_deltas[name] = face_delta

    # Create shape key
    sk = body_obj.shape_key_add(name=name, from_mix=False)

    # Apply delta only to face vertices (rest stays at basis)
    for fi, body_idx in enumerate(face_indices):
        d = face_delta[fi]
        basis_co = body_obj.data.shape_keys.key_blocks["Basis"].data[body_idx].co
        sk.data[body_idx].co = basis_co + Vector(d)

    n_added += 1

print(f"Added {n_added} ARKit blendshapes")

# Generate visemes
n_visemes = 0
for vis_name, definition in VISEME_DEFS.items():
    combined_delta = np.zeros((len(face_indices), 3))
    for bs_name, weight in definition.items():
        if bs_name in arkit_deltas:
            combined_delta += weight * arkit_deltas[bs_name]

    sk = body_obj.shape_key_add(name=vis_name, from_mix=False)
    for fi, body_idx in enumerate(face_indices):
        d = combined_delta[fi]
        basis_co = body_obj.data.shape_keys.key_blocks["Basis"].data[body_idx].co
        sk.data[body_idx].co = basis_co + Vector(d)

    n_visemes += 1

print(f"Added {n_visemes} visemes")
print(f"Total morph targets: {n_added + n_visemes}")

# Export
Path(output_glb).parent.mkdir(parents=True, exist_ok=True)
bpy.ops.export_scene.gltf(
    filepath=output_glb,
    export_format="GLB",
    export_morph=True,
    export_morph_normal=False,
    export_skins=True,
    export_animations=False,
    export_texcoords=True,
    export_normals=True,
    export_materials="EXPORT",
    export_image_format="JPEG",
)

print(f"\nExported: {output_glb}")
file_size = os.path.getsize(output_glb) / (1024 * 1024)
print(f"Size: {file_size:.1f} MB")
print("=== Done ===")
