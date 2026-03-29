"""Blender script to bake a photo texture onto a FLAME mesh via camera projection.

Usage:
    blender -b --python scripts/blender_bake_texture.py -- \
        --head-obj path/to/head.obj \
        --photo path/to/photo.jpg \
        --camera-npz path/to/camera.npz \
        --output path/to/texture.png \
        --resolution 2048
"""

import sys
import bpy
import numpy as np
from pathlib import Path
from mathutils import Vector

# Parse args
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []
args = {}
i = 0
while i < len(argv):
    if argv[i].startswith("--"):
        key = argv[i][2:].replace("-", "_")
        args[key] = argv[i + 1] if i + 1 < len(argv) else True
        i += 2
    else:
        i += 1

head_obj = args.get("head_obj")
photo_path = args.get("photo")
camera_npz = args.get("camera_npz")
output_path = args.get("output", "texture.png")
resolution = int(args.get("resolution", "2048"))

print(f"=== Blender Texture Bake ===")
print(f"Head: {head_obj}")
print(f"Photo: {photo_path}")
print(f"Camera: {camera_npz}")
print(f"Output: {output_path} ({resolution}x{resolution})")

# Clear scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()
for m in bpy.data.meshes:
    bpy.data.meshes.remove(m)
for m in bpy.data.materials:
    bpy.data.materials.remove(m)
for img in bpy.data.images:
    bpy.data.images.remove(img)

# Import mesh
bpy.ops.wm.obj_import(filepath=head_obj)
obj = bpy.context.selected_objects[0]
obj.name = "Head"

for poly in obj.data.polygons:
    poly.use_smooth = True

# Load camera params
cam_data = np.load(camera_npz)
scale = float(cam_data["scale"])
tx = float(cam_data["tx"])
ty = float(cam_data["ty"])
photo_size = int(cam_data["photo_size"])

print(f"Camera params: scale={scale:.4f}, tx={tx:.4f}, ty={ty:.4f}")

# Compute orthographic camera bounds
# Fitter projection: px_norm = vx * scale + tx, py_norm = -vy * scale + ty
# This means: vx = (px_norm - tx) / scale
# For px_norm in [0, 1]: vx ranges from -tx/scale to (1-tx)/scale
# Ortho camera sees: left = -tx/scale, right = (1-tx)/scale
ortho_left = -tx / scale
ortho_right = (1.0 - tx) / scale
ortho_bottom = -(1.0 - ty) / scale  # Flipped Y
ortho_top = ty / scale

ortho_width = ortho_right - ortho_left
ortho_height = ortho_top - ortho_bottom
ortho_center_x = (ortho_left + ortho_right) / 2
ortho_center_y = (ortho_bottom + ortho_top) / 2

print(f"Ortho bounds: L={ortho_left:.4f} R={ortho_right:.4f} B={ortho_bottom:.4f} T={ortho_top:.4f}")
print(f"Ortho size: {ortho_width:.4f} x {ortho_height:.4f}")

# Create camera
cam_data_bl = bpy.data.cameras.new("BakeCamera")
cam_data_bl.type = "ORTHO"
cam_data_bl.ortho_scale = max(ortho_width, ortho_height)

cam_obj = bpy.data.objects.new("BakeCamera", cam_data_bl)
bpy.context.scene.collection.objects.link(cam_obj)

# Position camera in front of face, looking at -Z
cam_obj.location = (ortho_center_x, ortho_center_y, 1.0)
cam_obj.rotation_euler = (0, 0, 0)  # Looking down -Z

# Set as active camera
bpy.context.scene.camera = cam_obj

# Set render resolution to match photo aspect
bpy.context.scene.render.resolution_x = photo_size
bpy.context.scene.render.resolution_y = photo_size

# Load photo as image
photo_img = bpy.data.images.load(photo_path)
print(f"Photo loaded: {photo_img.size[0]}x{photo_img.size[1]}")

# Create the bake target image
bake_img = bpy.data.images.new("BakeTarget", width=resolution, height=resolution, alpha=False)
bake_img.colorspace_settings.name = "sRGB"

# Set up material for projection baking
# We need TWO materials:
# 1. A material that projects the photo from the camera (for the "from active" source)
# 2. A material on the mesh with just the bake target image (for receiving the bake)

# Actually, Blender's "Bake from Active" approach is complex.
# Simpler: use "Project from View" via texture painting, or use emission bake.

# Simplest approach: create a material with the photo mapped via camera projection,
# then bake the diffuse/emission to the UV texture.

mat = obj.data.materials[0] if obj.data.materials else bpy.data.materials.new("BakeMat")
if not obj.data.materials:
    obj.data.materials.append(mat)

tree = mat.node_tree
nodes = tree.nodes
links = tree.links

# Clear existing nodes
for n in list(nodes):
    nodes.remove(n)

# Create nodes: TexCoord → Mapping → Image Texture → Emission → Output
# Use "Window" texture coordinate which projects from camera
tex_coord = nodes.new("ShaderNodeTexCoord")
tex_coord.location = (-800, 0)

# Camera projection: use Window coords
mapping = nodes.new("ShaderNodeMapping")
mapping.location = (-600, 0)

# Photo texture
img_tex = nodes.new("ShaderNodeTexImage")
img_tex.location = (-300, 0)
img_tex.image = photo_img
img_tex.interpolation = "Smart"

# Use emission for baking (captures color without lighting interference)
emission = nodes.new("ShaderNodeEmission")
emission.location = (0, 0)

output = nodes.new("ShaderNodeOutputMaterial")
output.location = (200, 0)

# Connect: Window coords → Image Texture → Emission → Output
links.new(tex_coord.outputs["Window"], img_tex.inputs["Vector"])
links.new(img_tex.outputs["Color"], emission.inputs["Color"])
links.new(emission.outputs["Emission"], output.inputs["Surface"])

# Also add the bake target image node (must be selected for bake destination)
bake_target_node = nodes.new("ShaderNodeTexImage")
bake_target_node.location = (-300, -300)
bake_target_node.image = bake_img
bake_target_node.select = True
nodes.active = bake_target_node

# Set up bake settings
bpy.context.scene.render.engine = "CYCLES"
bpy.context.scene.cycles.device = "CPU"
bpy.context.scene.cycles.samples = 1  # Just 1 sample for baking
bpy.context.scene.cycles.bake_type = "EMIT"

# Select the mesh
bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
bpy.context.view_layer.objects.active = obj

# Bake!
print("Baking texture...")
bpy.ops.object.bake(type="EMIT")
print("Bake complete!")

# Save baked texture
bake_img.filepath_raw = output_path
bake_img.file_format = "PNG"
bake_img.save()
print(f"Saved: {output_path}")

print("=== Bake Done ===")
