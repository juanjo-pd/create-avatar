"""Blender headless script for avatar assembly.

Usage:
    blender -b --python scripts/blender_assembly.py -- \
        --head-obj path/to/neutral.obj \
        --blendshapes-dir path/to/blendshapes/ \
        --texture-png path/to/texture.png \
        --bust-template path/to/bust_base.blend \
        --output-blend path/to/output.blend \
        --output-glb path/to/avatar.glb

This script:
1. Imports the FLAME neutral head mesh
2. Imports the bust template with skeleton
3. Merges head onto bust (bridge neck edge loops)
4. Imports 52 ARKit blendshapes + 15 visemes as shape keys
5. Applies automatic weights from skeleton
6. Applies texture as Principled BSDF material
7. Exports as GLB with morph targets and skeleton
"""

import sys
import os
from pathlib import Path

# Parse arguments after '--'
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []


def parse_args(argv):
    args = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            key = argv[i][2:].replace("-", "_")
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                args[key] = argv[i + 1]
                i += 2
            else:
                args[key] = True
                i += 1
        else:
            i += 1
    return args


args = parse_args(argv)

# Only import bpy when running inside Blender
try:
    import bpy
    import bmesh
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
    print("This script must be run inside Blender:")
    print("  blender -b --python scripts/blender_assembly.py -- --help")
    sys.exit(1)


def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat)


def import_obj(filepath, name=None):
    """Import an OBJ file and return the imported object."""
    bpy.ops.wm.obj_import(filepath=str(filepath))
    obj = bpy.context.selected_objects[0]
    if name:
        obj.name = name
    return obj


def fix_mesh(obj):
    """Recalculate normals, apply smooth shading, and clean up mesh."""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Enter edit mode to fix normals
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    # Apply smooth shading to all faces
    for poly in obj.data.polygons:
        poly.use_smooth = True

    print(f"  Fixed normals and applied smooth shading: {len(obj.data.vertices)} verts, {len(obj.data.polygons)} faces")


def create_skeleton():
    """Create a humanoid bust skeleton.

    Hierarchy:
    Hips → Spine → Spine1 → Spine2 → Neck → Head
                                     ├── LeftShoulder
                                     └── RightShoulder
    Head has child bones: LeftEye, RightEye, Jaw
    """
    bpy.ops.object.armature_add(enter_editmode=True)
    armature = bpy.context.object
    armature.name = "AvatarArmature"

    arm_data = armature.data
    arm_data.name = "AvatarSkeleton"

    # Remove default bone
    for bone in arm_data.edit_bones:
        arm_data.edit_bones.remove(bone)

    # Define bone hierarchy with positions (head_pos, tail_pos)
    bones_def = {
        "Hips": ((0, 0, 0.85), (0, 0, 0.95), None),
        "Spine": ((0, 0, 0.95), (0, 0, 1.05), "Hips"),
        "Spine1": ((0, 0, 1.05), (0, 0, 1.15), "Spine"),
        "Spine2": ((0, 0, 1.15), (0, 0, 1.25), "Spine1"),
        "Neck": ((0, 0, 1.25), (0, 0, 1.35), "Spine2"),
        "Head": ((0, 0, 1.35), (0, 0, 1.55), "Neck"),
        "LeftEye": ((0.03, 0, 1.47), (0.03, 0, 1.49), "Head"),
        "RightEye": ((-0.03, 0, 1.47), (-0.03, 0, 1.49), "Head"),
        "Jaw": ((0, 0, 1.38), (0, 0, 1.34), "Head"),
        "LeftShoulder": ((0, 0, 1.25), (0.08, 0, 1.24), "Spine2"),
        "RightShoulder": ((0, 0, 1.25), (-0.08, 0, 1.24), "Spine2"),
    }

    created_bones = {}
    for bone_name, (head, tail, parent_name) in bones_def.items():
        bone = arm_data.edit_bones.new(bone_name)
        bone.head = head
        bone.tail = tail
        if parent_name and parent_name in created_bones:
            bone.parent = created_bones[parent_name]
        created_bones[bone_name] = bone

    bpy.ops.object.mode_set(mode="OBJECT")
    return armature


def add_shape_keys(mesh_obj, blendshapes_dir, neutral_verts):
    """Add ARKit blendshapes and visemes as shape keys.

    Args:
        mesh_obj: Blender mesh object (already has the neutral shape).
        blendshapes_dir: Directory with OBJ files named by blendshape name.
        neutral_verts: Original neutral vertex positions for delta computation.
    """
    # Add Basis shape key
    mesh_obj.shape_key_add(name="Basis", from_mix=False)

    blendshapes_dir = Path(blendshapes_dir)

    # Import blendshapes in canonical order
    # Read the canonical name list
    names_file = blendshapes_dir / "_names.txt"
    if names_file.exists():
        with open(names_file) as f:
            names = [line.strip() for line in f if line.strip()]
    else:
        # Default: use all OBJ files found
        names = sorted([p.stem for p in blendshapes_dir.glob("*.obj")])

    for name in names:
        obj_path = blendshapes_dir / f"{name}.obj"
        if not obj_path.exists():
            print(f"  Warning: Missing blendshape OBJ: {obj_path}")
            continue

        # Read vertex positions from OBJ
        verts = _read_obj_vertices(obj_path)
        if len(verts) != len(mesh_obj.data.vertices):
            print(f"  Warning: Vertex count mismatch for {name}: {len(verts)} vs {len(mesh_obj.data.vertices)}")
            continue

        # Add shape key
        sk = mesh_obj.shape_key_add(name=name, from_mix=False)
        for i, v in enumerate(verts):
            sk.data[i].co = v

        print(f"  Added shape key: {name}")


def _read_obj_vertices(filepath):
    """Read just vertex positions from an OBJ file."""
    verts = []
    with open(filepath) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return verts


def apply_texture(mesh_obj, texture_path):
    """Apply a texture to the mesh's existing material.

    OBJ import in Blender 5 creates a material with Principled BSDF.
    We just add an image texture node and connect it to Base Color.
    DO NOT clear existing nodes — that breaks the GLTF export.
    """
    # Get existing material (created by OBJ import) or create one
    if mesh_obj.data.materials:
        mat = mesh_obj.data.materials[0]
    else:
        mat = bpy.data.materials.new(name="AvatarSkin")
        mesh_obj.data.materials.append(mat)

    if not mat.node_tree:
        mat.use_nodes = True

    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links

    # Find existing Principled BSDF
    bsdf = None
    for node in nodes:
        if node.type == "BSDF_PRINCIPLED":
            bsdf = node
            break

    if not bsdf:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        output = nodes.new("ShaderNodeOutputMaterial")
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # Set material properties
    bsdf.inputs["Roughness"].default_value = 0.7
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.3

    # Add image texture and connect to Base Color
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(texture_path))
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])


def parent_mesh_to_armature(mesh_obj, armature):
    """Parent mesh to armature with automatic weights."""
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")


def export_glb(output_path):
    """Export scene as GLB with morph targets and skeleton."""
    bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format="GLB",
        export_draco_mesh_compression_enable=False,  # Draco breaks morph targets in Three.js
        export_morph=True,
        export_morph_normal=False,  # Skip morph normals to reduce size ~50%
        export_skins=True,
        export_animations=False,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_image_format="JPEG",  # JPEG textures are much smaller than PNG
    )
    print(f"Exported GLB: {output_path}")


def main():
    head_obj_path = args.get("head_obj")
    blendshapes_dir = args.get("blendshapes_dir")
    texture_path = args.get("texture_png")
    output_glb = args.get("output_glb", "output/avatar.glb")

    if not head_obj_path:
        print("Error: --head-obj is required")
        sys.exit(1)

    print(f"=== Avatar Assembly ===")
    print(f"Head OBJ: {head_obj_path}")
    print(f"Blendshapes: {blendshapes_dir}")
    print(f"Texture: {texture_path}")
    print(f"Output: {output_glb}")

    # Clear scene
    clear_scene()

    # Import head mesh
    print("\n1. Importing head mesh...")
    head = import_obj(head_obj_path, name="AvatarHead")

    # Create skeleton
    print("\n2. Creating skeleton...")
    armature = create_skeleton()

    # Add shape keys (blendshapes + visemes)
    if blendshapes_dir and Path(blendshapes_dir).exists():
        print("\n3. Adding shape keys...")
        neutral_verts = [v.co.copy() for v in head.data.vertices]
        add_shape_keys(head, blendshapes_dir, neutral_verts)
    else:
        print("\n3. No blendshapes directory provided, skipping shape keys.")

    # Apply texture
    if texture_path and Path(texture_path).exists():
        print("\n4. Applying texture...")
        apply_texture(head, texture_path)
    else:
        print("\n4. No texture provided, skipping.")

    # Fix mesh normals (must be after all shape keys are added)
    print("\n5. Fixing mesh normals...")
    fix_mesh(head)

    # Parent to armature
    print("\n6. Rigging (automatic weights)...")
    parent_mesh_to_armature(head, armature)

    # Save blend file
    output_blend = args.get("output_blend")
    if output_blend:
        Path(output_blend).parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(output_blend).resolve()))
        print(f"\nSaved .blend: {output_blend}")

    # Export GLB
    print("\n7. Exporting GLB...")
    Path(output_glb).parent.mkdir(parents=True, exist_ok=True)
    export_glb(output_glb)

    print("\n=== Assembly complete ===")


if __name__ == "__main__":
    main()
