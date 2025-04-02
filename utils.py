import bpy
import numpy as np

from ElementData import *


def hex2rgbtuple(hexcode):
    """
    Convert 6-digit color hexcode to a tuple of floats
    """
    hexcode += "FF"
    hextuple = tuple([int(hexcode[i : i + 2], 16) / 255.0 for i in [0, 2, 4, 6]])

    return tuple([color_srgb_to_scene_linear(c) for c in hextuple])


def color_srgb_to_scene_linear(c):
    """
    Convert RGB to sRGB
    """
    if c < 0.04045:
        return 0.0 if c < 0.0 else c * (1.0 / 12.92)
    else:
        return ((c + 0.055) * (1.0 / 1.055)) ** 2.4


def createUVsphere(element, position, resolution="medium"):
    resolution = resolution.lower()
    if resolution not in ["low", "medium", "high", "ultra"]:
        msg = f"resolution should be one of 'low', 'medium', 'high' or 'ultra', but was '{resolution}'"
        raise ValueError(msg)
    segments = 64
    ring_count = 32

    if resolution == "low":
        segments /= 2
        ring_count /= 2
    elif resolution == "high":
        segments *= 2
        ring_count *= 2
    elif resolution == "ultra":
        segments *= 4
        ring_count *= 4

    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=segments,
        ring_count=ring_count,
        radius=vdwRadii[elementList.index(element)] * sphereScale,
        enter_editmode=False,
        align="WORLD",
        location=position,
    )
    obj = bpy.context.view_layer.objects.active

    obj.name = "atom-%s" % (element)
    bpy.ops.object.shade_auto_smooth()
    return obj


def createMeshAtoms(positions, referenceAtom, element):
    mesh = bpy.data.meshes.new(f"{element}_mesh")  # add the new mesh
    obj = bpy.data.objects.new(mesh.name, mesh)

    col = bpy.data.collections["Collection"]
    col.objects.link(obj)

    bpy.context.view_layer.objects.active = obj

    verts = positions
    edges = []
    faces = []

    mesh.from_pydata(verts, edges, faces)

    bpy.ops.object.parent_set(type="OBJECT", keep_transform=False)
    bpy.context.object.instance_type = "VERTS"


def create_material(name, color, alpha=1.0):
    """
    Build a new material
    """
    # early exit if material already exists and has the same color
    if (
        name in bpy.data.materials
    ):  # and np.allclose(bpy.data.materials[name].node_tree.nodes["Principled BSDF"].inputs[0].default_value, hex2rgbtuple(color)):
        return bpy.data.materials[name]

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    matsettings = {
        "Base Color": hex2rgbtuple(color),
        "Subsurface": 0.2,
        "Subsurface Radius": (0.3, 0.3, 0.3),
        "Subsurface Color": hex2rgbtuple("000000"),
        "Metallic": 0.0,
        "Roughness": 0.5,
        "Alpha": alpha,
    }

    for key, target in mat.node_tree.nodes["Principled BSDF"].inputs.items():
        for refkey, value in matsettings.items():
            if key == refkey:
                target.default_value = value

    return mat


def deleteAllObjects():
    """
    Deletes all objects in the current scene
    """
    deleteListObjects = [
        "MESH",
        "CURVE",
        "SURFACE",
        "META",
        "FONT",
        "HAIR",
        "POINTCLOUD",
        "VOLUME",
        "GPENCIL",
        "ARMATURE",
        "LATTICE",
        "EMPTY",
        "SPEAKER",
        "SPHERE",
    ]

    # Select all objects in the scene to be deleted:
    for o in bpy.context.scene.objects:
        if o.type in deleteListObjects:
            o.select_set(True)
        else:
            o.select_set(False)

    # Deletes all selected objects in the scene:
    bpy.ops.object.delete()


def createIsosurface(verts, faces, prefix, isovalue, assignMaterialBasedOnSign=True):
    name = f"{prefix}_{isovalue}"
    mesh = bpy.data.meshes.new(name=name)
    mesh.from_pydata(verts, [], faces, shade_flat=False)

    obj = bpy.data.objects.new(name, mesh)

    scene = bpy.context.scene
    scene.collection.objects.link(obj)

    if not assignMaterialBasedOnSign:
        return

    assignIsosurfaceMaterialBasedOnSign(obj, isovalue)


def loadPLY(filepath, assignMaterialBasedOnSign=True):
    bpy.ops.wm.ply_import(filepath=filepath)
    bpy.ops.object.shade_smooth()

    if not assignMaterialBasedOnSign:
        return

    isovalue = float(os.path.splitext(filepath)[0].split("_")[-1])
    obj = bpy.context.view_layer.objects.active
    assignIsosurfaceMaterialBasedOnSign(obj, isovalue)


def assignIsosurfaceMaterialBasedOnSign(isosurfaceObj, isovalue):
    # Perhaps add a positive or negative lobe material to it, depending on whether there's a '-' in the filepath
    if isovalue < 0:
        # Negative lobe material
        mat = create_material("Negative Lobe", "FF7743", alpha=0.5)
        isosurfaceObj.data.materials.append(mat)
    else:
        # Positive lobe material
        mat = create_material("Positive Lobe", "53B9FF", alpha=0.5)
        isosurfaceObj.data.materials.append(mat)
