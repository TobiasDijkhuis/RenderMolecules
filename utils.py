import bpy
import numpy as np

import math

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
    try_autosmooth()
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

    if assignMaterialBasedOnSign:
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


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta degrees.
    """
    theta *= math.pi / 180.0
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def try_autosmooth():
    try:
        bpy.ops.object.shade_auto_smooth()
    except AttributeError:
        msg = "AttributeError was raised because of shade_auto_smooth. This could be due to an old version of Blender.\n"
        msg += "Trying older syntax."
        print(msg)
        try:
            bpy.ops.object.shade_smooth(use_auto_smooth=True)
        except AttributeError:
            msg = "AttributeError was raised because of shade_smooth(use_auto_smooth=True). This could be due to I DONT KNOW.\n"
            msg += "Can still be applied manually"
            print(msg)


def angle_between(vector1, vector2) -> float:
    vector1 /= np.linalg.norm(vector1)
    vector2 /= np.linalg.norm(vector2)
    axis = np.cross(vector1, vector2)
    if np.linalg.norm(axis) < 1e-5:
        angle = 0.0
    else:
        angle = np.arccos(np.dot(vector1, vector2))
    return angle * 180 / math.pi


def set_background_transparency(transparency: bool) -> None:
    bpy.context.scene.render.film_transparent = transparency


def set_background_color(RGBA: tuple[float]) -> None:
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[
        0
    ].default_value = RGBA


def adjustSettings(isOneRender=True, transparentBackground=True):
    scene = bpy.context.scene

    scene.render.film_transparent = transparentBackground
    scene.render.use_persistent_data = not isOneRender
    scene.cycles.debug_use_spatial_slits = True


def outlineInRender(renderOutline=True, thickness=5):
    if not renderOutline:
        bpy.context.scene.render.use_freestyle = False
        return
    bpy.context.scene.render.use_freestyle = True

    viewLayer = bpy.data.scenes["Scene"].view_layers["ViewLayer"]
    viewLayer.use_freestyle = True

    lineset = viewLayer.freestyle_settings.linesets["LineSet"]

    lineset.select_external_contour = True

    lineset.select_suggestive_contour = False
    lineset.select_edge_mark = False
    lineset.select_material_boundary = False
    lineset.select_silhouette = False
    lineset.select_crease = False
    lineset.select_border = False
    lineset.select_ridge_valley = False
    lineset.select_contour = False

    bpy.data.linestyles["LineStyle"].caps = "SQUARE"
    bpy.data.linestyles["LineStyle"].texture_spacing = 20
    bpy.data.linestyles["LineStyle"].thickness = thickness


def findFirstStringInListOfStrings(
    stringToFind: str | list[str],
    listOfStrings: list[str],
    start: int = 0,
    end: int | None = None,
) -> int | None:
    """Finds the first instance of a string in a list of strings."""
    if isinstance(stringToFind, list):
        resultForEachSubstring = np.ndarray(len(stringToFind), dtype=list)

        for i, substring in enumerate(stringToFind):
            resultForEachSubstring[i] = findAllStringInListOfStrings(
                substring, listOfStrings, start, end
            )
        intersectionOfAllSubstrings = list(
            set.intersection(*map(set, resultForEachSubstring))
        )
        return intersectionOfAllSubstrings[0]

    seperatorString = "UNIQUE SEPERATOR STRING THAT DOES NOT OCCUR IN THE FILE ITSELF"
    joinedList = seperatorString.join(listOfStrings[start:end])
    try:
        stringIndex = joinedList.index(stringToFind)
    except ValueError:
        return
    listIndex = joinedList.count(seperatorString, 0, stringIndex)
    return listIndex + start


def findAllStringInListOfStrings(
    stringToFind: str | list[str],
    listOfStrings: list[str],
    start: int = 0,
    end: int | None = None,
) -> list[int]:
    """Finds all instances of a string stringToFind in a list of strings."""
    if isinstance(stringToFind, list):
        resultForEachSubstring = np.ndarray(len(stringToFind), dtype=list)

        for i, substring in enumerate(stringToFind):
            resultForEachSubstring[i] = findAllStringInListOfStrings(
                substring, listOfStrings, start, end
            )
        intersectionOfAllSubstrings = list(
            set.intersection(*map(set, resultForEachSubstring))
        )
        return intersectionOfAllSubstrings

    result = []
    newResult = start - 1
    while newResult is not None:
        start = newResult + 1
        newResult = findFirstStringInListOfStrings(
            stringToFind, listOfStrings, start, end
        )
        result.append(newResult)
    result.pop()
    return result


def selectObjectByName(name: str, select=True):
    bpy.data.objects[name].select_set(select)


def getObjectByName(name: str):
    return bpy.context.scene.objects[name]


def translateObject(object, displacementVector):
    location = object.location
    location.x += displacementVector[0]
    location.y += displacementVector[1]
    location.z += displacementVector[2]


def createCylinder(location, angle, thickness, length, name="Cylinder"):
    scale = (thickness, thickness, length)
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=128,
        enter_editmode=False,
        align="WORLD",
        location=location,
        scale=scale,
    )
    obj = bpy.context.view_layer.objects.active
    obj.rotation_mode = "AXIS_ANGLE"
    obj.rotation_axis_angle = angle
    obj.name = name
    try_autosmooth()
    return obj
