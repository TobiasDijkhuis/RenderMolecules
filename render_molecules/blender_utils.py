from __future__ import annotations

import os

import bmesh
import bpy
import numpy as np

from .constants import SPHERE_SCALE
from .element_data import element_list, vdw_radii


def hex_to_rgb_tuple(hexcode):
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


def create_uv_sphere(element, position, resolution="medium"):
    nsegments, nrings = scale_vertices(64, 32, resolution=resolution)

    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=nsegments,
        ring_count=nrings,
        radius=vdw_radii[element_list.index(element)] * SPHERE_SCALE,
        enter_editmode=False,
        align="WORLD",
        location=position,
    )
    obj = bpy.context.view_layer.objects.active

    obj.name = "atom-%s" % (element)
    try_autosmooth()
    return obj


def create_mesh_of_atoms(positions, reference_sphere, element):
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


def create_material(name, color, alpha=1.0, force: bool = False):
    """
    Build a new material
    """
    # early exit if material already exists
    if not force and name in bpy.data.materials:
        return bpy.data.materials[name]

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    matsettings = {
        "Base Color": hex_to_rgb_tuple(color),
        "Subsurface": 0.2,
        "Subsurface Radius": (0.3, 0.3, 0.3),
        "Subsurface Color": hex_to_rgb_tuple("000000"),
        "Metallic": 0.0,
        "Roughness": 0.5,
        "Alpha": alpha,
    }

    for key, target in mat.node_tree.nodes["Principled BSDF"].inputs.items():
        for refkey, value in matsettings.items():
            if key == refkey:
                target.default_value = value

    return mat


def delete_all_objects():
    """
    Deletes all objects in the current scene
    """
    delete_list_objects = [
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
        if o.type in delete_list_objects:
            o.select_set(True)
        else:
            o.select_set(False)

    # Deletes all selected objects in the scene:
    bpy.ops.object.delete()


def create_isosurface(
    verts, faces, isovalue, prefix="isosurface", assign_material_based_on_sign=True
):
    name = f"{prefix}_{isovalue}"
    mesh = bpy.data.meshes.new(name=name)
    mesh.from_pydata(verts, [], faces, shade_flat=False)

    obj = bpy.data.objects.new(name, mesh)

    scene = bpy.context.scene
    scene.collection.objects.link(obj)

    if assign_material_based_on_sign:
        assign_isosurface_material_based_on_sign(obj, isovalue)


def load_ply(filepath, assign_material_based_on_sign=True):
    bpy.ops.wm.ply_import(filepath=filepath)
    bpy.ops.object.shade_smooth()

    if not assign_material_based_on_sign:
        return

    obj = bpy.context.view_layer.objects.active

    isovalue = float(os.path.splitext(filepath)[0].split("_")[-1])
    assign_isosurface_material_based_on_sign(obj, isovalue)


def assign_isosurface_material_based_on_sign(isosurface_obj, isovalue):
    # Perhaps add a positive or negative lobe material to it, depending on whether there's a '-' in the filepath
    if isovalue < 0:
        # Negative lobe material
        mat = create_material("Negative Lobe", "FF7743", alpha=0.5)
        isosurface_obj.data.materials.append(mat)
    else:
        # Positive lobe material
        mat = create_material("Positive Lobe", "53B9FF", alpha=0.5)
        isosurface_obj.data.materials.append(mat)


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


def set_background_transparency(transparency: bool) -> None:
    bpy.context.scene.render.film_transparent = transparency


def set_background_color(rgba: tuple[float]) -> None:
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[
        0
    ].default_value = rgba


def adjust_settings(is_one_render: bool = True, transparent_background: bool = True):
    scene = bpy.context.scene

    scene.render.film_transparent = transparent_background
    scene.render.use_persistent_data = not is_one_render
    scene.cycles.debug_use_spatial_slits = True


def outline_in_render(render_outline=True, thickness=5):
    if not render_outline:
        bpy.context.scene.render.use_freestyle = False
        return
    bpy.context.scene.render.use_freestyle = True

    view_layer = bpy.data.scenes["Scene"].view_layers["ViewLayer"]
    view_layer.use_freestyle = True

    lineset = view_layer.freestyle_settings.linesets["LineSet"]

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


def select_object_by_name(name: str, select=True):
    bpy.data.objects[name].select_set(select)


def get_object_by_name(name: str):
    return bpy.context.scene.objects[name]


def create_cylinder(
    location, angle, thickness, length, resolution="medium", name="Cylinder"
):
    nvertices = scale_vertices(64, resolution=resolution)

    scale = (thickness, thickness, length)
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=nvertices,
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


def join_two_cylinders(cylinder1, cylinder2, center_position, select_within):
    deselect_all_selected()
    cylinder1.select_set(True)
    cylinder2.select_set(True)
    bpy.ops.object.join()
    mesh = bpy.context.object.data
    # Somehow select all edges of one side of each cylinder,
    # then bridge them.
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    for vert in bm.verts:
        vertex_distance = (
            sum([(vert.co[i] - center_position[i]) ** 2 for i in range(3)]) ** 0.5
        )
        if vertex_distance <= select_within:
            vert.select_set(True)
        else:
            vert.select_set(False)
        print(vert.co, center_position, vertex_distance, select_within, vert.select)
    for edge in bm.edges:
        if edge.verts[0].select and edge.verts[1].select:
            edge.select_set(True)
        else:
            edge.select_set(False)

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.loop_select(extend=True, ring=True)

    # deselectAllSelected()
    # bpy.ops.object.editmode_toggle()
    # bpy.ops.mesh.bridge_edge_loops(
    #     type="CLOSED",
    #     use_merge=False,
    #     number_cuts=20,
    #     smoothness=1,
    #     interpolation="SURFACE",
    #     profile_shape_factor=0,
    #     profile_shape="SPHERE",
    # )
    # bpy.ops.object.editmode_toggle()
    return
    # cylinder1.select_set(False)
    # cylinder2.select_set(False)
    # bpy.ops.object.editmode_toggle()
    # bm.to_mesh(mesh)
    # bm.free()


def join_cylinders(
    cylinders: list[object], atom_position: np.ndarray, vdw_radius: float
):
    for i, cylinder in enumerate(cylinders[:-1]):
        join_two_cylinders(cylinder, cylinders[i + 1], atom_position, vdw_radius)
        return


def put_cap_on_cylinder(cylinder):
    # If an atom is only bound on one side, the bond will have to be
    # terminated by a hemisphere at one end. Can be done like this?
    # https://blender.stackexchange.com/questions/84789/how-can-i-cap-a-cylinder-with-a-hemisphere
    pass


def scale_vertices(*args, resolution="medium"):
    resolution = resolution.lower()
    if resolution not in ["verylow", "low", "medium", "high", "veryhigh"]:
        msg = f"renderResolution should be one of ['verylow', 'low', 'medium', 'high', 'veryhigh'] but was '{resolution}'"
        raise ValueError(msg)

    if resolution == "verylow":
        scale = 1 / 4
    elif resolution == "low":
        scale = 1 / 2
    elif resolution == "medium":
        scale = 1
    elif resolution == "high":
        scale = 2
    elif resolution == "veryhigh":
        scale = 4

    if len(args) == 1:
        return int(args[0] * scale)
    elif len(args) > 1:
        return tuple([int(arg * scale) for arg in args])
    else:
        raise ValueError()


def deselect_all_selected():
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
