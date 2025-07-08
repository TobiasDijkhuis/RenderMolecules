import os

import bmesh
import bpy
import numpy as np

from .constants import FRAME_STEP, SPHERE_SCALE
from .element_data import element_list, manifest, vdw_radii
from .other_utils import color_srgb_to_scene_linear, hex2rgbtuple


def create_uv_sphere(
    element: str, position: np.ndarray, resolution: str = "medium"
) -> object:
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

    obj.name = f"atom-{element}"
    try_autosmooth()
    return obj


def get_all_materials() -> object:
    return bpy.data.materials


def delete_all_materials() -> None:
    materials = get_all_materials()
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    for object in bpy.context.scene.objects:
        if not object.material_slots:
            continue
        object.data.materials.clear()


def create_mesh_of_atoms(
    positions: np.ndarray, reference_sphere: object, element: str
) -> None:
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


def material_exists(mat):
    """Function to determine whether a material already exists. WIP"""
    for mat_name, mat in bpy.data.materials.items():
        if mat_name == name:
            return mat


def create_material(
    name: str, color: str, alpha: float = 1.0, force: bool = False
) -> object:
    """
    Build a new material

    Args:
        name (str): name of material
        color (str): color of material
        alpha (float): transparency of material
        force (bool): whether to force creation of the new material, regardless
            of if another material with that name already exists. If True,
            will remove other materials with that same name.

    Returns:
        mat (object): created material

    Notes:
        * If a material with name ``name`` already exists and
          ``force=False`` is used, returns that material instead.
    """
    # early exit if material already exists
    if not force and name in bpy.data.materials:
        return bpy.data.materials[name]

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    matsettings = {
        "Base Color": hex2rgbtuple(color),
        "Subsurface": 0.2,
        "Subsurface Radius": (0.3, 0.3, 0.3),
        "Subsurface Color": manifest["subsurface_color"],
        "Metallic": manifest["metallic"],
        "Roughness": manifest["roughness"],
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
    verts: np.ndarray,
    faces: np.ndarray,
    isovalue: float,
    prefix: str = "isosurface",
    color: str = "sign",
    alpha: float = manifest["isosurface_alpha"],
) -> None:
    """Creates isosurface from vertices and faces output from a marching cubes calculation.

    Args:
        verts (np.ndarray): Vx3 array of floats corresponding to vertex positions
        faces (np.ndarray): Fx3 array of integers corresponding to vertex indices
        isovalue (float): isovalue used to calculate the isosurface
        prefix (str): prefix to put in front of the isovalue to get its name
        color (str): color of isosurface. Can also be ``"sign"``, and then it is colored using the sign of ``isovalue``,
            and the ``constants.manifest['isosurface_color_negative']`` or ``constants.manifest['isosurface_color_positive']``.
        alpha (float): transparency of created isosurface.

    Notes:
        * :py:meth:`render_molecules.structure.CUBEfile.calculate_isosurface` can be used to calculate the vertices and faces.
        * This function creates a material called ``Isosurface``. Once created, any time ``create_isosurface`` is called again,
          it will call ``create_material`` again with the same name. That will return the original material again, so the new isosurface will
          have the same color and transparency. This is something I will have to fix somehow. Maybe by adding a ``material_name`` argument to
          this function, which can be changed by the user so that if they wish to have a different material on a second isosurface they
          simply specify a different material name for the two isosurfaces?
    """
    name = f"{prefix}_{isovalue}"
    mesh = bpy.data.meshes.new(name=name)
    mesh.from_pydata(verts, [], faces, shade_flat=False)

    obj = bpy.data.objects.new(name, mesh)

    scene = bpy.context.scene
    scene.collection.objects.link(obj)

    if color == "sign":
        assign_isosurface_material_based_on_sign(obj, isovalue, alpha=alpha)
    else:
        mat = create_material("Isosurface", color, alpha=alpha)
        mat.surface_render_method = "BLENDED"
        mat.use_transparency_overlap = False
        obj.data.materials.append(mat)


def load_ply(filepath, color: str = "sign", alpha=manifest["isosurface_alpha"]):
    bpy.ops.wm.ply_import(filepath=filepath)
    bpy.ops.object.shade_smooth()

    obj = bpy.context.view_layer.objects.active

    isovalue = float(os.path.splitext(filepath)[0].split("_")[-1])
    if color == "sign":
        assign_isosurface_material_based_on_sign(obj, isovalue)
    else:
        mat = create_material("Isosurface", color, alpha=alpha)
        mat.surface_render_method = "BLENDED"
        mat.use_transparency_overlap = False
        obj.data.materials.append(mat)


def assign_isosurface_material_based_on_sign(
    isosurface_obj: object,
    isovalue: float,
    alpha=manifest["isosurface_alpha"],
) -> None:
    # Perhaps add a positive or negative lobe material to it, depending on whether there's a '-' in the filepath

    if isovalue < 0:
        # Negative lobe material
        mat = create_material(
            "Negative Lobe",
            manifest["isosurface_color_negative"],
            alpha=alpha,
        )
        mat.surface_render_method = "BLENDED"
        mat.use_transparency_overlap = False
        isosurface_obj.data.materials.append(mat)
    else:
        # Positive lobe material
        mat = create_material(
            "Positive Lobe",
            manifest["isosurface_color_positive"],
            alpha=alpha,
        )
        mat.surface_render_method = "BLENDED"
        mat.use_transparency_overlap = False
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


def adjust_settings(
    is_one_render: bool = True, transparent_background: bool = True
) -> None:
    scene = bpy.context.scene

    scene.render.film_transparent = transparent_background
    scene.render.use_persistent_data = not is_one_render
    scene.cycles.debug_use_spatial_slits = True


def outline_in_render(render_outline: bool = True, thickness: float = 5) -> None:
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


def select_object_by_name(name: str, select=True) -> None:
    """Select an object in the Blender scene from its name

    Args:
        name (str): name of object to select
        select (bool): whether to select it or not (aka deselect)
    """
    bpy.data.objects[name].select_set(select)


def get_object_by_name(name: str) -> object:
    """Get an object in the Blender scene from its name

    Args:
        name (str): name of object to obtain

    Returns:
        object: object in Blender scene with the name ``name``
    """
    return bpy.context.scene.objects[name]


def create_cylinder(
    location: np.ndarray,
    angle: float,
    thickness: float,
    length: float,
    resolution: str = "medium",
    name: str = "Cylinder",
) -> object:
    """Create a cylinder in the Blender scene.

    Args:
        location (np.ndarray): midpoint position of cylinder
        angle (float): angle with z-axis in radians
        thickness (float): radius of created cylinder
        length (float): length of created cylinder
        resolution (str): desired object resolution.
            One of ``['verylow', 'low', 'medium', 'high', 'veryhigh']``.
        name (str): name of created cylinder

    Returns:
        obj (object): Blender object
    """
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


def scale_vertices(*args, resolution="medium") -> int | tuple[int]:
    """Scale number of vertices according to resolution

    Args:
        *args: number of vertices. Can also be multiple arguments.
        resolution (str): desired object resolution.
            One of ``['verylow', 'low', 'medium', 'high', 'veryhigh']``.

    Returns:
        int | tuple: scaled number of vertices. The same number of return values as the number of ``*args``.
    """
    if not isinstance(resolution, str):
        raise TypeError(
            f"resolution should be of type str, but was of type {type(resolution)}"
        )
    resolution = resolution.lower()
    if resolution not in ["verylow", "low", "medium", "high", "veryhigh"]:
        msg = f"resolution should be one of ['verylow', 'low', 'medium', 'high', 'veryhigh'] but was '{resolution}'"
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


def deselect_all_selected() -> None:
    """Deselect all selected objects"""
    for obj in bpy.context.selected_objects:
        obj.select_set(False)


def orbit_camera(
    radius: int | None = None,
    height: int | None = None,
    set_active: bool = True,
    nframes: int = 20,
):
    context = bpy.context
    scene = context.scene
    cam = scene.camera

    set_frame_step(FRAME_STEP)
    end_frame = 1 + FRAME_STEP * (nframes - 1)
    set_frame_end(end_frame)

    if not cam:
        cam_data = bpy.data.cameras.new("Camera")
        cam = bpy.data.objects.new("Camera", cam_data)
        bpy.context.collection.objects.link(cam)

        # Set position of camera to be at a certain radius away from origin and height
        cam.delta_location = (0, radius, height)

        # But still make it point at origin
        cam.delta_rotation_euler = (np.pi + np.arctan(radius / height), np.pi, 0)

    if "EMPTY FOR CAMERA ORBIT" in bpy.data.objects.keys():
        mt = bpy.data.objects["EMPTY FOR CAMERA ORBIT"]
    else:
        bpy.ops.object.empty_add(location=(0, 0, 0))
        mt = context.object
        mt.empty_display_type = "SPHERE"
        mt.empty_display_size = 4

        # Give it a distinctive name so that it can be found later
        mt.name = "EMPTY FOR CAMERA ORBIT"

        # Hide the orbit in the viewport
        mt.hide_set(True)

    if "ORBITING CAMERA" in bpy.data.objects.keys():
        cam2 = bpy.data.objects["ORBITING CAMERA"]
    else:
        # Copy other camera
        cam2 = cam.copy()
        cam2.name = "ORBITING CAMERA"

        # Set parent of cam2 to the created empty, so that if the empty rotates,
        # the camera does as well
        cam2.parent = mt

        context.collection.objects.link(cam2)

    # Set active scene to camera2
    if set_active:
        scene.camera = cam2

    # Add keyframes for animation and rotation
    # end_frame + 1 so that the final keyframe is also created
    frames = np.arange(1, end_frame + 1, step=FRAME_STEP)

    driver = mt.driver_add("rotation_euler", 2).driver
    driver.expression = f"2 * pi * (frame - 1) / {max(frames)}"

    for f in frames:
        mt.keyframe_insert("rotation_euler", index=2, frame=f)


def set_frame_step(frame_step: int) -> None:
    bpy.context.scene.frame_step = frame_step


def set_frame_end(frame_end: int) -> None:
    bpy.context.scene.frame_end = frame_end
