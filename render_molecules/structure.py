from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import bpy
import numpy as np

from .atom import Atom
from .blender_utils import (create_cylinder, create_material,
                            create_mesh_of_atoms, create_uv_sphere,
                            deselect_all_selected, get_object_by_name,
                            join_cylinders, put_cap_on_cylinder)
from .bond import Bond
from .constants import (AMU_TO_KG, ANGSTROM_TO_METERS, BOHR_TO_ANGSTROM,
                        BOHR_TO_METERS, CYLINDER_LENGTH_FRACTION,
                        CYLINDER_LENGTH_FRACTION_SPLIT)
from .element_data import bond_lengths, manifest
from .geometry import Geometry, angle_between, check_3d_vector, rotation_matrix
from .other_utils import (find_all_string_in_list_of_strings,
                          find_first_string_in_list_of_strings)


class Structure(Geometry):
    def __init__(self, atoms: list[Atom], bonds: list[Bond] | None = None):
        self._natoms = len(atoms)
        self._atoms = atoms

        if bonds is None:
            self._bonds = []
        else:
            self._bonds = bonds

        self._displacements = []
        self._affine_matrix = np.identity(4)

    def get_atoms(self) -> list[Atom]:
        """Get a list of all atoms in the structure"""
        return self._atoms

    def get_atom_positions(self) -> list[np.ndarray]:
        """Get a list of all atom positions"""
        return np.array([atom.get_position() for atom in self._atoms])

    def create_atoms(
        self,
        resolution: str = "medium",
        create_mesh: bool = True,
        force_material_creation: bool = False,
        atom_colors: dict | None = None,
    ) -> None:
        """Create the atoms in the scene

        Args:
            resolution (str): resolution of created spheres.
                One of ['verylow', 'low', 'medium', 'high', 'veryhigh']
            create_mesh (bool): create mesh of vertices. saves memory, but atom positions cannot be animated
            force_material_creation (bool): force creation of new materials with element names,
                even though materials with that name already exist. This is useful for if you want to
                change the atom colors
            atom_colors (dict): dictionary of atom colors, with keys elements and values hex-codes.
                If None, use the ``element_data.manifest['atom_colors']``. Can also be partially filled,
                e.g. only contain ``{'H': 'FFFFFF'}`` for H2O, and then the color of O atoms
                will be filled by the values in ``element_data.manifest['atom_colors']``.
        """

        atom_colors = self._check_atom_colors(atom_colors=atom_colors)

        if not create_mesh:
            # This is an old, naive method where we create a lot more spheres
            # It is still necessary for animations, since we cannot move vertex instances
            for atom in self._atoms:
                obj = create_uv_sphere(
                    atom.get_element(),
                    atom.get_position(),
                    resolution=resolution,
                )
                mat = create_material(
                    atom.get_element(),
                    atom_colors[atom.get_element()],
                    force=force_material_creation,
                )
                obj.data.materials.append(mat)
            return

        # Create a dictionary, with keys the atom element, and values a list of
        # all positions of atoms with that element.
        atom_vertices = {}
        for atom in self._atoms:
            if atom.get_element() in atom_vertices:
                atom_vertices[atom.get_element()].append(atom.get_position())
            else:
                atom_vertices[atom.get_element()] = [atom.get_position()]

        # For each element, create a reference UV sphere at the origin
        # Then, create a mesh with vertices at the positions and using vertex instancing,
        # copy the UV sphere to each of the vertices.
        for atom_type in atom_vertices:
            obj = create_uv_sphere(
                atom_type, np.array([0, 0, 0]), resolution=resolution
            )
            mat = create_material(
                atom_type, atom_colors[atom_type], force=force_material_creation
            )
            obj.data.materials.append(mat)

            create_mesh_of_atoms(atom_vertices[atom_type], obj, atom_type)
        deselect_all_selected()

    def find_bonds_from_distances(self) -> list[Bond]:
        """Create bonds based on the geometry"""
        all_positions = self.get_atom_positions()
        all_elements = [atom.get_element() for atom in self._atoms]

        # More efficient
        all_positions_tuples = (
            np.array([v[0] for v in all_positions]),
            np.array([v[1] for v in all_positions]),
            np.array([v[2] for v in all_positions]),
        )
        x, y, z = all_positions_tuples

        # Keep track of all indices that are marked as connected, so we do not bond them again
        connecting_indices = []

        # Can we skip looping over the last atom?
        for i, atom in enumerate(self._atoms):
            central_pos = all_positions[i]
            central_element = all_elements[i]

            # All possible bond types from the central element
            bond_types = [
                "-".join(sorted([central_element, other_element]))
                for other_element in all_elements
            ]
            allowed_bond_lengths_squared = [
                bond_lengths[bond_type] ** 2 for bond_type in bond_types
            ]

            # Calculate squared distance to all other atoms
            dx = x - central_pos[0]
            dy = y - central_pos[1]
            dz = z - central_pos[2]
            dist_squared = dx * dx + dy * dy + dz * dz

            # Atoms are bonded to the central atom if the distance squared is
            # less than the allowed squared distance for the corresponding bond type
            is_bonded_to_central = np.nonzero(
                dist_squared <= allowed_bond_lengths_squared
            )[0]

            # Loop over all bonds, and create Bond instances for them
            for atom_index in is_bonded_to_central:
                if atom_index == i:
                    # Do not allow atoms to bond to themselves
                    continue
                if (i, atom_index) in connecting_indices or (
                    atom_index,
                    i,
                ) in connecting_indices:
                    # If this bond was already made, continue
                    continue
                bond_midpoint = (all_positions[i] + all_positions[atom_index]) / 2.0
                bond_length = dist_squared[atom_index] ** 0.5
                bond_vector = all_positions[i] - all_positions[atom_index]
                new_bond = Bond(
                    i,
                    atom_index,
                    bond_types[atom_index],
                    bond_length,
                    bond_vector,
                    bond_midpoint,
                    (
                        atom.get_position(),
                        self._atoms[atom_index].get_position(),
                    ),
                    f"bond-{i}-{atom_index}",
                )
                connecting_indices.append((i, atom_index))
                self._bonds.append(new_bond)
        return self._bonds

    def generate_bond_order_bond(
        self,
        bond: Bond,
        bond_order: int,
        camera_position: np.ndarray[float],
        displacement_scaler=0.2,
    ) -> list[Bond]:
        """Way to generate multiple bonds, for example in CO2 molecule double bonds, or CO triple bonds."""
        if bond_order == 1:
            return self._bonds
        index = self._bonds.index(bond)
        self._bonds.pop(index)
        bond_vector = bond.get_interatomic_vector()

        # Get a vector that is perpendicular to the plane given by the bondVector and vector between camera and bond midpoint.
        displacement_vector = np.cross(
            bond_vector, bond.get_midpoint() - camera_position
        )
        displacement_vector /= np.linalg.norm(displacement_vector)

        # If bond_order is odd, then we also have displacementMag of 0.
        if bond_order % 2 == 0:
            displacement_magnitude = -displacement_scaler / 4 * bond_order
        else:
            displacement_magnitude = -displacement_scaler / 2 * (bond_order - 1)

        # Create the bonds, and add them to self._bonds
        for i in range(bond_order):  # noqa: B007
            bond_adjusted = deepcopy(bond)
            bond_adjusted.set_midpoint(
                bond_adjusted.get_midpoint()
                + displacement_vector * displacement_magnitude
            )
            self._bonds.append(bond_adjusted)
            displacement_magnitude += displacement_scaler
        return self._bonds

    def get_bonds(self) -> list[Bond]:
        """Get all bonds in the system"""
        return self._bonds

    def get_center_of_mass(self) -> np.ndarray[float]:
        """Get the center of mass position vector"""
        masses = np.array([atom.get_mass() for atom in self._atoms])
        atom_positions = self.get_atom_positions()
        com = np.array(
            sum(masses[i] * atom_positions[i] for i in range(self._natoms))
            / sum(masses)
        )
        return com

    def set_center_of_mass(self, new_center_of_mass: np.ndarray | list) -> None:
        """Set the Center Of Mass (COM) of the whole system to a new position

        Args:
            new_center_of_mass (ndarray): new COM position
        """
        new_center_of_mass = check_3d_vector(new_center_of_mass)
        translation_vector = new_center_of_mass - self.get_center_of_mass()
        self.translate(translation_vector)

    def set_average_position(self, new_average_position: np.ndarray | list):
        """Sets the average position of all atoms to a new position"""
        new_average_position = check_3d_vector(new_average_position)
        current_average_position = np.average(
            np.array([atom.get_position() for atom in self._atoms]), axis=0
        )
        translation_vector = new_average_position - current_average_position
        self.translate(translation_vector)

    def translate(self, translation_vector: np.ndarray) -> None:
        """Translate every atom in the Structure along a vector

        Args:
            translationVector (ndarray): vector to translate every atom
        """
        translation_vector = check_3d_vector(translation_vector)

        new_affine_matrix = np.identity(4)
        new_affine_matrix[:3, 3] = translation_vector
        self.add_transformation(new_affine_matrix)

        for atom in self._atoms:
            atom.set_position(atom.get_position() + translation_vector)

    def get_total_charge(self) -> int:
        """Get the total charge in the system

        Returns:
            total_charge (int): total charge of the system
        """
        charges = (atom.get_charge() for atom in self._atoms)
        if not all(isinstance(charge, int | float) for charge in charges):
            msg = "The charges of atoms are not all of type 'int' or 'float'"
            raise ValueError(msg)
        total_charge = int(sum(charges))
        return total_charge

    def get_nr_electrons(self) -> int:
        """Get the total amount of electrons in the system

        Returns:
            total_electrons (int): total number of electrons in the system
        """
        total_electrons_if_neutral = sum(
            atom.get_atomic_number() for atom in self._atoms
        )
        total_electrons = total_electrons_if_neutral - self.get_total_charge()
        return total_electrons

    def is_radical(self) -> bool:
        """Returns whether the studied structure is a radical (has an uneven amount of electrons)

        Returns:
            bool: whether the system is a radical or not
        """
        return self.get_nr_electrons() % 2 != 0

    def rotate_around_axis(self, axis: np.ndarray, angle: float) -> None:
        """Rotate the structure around a certain axis

        Args:
            axis (ndarray): 3D vector around which to rotate the structure
            angle (float): angle with which to rotate the structure. Given in angles, counterclockwise
        """
        rot_matrix = rotation_matrix(axis, angle)
        # create 4x4 matrix from the 3x3 rotation matrix
        new_affine_matrix = np.identity(4)
        new_affine_matrix[:3, :3] = rot_matrix
        self.add_transformation(new_affine_matrix)

        for atom in self._atoms:
            current_position = atom.get_position()
            rotated_position = np.dot(rot_matrix, current_position)
            atom.set_position(rotated_position)

    def get_inertia_tensor(self) -> np.ndarray:
        """Get the moment of inertia tensor

        Returns:
            inertiaTensor (ndarray): inertia tensor in kg m^2
        """
        # https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        center_of_mass = self.get_center_of_mass()

        inertia_tensor = np.zeros((3, 3))
        for atom in self._atoms:
            # Calculate moments of inertia to axes wrt COM
            coords = atom.get_position() - center_of_mass

            mass = atom.get_mass() * AMU_TO_KG  # Mass in kg

            # Convert coordinates to meters
            if atom.is_angstrom:
                coords *= ANGSTROM_TO_METERS
            else:
                coords *= BOHR_TO_METERS

            inertia_tensor[0, 0] += mass * (
                coords[1] * coords[1] + coords[2] * coords[2]
            )
            inertia_tensor[1, 1] += mass * (
                coords[0] * coords[0] + coords[2] * coords[2]
            )
            inertia_tensor[2, 2] += mass * (
                coords[0] * coords[0] + coords[1] * coords[1]
            )
            inertia_tensor[0, 1] -= mass * coords[0] * coords[1]
            inertia_tensor[0, 2] -= mass * coords[0] * coords[2]
            inertia_tensor[1, 2] -= mass * coords[1] * coords[2]
        inertia_tensor[1, 0] = inertia_tensor[0, 1]
        inertia_tensor[2, 0] = inertia_tensor[0, 2]
        inertia_tensor[2, 1] = inertia_tensor[1, 2]
        return inertia_tensor

    def get_principal_moments_of_inertia(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the principal moments of inertia and the principal axes

        Returns:
            principalMoments (ndarray): array of length 3 with the three principal moments of inertia in kg m^2
            principalAxes (ndarray): matrix of shape 3x3 with three principal moment axes
        """
        inertia_tensor = self.get_inertia_tensor()
        principal_moments, principal_axes = np.linalg.eig(inertia_tensor)
        indeces = np.argsort(principal_moments)
        return principal_moments[indeces], principal_axes[indeces]

    def create_hydrogen_bonds(self) -> None:
        """Adds hydrogen bonds to each molecule"""
        hbond_forming_elements = ["H", "O", "N"]
        atoms = self._atoms

        z = np.array([0, 0, 1])

        hbond_curves = []
        for i, at1 in enumerate(atoms):
            if at1.get_element() not in hbond_forming_elements:
                # If the atom is a C, it can not form a hydrogen bond (in our system at least), so skip
                continue
            r1 = at1.get_position()
            atom1_bound_indices = at1.find_bound_atoms(self)
            for j, at2 in enumerate(atoms):
                if i == j:  # Skip same atom
                    continue
                if j in atom1_bound_indices:  # If j is bound to i, skip
                    continue
                if (
                    at2.get_element() not in hbond_forming_elements
                ):  # Skip if atom 2 cannot form hydrogen bonds
                    continue
                if (
                    at1.get_element() == at2.get_element()
                ):  # OO, HH or NN cannot form hydrogen bonds.
                    continue
                if at1.get_element() in ["C", "O", "N"] and at2.get_element() in [
                    "C",
                    "O",
                    "N",
                ]:
                    # Assume that a C, N or O atom cannot form a hydrogen bond to another C, N or O atom
                    continue
                r2 = at2.get_position()

                dist = np.linalg.norm(r2 - r1)
                if dist > manifest["hbond_max_length"]:
                    continue

                atom2_bound_indices = at2.find_bound_atoms(self)

                if at2.get_element() == "H":
                    # Use some boolean arithmetic to find the position of the O/C/N that the H is bonded to
                    bonded_atom_position = atoms[atom2_bound_indices[0]].get_position()

                    # Calculate intramolecular vector
                    intramol_vector = bonded_atom_position - r2
                elif at1.get_element() == "H":
                    bonded_atom_position = atoms[atom1_bound_indices[0]].get_position()

                    # Calculate intramolecular vector
                    intramol_vector = bonded_atom_position - r1
                else:
                    raise NotImplementedError()

                angle = angle_between(intramol_vector, r2 - r1)

                # create a hydrogen bond when the interatomic distance and O-H----O angle are less than the specified threshold value
                if np.abs(angle) > 180 - manifest["hbond_max_angle"]:
                    axis = np.cross(z, r2 - r1)
                    if np.linalg.norm(axis) < 1e-5:
                        axis = np.array([0, 0, 1])
                        angle = 0.0
                    else:
                        axis /= np.linalg.norm(axis)
                        angle = np.arccos(np.dot(r2 - r1, z) / dist)

                    bpy.ops.curve.primitive_nurbs_path_add(
                        enter_editmode=False,
                        align="WORLD",
                        location=tuple((r1 + 0.35 * r2) / 1.35),
                    )

                    obj = bpy.context.view_layer.objects.active
                    obj.scale = (
                        manifest["hbond_thickness"],
                        manifest["hbond_thickness"],
                        dist * 2.2,
                    )
                    obj.rotation_mode = "AXIS_ANGLE"
                    obj.rotation_axis_angle = (angle, axis[0], axis[1], axis[2])

                    obj.name = "Hbond-%s-%03i-%s-%03i" % (
                        at1.get_element(),
                        i,
                        at2.get_element(),
                        j,
                    )
                    hbond_curves.append(obj)

        hbond_material = create_material("H-bond", manifest["hbond_color"])

        for o in hbond_curves:
            rot_axis = o.rotation_axis_angle
            bpy.ops.surface.primitive_nurbs_surface_cylinder_add(
                enter_editmode=False,
                align="WORLD",
                location=o.location,
            )
            obj = bpy.context.view_layer.objects.active
            obj.name = "Hbond_cyl"

            obj.scale = (manifest["hbond_thickness"], manifest["hbond_thickness"], 0.1)
            obj.data.materials.append(hbond_material)

            obj.rotation_mode = "AXIS_ANGLE"
            obj.rotation_axis_angle = rot_axis

            _ = obj.modifiers.new(name="FollowCurve", type="ARRAY")
            bpy.context.object.modifiers["FollowCurve"].fit_type = "FIT_CURVE"
            bpy.context.object.modifiers["FollowCurve"].curve = o
            bpy.context.object.modifiers["FollowCurve"].relative_offset_displace[0] = 0
            bpy.context.object.modifiers["FollowCurve"].relative_offset_displace[1] = 0
            bpy.context.object.modifiers["FollowCurve"].relative_offset_displace[2] = (
                1.3
            )

    def _check_atom_colors(self, atom_colors: dict | None = None):
        element_types = list(set([atom.get_element() for atom in self._atoms]))
        if atom_colors is None:
            atom_colors = manifest["atom_colors"]

            for element_type in element_types:
                if not element_type in atom_colors:
                    msg = f"element_data.manifest['atom_colors'] dictionary needs to contain all elements in Structure, but did not contain element {element_type}."
                    msg += f"Alternatively, pass a custom atom_colors dictionary that contains all the elements in the Structure."
                    raise ValueError(msg)
        else:
            for element_type in element_types:
                if element_type in atom_colors:
                    continue
                if element_type in manifest["atom_colors"]:
                    atom_colors[element_type] = manifest["atom_colors"][element_type]
                else:
                    msg = f"atom_colors dictionary needs to contain all elements in Structure, but did not contain element {element_type}."
                    msg += f"The element was also not found in element_data.manifest['atom_colors'] dictionary, so could not fill it."
                    raise ValueError(msg)

        return atom_colors

    def create_bonds(
        self,
        bonds: list[Bond],
        split_bond_to_atom_materials: bool = True,
        resolution: str = "medium",
        atom_colors: dict | None = None,
        force_material_creation: bool = False,
    ) -> None:
        """Create the bonds in the Blender scene

        Args:
            bonds (list[Bond]): list of bonds to be drawn
            split_bond_to_atom_materials (bool): whether to split up the bonds to the two atom materials connecting them
            resolution (str): render resolution. One of ['verylow', 'low', 'medium', 'high', 'veryhigh']
            atom_colors (dict): dictionary of atom colors, with keys elements and values hex-codes.
                If None, use the ``element_data.manifest['atom_colors']``. Can also be partially filled,
                e.g. only contain ``{'H': 'FFFFFF'}`` for H2O, and then the color of O atoms
                will be filled by the values in ``element_data.manifest['atom_colors']``.
            force_material_creation (bool): force creation of new materials with element names,
                even though materials with that name already exist. This is useful for if you want to
                change the atom colors
        """
        atom_colors = self._check_atom_colors(atom_colors=atom_colors)

        all_elements = [atom.get_element() for atom in self._atoms]

        for bond in bonds:
            axis_angle_with_z = bond.get_axis_angle_with_z()
            bond_length = bond.get_length()
            bond_midpoint = bond.get_midpoint()

            if split_bond_to_atom_materials:
                # We will move the two cylinders according to their vdw radii, such that each atom
                # has about the same of its material shown in the bond. This means that for, e.g.
                # an O-H bond, the cylinder closer to O will move less than the cylinder closer to H
                atom1_index = bond.get_atom1_index()
                atom1_element = all_elements[atom1_index]

                atom2_index = bond.get_atom2_index()
                atom2_element = all_elements[atom2_index]

                if atom1_element == atom2_element:
                    mat1 = create_material(
                        atom1_element,
                        atom_colors[atom1_element],
                        force=force_material_creation,
                    )
                    obj = create_cylinder(
                        bond_midpoint,
                        axis_angle_with_z,
                        manifest["bond_thickness"],
                        bond_length * CYLINDER_LENGTH_FRACTION_SPLIT,
                        resolution=resolution,
                        name=f"bond-{atom1_index}-{atom2_index}",
                    )
                    obj.data.materials.append(mat1)
                    continue

                vdw_weighted_midpoints = bond.get_vdw_weighted_cylinder_midpoints(
                    atom1_element, atom2_element
                )

                # Because of how we calculate the bonds, the first cylinder (where we subtract the direction
                # from the midpoint) will be the one closest to the atom with the higher index.
                # So, we take the element and material from that one, and assign it to the first cylinder.
                mat2 = create_material(atom2_element, atom_colors[atom2_element])

                # First cylinder
                obj = create_cylinder(
                    vdw_weighted_midpoints[0],
                    axis_angle_with_z,
                    manifest["bond_thickness"],
                    bond_length * CYLINDER_LENGTH_FRACTION_SPLIT,
                    resolution=resolution,
                    name=f"bond-{atom1_index}-{atom2_index}",
                )
                obj.data.materials.append(mat2)

                mat1 = create_material(
                    atom1_element,
                    atom_colors[atom1_element],
                    force=force_material_creation,
                )

                # First cylinder
                obj = create_cylinder(
                    vdw_weighted_midpoints[1],
                    axis_angle_with_z,
                    manifest["bond_thickness"],
                    bond_length * CYLINDER_LENGTH_FRACTION_SPLIT,
                    resolution=resolution,
                    name=f"bond-{atom1_index}-{atom2_index}",
                )
                obj.data.materials.append(mat1)
            else:
                obj = create_cylinder(
                    bond_midpoint,
                    axis_angle_with_z,
                    manifest["bond_thickness"],
                    bond_length * CYLINDER_LENGTH_FRACTION,
                    resolution=resolution,
                    name=f"bond-{atom1_index}-{atom2_index}",
                )
        deselect_all_selected()

    def create_structure(
        self,
        resolution: str = "medium",
        create_mesh: bool = True,
        atom_colors: dict | None = None,
        split_bond_to_atom_materials: bool = True,
        force_material_creation: bool = False,
    ) -> None:
        """Create the atoms and bond in the scene

        Args:
            resolution (str): resolution of created spheres.
                One of ['verylow', 'low', 'medium', 'high', 'veryhigh']
            create_mesh (bool): create mesh of vertices. saves memory, but atom positions cannot be animated
            force_material_creation (bool): force creation of new materials with element names,
                even though materials with that name already exist. This is useful for if you want to
                change the atom colors
            atom_colors (dict): dictionary of atom colors, with keys elements and values hex-codes.
                If None, use the ``element_data.manifest['atom_colors']``. Can also be partially filled,
                e.g. only contain ``{'H': 'FFFFFF'}`` for H2O, and then the color of O atoms
                will be filled by the values in ``element_data.manifest['atom_colors']``.
            split_bond_to_atom_materials (bool): whether to split up the bonds to the two atom materials connecting them
            force_material_creation (bool): force creation of new materials with element names,
                even though materials with that name already exist. This is useful for if you want to
                change the atom colors
        """
        self.create_atoms(
            resolution=resolution,
            create_mesh=create_mesh,
            atom_colors=atom_colors,
            force_material_creation=force_material_creation,
        )

        bonds = self.find_bonds_from_distances()

        self.create_bonds(
            bonds,
            split_bond_to_atom_materials=split_bond_to_atom_materials,
            resolution=resolution,
            atom_colors=atom_colors,
            force_material_creation=force_material_creation,
        )

    def join_bonds(self):
        """Join bonds. DOES NOT WORK YET"""
        for atom_index, atom in enumerate(self._atoms):
            bonds_to_join = []
            for bond in self._bonds:
                if (
                    bond.get_atom1_index() == atom_index
                    or bond.get_atom2_index() == atom_index
                ):
                    bonds_to_join.append(get_object_by_name(bond.getName()))
            if not bonds_to_join:
                continue
            if len(bonds_to_join) == 1:
                # This also needs to extend the cylinder somehow to the atom position
                # Maybe by creating a second, very narrow cylinder at the atom position,
                # joining that with the original cylinder and then putting the cap on that?
                put_cap_on_cylinder(bonds_to_join[0])
            else:
                join_cylinders(
                    bonds_to_join, atom.get_position(), atom.get_vdw_radius()
                )
                return

    def displace_atoms(self, displacements: np.ndarray) -> None:
        """Displace all atoms along different displacement vectors

        Args:
            displacements (ndarray): matrix of shape natoms * 3, vectors to displace atoms
        """
        if not np.shape(displacements) == (self._natoms, 3):
            raise ValueError()

        for i, atom in enumerate(self._atoms):
            atom.set_position(atom.get_position() + displacements[i, :])

    @classmethod
    def from_xyz(cls, filepath: str | Path, index: int = -1):
        """Create a Structure from an XYZ file

        Args:
            filepath (str): XYZ file to read
            index (int): index to read. Default: -1, last. TODO: Implement
        """
        with open(filepath) as file:
            _lines = file.readlines()

        _natoms = int(_lines[0].strip())
        _atoms = [0] * _natoms

        for i in range(_natoms):
            _atoms[i] = Atom.from_xyz_string(_lines[2 + i])

        return cls(_atoms)

    @classmethod
    def from_orca(cls, filepath: str | Path, index: int = -1):
        """Create a structure from an ORCA output file. Reads the
        cartesian coordinates in the input.

        Args:
            filepath (str | Path): orca output file to read
        """

        with open(filepath) as file:
            _lines = file.readlines()

        geometry_blocks = find_all_string_in_list_of_strings(
            "CARTESIAN COORDINATES (ANGSTROEM)", _lines
        )
        geometry_block = geometry_blocks[index]

        _natoms = int(
            _lines[
                find_first_string_in_list_of_strings("Number of atoms", _lines)
            ].split()[-1]
        )
        _atoms = [0] * _natoms

        for i in range(_natoms):
            _atoms[i] = Atom.from_xyz_string(_lines[geometry_block + 2 + i])

        return cls(_atoms)

    @classmethod
    def from_sdf(cls, filepath: str):
        """Creates a Structure from an SDF file

        Args:
            filepath (str | Path): SDF file to read
        """
        with open(filepath) as file:
            _lines = file.readlines()

        _natoms = int(_lines[3].split()[0].strip())
        _atoms = [0] * _natoms

        for i in range(_natoms):
            _atoms[i] = Atom.from_sdf_string(_lines[4 + i])

        # SDF already contains connectivity, so maybe we can somehow read them and create the Bond instances?
        _bonds = []
        return cls(_atoms, _bonds)

    @classmethod
    def from_xsf(cls, filepath: str):
        """Creates a Structure from an XSF file. To be tested

        Args:
            filepath(str): XSF file to read
        """
        # http://www.xcrysden.org/doc/XSF.html#__toc__2
        with open(filepath) as file:
            _lines = file.readlines()
        joined_lines = "".join(_lines)

        if "CRYSTAL" in joined_lines:
            is_periodic = True
        if "ANIMSTEPS" in joined_lines:
            msg = "Structure class cannot read changing structures. Use Trajectory for that"
            raise TypeError(msg)

        if is_periodic:
            # Does not use that it is periodic, but just reads differently
            # (at least for now)
            for i, line in enumerate(_lines):
                if line[0] == "#":
                    continue
                if "PRIMVEC" in line:
                    primvec = np.fromstring(_lines[i + 1 : i + 4], dtype=float)
                elif "CONVVEC" in line:
                    convvec = np.fromstring(_lines[i + 1 : i + 4], dtype=float)
                elif "PRIMCOORD" in line:
                    _natoms = int(_lines[i + 1].split()[0])
                    _atoms = [0] * _natoms
                    for j in range(_natoms):
                        _atoms[j] = Atom.from_xyz_string(_lines[i + 2 + j])
        else:
            for i, line in enumerate(_lines):
                if line[0] == "#":
                    continue
                if "ATOMS" in line:
                    _atoms = []
                    j = 0
                    while _lines[i + 1 + j][0] != "#":
                        try:
                            # If we're still able to convert the first column to an integer,
                            # there's still a new atom. If not, the atom list is done
                            # and we need to stop the reading.
                            int(_lines[i + 1 + j].split()[0])
                        except ValueError:
                            break
                        _atoms.append(Atom.from_xyz_string(_lines[i + 1 + j]))
                        j += 1
        return cls(_atoms)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Structure with {self._natoms} atoms"


class CUBEfile(Structure):
    def __init__(self, filepath: str):
        with open(filepath) as file:
            self._lines = file.readlines()
        natoms = int(self._lines[2].split()[0].strip())

        if natoms < 0:
            natoms = -natoms

        atoms = [0] * natoms

        for i in range(natoms):
            atoms[i] = Atom.from_cube_string(self._lines[6 + i].strip())
            atoms[i].position_bohr_to_angstrom()

        super().__init__(atoms)

    def read_volumetric_data(self) -> None:
        """Read the volumetric data in the CUBE file"""
        self._NX, self._NY, self._NZ = (
            int(self._lines[i].split()[0].strip()) for i in [3, 4, 5]
        )
        self._volumetric_origin_vector = (
            np.array([float(i) for i in self._lines[2].split()[1:]]) * BOHR_TO_ANGSTROM
        )

        self._volumetric_axis_vectors = (
            np.array(
                [[float(i) for i in self._lines[3 + i].split()[1:]] for i in [0, 1, 2]]
            )
            * BOHR_TO_ANGSTROM
        )
        if not np.all(
            np.diag(np.diagonal(self._volumetric_axis_vectors))
            == self._volumetric_axis_vectors
        ):
            warning = "WARNING: Volumetric data axis vectors are not diagonal. Not sure if this works"
            warning += (
                f" Volumetric data axis vectors:\n{self._volumetric_axis_vectors}"
            )
            print(warning)

        try:
            self._volumetric_data = np.fromiter(
                (
                    float(num)
                    for line in self._lines[6 + self._natoms :]
                    for num in line.split()
                ),
                dtype=np.float32,
                count=-1,
            ).reshape((self._NX, self._NY, self._NZ))
        except ValueError:
            self._volumetric_data = np.fromiter(
                (
                    float(num)
                    for line in self._lines[7 + self._natoms :]
                    for num in line.split()
                ),
                dtype=np.float32,
                count=-1,
            ).reshape((self._NX, self._NY, self._NZ))

        # Old, much slower way to read the data.
        # volumetricLines = " ".join(
        #     line.strip() for line in self._lines[6 + self._natoms :]
        # ).split()

        # self._volumetricData = np.zeros((self._NX, self._NY, self._NZ))
        # for ix in range(self._NX):
        #     for iy in range(self._NY):
        #         for iz in range(self._NZ):
        #             dataIndex = ix * self._NY * self._NZ + iy * self._NZ + iz
        #             self._volumetricData[ix, iy, iz] = float(volumetricLines[dataIndex])

    def get_volumetric_origin_vector(self) -> np.ndarray[float]:
        """Get the origin vector of the volumetric data

        Returns:
            ndarray: np.ndarray of length 3 corresponding to x, y and z coordinates of volumetric origin
        """
        return self._volumetric_origin_vector

    def get_volumetric_axis_vectors(self) -> np.ndarray[float]:
        """Get the axis vectors of the volumetric data

        Returns:
            ndarray: np.ndarray of length 3 corresponding to i, j and k axis vectors of volumetric data
        """
        return self._volumetric_axis_vectors

    def get_volumetric_data(self) -> np.ndarray[float]:
        """Get the volumetric data

        Returns:
            ndarray: matrix of shape NX*NY*NZ containing volumetric data
        """
        return self._volumetric_data

    def write_ply(self, filepath: str, isovalue: float) -> None:
        """Write the volumetric data to a filepath

        Args:
            filepath (str):
            isovalue (float):
        """
        from pytessel import PyTessel

        self._check_isovalue(isovalue)

        pytessel = PyTessel()

        unit_cell = self._volumetric_axis_vectors * self._volumetric_data.shape

        # Flatten the volumetric data such that X is the fastest moving index, according to the PyTessel documentation.
        vertices, normals, indices = pytessel.marching_cubes(
            self._volumetric_data.flatten(order="F"),
            reversed(self._volumetric_data.shape),
            unit_cell.flatten(),
            isovalue,
        )

        vertices += np.diag(0.5 * unit_cell) + self._volumetric_origin_vector

        nvertices = np.shape(vertices)[0]
        vertices_4d = np.concatenate([vertices, np.ones((nvertices, 1))], axis=1)
        vertices = (self._affine_matrix @ vertices_4d.T).T[:, :3]

        pytessel.write_ply(filepath, vertices, normals, indices)

    def calculate_isosurface(
        self,
        isovalue: float,
        step_size: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Calculate the isosurface from the volumetric data and an isovalue

        Args:
            isovalue (float): value where to calculate the isosurface
            step_size (int): step size in the grid. Larger values result in coarser, but quicker, results. Default = 1

        Returns:
            vertices (ndarray): Vx3 array of floats corresponding to vertex positions
            faces (ndarray): Fx3 array of integers corresponding to vertex indices
            normals (ndarray): Vx3 array of floats corresponding to normal direction at each vertex
            values (ndarray): Vx1 array of maximum value in data close to each vertex
        """
        from skimage.measure import marching_cubes

        self._check_isovalue(isovalue)

        vertices, faces, normals, values = marching_cubes(
            self._volumetric_data,
            level=isovalue,
            spacing=np.diag(self._volumetric_axis_vectors),
            step_size=step_size,
        )

        vertices += self._volumetric_origin_vector

        nvertices = np.shape(vertices)[0]
        vertices_4d = np.concatenate([vertices, np.ones((nvertices, 1))], axis=1)
        vertices = (self._affine_matrix @ vertices_4d.T).T[:, :3]

        # Blender has opposite clockwise-ness (handedness) that skimage has, so backface culling is incorrect
        # Flip order of faces, e.g. [0 1 2] -> [0 1 2]
        faces = np.flip(faces, axis=1)

        return vertices, faces, normals, values

    def _check_isovalue(self, isovalue: float) -> None:
        """Checks whether the supplied isovalue is valid

        Args:
            isovalue (float): value where to draw the isosurface

        Raises:
            ValueError: if the supplied isovalue is below the minimum value in the volumetric data, or above the maximum
        """
        if isovalue <= np.min(self._volumetric_data):
            msg = f"Set isovalue ({isovalue}) was less than or equal to the minimum value in the volumetric data ({np.min(self._volumetric_data)}). This will result in an empty isosurface. Set a larger isovalue."
            raise ValueError(msg)
        if isovalue >= np.max(self._volumetric_data):
            msg = f"Set isovalue ({isovalue}) was more than or equal to the maximum value in the volumetric data ({np.max(self._volumetric_data)}). This will result in an empty isosurface. Set a smaller isovalue."
            raise ValueError(msg)


class JSONfile(Structure):
    """WORK IN PROGRESS"""

    def __init__(self, filepath):
        import json

        self._filepath = filepath
        with open(self._filepath) as file:
            self._lines = file.readlines()
            json_data = json.load(file)

        print(json_data)
