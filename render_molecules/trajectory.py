from __future__ import annotations

from copy import deepcopy

import bpy
import numpy as np

from .atom import Atom
from .blender_utils import get_object_by_name
from .geometry import Geometry
from .other_utils import (find_all_string_in_list_of_strings,
                          find_first_string_in_list_of_strings)
from .structure import Structure


class Trajectory(Geometry):
    def __init__(self, frames: list[Structure]):
        self._nframes = len(frames)
        self._frames = frames

    def get_frames(self) -> list[Structure]:
        """Get all frames in the trajectory"""
        return self._frames

    def get_nframes(self) -> int:
        """Get the number of frames in the trajectory"""
        return self._nframes

    def set_center_of_mass(
        self, new_center_of_mass: np.ndarray, frame_index: int = 0
    ) -> None:
        """Set the Center Of Mass (COM) of the structure at index 'frameIndex' to a new position

        Args:
            newCOMposition (np.ndarray): new COM position [x, y, z], in Angstrom
            frameIndex (int): frame index to position according to COM
        """
        original_com = self._frames[frame_index].get_center_of_mass()
        displacement = new_center_of_mass - original_com
        self.translate(displacement)

    def create_animation(
        self,
        create_bonds: bool = True,
        resolution: str = "medium",
        split_bond_to_atom_materials: bool = True,
    ) -> None:
        """Create the animation of the trajectory in the Blender scene"""
        # At the moment, this looks a bit weird if bond lengths change a lot.
        # Also, does not support bond creation/destruction.
        frame_step = 10
        bpy.context.scene.frame_step = frame_step
        bpy.context.scene.frame_end = 1 + frame_step * (self._nframes - 1)

        initial_frame = self._frames[0]

        initial_frame.create_atoms(create_mesh=False, resolution=resolution)

        initial_bonds = initial_frame.find_bonds_from_distances()
        initial_frame.create_bonds(
            initial_bonds,
            split_bond_to_atom_materials=split_bond_to_atom_materials,
            resolution=resolution,
        )

        all_elements = [atom.get_element() for atom in initial_frame.get_atoms()]
        all_element_indices = [
            all_elements[:i].count(all_elements[i]) for i in range(len(all_elements))
        ]

        starting_bond_lengths = [bond.get_length() for bond in initial_bonds]

        for i, frame in enumerate(self._frames):
            current_frame_nr = 1 + i * frame_step
            current_positions = frame.get_atom_positions()
            for j in range(len(all_elements)):
                object_name = f"atom-{all_elements[j]}"

                if all_element_indices[j] > 0:
                    object_name += "." + str(all_element_indices[j]).rjust(3, "0")

                # Select UV sphere with correct name
                obj = get_object_by_name(object_name)

                obj.location.x = current_positions[j, 0]
                obj.location.y = current_positions[j, 1]
                obj.location.z = current_positions[j, 2]

                obj.keyframe_insert(data_path="location", frame=current_frame_nr)

            # Now reposition and rerotate all the bonds.
            if i >= 1:
                current_bonds = frame.find_bonds_from_distances()
            else:
                current_bonds = initial_bonds

            for j, bond in enumerate(current_bonds):
                atom1_index = bond.get_atom1_index()
                atom2_index = bond.get_atom2_index()
                try:
                    obj = get_object_by_name(f"bond-{atom1_index}-{atom2_index}")
                except KeyError as e:
                    # This bond did not exist yet, needs to be created
                    raise NotImplementedError(
                        "Does not support bond creation yet"
                    ) from e

                scale = bond.get_length() / starting_bond_lengths[j]

                vdw_weighted_midpoints = bond.get_vdw_weighted_cylinder_midpoints(
                    all_elements[atom1_index], all_elements[atom2_index]
                )

                obj.location.x = vdw_weighted_midpoints[0, 0]
                obj.location.y = vdw_weighted_midpoints[0, 1]
                obj.location.z = vdw_weighted_midpoints[0, 2]

                obj.scale[2] = scale
                obj.rotation_axis_angle = bond.get_axis_angle_with_z()
                obj.keyframe_insert(data_path="location", frame=current_frame_nr)
                obj.keyframe_insert(
                    data_path="rotation_axis_angle", frame=current_frame_nr
                )
                obj.keyframe_insert(data_path="scale", frame=current_frame_nr)

                if (
                    not split_bond_to_atom_materials
                    or all_elements[atom1_index] == all_elements[atom2_index]
                ):
                    continue

                try:
                    obj = get_object_by_name(f"bond-{atom1_index}-{atom2_index}.001")
                except KeyError as e:
                    # This bond did not exist yet, needs to be created
                    raise NotImplementedError(
                        "Does not support bond creation yet"
                    ) from e

                obj.location.x = vdw_weighted_midpoints[1, 0]
                obj.location.y = vdw_weighted_midpoints[1, 1]
                obj.location.z = vdw_weighted_midpoints[1, 2]

                obj.scale[2] = scale

                obj.rotation_axis_angle = bond.get_axis_angle_with_z()

                obj.keyframe_insert(data_path="location", frame=current_frame_nr)
                obj.keyframe_insert(
                    data_path="rotation_axis_angle", frame=current_frame_nr
                )
                obj.keyframe_insert(data_path="scale", frame=current_frame_nr)

    def get_frame(self, frame_index: int) -> Structure:
        """Get the Structure at a certain frame index"""
        return self._frames[frame_index]

    def rotate_around_axis(self, axis: np.ndarray, angle: float) -> None:
        """Rotate all frames in the trajectory around an axis counterclockwise with a certain angle in degrees"""
        for frame in self._frames:
            frame.rotate_around_axis(axis, angle)

    @classmethod
    def _from_orca_geom_opt(cls, filepath):
        """Generate a trajectory from an ORCA geometry optimzation output file"""
        with open(filepath) as file:
            _lines = file.readlines()

        begin_structure = find_all_string_in_list_of_strings(
            "CARTESIAN COORDINATES (ANGSTROEM)", _lines
        )
        begin_structure = [beginIndex + 2 for beginIndex in begin_structure]

        end_structure = find_all_string_in_list_of_strings(
            "CARTESIAN COORDINATES (A.U.)", _lines
        )
        end_structure = [endIndex - 2 for endIndex in end_structure]

        _nframes = len(end_structure)
        structure_lines = [
            (begin_structure[i], end_structure[i]) for i in range(_nframes)
        ]

        _frames = [0] * _nframes
        for i, structure_tuple in enumerate(structure_lines):
            cartesian_coord_lines = _lines[structure_tuple[0] : structure_tuple[1]]

            _natoms = len(cartesian_coord_lines)
            _atoms = [0] * _natoms
            for j in range(_natoms):
                _atoms[j] = Atom.from_xyz_string(cartesian_coord_lines[j])
            _frames[i] = Structure(_atoms)
        return cls(_frames)

    @classmethod
    def from_xyz(cls, filepath: str):
        """Generate a trajectory from an XYZ file containing multiple frames"""
        with open(filepath) as file:
            _lines = file.readlines()

        _frames = []
        for i, line in enumerate(_lines):
            line = line.strip()
            try:
                # If the line is just an integer, it is the amount of atoms in that frame
                _natoms = int(line)
            except ValueError:
                # If it is not, we skip the line
                continue

            _atoms = [0] * _natoms
            for j in range(_natoms):
                _atoms[j] = Atom.from_xyz_string(_lines[i + 2 + j])
            _frames.append(Structure(_atoms))
        return cls(_frames)

    @classmethod
    def from_xsf(cls, filepath: str):
        raise NotImplementedError()

    @classmethod
    def from_orca(
        cls,
        filepath: str,
        use_vibrations: bool = True,
        use_geometry_optimization: bool = False,
        vibration_nr: str | int = "imag",
        n_frames_per_oscillation: int = 20,
        amplitude: float = 0.5,
    ):
        """Get a Trajectory from an ORCA output file.

        Args:
            filepath (str): filepath of ORCA output file
            use_vibrations (bool): whether to use the vibrations
            use_geometry_optimization (bool):
            vibration_nr (str | int): vibration number index, or string 
                (then has to be one of ``['i', 'im', 'imag', 'imaginary']``). Vibration number index
                is 0-indexed (i.e. 0 is the first vibrational mode, 1 is the second, etc.)
            n_frames_per_oscillation (int): number of keyframes per full vibrational oscillation (phase= :math:`2\\pi`)
            amplitude (float): amplitude of vibrations. 0.5 seems to be a good default value.

        Notes:
            * Tries to find imaginary mode from the output. If multiple imaginary modes are found, 
              will raise an error. Please investigate the output file manually to see which one
              you want to visualize, and give the integer index to this function instead.
            * Calculates the vibrational trajectory from the normal modes. ORCA outputs the translational
              and rotational modes as vibrations too, meaning that often the first 6 vibrations
              (so up to and including ``vibration_nr=5``) have no displacements, 
              and the resulting Trajectory will show no movement.
        """
        # Get trajectory corresponding to either vibrations of the normal modes or the geometry optimization
        if use_vibrations == use_geometry_optimization:
            raise ValueError(
                f"One (and not two) of useVibrations and useGeometryOptimzation must be True, but both were {use_vibrations}"
            )
        if use_geometry_optimization:
            return cls._from_orca_geom_opt(filepath)

        with open(filepath) as file:
            _lines = file.readlines()

        result_frequencies = find_first_string_in_list_of_strings(
            "VIBRATIONAL FREQUENCIES", _lines
        )
        if result_frequencies is None:
            raise ValueError("No 'VIBRATIONAL FREQUENCIES' found in ORCA output")

        result_normal = find_first_string_in_list_of_strings("NORMAL MODES", _lines)
        if result_normal is None:
            raise ValueError("No 'NORMAL MODES' found in ORCA output")

        # resultIRspec = findFirstStringInListOfStrings(
        #     "IR SPECTRUM", _lines, start=result_normal
        # )

        frequencies_lines = _lines[result_frequencies + 5 : result_normal - 3]
        nfreqs = len(frequencies_lines)

        if isinstance(vibration_nr, str):
            # Try to infer which vibration is the imaginary one, and then give trajectory corresponding to that one.
            if vibration_nr not in ["i", "im", "imag", "imaginary"]:
                raise ValueError(
                    f"If vibrationNr is a string, it should be one of ['i', 'im', 'imag', imaginary'], but was '{vibration_nr}'"
                )
            result_imaginary = find_all_string_in_list_of_strings(
                r"***imaginary mode***", _lines, start=result_normal
            )

            if not result_imaginary:
                raise ValueError(
                    "Tried to visualize imaginary normal mode, but no imaginary frequency found"
                )

            if len(result_imaginary) > 1:
                raise ValueError(
                    "Tried to visualize imaginary mode, but multiple imaginary frequencies found. This code cannot decide which to use"
                )
            vibration_nr = int(_lines[result_imaginary[0]].split()[0].strip(":"))
            print(f"Imaginary vibration was found to be vibrationNr {vibration_nr}")
        elif not isinstance(vibration_nr, int):
            raise TypeError(
                f"vibrationNr should be of type int or string, but was type {type(vibration_nr)}"
            )

        if vibration_nr > nfreqs - 1:
            raise ValueError(
                f"Tried to visualize vibrationNr {vibration_nr}, but only {nfreqs} vibrations found"
            )

        frequency = float(frequencies_lines[vibration_nr].split()[1])
        if frequency == 0.0:
            print(
                "Warning: Trying to visualize normal mode with frequency 0. Amplitudes will be 0 too so animation shows no vibrations"
            )

        n_atoms_result = find_first_string_in_list_of_strings("Number of atoms", _lines)
        natoms = int(_lines[n_atoms_result].split()[-1])

        # Read displacements. ORCA has the displacements as rows, and frequencies as columns
        # Each "block" has 6 columns (or less if it's the last block), and 3*natoms rows.
        n_lines_amplitudes = 3 * natoms
        block_nr = vibration_nr // 6
        col_nr = vibration_nr % 6

        # Get the lines corresponding to the correct vibration
        lines_modes = _lines[
            result_normal + 8 + block_nr * (n_lines_amplitudes + 1) : result_normal
            + 8
            + (block_nr + 1) * (n_lines_amplitudes + 1)
            - 1
        ]
        assert len(lines_modes) == n_lines_amplitudes, (
            f"NOT CORRECT LENGTH. SHOULD HAVE BEEN 3*natoms={n_lines_amplitudes}, but was {len(lines_modes)}"
        )
        # Read correct column, and reshape displacements to natoms*3 matrix for x, y, z
        mass_weighed_displacements = np.array(
            [float(i.split()[col_nr + 1]) for i in lines_modes]
        ).reshape(natoms, 3)

        begin_structure = find_all_string_in_list_of_strings(
            "CARTESIAN COORDINATES (ANGSTROEM)", _lines, end=result_frequencies
        )
        if begin_structure is None:
            raise ValueError()

        # In case there was a geometry optimization, use the final (optimized) structure
        final_structure_line = begin_structure[-1]
        base_structure = Structure(
            [
                Atom.from_xyz_string(line)
                for line in _lines[
                    final_structure_line + 2 : final_structure_line + 2 + natoms
                ]
            ]
        )

        # The displacements in the ORCA output are mass-weighted (i.e. scaled by 1/sqrt(mass))
        # so if you want you can use these, but generally it looks better to use the mass-weighted ones.
        # masses = np.array([atom.get_mass() for atom in baseStructure.get_atoms()])
        # displacements = displacementsMassWeighed * np.sqrt(masses)[:, np.newaxis]

        frames = [deepcopy(base_structure) for i in range(n_frames_per_oscillation)]
        # Displace atoms in structure according to phase.
        # We do not need to loop over the first and last frame (i.e. phase = 0 and 2*pi),
        # since the displacement will be np.sin(0)=np.sin(2pi)=0,
        # so the baseStructure is unaltered.
        for i in range(1, n_frames_per_oscillation - 1):
            phase = i * 2.0 * np.pi / (n_frames_per_oscillation - 1)
            frames[i].displace_atoms(
                np.sin(phase) * amplitude * mass_weighed_displacements
            )
        return cls(frames)

    def translate(self, translation_vector: np.ndarray | list) -> None:
        for frame in self._frames:
            frame.translate(translation_vector)
