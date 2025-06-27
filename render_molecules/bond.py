from __future__ import annotations

import numpy as np

from .constants import (CYLINDER_LENGTH_FRACTION,
                        CYLINDER_LENGTH_FRACTION_SPLIT, SPHERE_SCALE)
from .element_data import element_list, vdw_radii
from .geometry import check_3d_vector


class Bond:
    def __init__(
        self,
        atom1_index: int,
        atom2_index: int,
        bond_type: str,
        bond_length: float,
        interatomic_vector: np.ndarray,
        midpoint_position: np.ndarray,
        atom1_and_2_pos: np.ndarray,
        name: str,
    ):
        self._atom1_index, self._atom2_index = atom1_index, atom2_index
        self._bond_type = bond_type
        self._bond_length = bond_length
        self._interatomic_vector = interatomic_vector
        self.set_midpoint(midpoint_position)
        self._atom1_pos = atom1_and_2_pos[0]
        self._atom2_pos = atom1_and_2_pos[1]
        self.set_name(name)

    def get_atom1_index(self) -> int:
        """Get the index of the first atom that is connected to this bond"""
        return self._atom1_index

    def get_atom2_index(self) -> int:
        """Get the index of the second atom that is connected to this bond"""
        return self._atom2_index

    def get_bond_type(self) -> str:
        """Get the bond type (the two connecting elements in alphabetical order)"""
        return self._bond_type

    def get_length(self) -> float:
        """Get the bond length"""
        return self._bond_length

    def get_interatomic_vector(self) -> np.ndarray[float]:
        """Get the vector connecting the two atoms"""
        return self._interatomic_vector

    def get_midpoint(self) -> np.ndarray[float]:
        """Get the midpoint position of the two atoms"""
        return self._midpointPosition

    def set_midpoint(self, midpoint_position: np.ndarray) -> None:
        """Set the midpoint position of the two atoms"""
        midpoint_position = check_3d_vector(midpoint_position)
        self._midpointPosition = midpoint_position

    def get_direction(self) -> np.ndarray[float]:
        """Get the unit vector in the direction of the bond"""
        return self._interatomic_vector / self._bond_length

    def get_atom1_position(self) -> np.ndarray[float]:
        """Get position of atom 1"""
        return self._atom1_pos

    def get_atom2_position(self) -> np.ndarray[float]:
        """Get position of atom 2"""
        return self._atom2_pos

    def get_name(self) -> str:
        return self._name

    def set_name(self, name) -> None:
        if not isinstance(name, str):
            raise TypeError(f"name was supposed to be string but was type {type(name)}")
        self._name = name

    def get_axis_angle_with_z(self) -> tuple[float, float, float, float]:
        """Get the axis angle such that a created cylinder in the direction of the bond"""
        z = np.array([0, 0, 1])
        axis = np.cross(z, self._interatomic_vector)
        if np.linalg.norm(axis) < 1e-5:
            axis = np.array([0, 0, 1])
            angle = 0.0
        else:
            axis /= np.linalg.norm(axis)
            angle = np.arccos(np.dot(self._interatomic_vector, z) / self._bond_length)
        return angle, axis[0], axis[1], axis[2]

    def get_vdw_weighted_cylinder_midpoints(
        self, element1: str, element2: str
    ) -> np.ndarray:
        """Get the Van der Waals-radii weighted location where both cylinders need to be placed"""
        vdw_midpoint = self.get_vdw_weighted_midpoint(element1, element2)

        loc1 = (
            vdw_midpoint
            - self.get_direction() * self._bond_length * CYLINDER_LENGTH_FRACTION_SPLIT
        )
        loc2 = (
            vdw_midpoint
            + self.get_direction() * self._bond_length * CYLINDER_LENGTH_FRACTION_SPLIT
        )
        return np.array([loc1, loc2])

    def get_vdw_weighted_midpoint(self, element1: str, element2: str) -> np.ndarray:
        """Get the Van der Waals-radii weighted bond-midpoints"""
        element1_index = element_list.index(element1)
        vdw_radius1 = vdw_radii[element1_index]

        element2_index = element_list.index(element2)
        vdw_radius2 = vdw_radii[element2_index]

        # sum_vdw_radii = vdw_radius1 + vdw_radius2
        # fraction_vdw_radius1 = vdw_radius1 / sum_vdw_radii
        # fraction_vdw_radius2 = vdw_radius2 / sum_vdw_radii

        # Starting from the first atom (let's call it atom A),
        # the VdW weighed midpoint is (where the two cylinders meet)
        # position_A + (r_AB + r_A - rB)/2 * bondDirectionVector
        return (
            self._atom1_pos
            + (
                self._bond_length
                + vdw_radius1 * SPHERE_SCALE
                - vdw_radius2 * SPHERE_SCALE
            )
            / 2
            * -self.get_direction()
        )
