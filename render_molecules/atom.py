"""
The Atom class is a useful way of keeping track of properties of an atom, such as its atomic number, charge and position
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .constants import BOHR_TO_ANGSTROM
from .element_data import (
    element_mass,
    get_atomic_number_from_element,
    get_element_from_atomic_number,
    vdw_radii,
)

if TYPE_CHECKING:
    from .structure import Structure


class Atom:
    def __init__(
        self,
        atomic_number: int,
        element: str,
        charge: float,
        x: float,
        y: float,
        z: float,
        is_angstrom: bool,
    ):
        """
        Args:
            atomicNumber (int): atomic number. For example, for iron atom, 26
            element (str): element. For example, for iron atom, Fe
            charge (float): charge of atom
            x (float): x-coordinate
            y (float): y-coordinate
            z (float): z-coordinate
            isAngstrom (bool): whether coordinates are in Angstrom. If False, assume in Bohr
        """
        self._atomic_number = atomic_number
        self._element = element
        self._charge = charge
        self._x = x
        self._y = y
        self._z = z
        self._position = np.array([self._x, self._y, self._z])
        self.is_angstrom = is_angstrom

        try:
            self._mass = element_mass[self._atomic_number - 1]
        except ValueError:
            msg = f"Could not determine mass of Atom with atomic number {self._atomic_number}"
            print(msg)
            self._mass = -1

        try:
            self._vdw_radius = vdw_radii[self._atomic_number - 1]
        except ValueError:
            msg = f"Could not determine Van der Waals radius of Atom with atomic number {self._atomic_number}"
            print(msg)
            self._vdw_radius = -1.0

    @classmethod
    def from_cube_string(cls, string: str):
        """Create Atom instance from a line in a CUBE file

        Args:
            string (str): line in CUBE file"""
        split_string = string.split()
        atomic_number = int(split_string[0])
        element = get_element_from_atomic_number(atomic_number)
        charge, x, y, z = (float(field) for field in split_string[1:])
        is_angstrom = False  # Bohr by default
        return cls(atomic_number, element, charge, x, y, z, is_angstrom)

    @classmethod
    def from_xyz_string(cls, string: str):
        """Create Atom instance from a line in an XYZ file

        Args:
            string (str): line from an XYZ file. Formatted as A x y z ..., where A is either atomic number or element string
        """
        split_string = string.split()
        element = split_string[0].strip()
        try:
            atomic_number = int(element)
            element = get_element_from_atomic_number(atomic_number)
        except ValueError:
            atomic_number = get_atomic_number_from_element(element)
        x, y, z = (float(field) for field in split_string[1:4])
        is_angstrom = True  # Angstrom by default
        return cls(atomic_number, element, "UNKNOWN", x, y, z, is_angstrom)

    @classmethod
    def from_sdf_string(cls, string: str):
        """Create Atom instance from a line in an SDF file

        Args:
            string (str): line from an SDF file. Formatted as x y z A, where A is an element string.
        """
        split_string = string.split()
        element = split_string[3].strip()
        atomic_number = get_atomic_number_from_element(element)
        x, y, z = (float(field) for field in split_string[:3])
        is_angstrom = True  # SDF is in Angstrom
        return cls(atomic_number, element, "UNKNOWN", x, y, z, is_angstrom)

    def get_atomic_number(self) -> int:
        """Get the atomic number of the atom

        Returns:
            int: atomic number"""
        return self._atomic_number

    def get_charge(self) -> float:
        """Get the charge of the Atom (undefined if created from XYZ file)

        Returns:
            float: charge of Atom"""
        return self._charge

    def get_x(self) -> float:
        """Get the x-coordinate of the atom

        Returns:
            float: x-coordinate of Atom
        """
        return self._x

    def get_y(self) -> float:
        """Get the y-coordinate of the atom

        Returns:
            float: y-coordinate of Atom
        """
        return self._y

    def get_z(self) -> float:
        """Get the z-coordinate of the atom

        Returns:
            float: z-coordinate of Atom
        """
        return self._z

    def get_position(self) -> np.ndarray[float]:
        """Get position of the atom

        Returns:
            ndarray: array with x, y and z coordinates of Atom"""
        return self._position

    def position_bohr_to_angstrom(self) -> None:
        """Convert the position vector from Bohr to Angstrom"""
        if self.is_angstrom:
            raise ValueError()
        self.is_angstrom = True
        self.set_position(self._position * BOHR_TO_ANGSTROM)

    def set_position(self, new_position) -> None:
        """Set the position of the atom to a new position

        Args:
            newPosition (ndarray): x, y and z coordinates of new position
        """
        self._position = new_position
        self._x, self._y, self._z = new_position

    def get_element(self) -> str:
        """Get the element of the atom

        Returns:
            str: element of Atom
        """
        return self._element

    def get_mass(self) -> float:
        """Get the mass of the atom

        Returns:
            float: mass of Atom"""
        return self._mass

    def get_vdw_radius(self) -> float:
        """Get the Van der Waals radius of the atom

        Returns:
            float: Van der Waals radius of the Atom"""
        return self._vdw_radius

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Atom with atomic number {self._atomic_number} at position {self._position}"

    def find_bound_atoms(self, structure: Structure) -> list[int]:
        """Find which Atom indeces are bound to this Atom in the structure"""
        bound_atom_indeces = []
        for bond in structure.get_bonds():
            atom1_pos = bond.get_atom1_position()
            atom2_pos = bond.get_atom2_position()
            if np.all(self._position == atom1_pos):
                bound_atom_indeces.append(bond.get_atom2_index())
            elif np.all(self._position == atom2_pos):
                bound_atom_indeces.append(bond.get_atom1_index())
        return bound_atom_indeces
