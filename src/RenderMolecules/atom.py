import numpy as np

from .constants import BOHR_TO_ANGSTROM, KGM2_TO_AMU_ANGSTROM2
from .ElementData import (
    elementMass,
    getAtomicNumberFromElement,
    getElementFromAtomicNumber,
    vdwRadii,
)


class Atom:
    def __init__(
        self,
        atomicNumber: int,
        element: str,
        charge: float,
        x: float,
        y: float,
        z: float,
        isAngstrom: bool,
    ):
        """Atom

        Args:
            atomicNumber (int): atomic number. For example, for iron atom, 26
            element (str): element. For example, for iron atom, Fe
            charge (float): charge of atom
            x (float): x-coordinate
            y (float): y-coordinate
            z (float): z-coordinate
            isAngstrom (bool): whether coordinates are in Angstrom. If False, assume in Bohr
        """
        self._atomicNumber = atomicNumber
        self._element = element
        self._charge = charge
        self._x = x
        self._y = y
        self._z = z
        self._positionVector = np.array([self._x, self._y, self._z])
        self.isAngstrom = isAngstrom

        try:
            self._mass = elementMass[self._atomicNumber - 1]
        except ValueError:
            msg = f"Could not determine mass of Atom with atomic number {self._atomicNumber}"
            print(msg)
            self._mass = -1

        try:
            self._vdwRadius = vdwRadii[self._atomicNumber - 1]
        except ValueError:
            msg = f"Could not determine Van der Waals radius of Atom with atomic number {self._atomicNumber}"
            print(msg)
            self._vdwRadius = -1.0

    @classmethod
    def fromCUBE(cls, string: str):
        """Create Atom instance from a line in a CUBE file

        Args:
            string (str): line in CUBE file"""
        splitString = string.split()
        atomicNumber = int(splitString[0])
        element = getElementFromAtomicNumber(atomicNumber)
        charge, x, y, z = [float(field) for field in splitString[1:]]
        isAngstrom = False  # Bohr by default
        return cls(atomicNumber, element, charge, x, y, z, isAngstrom)

    @classmethod
    def fromXYZ(cls, string: str):
        """Create Atom instance from a line in an XYZ file

        Args:
            string (str): line from an XYZ file. Formatted as A x y z ..., where A is either atomic number or element string
        """
        splitString = string.split()
        element = splitString[0].strip()
        try:
            atomicNumber = int(element)
            element = getElementFromAtomicNumber(atomicNumber)
        except ValueError:
            atomicNumber = getAtomicNumberFromElement(element)
        x, y, z = [float(field) for field in splitString[1:4]]
        isAngstrom = True  # Angstrom by default
        return cls(atomicNumber, element, "UNKNOWN", x, y, z, isAngstrom)

    @classmethod
    def fromSDF(cls, string: str):
        """Create Atom instance from a line in an SDF file

        Args:
            string (str): line from an SDF file. Formatted as x y z A, where A is an element string.
        """
        splitString = string.split()
        element = splitString[3].strip()
        atomicNumber = getAtomicNumberFromElement(element)
        x, y, z = [float(field) for field in splitString[:3]]
        isAngstrom = True  # SDF is in Angstrom
        return cls(atomicNumber, element, "UNKNOWN", x, y, z, isAngstrom)

    def getAtomicNumber(self) -> int:
        """Get the atomic number of the atom

        Returns:
            int: atomic number"""
        return self._atomicNumber

    def getCharge(self) -> float:
        """Get the charge of the Atom (undefined if created from XYZ file)

        Returns:
            float: charge of Atom"""
        return self._charge

    def getX(self) -> float:
        """Get the x-coordinate of the atom

        Returns:
            float: x-coordinate of Atom
        """
        return self._x

    def getY(self) -> float:
        """Get the y-coordinate of the atom

        Returns:
            float: y-coordinate of Atom
        """
        return self._y

    def getZ(self) -> float:
        """Get the z-coordinate of the atom

        Returns:
            float: z-coordinate of Atom
        """
        return self._z

    def getPositionVector(self) -> np.ndarray[float]:
        """Get position of the atom

        Returns:
            ndarray: array with x, y and z coordinates of Atom"""
        return self._positionVector

    def positionBohrToAngstrom(self) -> None:
        """Convert the position vector from Bohr to Angstrom"""
        if self.isAngstrom:
            raise ValueError()
        self.isAngstrom = True
        self.setPositionVector(self._positionVector * BOHR_TO_ANGSTROM)

    def setPositionVector(self, newPosition) -> None:
        """Set the position of the atom to a new position

        Args:
            newPosition (ndarray): x, y and z coordinates of new position
        """
        self._positionVector = newPosition
        self._x, self._y, self._z = newPosition

    def getElement(self) -> str:
        """Get the element of the atom

        Returns:
            str: element of Atom
        """
        return self._element

    def getMass(self) -> float:
        """Get the mass of the atom

        Returns:
            float: mass of Atom"""
        return self._mass

    def getVdWRadius(self) -> float:
        """Get the Van der Waals radius of the atom

        Returns:
            float: Van der Waals radius of the Atom"""
        return self._vdwRadius

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Atom with atomic number {self._atomicNumber} at position {self._positionVector}"

    # def findBoundAtoms(self, structure: Structure) -> list[int]:
    #     """Find which Atom indeces are bound to this Atom in the structure"""
    #     boundAtomIndeces = []
    #     for i, bond in enumerate(structure.getBonds()):
    #         atom1Pos = bond.getAtom1Pos()
    #         atom2Pos = bond.getAtom2Pos()
    #         if np.all(self._positionVector == atom1Pos):
    #             boundAtomIndeces.append(bond.getAtom2Index())
    #         elif np.all(self._positionVector == atom2Pos):
    #             boundAtomIndeces.append(bond.getAtom1Index())
    #     return boundAtomIndeces
