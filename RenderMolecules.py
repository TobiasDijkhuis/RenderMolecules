import math
import os
import sys
from copy import deepcopy

import numpy as np
from pytessel import PyTessel

from ElementData import *


def getElementFromAtomicNumber(atomicNumber: int) -> str:
    try:
        element = elementList[atomicNumber - 1]
    except ValueError:
        msg = f"Could not determine element from atomic number {atomicNumber}"
        raise ValueError(msg)
    return element


def getAtomicNumberFromElement(element: str) -> int:
    try:
        atomicNumber = elementList.index(element) + 1
    except ValueError:
        msg = f"Could not determine atomic number from element {element}"
        raise ValueError()
    return atomicNumber


class Atom:
    def __init__(self, atomicNumber, element, charge, x, y, z, isAngstrom):
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
        """Create Atom instance from a line in a CUBE file"""
        splitString = string.split()
        atomicNumber = int(splitString[0])
        element = getElementFromAtomicNumber(atomicNumber)
        charge, x, y, z = [float(field) for field in splitString[1:]]
        isAngstrom = False  # Bohr by default
        return cls(atomicNumber, element, charge, x, y, z, isAngstrom)

    @classmethod
    def fromXYZ(cls, string: str):
        """Create Atom instance from a line in an XYZ file"""
        splitString = string.split()
        element = splitString[0].strip()
        atomicNumber = getAtomicNumberFromElement(element)
        x, y, z = [float(field) for field in self._splitString[1:]]
        isAngstrom = True  # Angstrom by default
        return cls(atomicNumber, element, "UNKNOWN", x, y, z, isAngstrom)

    def getAtomicNumber(self) -> int:
        """Get the atomic number of the atom"""
        return self._atomicNumber

    def getCharge(self) -> float:
        """Get the charge of the atom (undefined if created from XYZ file)"""
        return self._charge

    def getX(self) -> float:
        """Get the x-coordinate of the atom"""
        return self._x

    def getY(self) -> float:
        """Get the y-coordinate of the atom"""
        return self._y

    def getZ(self) -> float:
        """Get the z-coordinate of the atom"""
        return self._z

    def getPositionVector(self) -> np.ndarray[float]:
        """Get position of the atom"""
        return self._positionVector

    def positionBohrToAngstrom(self) -> None:
        """Convert the position vector from Bohr to Angstrom"""
        if self.isAngstrom:
            raise ValueError()
        self.isAngstrom = True
        self.setPositionVector(self._positionVector * BOHR_TO_ANGSTROM)

    def setPositionVector(self, newPosition) -> None:
        """Set the position of the atom to a new position"""
        self._positionVector = newPosition
        self._x, self._y, self._z = newPosition

    def getElement(self) -> str:
        """Get the element of the atom"""
        return self._element

    def getMass(self) -> str:
        """Get the mass of the atom"""
        return self._mass

    def getVdWRadius(self) -> float:
        """Get the Van der Waals radius of the atom"""
        return self._vdwRadius

    def __repr__(self):
        return f"Atom('{self._string}')"

    def __str__(self):
        return f"Atom with atomic number {self._atomicNumber} at position {self._positionVector}"


class Bond:
    def __init__(
        self,
        atom1Index,
        atom2Index,
        bondType,
        bondLength,
        interatomicVector,
        midpointPosition,
    ):
        self._atom1Index, self._atom2Index = atom1Index, atom2Index
        self._bondType = bondType
        self._bondLength = bondLength
        self._interatomicVector = interatomicVector
        self._midpointPosition = midpointPosition

    def getAtom1Index(self) -> int:
        """Get the index of the first atom that is connected to this bond"""
        return self._atom1Index

    def getAtom2Index(self) -> int:
        """Get the index of the second atom that is connected to this bond"""
        return self._atom2Index

    def getBondType(self) -> str:
        """Get the bond type (the two connecting elements in alphabetical order)"""
        return self._bondType

    def getBondLength(self) -> float:
        """Get the bond length"""
        return self._bondLength

    def getInteratomicVector(self) -> np.ndarray[float]:
        """Get the vector connecting the two atoms"""
        return self._interatomicVector

    def getMidpointPosition(self) -> np.ndarray[float]:
        """Get the midpoint position of the two atoms"""
        return self._midpointPosition

    def setMidpointPosition(self, midpointPosition: np.ndarray) -> None:
        """Set the midpoint position of the two atoms"""
        self._midpointPosition = midpointPosition

    def getDirection(self) -> np.ndarray[float]:
        """Get the unit vector in the direction of the bond"""
        return self._interatomicVector / self._bondLength

    def getAxisAngleWithZaxis(self) -> tuple[float, float, float, float]:
        """Get the axis angle such that a created cylinder in the direction of the bond"""
        z = np.array([0, 0, 1])
        axis = np.cross(z, self._interatomicVector)
        if np.linalg.norm(axis) < 1e-5:
            axis = np.array([0, 0, 1])
            angle = 0.0
        else:
            axis /= np.linalg.norm(axis)
            angle = np.arccos(np.dot(self._interatomicVector, z) / self._bondLength)
        return angle, axis[0], axis[1], axis[2]


class Structure:
    def getAtoms(self) -> list[Atom]:
        """Get a list of all atoms in the structure"""
        return self._atoms

    def getAtomPositionVectors(self) -> list[np.ndarray]:
        """Get a list of all atom positions"""
        return [atom.getPositionVector() for atom in self._atoms]

    def getFilepath(self) -> str:
        """Get the filepath of the file that created the structure"""
        return self._filepath

    def getLines(self) -> list[str]:
        """Get the lines of the file that created the structure"""
        return self._lines

    def createBonds(self) -> list[Bond]:
        """Create bonds based on the geometry"""
        allAtomPositions = self.getAtomPositionVectors()
        allAtomElements = [atom.getElement() for atom in self._atoms]
        allAtomPositionsTuples = (
            np.array([v[0] for v in allAtomPositions]),
            np.array([v[1] for v in allAtomPositions]),
            np.array([v[2] for v in allAtomPositions]),
        )
        x, y, z = allAtomPositionsTuples

        connectingIndices = []
        for i, atom in enumerate(self._atoms):
            centralPos = allAtomPositions[i]
            centralElement = allAtomElements[i]
            allowedBondLengthsSquared = [
                bondLengths["".join(sorted(f"{centralElement}{otherElement}"))] ** 2
                for otherElement in allAtomElements
            ]
            dx = x - centralPos[0]
            dy = y - centralPos[1]
            dz = z - centralPos[2]
            distSquared = dx * dx + dy * dy + dz * dz
            isBondedToCentral = np.nonzero(distSquared <= allowedBondLengthsSquared)[0]
            for atomIndex in isBondedToCentral:
                # connectingIndices = [(a[0], a[1]) for a in bondTuples]
                if atomIndex == i:
                    continue
                if (i, atomIndex) not in connectingIndices and (
                    atomIndex,
                    i,
                ) not in connectingIndices:
                    bondMidpoint = (
                        allAtomPositions[i] + allAtomPositions[atomIndex]
                    ) / 2.0
                    bondLength = distSquared[atomIndex] ** 0.5
                    bondVector = allAtomPositions[i] - allAtomPositions[atomIndex]
                    newBond = Bond(
                        i,
                        atomIndex,
                        "".join(
                            sorted(f"{centralElement}{allAtomElements[atomIndex]}")
                        ),
                        bondLength,
                        bondVector,
                        bondMidpoint,
                    )
                    connectingIndices.append((i, atomIndex))
                    self._bonds.append(newBond)

    def generateBondOrderBond(
        self,
        bond: Bond,
        bondOrder: int,
        cameraPos: np.ndarray[float],
        displacementScaler=0.2,
    ) -> list[Bond]:
        """Way to generate multiple bonds, for example in CO2 molecule double bonds, or CO triple bonds."""
        if bondOrder == 1:
            return
        index = self._bonds.index(bond)
        self._bonds.pop(index)
        bondVector = bond.getInteratomicVector()

        # Get a vector that is perpendicular to the plane given by the bondVector and vector between camera and bond midpoint.
        displacementVector = np.cross(
            bondVector, bond.getMidpointPosition() - cameraPos
        )
        displacementVector /= np.linalg.norm(displacementVector)

        # If bondOrder is odd, then we also have displacementMag of 0.
        if bondOrder % 2 == 0:
            displacementMag = -displacementScaler / 4 * bondOrder
        else:
            displacementMag = -displacementScaler / 2 * (bondOrder - 1)

        # Create the bonds, and add them to self._bonds
        for i in range(bondOrder):
            bondAdjusted = deepcopy(bond)
            bondAdjusted.setMidpointPosition(
                bondAdjusted.getMidpointPosition()
                + displacementVector * displacementMag
            )
            self._bonds.append(bondAdjusted)
            displacementMag += displacementScaler
        return self._bonds

    def getBonds(self) -> list[Bond]:
        """Get all bonds in the system"""
        return self._bonds

    def getCenterOfMass(self) -> np.ndarray[float]:
        """Get the center of mass position"""
        masses = np.array([atom.getMass() for atom in self._atoms])
        atomPositions = self.getAtomPositionVectors()
        COM = np.array(
            sum(masses[i] * atomPositions[i] for i in range(self._nAtoms)) / sum(masses)
        )
        return COM

    def setCOMto(self, newCOMposition):
        """Set the center of mass of the whole system to a new position"""
        COM = self.getCenterOfMass()
        for atom in self._atoms:
            atom.setPositionVector(atom.getPositionVector() - (COM - newCOMposition))
        self._displacements.append(-(COM - newCOMposition))

    def setAveragePositionToOrigin(self):
        """Sets the average position of all atoms to the origin (0,0,0)"""
        averagePosition = np.average(
            np.array([atom.getPositionVector() for atom in self._atoms]), axis=0
        )
        for atom in self._atoms:
            atom.setPositionVector(atom.getPositionVector() - averagePosition)
        self._displacements.append(-averagePosition)


class CUBEfile(Structure):
    def __init__(self, filepath):
        self._filepath = filepath
        with open(self._filepath, "r") as file:
            self._lines = file.readlines()
        self._nAtoms = int(self._lines[2].split()[0].strip())

        if self._nAtoms < 0:
            self._nAtoms = -self._nAtoms
        self._atoms = [0] * self._nAtoms

        for i in range(self._nAtoms):
            self._atoms[i] = Atom.fromCUBE(self._lines[6 + i].strip())

            self._atoms[i].positionBohrToAngstrom()

        self._displacements = []
        self._bonds = []

    def readVolumetricData(self) -> None:
        """Read the volumetric data in the CUBE file"""
        self._NX, self._NY, self._NZ = [
            int(self._lines[i].split()[0].strip()) for i in [3, 4, 5]
        ]
        self._volumetricOriginVector = (
            np.array([float(i) for i in self._lines[2].split()[1:]]) * BOHR_TO_ANGSTROM
        )

        self._volumetricAxisVectors = (
            np.array(
                [[float(i) for i in self._lines[3 + i].split()[1:]] for i in [0, 1, 2]]
            )
            * BOHR_TO_ANGSTROM
        )

        volumetricLines = " ".join(
            line.strip() for line in self._lines[6 + self._nAtoms :]
        ).split()

        self._volumetricData = np.zeros((self._NX, self._NY, self._NZ))
        for ix in range(self._NX):
            for iy in range(self._NY):
                for iz in range(self._NZ):
                    dataIndex = ix * self._NY * self._NZ + iy * self._NZ + iz
                    self._volumetricData[ix, iy, iz] = float(volumetricLines[dataIndex])

    def getVolumetricOriginVector(self) -> np.ndarray[float]:
        """Get the origin vector of the volumetric data"""
        return self._volumetricOriginVector

    def getVolumetricAxisVectors(self) -> np.ndarray[float]:
        """Get the axis vectors of the volumetric data"""
        return self._volumetricAxisVectors

    def getVolumetricData(self) -> np.ndarray[float]:
        """Get the volumetric data"""
        return self._volumetricData

    def writePLY(self, filepath, isovalue) -> None:
        """Write the volumetric data to a filepath"""
        if isovalue <= np.min(self._volumetricData):
            msg = f"Set isovalue ({isovalue}) was less than or equal to the minimum value in the volumetric data ({np.min(self._volumetricData)}). This will result in an empty PLY. Set a larger isovalue."
            raise ValueError(msg)
        if isovalue >= np.max(self._volumetricData):
            msg = f"Set isovalue ({isovalue}) was more than or equal to the maximum value in the volumetric data ({np.max(self._volumetricData)}). This will result in an empty PLY. Set a smaller isovalue."
            raise ValueError(msg)

        pytessel = PyTessel()

        unitCell = self._volumetricAxisVectors * self._volumetricData.shape

        # Flatten the volumetric data such that X is the fastest moving index, according to the PyTessel documentation.
        vertices, normals, indices = pytessel.marching_cubes(
            self._volumetricData.flatten(order="F"),
            reversed(self._volumetricData.shape),
            unitCell.flatten(),
            isovalue,
        )

        vertices += np.diag(0.5 * unitCell) + self._volumetricOriginVector
        for displacement in self._displacements:
            vertices += displacement

        pytessel.write_ply(filepath, vertices, normals, indices)

    def getTotalCharge(self) -> int:
        """Get the total charge in the system"""
        totalCharge = int(sum(atom.getCharge() for atom in self._atoms))
        return totalCharge

    def getAmountOfElectrons(self) -> int:
        """Get the total amount of electrons in the system"""
        totalElectronsIfNeutral = sum(atom.getAtomicNumber() for atom in self._atoms)
        totalElectrons = totalElectronsIfNeutral - self.getTotalCharge()
        return totalElectrons

    def isRadical(self) -> bool:
        """Returns whether the studied structure is a radical (has an uneven amount of electrons)"""
        return self.getAmountOfElectrons() % 2 != 0


class XYZfile(Structure):
    def __init__(self, filepath):
        self._filepath = filepath
        with open(self._filepath, "r") as file:
            self._lines = file.readlines()

        self._nAtoms = int(self._lines[0].strip())
        self._atoms = [0] * self._nAtoms

        for i in range(self._nAtoms):
            self._atoms[i] = Atom.fromXYZ(self._lines[2:])

        self._displacements = []
        self._bonds = []


if __name__ == "__main__":
    CUBEfilepath = "H2C3N_B3LYP-D4_spindensity.cube"
    # CUBEfilepath = "H2O_elf.cube"
    # CUBEfilepath = "hexazine.cube"
    CUBEfilepath_noext = os.path.splitext(CUBEfilepath)[0]

    CUBE = CUBEfile(CUBEfilepath)
    CUBE.setCOMto(np.array([0, 0, 0]))
    CUBE.readVolumetricData()
    value = 0.02
    CUBE.writePLY(f"{CUBEfilepath_noext}_{value}.ply", value)
