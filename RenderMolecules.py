import math
import os
import sys

import numpy as np
from pytessel import PyTessel

from copy import deepcopy

elementList = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
]

elementMass = [
    1.008,
    4,
    7,
    9,
    11,
    12.01,
    14.007,
    16.00,
    19,
    20,
    23,
    24,
    27,
    28,
    31,
    32,
    35,
    40,
    39,
    40,
    45,
]

# Van der Waals radii in Angstrom (from https://en.wikipedia.org/wiki/Van_der_Waals_radius)
vdwRadii = [
    1.1,
    1.4,
    1.82,
    1.53,
    1.92,
    1.70,
    1.55,
    1.52,
    1.47,
    1.54,
    2.27,
    1.73,
    1.84,
    2.10,
    1.80,
    1.80,
    1.75,
    1.88,
    2.75,
    2.31,
    2.11,
]

# Bond lengths in Angstrom
bondLengths = {
    "HO": 2.0,
    "CO": 2.0,
    "CH": 2.0,
    "OO": 2.0,
    "HH": 0.75,
    "NN": 1.5,
    "HN": 1.0,
    "CN": 1.5,
    "CC": 1.5,
}
hydrogenBondLength = 3.5
hydrogenBondAngle = 35
sphereScale = 0.2

BOHR_TO_ANGSTROM = 0.5291177249
ANGSTROM_TO_BOHR = 1 / BOHR_TO_ANGSTROM


class Atom:
    def __init__(self, string):
        self._string = string
        self._splitString = self._string.split()
        self._atomicNumber = int(self._splitString[0])
        self._charge, self._x, self._y, self._z = [
            float(field) for field in self._splitString[1:]
        ]
        self._positionVector = np.array([self._x, self._y, self._z])
        self.isAngstrom = False

        try:
            self._element = elementList[self._atomicNumber - 1]
        except ValueError:
            msg = f"Could not determine element of Atom with atomic number {self._atomicNumber}"
            print(msg)
            self._element = "Unknown"

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

    def getAtomicNumber(self) -> int:
        return self._atomicNumber

    def getCharge(self) -> float:
        return self._charge

    def getX(self) -> float:
        return self._x

    def getY(self) -> float:
        return self._y

    def getZ(self) -> float:
        return self._z

    def getPositionVector(self) -> np.ndarray[float]:
        return self._positionVector

    def positionBohrToAngstrom(self) -> None:
        if self.isAngstrom:
            raise ValueError()
        self.isAngstrom = True
        self.setPositionVector(self._positionVector*BOHR_TO_ANGSTROM)

    def setPositionVector(self, newPosition) -> None:
        self._positionVector = newPosition
        self._x, self._y, self._z = newPosition

    def getElement(self) -> str:
        return self._element

    def getMass(self) -> str:
        return self._mass

    def getVdWRadius(self) -> float:
        return self._vdwRadius

    def __repr__(self):
        return f"Atom('{self._string}')"

    def __str__(self):
        return f"Atom with atomic number {self._atomicNumber} at position {self._positionVector}"

class Bond:
    def __init__(self, atom1Index, atom2Index, bondType, bondLength, interatomicVector, midpointPosition):
        self._atom1Index, self._atom2Index = atom1Index, atom2Index
        self._bondType = bondType
        self._bondLength = bondLength
        self._interatomicVector = interatomicVector
        self._midpointPosition = midpointPosition

    def getAtom1Index(self) -> int:
        return self._atom1Index
    
    def getAtom2Index(self) -> int:
        return self._atom2Index

    def getBondType(self) -> str:
        return self._bondType
        
    def getBondLength(self) -> float:
        return self._bondLength

    def getInteratomicVector(self)-> np.ndarray[float]:
        return self._interatomicVector
    
    def getMidpointPosition(self) -> np.ndarray[float]:
        return self._midpointPosition

    def setMidpointPosition(self, midpointPosition: np.ndarray) -> None:
        self._midpointPosition = midpointPosition

    def getDirection(self) -> np.ndarray[float]:
        return self._interatomicVector / self._bondLength

    def getAxisAngleWithZaxis(self) -> tuple[float, float, float, float]:
        z = np.array([0, 0, 1])
        axis = np.cross(z,self._interatomicVector)
        if np.linalg.norm(axis) < 1e-5:
            axis = np.array([0,0,1])
            angle = 0.0
        else:
            axis /= np.linalg.norm(axis)
            angle = np.arccos(np.dot(self._interatomicVector,z)/self._bondLength)
        return angle, axis[0], axis[1], axis[2]


class CUBEfile:
    def __init__(self, filepath):
        self._filepath = filepath
        with open(self._filepath, "r") as file:
            self._lines = file.readlines()
        self._nAtoms = int(self._lines[2].split()[0].strip())

        if self._nAtoms < 0:
            self._nAtoms = -self._nAtoms
        self._atoms = [0] * self._nAtoms
        
        for i in range(self._nAtoms):
            self._atoms[i] = Atom(self._lines[6 + i].strip())

            self._atoms[i].positionBohrToAngstrom()
        
        self._displacements = []
        self._bonds = []

    def getAtoms(self):
        return self._atoms

    def getAtomPositionVectors(self):
        return [atom.getPositionVector() for atom in self._atoms]

    def readVolumetricData(self) -> None:
        self._NX, self._NY, self._NZ = [
            int(self._lines[i].split()[0].strip()) for i in [3, 4, 5]
        ] 
        self._volumetricOriginVector = np.array(
            [float(i) for i in self._lines[2].split()[1:]]
        ) * BOHR_TO_ANGSTROM

        self._volumetricAxisVectors = np.array(
            [[float(i) for i in self._lines[3 + i].split()[1:]] for i in [0, 1, 2]]
        ) * BOHR_TO_ANGSTROM

        volumetricLines = " ".join(
            line.strip() for line in self._lines[6 + self._nAtoms :]
        ).split()

        self._volumetricData = np.zeros((self._NX, self._NY, self._NZ))
        for ix in range(self._NX):
            for iy in range(self._NY):
                for iz in range(self._NZ):
                    dataIndex = ix * self._NY * self._NZ + iy * self._NZ + iz
                    self._volumetricData[ix, iy, iz] = float(
                        volumetricLines[dataIndex]
                    )
                    
    def getFilepath(self) -> str:
        return self._filepath

    def getLines(self) -> list[str]:
        return self._lines

    def getVolumetricOriginVector(self) -> np.ndarray[float]:
        return self._volumetricOriginVector

    def getVolumetricAxisVectors(self) -> np.ndarray[float]:
        return self._volumetricAxisVectors

    def getVolumetricData(self) -> np.ndarray[float]:
        return self._volumetricData

    def writePLY(self, filepath, isovalue) -> None:
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
            self._volumetricData.flatten(order='F'),
            reversed(self._volumetricData.shape),
            unitCell.flatten(),
            isovalue,
        )

        vertices += np.diag(0.5 * unitCell) + self._volumetricOriginVector
        for displacement in self._displacements:
            vertices += displacement

        pytessel.write_ply(filepath, vertices, normals, indices)

    def getTotalCharge(self) -> int:
        totalCharge = int(sum(atom.getCharge() for atom in self._atoms))
        return totalCharge

    def getAmountOfElectrons(self) -> int:
        totalElectronsIfNeutral = sum(atom.getAtomicNumber() for atom in self._atoms)
        totalElectrons = totalElectronsIfNeutral - self.getTotalCharge()
        return totalElectrons

    def isRadical(self) -> bool:
        return self.getAmountOfElectrons() % 2 != 0

    def createBonds(
        self,
    ) -> list[tuple[int, int, str, float, np.ndarray[float], np.ndarray[float]]]:
        """For each bond, get the indices of atoms it connects, bond type, bond length, bond vector and bond midpoint vector"""
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
                #connectingIndices = [(a[0], a[1]) for a in bondTuples]
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
        displacementVector = np.cross(bondVector, bond.getMidpointPosition() - cameraPos)
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
                bondAdjusted.getMidpointPosition() + displacementVector * displacementMag
            )
            self._bonds.append(bondAdjusted)
            displacementMag += displacementScaler
        return self._bonds

    def getBonds(self) -> list[Bond]:
        return self._bonds

    def getCenterOfMass(self) -> np.ndarray[float]:
        masses = np.array([atom.getMass() for atom in self._atoms])
        atomPositions = self.getAtomPositionVectors()
        COM = np.array(
            sum(masses[i] * atomPositions[i] for i in range(self._nAtoms)) / sum(masses)
        )
        return COM

    def setCOMto(self, newCOMposition):
        COM = self.getCenterOfMass()
        for atom in self._atoms:
            atom.setPositionVector(atom.getPositionVector() - (COM-newCOMposition))
        self._displacements.append(-(COM-newCOMposition))

    def setAveragePositionToOrigin(self):
        averagePosition = np.average(np.array([atom.getPositionVector() for atom in self._atoms]), axis=0)
        for atom in self._atoms:
            atom.setPositionVector(atom.getPositionVector() - averagePosition)
        self._displacements.append(-averagePosition)


if __name__ == "__main__":
    CUBEfilepath = "H2C3N_B3LYP-D4_spindensity.cube"
    #CUBEfilepath = "H2O_elf.cube"
    #CUBEfilepath = "hexazine.cube"
    CUBEfilepath_noext = os.path.splitext(CUBEfilepath)[0]
    
    CUBE = CUBEfile(CUBEfilepath)
    CUBE.setCOMto(np.array([0, 0, 0]))
    CUBE.readVolumetricData()
    value = 0.02
    #CUBE.writePLY(f"{CUBEfilepath_noext}_{value}.ply", value)
