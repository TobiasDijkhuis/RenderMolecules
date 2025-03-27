import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from pytessel import PyTessel

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
    1,
    4,
    7,
    9,
    11,
    12,
    14,
    16,
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

bondLengths = {"HO": 2.0, "CO": 2.0, "CH": 2.0, "OO": 2.0, "HH": 0.75}
hydrogenBondLength = 3.5
hydrogenBondAngle = 35
sphereScale = 0.5


class Atom:
    def __init__(self, string):
        self._string = string
        self._splitString = self._string.split()
        self._atomicNumber = int(self._splitString[0])
        self._charge, self._x, self._y, self._z = [
            float(field) for field in self._splitString[1:]
        ]
        self._positionVector = np.array([self._x, self._y, self._z])

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


class CUBEfile:
    def __init__(self, filepath):
        self._filepath = filepath
        with open(self._filepath, "r") as file:
            self._lines = file.readlines()
        self._nAtoms = int(self._lines[2].split()[0].strip())

        self._atoms = [0] * self._nAtoms
        for i in range(self._nAtoms):
            self._atoms[i] = Atom(self._lines[6 + i].strip())

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
        )
        self._volumetricAxisVectors = np.array(
            [[float(i) for i in self._lines[3 + i].split()[1:]] for i in [0, 1, 2]]
        )

        self._volumetricData = np.zeros((self._NX, self._NY, self._NZ))
        volumetricLines = " ".join(
            line.strip() for line in self._lines[6 + self._nAtoms :]
        ).split()
        dataIndexUpdate = 0
        for ix in range(self._NX):
            for iy in range(self._NY):
                for iz in range(self._NZ):
                    dataIndex = ix * self._NY * self._NZ + iy * self._NZ + iz
                    # print(dataIndex)
                    self._volumetricData[ix, iy, iz] = float(volumetricLines[dataIndex])
                    dataIndexUpdate += 1

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
        vertices, normals, indices = pytessel.marching_cubes(
            self._volumetricData.flatten(),
            self._volumetricData.shape,
            self._volumetricAxisVectors.flatten(),
            isovalue,
        )
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

    def getBondTuples(
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
        bondTuples = []
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
                connectingIndices = [(a[0], a[1]) for a in bondTuples]
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
                    bondTuples.append(
                        (
                            i,
                            atomIndex,
                            "".join(
                                sorted(f"{centralElement}{allAtomElements[atomIndex]}")
                            ),
                            bondLength,
                            bondVector,
                            bondMidpoint,
                        )
                    )
        self._bondTuples = []
        return bondTuples

    def generateBondOrderBond(self, bondTuple, bondOrder: int, angle=0.0):
        """TODO: Add way to generate double bond if desired for a certain bond"""
        index = self._bondTuples.index(bondTuple)
        self._bondTuples.pop(index)
        bondVector = bondTuple[-2]
        raise NotImplementedError()


def generatePerpendicularVector(vector: np.ndarray[float]) -> np.ndarray[float]:
    # https://math.stackexchange.com/questions/137362/how-to-find-perpendicular-vector-to-another-vector
    x, y, z = vector
    perpVector = np.array(
        [
            math.copysign(vector[2], vector[0]),
            math.copysign(vector[2], vector[1]),
            -math.copysign(abs(vector[0]) + abs(vector[1]), vector[2]),
        ]
    )
    perpVector /= np.linalg.norm(perpVector)

    # Instead of doing this, we could also describe the plane generated by the two vectors
    # in R3, the bondvector and the vector from the camera to the bond. Then, the displacement vector
    # should be perpendicular to both these vectors, i.e. the third vector should be the cross product of
    # the other two vectors. This third vector is then the displacement vector to generate a double bond.
    # The camera location can be found as bpy.data.objects["Camera"].location,
    # and then vector connecting camera and bond can be calculated as cam_log - bondMidpoint
    return perpVector


if __name__ == "__main__":
    CUBE = CUBEfile("H2O_elf.cube")
    atomPositions = CUBE.getAtomPositionVectors()
    atoms = CUBE.getAtoms()
    masses = [atom.getMass() for atom in atoms]

    bondTuples = CUBE.getBondTuples()
    # CUBE.generateDoubleBond(bondTuples[0])
    print(bondTuples)
    print(generatePerpendicularVector(bondTuples[0][-2]))
    # CUBE.readVolumetricData()
    # CUBE.writePLY("H2O_elf.ply", 0.5)
