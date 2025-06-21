import numpy as np

from .ElementData import elementList, elementMass, vdwRadii


class Bond:
    def __init__(
        self,
        atom1Index: int,
        atom2Index: int,
        bondType: str,
        bondLength: float,
        interatomicVector: np.ndarray,
        midpointPosition: np.ndarray,
        atom1and2Pos: np.ndarray,
        name: str,
    ):
        self._atom1Index, self._atom2Index = atom1Index, atom2Index
        self._bondType = bondType
        self._bondLength = bondLength
        self._interatomicVector = interatomicVector
        self.setMidpointPosition(midpointPosition)
        self._atom1Pos = atom1and2Pos[0]
        self._atom2Pos = atom1and2Pos[1]
        self.setName(name)

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
        if isinstance(midpointPosition, list):
            midpointPosition = np.asarray(list)
        if not np.shape(midpointPosition) == (3,):
            raise ValueError()
        self._midpointPosition = midpointPosition

    def getDirection(self) -> np.ndarray[float]:
        """Get the unit vector in the direction of the bond"""
        return self._interatomicVector / self._bondLength

    def getAtom1Pos(self) -> np.ndarray[float]:
        """Get position of atom 1"""
        return self._atom1Pos

    def getAtom2Pos(self) -> np.ndarray[float]:
        """Get position of atom 2"""
        return self._atom2Pos

    def getName(self) -> str:
        return self._name

    def setName(self, name) -> None:
        if not isinstance(name, str):
            raise TypeError(f"name was supposed to be string but was type {type(name)}")
        self._name = name

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

    def getVdWWeightedMidpoints(self, element1: str, element2: str) -> np.ndarray:
        """Get the Van der Waals-radii weighted bond-midpoints"""
        element1Index = elementList.index(element1)
        VdWRadius1 = vdwRadii[element1Index]

        element2Index = elementList.index(element2)
        VdWRadius2 = vdwRadii[element2Index]

        sumVdWRadius = VdWRadius1 + VdWRadius2
        fractionVdWRadius1 = VdWRadius1 / sumVdWRadius
        fractionVdWRadius2 = VdWRadius2 / sumVdWRadius

        loc1 = self._midpointPosition - self.getDirection() * self._bondLength / (
            2 / fractionVdWRadius1
        )
        loc2 = self._midpointPosition + self.getDirection() * self._bondLength / (
            2 / fractionVdWRadius2
        )
        return np.array([loc1, loc2])
