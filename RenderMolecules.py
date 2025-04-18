from __future__ import annotations

import json
import math
import os
import sys
from copy import deepcopy

import bpy
import numpy as np

from ElementData import *
from utils import *


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
        print(splitString)
        element = splitString[0].strip()
        atomicNumber = getAtomicNumberFromElement(element)
        x, y, z = [float(field) for field in splitString[1:]]
        isAngstrom = True  # Angstrom by default
        return cls(atomicNumber, element, "UNKNOWN", x, y, z, isAngstrom)

    @classmethod
    def fromSDF(cls, string: str):
        """Create Atom instance from a line in an SDF file"""
        splitString = string.split()
        element = splitString[3].strip()
        atomicNumber = getAtomicNumberFromElement(element)
        x, y, z = [float(field) for field in splitString[:3]]
        isAngstrom = True  # SDF is in Angstrom
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
        return self.__str__()

    def __str__(self):
        return f"Atom with atomic number {self._atomicNumber} at position {self._positionVector}"

    def findBoundAtoms(self, structure: Structure) -> list[int]:
        boundAtomIndeces = []
        for i, bond in enumerate(structure.getBonds()):
            atom1Pos = bond.getAtom1Pos()
            atom2Pos = bond.getAtom2Pos()
            if np.all(self._positionVector == atom1Pos):
                boundAtomIndeces.append(bond.getAtom2Index())
            elif np.all(self._positionVector == atom2Pos):
                boundAtomIndeces.append(bond.getAtom1Index())
        return boundAtomIndeces


class Bond:
    def __init__(
        self,
        atom1Index,
        atom2Index,
        bondType,
        bondLength,
        interatomicVector,
        midpointPosition,
        atom1and2Pos,
    ):
        self._atom1Index, self._atom2Index = atom1Index, atom2Index
        self._bondType = bondType
        self._bondLength = bondLength
        self._interatomicVector = interatomicVector
        self._midpointPosition = midpointPosition
        self._atom1Pos = atom1and2Pos[0]
        self._atom2Pos = atom1and2Pos[1]

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

    def getAtom1Pos(self) -> np.ndarray[float]:
        return self._atom1Pos

    def getAtom2Pos(self) -> np.ndarray[float]:
        return self._atom2Pos

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
    def __init__(self, atomList: list[Atom]):
        self._nAtoms = len(atomList)
        self._atoms = atomList

    def getAtoms(self) -> list[Atom]:
        """Get a list of all atoms in the structure"""
        return self._atoms

    def getAllAtomPositions(self) -> list[np.ndarray]:
        """Get a list of all atom positions"""
        return np.array([atom.getPositionVector() for atom in self._atoms])

    def getFilepath(self) -> str:
        """Get the filepath of the file that created the structure"""
        return self._filepath

    def getLines(self) -> list[str]:
        """Get the lines of the file that created the structure"""
        return self._lines

    def createAtoms(self, renderResolution="medium", createMesh=True) -> None:
        if not createMesh:
            # This is an old, naive method where we create a lot more spheres
            for atom in self._atoms:
                obj = createUVsphere(
                    atom.getElement(),
                    atom.getPositionVector(),
                    resolution=renderResolution,
                )
                mat = create_material(
                    atom.getElement(), manifest["atom_colors"][atom.getElement()]
                )
                obj.data.materials.append(mat)
            return

        # Create a dictionary, with keys the atom element, and values a list of
        # all positions of atoms with that element.
        atomVertices = {}
        for atom in self._atoms:
            if atom.getElement() in atomVertices.keys():
                atomVertices[atom.getElement()].append(atom.getPositionVector())
            else:
                atomVertices[atom.getElement()] = [atom.getPositionVector()]

        # For each element, create a reference UV sphere at the origin
        # Then, create a mesh with vertices at the positions and using vertex instancing,
        # copy the UV sphere to each of the vertices.
        for atomType in atomVertices.keys():
            obj = createUVsphere(atomType, np.array([0, 0, 0]), renderResolution)
            mat = create_material(atomType, manifest["atom_colors"][atomType])
            obj.data.materials.append(mat)

            createMeshAtoms(atomVertices[atomType], obj, atomType)

    def findBondsBasedOnDistance(self) -> list[Bond]:
        """Create bonds based on the geometry"""
        allAtomPositions = self.getAllAtomPositions()
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
                if atomIndex == i:
                    # Do not allow atoms to bond to themselves
                    continue
                if (i, atomIndex) in connectingIndices or (
                    atomIndex,
                    i,
                ) in connectingIndices:
                    # If this bond was already made, continue
                    continue
                bondMidpoint = (allAtomPositions[i] + allAtomPositions[atomIndex]) / 2.0
                bondLength = distSquared[atomIndex] ** 0.5
                bondVector = allAtomPositions[i] - allAtomPositions[atomIndex]
                newBond = Bond(
                    i,
                    atomIndex,
                    "".join(sorted(f"{centralElement}{allAtomElements[atomIndex]}")),
                    bondLength,
                    bondVector,
                    bondMidpoint,
                    (
                        atom.getPositionVector(),
                        self._atoms[atomIndex].getPositionVector(),
                    ),
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
        atomPositions = self.getAllAtomPositions()
        COM = np.array(
            sum(masses[i] * atomPositions[i] for i in range(self._nAtoms)) / sum(masses)
        )
        return COM

    def setCenterOfMass(self, newCOMposition):
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

    def getTotalCharge(self) -> int:
        """Get the total charge in the system"""
        charges = (atom.getCharge() for atom in self._atoms)
        print(list(charges))
        if not all(
            isinstance(charge, int) or isinstance(charge, float) for charge in charges
        ):
            msg = "The charges of atoms are not all of type 'int' or 'float'"
            raise ValueError(msg)
        totalCharge = int(sum(charges))
        return totalCharge

    def getAmountOfElectrons(self) -> int:
        """Get the total amount of electrons in the system"""
        totalElectronsIfNeutral = sum(atom.getAtomicNumber() for atom in self._atoms)
        totalElectrons = totalElectronsIfNeutral - self.getTotalCharge()
        return totalElectrons

    def isRadical(self) -> bool:
        """Returns whether the studied structure is a radical (has an uneven amount of electrons)"""
        return self.getAmountOfElectrons() % 2 != 0

    def rotateAroundX(self, angle):
        self.rotateAroundAxis([1, 0, 0], angle)

    def rotateAroundY(self, angle):
        self.rotateAroundAxis([0, 1, 0], angle)

    def rotateAroundZ(self, angle):
        self.rotateAroundAxis([0, 0, 1], angle)

    def rotateAroundAxis(self, axis, angle):
        rotMatrix = rotation_matrix(axis, angle)

        for atom in self._atoms:
            currentPos = atom.getPositionVector()
            rotatedPos = np.dot(rotMatrix, currentPos)
            atom.setPositionVector(rotatedPos)

    def createHydrogenBonds(self):
        """Adds hydrogen bonds to each molecule"""
        hbondFormingElements = ["H", "O", "N"]
        atoms = self._atoms

        z = np.array([0, 0, 1])

        hbondingCurves = []
        for i, at1 in enumerate(atoms):
            if at1.getElement() not in hbondFormingElements:
                # If the atom is a C, it can not form a hydrogen bond (in our system at least), so skip
                continue
            r1 = at1.getPositionVector()
            atom1BoundIndeces = at1.findBoundAtoms(self)
            for j, at2 in enumerate(atoms):
                if i == j:  # Skip same atom
                    continue
                if j in atom1BoundIndeces:  # If j is bound to i, skip
                    continue
                if (
                    at2.getElement() not in hbondFormingElements
                ):  # Skip if atom 2 cannot form hydrogen bonds
                    continue
                if (
                    at1.getElement() == at2.getElement()
                ):  # OO, HH or NN cannot form hydrogen bonds.
                    continue
                if at1.getElement() in ["C", "O", "N"] and at2.getElement() in [
                    "C",
                    "O",
                    "N",
                ]:
                    # Assume that a C, N or O atom cannot form a hydrogen bond to another C, N or O atom
                    continue
                r2 = at2.getPositionVector()

                dist = np.linalg.norm(r2 - r1)
                if dist > manifest["hbond_distance"]:
                    continue

                atom2BoundIndeces = at2.findBoundAtoms(self)

                if at2.getElement() == "H":
                    # Use some boolean arithmetic to find the position of the O/C/N that the H is bonded to
                    bondedAtomPosition = atoms[atom2BoundIndeces[0]].getPositionVector()

                    # Calculate intramolecular vector
                    intramolOwHw = bondedAtomPosition - r2
                elif at1.getElement() == "H":
                    bondedAtomPosition = atoms[atom1BoundIndeces[0]].getPositionVector()

                    # Calculate intramolecular vector
                    intramolOwHw = bondedAtomPosition - r1
                else:
                    raise NotImplementedError()

                angle = angle_between(intramolOwHw, r2 - r1)

                # create a hydrogen bond when the interatomic distance and O-H----O angle are less than the specified threshold value
                if np.abs(angle) > 180 - manifest["hbond_angle"]:
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
                        at1.getElement(),
                        i,
                        at2.getElement(),
                        j,
                    )
                    hbondingCurves.append(obj)

        mathbond = create_material("H-bond", manifest["hbond_color"])

        for o in hbondingCurves:
            rot_axis = o.rotation_axis_angle
            bpy.ops.surface.primitive_nurbs_surface_cylinder_add(
                enter_editmode=False,
                align="WORLD",
                location=o.location,
            )
            obj = bpy.context.view_layer.objects.active
            obj.name = "Hbond_cyl"

            obj.scale = (manifest["hbond_thickness"], manifest["hbond_thickness"], 0.1)
            obj.data.materials.append(mathbond)

            obj.rotation_mode = "AXIS_ANGLE"
            obj.rotation_axis_angle = rot_axis

            mod = obj.modifiers.new(name="FollowCurve", type="ARRAY")
            bpy.context.object.modifiers["FollowCurve"].fit_type = "FIT_CURVE"
            bpy.context.object.modifiers["FollowCurve"].curve = o
            bpy.context.object.modifiers["FollowCurve"].relative_offset_displace[0] = 0
            bpy.context.object.modifiers["FollowCurve"].relative_offset_displace[1] = 0
            bpy.context.object.modifiers["FollowCurve"].relative_offset_displace[2] = (
                1.3
            )


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
            print(self._lines[6 + i], self._atoms[i].getCharge())

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
        if np.all(
            np.diag(np.diagonal(self._volumetricAxisVectors))
            != self._volumetricAxisVectors
        ):
            warning = "WARNING: Volumetric data axis vectors are not diagonal. Not sure if this works"
            warning += f" Volumetric data axis vectors:\n{self._volumetricAxisVectors}"
            print(warning)

        self._volumetricData = np.fromiter(
            (
                float(num)
                for line in self._lines[6 + self._nAtoms :]
                for num in line.split()
            ),
            dtype=np.float32,
            count=-1,
        ).reshape((self._NX, self._NY, self._NZ))

        # Old, much slower way to read the data.
        # volumetricLines = " ".join(
        #     line.strip() for line in self._lines[6 + self._nAtoms :]
        # ).split()

        # self._volumetricData = np.zeros((self._NX, self._NY, self._NZ))
        # for ix in range(self._NX):
        #     for iy in range(self._NY):
        #         for iz in range(self._NZ):
        #             dataIndex = ix * self._NY * self._NZ + iy * self._NZ + iz
        #             self._volumetricData[ix, iy, iz] = float(volumetricLines[dataIndex])

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
        from pytessel import PyTessel

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

    def calculateIsosurface(self, isovalue) -> tuple[np.ndarray, np.ndarray, int]:
        from skimage.measure import marching_cubes

        """Write the volumetric data to a filepath"""
        if isovalue <= np.min(self._volumetricData):
            msg = f"Set isovalue ({isovalue}) was less than or equal to the minimum value in the volumetric data ({np.min(self._volumetricData)}). This will result in an empty PLY. Set a larger isovalue."
            raise ValueError(msg)
        if isovalue >= np.max(self._volumetricData):
            msg = f"Set isovalue ({isovalue}) was more than or equal to the maximum value in the volumetric data ({np.max(self._volumetricData)}). This will result in an empty PLY. Set a smaller isovalue."
            raise ValueError(msg)

        vertices, faces, normals, values = marching_cubes(
            self._volumetricData,
            level=isovalue,
            spacing=np.diag(self._volumetricAxisVectors),
        )

        vertices += self._volumetricOriginVector
        for displacement in self._displacements:
            vertices += displacement
        return vertices, faces, normals, values


class XYZfile(Structure):
    def __init__(self, filepath):
        self._filepath = filepath
        with open(self._filepath, "r") as file:
            self._lines = file.readlines()

        self._nAtoms = int(self._lines[0].strip())
        self._atoms = [0] * self._nAtoms

        for i in range(self._nAtoms):
            self._atoms[i] = Atom.fromXYZ(self._lines[2 + i])

        self._displacements = []
        self._bonds = []


class SDFfile(Structure):
    def __init__(self, filepath):
        self._filepath = filepath
        with open(self._filepath, "r") as file:
            self._lines = file.readlines()

        self._nAtoms = int(self._lines[3].split()[0].strip())
        self._atoms = [0] * self._nAtoms

        for i in range(self._nAtoms):
            self._atoms[i] = Atom.fromSDF(self._lines[4 + i])

        self._displacements = []

        # SDF already contains connectivity, so maybe we can somehow read them and create the Bond instances?
        self._bonds = []


class JSONfile(Structure):
    def __init__(self, filepath):
        self._filepath = filepath
        with open(self._filepath, "r") as file:
            self._lines = file.readlines()
            jsonData = json.load(file)

        print(jsonData)

        # self._nAtoms = int(self._lines[3].split()[0].strip())
        # self._atoms = [0] * self._nAtoms

        # for i in range(self._nAtoms):
        #     self._atoms[i] = Atom.fromSDF(self._lines[4 + i])

        # self._displacements = []

        # # SDF already contains connectivity, so maybe we can somehow read them and create the Bond instances?
        # self._bonds = []


class Trajectory:
    def __init__(self, frames: list[Structure]):
        self._nframes = len(frames)
        self._frames = frames

    def get_frames(self) -> list[Structure]:
        return self._frames

    def get_nframes(self) -> int:
        return self._nframes

    def setCenterOfMass(self, newCOMposition, frameIndex=0):
        originalCOM = self._frames[frameIndex].getCenterOfMass()
        displacement = newCOMposition - originalCOM

        for frame in self._frames:
            for atom in frame.getAtoms():
                newPosition = atom.getPositionVector() + displacement
                atom.setPositionVector(newPosition)

    def createAnimation(self) -> None:
        frame_step = 10
        bpy.context.scene.frame_step = frame_step
        bpy.context.scene.frame_end = 1 + frame_step * (self._nframes - 1)

        initialFrame = self._frames[0]

        initialFrame.createAtoms(createMesh=False)
        previousPositions = initialFrame.getAllAtomPositions()

        allAtomElements = [atom.getElement() for atom in initialFrame.getAtoms()]
        allElementIndeces = [
            allAtomElements[:i].count(allAtomElements[i])
            for i in range(len(allAtomElements))
        ]

        for i, frame in enumerate(self._frames):
            currentFrameNr = 1 + i * frame_step
            currentPositions = frame.getAllAtomPositions()
            displacements = currentPositions - previousPositions

            previousPositions = currentPositions

            for j, atom in enumerate(frame.getAtoms()):
                objectName = f"atom-{allAtomElements[j]}"
                if allElementIndeces[j] > 0:
                    objectName += "." + str(allElementIndeces[j]).rjust(3, "0")

                # Select UV sphere with correct name
                obj = getObjectByName(objectName)

                translateObject(obj, displacements[j])

                obj.keyframe_insert(data_path="location", frame=currentFrameNr)

    def get_frame(self, frameIndex) -> Structure:
        return self._frames[frameIndex]


class ORCAgeomOptFile(Trajectory):
    def __init__(self, filepath):
        self._filepath = filepath
        with open(self._filepath, "r") as file:
            self._lines = file.readlines()

        beginStructure = findAllStringInListOfStrings(
            "CARTESIAN COORDINATES (ANGSTROEM)", self._lines
        )
        beginStructure = [beginIndex + 2 for beginIndex in beginStructure]

        endStructure = findAllStringInListOfStrings(
            "CARTESIAN COORDINATES (A.U.)", self._lines
        )
        endStructure = [endIndex - 2 for endIndex in endStructure]

        self._nframes = len(endStructure)
        structureLines = [
            (beginStructure[i], endStructure[i]) for i in range(self._nframes)
        ]

        self._frames = [0] * self._nframes
        for i, structureTuple in enumerate(structureLines):
            cartesianCoordLines = self._lines[structureTuple[0] : structureTuple[1]]
            print(cartesianCoordLines)

            _nAtoms = len(cartesianCoordLines)
            _atoms = [0] * _nAtoms
            for j in range(_nAtoms):
                _atoms[j] = Atom.fromXYZ(cartesianCoordLines[j])
            self._frames[i] = Structure(_atoms)


if __name__ == "__main__":
    structure = JSONfile(
        "/home/tobiasdijkhuis/Downloads/Structure2D_COMPOUND_CID_3034819.json"
    )
