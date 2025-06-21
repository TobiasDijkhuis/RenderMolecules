import bpy
import numpy as np

from .atom import Atom
from .blenderUtils import getObjectByName
from .geometry import Geometry
from .otherUtils import (findAllStringInListOfStrings,
                         findFirstStringInListOfStrings)
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

    def setCenterOfMass(self, newCOMposition: np.ndarray, frameIndex: int = 0) -> None:
        """Set the Center Of Mass of the structure at index 'frameIndex' to a new position"""
        originalCOM = self._frames[frameIndex].getCenterOfMass()
        displacement = newCOMposition - originalCOM

        for frame in self._frames:
            for atom in frame.getAtoms():
                newPosition = atom.getPositionVector() + displacement
                atom.setPositionVector(newPosition)

    def createAnimation(
        self,
        createBonds: bool = True,
        renderResolution: str = "medium",
        splitBondToAtomMaterials: bool = True,
    ) -> None:
        """Create the animation of the trajectory in the Blender scene"""
        # At the moment, this looks a bit weird if bond lengths change a lot.
        # Also, does not support bond creation/destruction.
        frame_step = 10
        bpy.context.scene.frame_step = frame_step
        bpy.context.scene.frame_end = 1 + frame_step * (self._nframes - 1)

        initialFrame = self._frames[0]

        initialFrame.createAtoms(createMesh=False, renderResolution=renderResolution)

        initialBonds = initialFrame.findBondsBasedOnDistance()
        initialFrame.createBonds(
            initialBonds, splitBondToAtomMaterials, renderResolution=renderResolution
        )

        allAtomElements = [atom.getElement() for atom in initialFrame.getAtoms()]
        allElementIndeces = [
            allAtomElements[:i].count(allAtomElements[i])
            for i in range(len(allAtomElements))
        ]

        previousBondLengths = [bond.getBondLength() for bond in initialBonds]

        for i, frame in enumerate(self._frames):
            currentFrameNr = 1 + i * frame_step
            currentPositions = frame.getAllAtomPositions()
            for j, atom in enumerate(frame.getAtoms()):
                objectName = f"atom-{allAtomElements[j]}"
                if allElementIndeces[j] > 0:
                    objectName += "." + str(allElementIndeces[j]).rjust(3, "0")

                # Select UV sphere with correct name
                obj = getObjectByName(objectName)

                obj.location.x = currentPositions[j, 0]
                obj.location.y = currentPositions[j, 1]
                obj.location.z = currentPositions[j, 2]

                obj.keyframe_insert(data_path="location", frame=currentFrameNr)

            # Now reposition and rerotate all the bonds.
            if i >= 1:
                currentBonds = frame.findBondsBasedOnDistance()
            else:
                currentBonds = initialBonds

            for j, bond in enumerate(currentBonds):
                atom1Index = bond.getAtom1Index()
                atom2Index = bond.getAtom2Index()
                try:
                    obj = getObjectByName(f"bond-{atom1Index}-{atom2Index}")
                except KeyError:
                    # This bond did not exist yet, needs to be created
                    continue

                scale = bond.getBondLength() / previousBondLengths[j]
                previousBondLengths[j] = bond.getBondLength()

                vdwWeightedLocations = bond.getVdWWeightedMidpoints(
                    allAtomElements[atom1Index], allAtomElements[atom2Index]
                )

                obj.location.x = vdwWeightedLocations[1, 0]
                obj.location.y = vdwWeightedLocations[1, 1]
                obj.location.z = vdwWeightedLocations[1, 2]

                obj.scale[2] = scale
                obj.rotation_axis_angle = bond.getAxisAngleWithZaxis()
                obj.keyframe_insert(data_path="location", frame=currentFrameNr)
                obj.keyframe_insert(
                    data_path="rotation_axis_angle", frame=currentFrameNr
                )
                obj.keyframe_insert(data_path="scale", frame=currentFrameNr)

                if (
                    not splitBondToAtomMaterials
                    or allAtomElements[atom1Index] == allAtomElements[atom2Index]
                ):
                    continue

                try:
                    obj = getObjectByName(f"bond-{atom1Index}-{atom2Index}.001")
                except KeyError:
                    # This bond did not exist yet, needs to be created
                    continue

                obj.location.x = vdwWeightedLocations[0, 0]
                obj.location.y = vdwWeightedLocations[0, 1]
                obj.location.z = vdwWeightedLocations[0, 2]

                obj.scale[2] = scale

                obj.rotation_axis_angle = bond.getAxisAngleWithZaxis()

                obj.keyframe_insert(data_path="location", frame=currentFrameNr)
                obj.keyframe_insert(
                    data_path="rotation_axis_angle", frame=currentFrameNr
                )
                obj.keyframe_insert(data_path="scale", frame=currentFrameNr)

    def get_frame(self, frameIndex: int) -> Structure:
        """Get the Structure at a certain frame index"""
        return self._frames[frameIndex]

    def rotateAroundAxis(self, axis: np.ndarray, angle: float) -> None:
        """Rotate all frames in the trajectory around an axis counterclockwise with a certain angle in degrees"""
        for frame in self._frames:
            frame.rotateAroundAxis(axis, angle)

    @classmethod
    def fromORCAgeomOpt(cls, filepath):
        """Generate a trajectory from an ORCA geometry optimzation output file"""
        with open(filepath, "r") as file:
            _lines = file.readlines()

        beginStructure = findAllStringInListOfStrings(
            "CARTESIAN COORDINATES (ANGSTROEM)", _lines
        )
        beginStructure = [beginIndex + 2 for beginIndex in beginStructure]

        endStructure = findAllStringInListOfStrings(
            "CARTESIAN COORDINATES (A.U.)", _lines
        )
        endStructure = [endIndex - 2 for endIndex in endStructure]

        _nframes = len(endStructure)
        structureLines = [(beginStructure[i], endStructure[i]) for i in range(_nframes)]

        _frames = [0] * _nframes
        for i, structureTuple in enumerate(structureLines):
            cartesianCoordLines = _lines[structureTuple[0] : structureTuple[1]]

            _nAtoms = len(cartesianCoordLines)
            _atoms = [0] * _nAtoms
            for j in range(_nAtoms):
                _atoms[j] = Atom.fromXYZ(cartesianCoordLines[j])
            _frames[i] = Structure(_atoms)
        return cls(_frames)

    @classmethod
    def fromXYZ(cls, filepath):
        """Generate a trajectory from an XYZ file containing multiple frames"""
        with open(filepath, "r") as file:
            _lines = file.readlines()

        _frames = []
        for i, line in enumerate(_lines):
            line = line.strip()
            try:
                # If the line is just an integer, it is the amount of atoms in that frame
                _nAtoms = int(line)
            except ValueError:
                # If it is not, we skip the line
                continue

            _atoms = [0] * _nAtoms
            for j in range(_nAtoms):
                _atoms[j] = Atom.fromXYZ(_lines[i + 2 + j])
            _frames.append(Structure(_atoms))
        return cls(_frames)

    @classmethod
    def fromXSF(cls, filepath):
        raise NotImplementedError()

    def translate(self, translationVector) -> None:
        for frame in self._frames:
            frame.translate(translationVector)
