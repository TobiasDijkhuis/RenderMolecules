from copy import deepcopy

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
            initialBonds,
            splitBondToAtomMaterials=splitBondToAtomMaterials,
            renderResolution=renderResolution,
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
                    raise NotImplementedError("Does not support bond creation yet")

                scale = bond.getBondLength() / previousBondLengths[j]
                # previousBondLengths[j] = bond.getBondLength()

                vdwWeightedLocations = bond.getVdWWeightedMidpoints(
                    allAtomElements[atom1Index], allAtomElements[atom2Index]
                )

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

                if (
                    not splitBondToAtomMaterials
                    or allAtomElements[atom1Index] == allAtomElements[atom2Index]
                ):
                    continue

                try:
                    obj = getObjectByName(f"bond-{atom1Index}-{atom2Index}.001")
                except KeyError:
                    # This bond did not exist yet, needs to be created
                    raise NotImplementedError("Does not support bond creation yet")

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

    def get_frame(self, frameIndex: int) -> Structure:
        """Get the Structure at a certain frame index"""
        return self._frames[frameIndex]

    def rotateAroundAxis(self, axis: np.ndarray, angle: float) -> None:
        """Rotate all frames in the trajectory around an axis counterclockwise with a certain angle in degrees"""
        for frame in self._frames:
            frame.rotateAroundAxis(axis, angle)

    @classmethod
    def _fromORCAgeomOpt(cls, filepath):
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
    def fromXSF(cls, filepath: str):
        raise NotImplementedError()

    @classmethod
    def fromORCAoutput(
        cls,
        filepath: str,
        useVibrations: bool = True,
        useGeometryOptimization: bool = False,
        vibrationNr: str | int = "imag",
        nFramesPerOscillation: float = 20,
        amplitude: float = 0.5,
    ):
        # Get trajectory corresponding to either vibrations of the normal modes or the geometry optimization
        if useVibrations == useGeometryOptimization:
            raise ValueError(
                f"One (and not two) of useVibrations and useGeometryOptimzation must be True, but both were {useVibrations}"
            )
        if useGeometryOptimization:
            return cls._fromORCAgeomOpt(filepath)

        with open(filepath, "r") as file:
            _lines = file.readlines()

        resultFrequencies = findFirstStringInListOfStrings(
            "VIBRATIONAL FREQUENCIES", _lines
        )
        if resultFrequencies is None:
            raise ValueError("No 'VIBRATIONAL FREQUENCIES' found in ORCA output")

        resultNormal = findFirstStringInListOfStrings("NORMAL MODES", _lines)
        if resultNormal is None:
            raise ValueError("No 'NORMAL MODES' found in ORCA output")

        resultIRspec = findFirstStringInListOfStrings(
            "IR SPECTRUM", _lines, start=resultNormal
        )

        frequenciesLines = _lines[resultFrequencies + 5 : resultNormal - 3]
        nfreqs = len(frequenciesLines)

        if isinstance(vibrationNr, str):
            # Try to infer which vibration is the imaginary one, and then give trajectory corresponding to that one.
            if not vibrationNr in ["i", "im", "imag", "imaginary"]:
                raise ValueError(
                    f"If vibrationNr is a string, it should be one of ['i', 'im', 'imag', imaginary'], but was '{vibrationNr}'"
                )
            resultImag = findAllStringInListOfStrings(
                r"***imaginary mode***", _lines, start=resultNormal
            )

            if not resultImag:
                raise ValueError(
                    "Tried to visualize imaginary normal mode, but no imaginary frequency found"
                )

            if len(resultImag) > 1:
                raise ValueError(
                    "Tried to visualize imaginary mode, but multiple imaginary frequencies found. This code cannot decide which to use"
                )
            vibrationNr = int(_lines[resultImag[0]].split()[0].strip(":"))
            print(f"Imaginary vibration was found to be vibrationNr {vibrationNr}")
        elif not isinstance(vibrationNr, int):
            raise TypeError(
                f"vibrationNr should be of type int or string, but was type {type(vibrationNr)}"
            )

        if vibrationNr > nfreqs - 1:
            raise ValueError(
                f"Tried to visualize vibrationNr {vibrationNr}, but only {nfreqs} vibrations found"
            )

        frequency = float(frequenciesLines[vibrationNr].split()[1])
        if frequency == 0.0:
            print(
                "Warning: Trying to visualize normal mode with frequency 0. Amplitudes will be 0 too so animation shows no vibrations"
            )

        natomsResult = findFirstStringInListOfStrings("Number of atoms", _lines)
        natoms = int(_lines[natomsResult].split()[-1])

        # Read displacements. ORCA has the displacements as rows, and frequencies as columns
        # Each "block" has 6 columns (or less if it's the last block), and 3*nAtoms rows.
        nlinesAmplitudes = 3 * natoms
        blockNr = vibrationNr // 6
        colNr = vibrationNr % 6

        # Get the lines corresponding to the correct vibration
        linesModes = _lines[
            resultNormal + 8 + blockNr * (nlinesAmplitudes + 1) : resultNormal
            + 8
            + (blockNr + 1) * (nlinesAmplitudes + 1)
            - 1
        ]
        assert len(linesModes) == nlinesAmplitudes, (
            f"NOT CORRECT LENGTH. SHOULD HAVE BEEN 3*natoms={nlinesAmplitudes}, but was {len(linesModes)}"
        )
        # Read correct column, and reshape displacements to nAtoms*3 matrix for x, y, z
        displacementsMassWeighed = np.array(
            [float(i.split()[colNr + 1]) for i in linesModes]
        ).reshape(natoms, 3)

        beginStructure = findAllStringInListOfStrings(
            "CARTESIAN COORDINATES (ANGSTROEM)", _lines, end=resultFrequencies
        )
        if beginStructure is None:
            raise ValueError()

        # In case there was a geometry optimization, use the final (optimized) structure
        finalStructureLine = beginStructure[-1]
        baseStructure = Structure(
            [
                Atom.fromXYZ(line)
                for line in _lines[
                    finalStructureLine + 2 : finalStructureLine + 2 + natoms
                ]
            ]
        )

        # The displacements in the ORCA output are mass-weighted (i.e. scaled by 1/sqrt(mass))
        # so if you want you can use these, but generally it looks better to use the mass-weighted ones.
        masses = np.array([atom.getMass() for atom in baseStructure.getAtoms()])
        displacements = displacementsMassWeighed * np.sqrt(masses)[:, np.newaxis]

        frames = [deepcopy(baseStructure) for i in range(nFramesPerOscillation)]
        # Displace atoms in structure according to phase.
        # We do not need to loop over the first and last frame (i.e. phase = 0 and 2*pi),
        # since the displacement will be np.sin(0)=np.sin(2pi)=0,
        # so the baseStructure is unaltered.
        for i in range(1, nFramesPerOscillation - 1):
            phase = i * 2.0 * np.pi / (nFramesPerOscillation - 1)
            frames[i].displaceAtoms(
                np.sin(phase) * amplitude * displacementsMassWeighed
            )
        return cls(frames)

    def translate(self, translationVector: np.ndarray | list) -> None:
        for frame in self._frames:
            frame.translate(translationVector)
