from .atom import Atom
from .bond import Bond
from .ElementData import manifest, bondLengths
import numpy as np

from .blenderUtils import (
    createCylinder,
    createUVsphere,
    create_material,
    createIsosurface,
    createMeshAtoms,
)

from .geometry import Geometry, rotation_matrix
from .constants import AMU_TO_KG, BOHR_TO_ANGSTROM, ANGSTROM_TO_METERS, BOHR_TO_METERS

class Structure(Geometry):
    def __init__(self, atomList: list[Atom], bonds: list[Bond] | None = None):
        self._nAtoms = len(atomList)
        self._atoms = atomList

        if bonds is None:
            self._bonds = []
        else:
            self._bonds = bonds

        self._displacements = []
        self._affineMatrix = np.identity(4)

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
        """Create the atoms in the scene"""
        if not createMesh:
            # This is an old, naive method where we create a lot more spheres
            for atom in self._atoms:
                obj = createUVsphere(
                    atom.getElement(),
                    atom.getPositionVector(),
                    renderResolution=renderResolution,
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
            obj = createUVsphere(
                atomType, np.array([0, 0, 0]), renderResolution=renderResolution
            )
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
        return self._bonds

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
        """Get the center of mass position vector"""
        masses = np.array([atom.getMass() for atom in self._atoms])
        atomPositions = self.getAllAtomPositions()
        COM = np.array(
            sum(masses[i] * atomPositions[i] for i in range(self._nAtoms)) / sum(masses)
        )
        return COM

    def setCenterOfMass(self, newCOMposition):
        """Set the center of mass of the whole system to a new position"""
        COM = self.getCenterOfMass()
        translationVector = newCOMposition - COM
        self.translate(translationVector)

    def setAveragePosition(self, newAveragePosition):
        """Sets the average position of all atoms to a new position"""
        averagePosition = np.average(
            np.array([atom.getPositionVector() for atom in self._atoms]), axis=0
        )
        translationVector = newAveragePosition - averagePosition
        self.translate(translationVector)

    def translate(self, translationVector: np.ndarray) -> None:
        newTransform = np.identity(4)
        newTransform[:3, 3] = translationVector
        self.addTransformation(newTransform)

        for atom in self._atoms:
            atom.setPositionVector(atom.getPositionVector()+translationVector)


    def getTotalCharge(self) -> int:
        """Get the total charge in the system"""
        charges = (atom.getCharge() for atom in self._atoms)
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

    def rotateAroundAxis(self, axis: np.ndarray, angle: float) -> None:
        """Rotate the structure around a certain axis counterclockwise with a certain angle in degrees"""
        rotMatrix = rotation_matrix(axis, angle)
        # create 4x4 matrix from the 3x3 rotation matrix
        newTransform = np.identity(4)
        newTransform[:3, :3] = rotMatrix
        self.addTransformation(newTransform)

        for atom in self._atoms:
            currentPos = atom.getPositionVector()
            rotatedPos = np.dot(rotMatrix, currentPos)
            atom.setPositionVector(rotatedPos)

    def getInertiaTensor(self) -> np.ndarray:
        """Get the moment of inertia tensor, in kg m2"""
        # https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        centerOfMass = self.getCenterOfMass()

        inertiaTensor = np.zeros((3, 3))
        for atom in self._atoms:
            # Calculate moments of inertia to axes wrt COM
            coords = atom.getPositionVector() - centerOfMass

            mass = atom.getMass() * AMU_TO_KG  # Mass in kg

            # Convert coordinates to meters
            if atom.isAngstrom:
                coords *= ANGSTROM_TO_METERS
            else:
                coords *= BOHR_TO_METERS

            inertiaTensor[0, 0] += mass * (
                coords[1] * coords[1] + coords[2] * coords[2]
            )
            inertiaTensor[1, 1] += mass * (
                coords[0] * coords[0] + coords[2] * coords[2]
            )
            inertiaTensor[2, 2] += mass * (
                coords[0] * coords[0] + coords[1] * coords[1]
            )
            inertiaTensor[0, 1] -= mass * coords[0] * coords[1]
            inertiaTensor[0, 2] -= mass * coords[0] * coords[2]
            inertiaTensor[1, 2] -= mass * coords[1] * coords[2]
        inertiaTensor[1, 0] = inertiaTensor[0, 1]
        inertiaTensor[2, 0] = inertiaTensor[0, 2]
        inertiaTensor[2, 1] = inertiaTensor[1, 2]
        return inertiaTensor


    def getPrincipalMomentsOfInertia(self):
        """Get the principal moments of inertia (in kg m2) and the principal axes"""
        inertiaTensor = self.getInertiaTensor()
        principalMoments, principalAxes = np.linalg.eig(inertiaTensor)
        return principalMoments, principalAxes

    def createHydrogenBonds(self) -> None:
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

    def createBonds(
        self,
        bonds: list[Bond],
        splitBondToAtomMaterials: bool = True,
        renderResolution: str = "medium",
    ) -> None:
        """Create the bonds in the Blender scene"""
        allAtomElements = [atom.getElement() for atom in self._atoms]
        allAtomVdWRadii = [atom.getVdWRadius() for atom in self._atoms]

        for bond in bonds:
            direction = bond.getDirection()
            axisAngleWithZ = bond.getAxisAngleWithZaxis()
            bondLength = bond.getBondLength()
            bondMidpoint = bond.getMidpointPosition()

            if splitBondToAtomMaterials:
                # We will move the two cylinders according to their vdw radii, such that each atom
                # has about the same of its material shown in the bond. This means that for, e.g.
                # an O-H bond, the cylinder closer to O will move less than the cylinder closer to H
                atom1Index = bond.getAtom1Index()
                atom1Element = allAtomElements[atom1Index]

                atom2Index = bond.getAtom2Index()
                atom2Element = allAtomElements[atom2Index]

                if atom1Element == atom2Element:
                    mat1 = create_material(
                        atom1Element, manifest["atom_colors"][atom1Element]
                    )
                    obj = createCylinder(
                        bondMidpoint,
                        axisAngleWithZ,
                        manifest["bond_thickness"],
                        bondLength / 4,
                        renderResolution=renderResolution,
                        name=f"bond-{atom1Index}-{atom2Index}",
                    )
                    obj.data.materials.append(mat1)
                    continue

                vdwWeightedLocations = bond.getVdWWeightedMidpoints(
                    atom1Element, atom2Element
                )

                # Because of how we calculate the bonds, the first cylinder (where we subtract the direction
                # from the midpoint) will be the one closest to the atom with the higher index.
                # So, we take the element and material from that one, and assign it to the first cylinder.
                mat2 = create_material(
                    atom2Element, manifest["atom_colors"][atom2Element]
                )

                # First cylinder
                obj = createCylinder(
                    vdwWeightedLocations[0],
                    axisAngleWithZ,
                    manifest["bond_thickness"],
                    bondLength / 4,
                    renderResolution=renderResolution,
                    name=f"bond-{atom1Index}-{atom2Index}",
                )
                obj.data.materials.append(mat2)

                mat1 = create_material(
                    atom1Element, manifest["atom_colors"][atom1Element]
                )

                # First cylinder
                obj = createCylinder(
                    vdwWeightedLocations[1],
                    axisAngleWithZ,
                    manifest["bond_thickness"],
                    bondLength / 4,
                    renderResolution=renderResolution,
                    name=f"bond-{atom1Index}-{atom2Index}",
                )
                obj.data.materials.append(mat1)
            else:
                obj = createCylinder(
                    bondMidpoint,
                    axisAngleWithZ,
                    manifest["bond_thickness"],
                    bondLength / 2,
                    renderResolution=renderResolution,
                    name=f"bond-{atom1Index}-{atom2Index}",
                )

    @classmethod
    def fromXYZ(cls, filepath: str):
        """Create a Structure from an XYZ file"""
        with open(filepath, "r") as file:
            _lines = file.readlines()

        _nAtoms = int(_lines[0].strip())
        _atoms = [0] * _nAtoms

        for i in range(_nAtoms):
            _atoms[i] = Atom.fromXYZ(_lines[2 + i])

        return cls(_atoms)

    @classmethod
    def fromSDF(cls, filepath: str):
        """Creates a Structure from an SDF file"""
        with open(_filepath, "r") as file:
            _lines = file.readlines()

        _nAtoms = int(_lines[3].split()[0].strip())
        _atoms = [0] * _nAtoms

        for i in range(_nAtoms):
            _atoms[i] = Atom.fromSDF(_lines[4 + i])

        # SDF already contains connectivity, so maybe we can somehow read them and create the Bond instances?
        _bonds = []
        return cls(_atoms, _bonds)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Structure with {self._nAtoms} atoms"


class CUBEfile(Structure):
    def __init__(self, filepath: str):
        with open(filepath, "r") as file:
            self._lines = file.readlines()
        nAtoms = int(self._lines[2].split()[0].strip())

        if nAtoms < 0:
            nAtoms = -nAtoms

        atoms = [0] * nAtoms

        for i in range(nAtoms):
            atoms[i] = Atom.fromCUBE(self._lines[6 + i].strip())
            atoms[i].positionBohrToAngstrom()

        super().__init__(atoms)

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

        try:
            self._volumetricData = np.fromiter(
                (
                    float(num)
                    for line in self._lines[6 + self._nAtoms :]
                    for num in line.split()
                ),
                dtype=np.float32,
                count=-1,
            ).reshape((self._NX, self._NY, self._NZ))
        except ValueError:
            self._volumetricData = np.fromiter(
                (
                    float(num)
                    for line in self._lines[7 + self._nAtoms :]
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

    def writePLY(self, filepath: str, isovalue: float) -> None:
        """Write the volumetric data to a filepath"""
        from pytessel import PyTessel

        self._checkIsovalue(isovalue)

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

        nvertices = np.shape(vertices)[0]
        vertices4D = np.concatenate([vertices, np.ones((nvertices, 1))], axis=1)
        vertices = (self._affineMatrix @ vertices4D.T).T[:, :3]

        pytessel.write_ply(filepath, vertices, normals, indices)

    def calculateIsosurface(
        self, isovalue: float
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Calculate the isosurface from the volumetric data and an isovalue"""
        from skimage.measure import marching_cubes
        
        self._checkIsovalue(isovalue)

        vertices, faces, normals, values = marching_cubes(
            self._volumetricData,
            level=isovalue,
            spacing=np.diag(self._volumetricAxisVectors),
        )

        vertices += self._volumetricOriginVector
        
        nvertices = np.shape(vertices)[0]
        vertices4D = np.concatenate([vertices, np.ones((nvertices, 1))], axis=1)
        vertices = (self._affineMatrix @ vertices4D.T).T[:, :3]

        return vertices, faces, normals, values

    def _checkIsovalue(self, isovalue: float) -> None:
        """Checks whether the supplied isovalue is valid"""
        if isovalue <= np.min(self._volumetricData):
            msg = f"Set isovalue ({isovalue}) was less than or equal to the minimum value in the volumetric data ({np.min(self._volumetricData)}). This will result in an empty isosurface. Set a larger isovalue."
            raise ValueError(msg)
        if isovalue >= np.max(self._volumetricData):
            msg = f"Set isovalue ({isovalue}) was more than or equal to the maximum value in the volumetric data ({np.max(self._volumetricData)}). This will result in an empty isosurface. Set a smaller isovalue."
            raise ValueError(msg)

class JSONfile(Structure):
    def __init__(self, filepath):
        import json

        self._filepath = filepath
        with open(self._filepath, "r") as file:
            self._lines = file.readlines()
            jsonData = json.load(file)

        print(jsonData)
