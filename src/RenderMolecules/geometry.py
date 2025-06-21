import numpy as np


class Geometry:
    def __init__(self):
        self._affineMatrix = np.identity(4)

    def getAffineMatrix(self) -> np.ndarray:
        return self._affineMatrix

    def setAffineMatrix(self, newAffineMatrix) -> None:
        shape = np.shape(newAffineMatrix)
        if not shape == (4, 4):
            msg = f"affineTransformationMatrix was not the correct shape. Should have been (4, 4), but was {shape}"
            raise ValueError(msg)
        self._affineMatrix = newAffineMatrix

    def addTransformation(self, affineTransformationMatrix) -> None:
        shape = np.shape(affineTransformationMatrix)
        if not shape == (4, 4):
            msg = f"affineTransformationMatrix was not the correct shape. Should have been (4, 4), but was {shape}"
            raise ValueError(msg)
        self.setAffineMatrix(affineTransformationMatrix @ self._affineMatrix)

    def translate(self, translationVector) -> None:
        msg = "Geometry.translate method should be implemented by child class"
        raise NotImplementedError(msg)

    def rotateAroundAxis(self, axis, angle) -> None:
        msg = "Geometry.rotateAroundAxis method should be implemented by child class"
        raise NotImplementedError(msg)

    def rotateAroundX(self, angle: float) -> None:
        """Rotate the geometry around the x-axis counterclockwise with a certain angle in degrees"""
        self.rotateAroundAxis([1, 0, 0], angle)

    def rotateAroundY(self, angle: float) -> None:
        """Rotate the geometry around the y-axis counterclockwise with a certain angle in degrees"""
        self.rotateAroundAxis([0, 1, 0], angle)

    def rotateAroundZ(self, angle: float) -> None:
        """Rotate the geometry around the z-axis counterclockwise with a certain angle in degrees"""
        self.rotateAroundAxis([0, 0, 1], angle)


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta degrees.
    """
    check3Dvector(axis)

    theta *= np.pi / 180.0
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def angle_between(vector1, vector2) -> float:
    vector1 = check3Dvector(vector1)
    vector2 = check3Dvector(vector2)

    vector1 /= np.linalg.norm(vector1)
    vector2 /= np.linalg.norm(vector2)
    axis = np.cross(vector1, vector2)
    if np.linalg.norm(axis) < 1e-5:
        angle = 0.0
    else:
        angle = np.arccos(np.dot(vector1, vector2))
    return angle * 180 / np.pi


def check3Dvector(vector: list | np.ndarray) -> np.ndarray:
    if isinstance(vector, list):
        vector = np.asarray(vector)

    if not np.shape(vector) == (3,):
        msg = f"vector was supposed to be shape 3, but was shape {np.shape(vector)}"
        raise ValueError(msg)

    return vector
