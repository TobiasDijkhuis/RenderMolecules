from __future__ import annotations

import numpy as np


class Geometry:
    def __init__(self):
        self._affine_matrix = np.identity(4)

    def get_affine_matrix(self) -> np.ndarray:
        return self._affine_matrix

    def set_affine_matrix(self, new_affine_matrix: np.ndarray) -> None:
        new_affine_matrix = check_4x4_matrix(new_affine_matrix)
        self._affine_matrix = new_affine_matrix

    def add_transformation(self, affine_transform_matrix: np.ndarray) -> None:
        affine_transform_matrix = check_4x4_matrix(affine_transform_matrix)
        self.set_affine_matrix(affine_transform_matrix @ self._affine_matrix)

    def translate(self, translation_vector) -> None:
        msg = "Geometry.translate method should be implemented by child class"
        raise NotImplementedError(msg)

    def rotate_around_axis(self, axis, angle) -> None:
        msg = "Geometry.rotateAroundAxis method should be implemented by child class"
        raise NotImplementedError(msg)

    def rotate_around_x(self, angle: float) -> None:
        """Rotate the geometry around the x-axis counterclockwise with a certain angle in degrees"""
        self.rotate_around_axis([1, 0, 0], angle)

    def rotate_around_y(self, angle: float) -> None:
        """Rotate the geometry around the y-axis counterclockwise with a certain angle in degrees"""
        self.rotate_around_axis([0, 1, 0], angle)

    def rotate_around_z(self, angle: float) -> None:
        """Rotate the geometry around the z-axis counterclockwise with a certain angle in degrees"""
        self.rotate_around_axis([0, 0, 1], angle)


def rotation_matrix(axis: list | np.ndarray, theta: float) -> np.ndarray:
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta degrees.
    """
    check_3d_vector(axis)

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


def angle_between(vector1: np.ndarray | list, vector2: np.ndarray | list) -> float:
    """Get the angle between two vectors.

    Args:
        vector1 (ndarray): list or np array of length 3
        vector2 (ndarray): list or np array of length 3

    Returns:
        float: angle between two vectors in degrees
    """
    vector1 = check_3d_vector(vector1)
    vector2 = check_3d_vector(vector2)

    vector1 /= np.linalg.norm(vector1)
    vector2 /= np.linalg.norm(vector2)
    axis = np.cross(vector1, vector2)
    if np.linalg.norm(axis) < 1e-5:
        angle = 0.0
    else:
        angle = np.arccos(np.dot(vector1, vector2))
    return angle * 180 / np.pi


def check_3d_vector(vector: list | np.ndarray) -> np.ndarray:
    if isinstance(vector, list):
        vector = np.asarray(vector)
    if not isinstance(vector, np.ndarray):
        msg = f"vector was supposed to be np array, but was instead type {type(vector)}"
        raise TypeError(msg)
    if np.shape(vector) != (3,):
        msg = f"vector was supposed to be shape 3, but was shape {np.shape(vector)}"
        raise ValueError(msg)
    return vector


def check_4x4_matrix(matrix: np.ndarray) -> np.ndarray:
    if not isinstance(matrix, np.ndarray):
        msg = f"matrix was supposed to be np array, but was instead type {type(matrix)}"
        raise TypeError(msg)
    if np.shape(matrix) != (4, 4):
        msg = f"matrix was supposed to be shape 4, but was shape {np.shape(matrix)}"
        raise ValueError(msg)
    return matrix
