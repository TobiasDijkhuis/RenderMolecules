import numpy as np

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta degrees.
    """
    theta *= np.pi / 180.0
    axis = np.asarray(axis)
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
    vector1 /= np.linalg.norm(vector1)
    vector2 /= np.linalg.norm(vector2)
    axis = np.cross(vector1, vector2)
    if np.linalg.norm(axis) < 1e-5:
        angle = 0.0
    else:
        angle = np.arccos(np.dot(vector1, vector2))
    return angle * 180 / np.pi


def translateObject(object, displacementVector):
    location = object.location
    location.x += displacementVector[0]
    location.y += displacementVector[1]
    location.z += displacementVector[2]
