from __future__ import annotations


def get_element_from_atomic_number(atomic_number: int) -> str:
    try:
        element = element_list[atomic_number - 1]
    except ValueError as e:
        msg = f"Could not determine element from atomic number {atomic_number}"
        raise ValueError(msg) from e
    return element


def get_atomic_number_from_element(element: str) -> int:
    try:
        atomic_number = element_list.index(element) + 1
    except ValueError as e:
        msg = f"Could not determine atomic number from element {element}"
        raise ValueError(msg) from e
    return atomic_number


# manifest that contains atomic information
manifest = {
    "atom_colors": {
        "C": "555555",
        "H": "DDDDDD",
        "O": "FF0000",
        "N": "0000FF",
        "S": "FFFF30",
        "Si": "F0C8A0",
    },
    "bond_thickness": 0.2,
    "bond_color": "4444444",
    "hbond_color": "999999",
    "hbond_thickness": 0.035,
}

element_list = [
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


element_mass = [
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
vdw_radii = [
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
bond_lengths = {
    "H-O": 1.5,
    "C-O": 1.5,
    "C-H": 1.5,
    "O-O": 1.5,
    "H-H": 1.2,
    "N-N": 1.5,
    "H-N": 1.5,
    "C-N": 1.5,
    "C-C": 1.5,
    "N-O": 1.5,
    "S-S": 1.5,
    "N-S": 2.0,
    "H-S": 1.5,
    "C-S": 1.5,
    "O-S": 1.5,
    "Si-Si": 2.7,
    "O-Si": 2.0,
}
