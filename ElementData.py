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
    1.008,
    4,
    7,
    9,
    11,
    12.01,
    14.007,
    16.00,
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

# Bond lengths in Angstrom
bondLengths = {
    "HO": 2.0,
    "CO": 2.0,
    "CH": 2.0,
    "OO": 2.0,
    "HH": 0.75,
    "NN": 1.5,
    "HN": 1.0,
    "CN": 1.5,
    "CC": 1.5,
}
hydrogenBondLength = 3.5
hydrogenBondAngle = 35
sphereScale = 0.2

BOHR_TO_ANGSTROM = 0.5291177249
ANGSTROM_TO_BOHR = 1 / BOHR_TO_ANGSTROM
