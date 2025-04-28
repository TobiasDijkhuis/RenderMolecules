def getElementFromAtomicNumber(atomicNumber: int) -> str:
    try:
        element = elementList[atomicNumber - 1]
    except ValueError:
        msg = f"Could not determine element from atomic number {atomicNumber}"
        raise ValueError(msg)
    return element


def getAtomicNumberFromElement(element: str) -> int:
    try:
        atomicNumber = elementList.index(element) + 1
    except ValueError:
        msg = f"Could not determine atomic number from element {element}"
        raise ValueError()
    return atomicNumber


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

# manifest that contains atomic information
manifest = {
    "atom_colors": {
        "C": "555555",
        "H": "DDDDDD",
        "O": "FF0000",
        "N": "0000FF",
        "S": "FFFF30",
    },
    "bond_thickness": 0.2,
    "bond_color": "4444444",
    "hbond_color": "999999",
    "hbond_thickness": 0.035,
    "hbond_distance": 3.5,
    "hbond_angle": 30,
}

elementMass = [
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
    "HO": 1.5,
    "CO": 1.5,
    "CH": 1.5,
    "OO": 1.5,
    "HH": 1.2,
    "NN": 1.5,
    "HN": 1.5,
    "CN": 1.5,
    "CC": 1.5,
    "NO": 1.5,
    "SS": 1.5,
    "NS": 2.0,
    "HS": 1.5,
    "CS": 1.5,
    "OS": 1.5,
}

hydrogenBondLength = 3.5  # Maximum hydrogen bond length is 3.5 Angstrom
hydrogenBondAngle = 35  # Maximum hydrogen bond angle is 35 degrees
sphereScale = 0.3  # Created atoms have a radius of 0.3*r_VDW

BOHR_TO_ANGSTROM = 0.5291177249
ANGSTROM_TO_BOHR = 1 / BOHR_TO_ANGSTROM
BOHR_TO_METERS = 5.2917721067121 * 1e-11
METERS_TO_BOHR = 1 / BOHR_TO_METERS
ANGSTROM_TO_METERS = 1e-10
METERS_TO_ANGSTROM = 1e10
AMU_TO_KG = 1.66053907 * 1e-27
KG_TO_AMU = 1 / AMU_TO_KG

KGM2_TO_AMU_ANGSTROM2 = KG_TO_AMU * METERS_TO_ANGSTROM * METERS_TO_ANGSTROM
