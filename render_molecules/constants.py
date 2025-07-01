from __future__ import annotations

BOHR_TO_ANGSTROM = 0.5291177249
"""Angstrom per Bohr"""

ANGSTROM_TO_BOHR = 1 / BOHR_TO_ANGSTROM
"""Bohr per Angstrom"""

BOHR_TO_METERS = 5.2917721067121e-11
"""Meter per Bohr"""

METERS_TO_BOHR = 1 / BOHR_TO_METERS
"""Bohr per Meter"""

ANGSTROM_TO_METERS = 1e-10
"""Meter per Angstrom"""

METERS_TO_ANGSTROM = 1e10
"""Angstrom per Meter"""

AMU_TO_KG = 1.66053907e-27
"""kg per amu"""

KG_TO_AMU = 1 / AMU_TO_KG
"""amu per kg"""

KGM2_TO_AMU_ANGSTROM2 = KG_TO_AMU * METERS_TO_ANGSTROM * METERS_TO_ANGSTROM
"""amu angstrom2 per kg m2"""

HYDROGEN_BOND_LENGTH = 3.5
"""Maximum hydrogen bond length"""
HYDROGEN_BOND_ANGLE = 35.0
"""Maximum hydrogen bond angle"""

SPHERE_SCALE = 0.3
"""Fraction of Van der Waals radius for created spheres"""

CYLINDER_LENGTH_FRACTION_SPLIT = 1.0 / 4.0
"""Cylinder length fraction for a bond split between two materials"""

CYLINDER_LENGTH_FRACTION = 1.0 / 2.0
"""Cylinder length fraction for a bond"""
