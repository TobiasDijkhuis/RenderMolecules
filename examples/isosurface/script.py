from __future__ import annotations

import os

from render_molecules.blender_utils import create_isosurface, delete_all_objects
from render_molecules.other_utils import get_render_molecules_dir
from render_molecules.structure import CUBEfile


def main():
    delete_all_objects()

    # Read the CUBE file
    render_molecules_dir = get_render_molecules_dir()
    cube_path = os.path.join(
        render_molecules_dir, "../examples/isosurface/CH3OH_OPT.eldens.cube"
    )
    cube = CUBEfile(cube_path)

    # Read the electron density in the CUBE file
    cube.read_volumetric_data()

    # Set center of mass to origin
    cube.set_center_of_mass([0, 0, 0])

    # Create the atoms and bonds
    cube.create_structure()

    # Calculate and render the isosurface
    isovalue = 0.025
    verts, edges, _, _ = cube.calculate_isosurface(isovalue)
    create_isosurface(verts, edges, isovalue, "CH3OH")
    return


if __name__ == "__main__":
    main()
