import os

from render_molecules.structure import CUBEfile
from render_molecules.blender_utils import delete_all_objects, create_isosurface
from render_molecules.other_utils import get_render_molecules_dir

def main():
    delete_all_objects()
    
    # Read the CUBE file
    render_molecules_dir = get_render_molecules_dir()
    CUBEpath = os.path.join(render_molecules_dir, "../examples/isosurface/CH3OH_OPT.eldens.cube")
    CUBE = CUBEfile(CUBEpath)
    
    # Read the electron density in the CUBE file
    CUBE.read_volumetric_data()
    
    # Set center of mass to origin
    CUBE.set_center_of_mass([0, 0,0])
    
    # Create the atoms and bonds
    CUBE.create_structure()
    
    # Calculate and render the isosurface
    isovalue = 0.025
    verts, edges, _, _ = CUBE.calculate_isosurface(isovalue)
    create_isosurface(verts, edges, isovalue, "CH3OH")
    return

if __name__ == '__main__':
    main()
