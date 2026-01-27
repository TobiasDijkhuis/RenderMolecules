import os

from render_molecules.structure import Structure
from render_molecules.blender_utils import delete_all_objects, orbit_camera
from render_molecules.other_utils import get_render_molecules_dir

def main():
    delete_all_objects()
    
    # Read the CUBE file
    render_molecules_dir = get_render_molecules_dir()
    xyz_path = os.path.join(render_molecules_dir, "../examples/orbit_camera/CH3OH_OPT.xyz")
    structure = Structure.from_xyz(xyz_path)
    
    # Set center of mass to origin
    structure.set_center_of_mass([0, 0,0])
    
    # Create the atoms and bonds
    structure.create_structure()
    
    orbit_camera(radius=8, height=4, set_active=True)
    return

if __name__ == '__main__':
    main()
