import os

from render_molecules.trajectory import Trajectory
from render_molecules.blender_utils import delete_all_objects
from render_molecules.other_utils import get_render_molecules_dir

def main():
    delete_all_objects()
    
    # ORCA output file
    render_molecules_dir = get_render_molecules_dir()
    out_path = os.path.join(render_molecules_dir, "../examples/vibrations/CH3OH.out")
    
    # Use vibrations (and not geometry optimization), index (starting from 0) of vibrational mode.
    trajectory = Trajectory.from_orca(
        out_path, 
        use_vibrations=True, 
        vibration_nr = 10, 
        n_frames_per_oscillation=100
    )
    
    # Set center of mass to origin
    trajectory.set_center_of_mass([0,0,0])
    
    # Create the animation
    trajectory.create_animation()
    return  

if __name__ == '__main__':
    main()
