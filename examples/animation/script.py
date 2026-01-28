from __future__ import annotations

import os

from render_molecules.blender_utils import delete_all_objects
from render_molecules.other_utils import get_render_molecules_dir
from render_molecules.trajectory import Trajectory


def main():
    delete_all_objects()

    # Read the xyz file
    render_molecules_dir = get_render_molecules_dir()
    xyz_path = os.path.join(
        render_molecules_dir, "../examples/animation/CH3OH_OPT_trj.xyz"
    )
    trajectory = Trajectory.from_xyz(xyz_path)

    # Set center of mass to origin
    trajectory.set_center_of_mass([0, 0, 0])

    # Create the animation
    trajectory.create_animation()
    return


if __name__ == "__main__":
    main()
