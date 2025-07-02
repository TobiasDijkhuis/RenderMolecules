.. _examples:
.. index:: Examples

Examples
========

PUTTING EXAMPLES HERE. TODO

Simple structure
----------------

.. code-block:: python

   import os

   from render_molecules.structure import Structure
   from render_molecules.blender_utils import delete_all_objects
   from render_molecules.other_utils import get_render_molecules_dir

   def main():
      delete_all_objects()
      
      # Read the CUBE file
      render_molecules_dir = get_render_molecules_dir()
      xyz_path = os.path.join(render_molecules_dir, "../examples/structure/CH3OH.xyz")
      structure = Structure.from_xyz(xyz_path)
      
      # Set center of mass to origin
      structure.set_center_of_mass([0, 0,0])
      
      # Create the atoms
      structure.create_atoms()
      
      # Find bonds based on distances between atoms
      bonds = structure.find_bonds_from_distances()
      
      # Create the bonds
      structure.create_bonds(bonds)

   if __name__ == '__main__':
      main()

Isosurface
----------

.. code-block:: python
   
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
      CUBE.set_center_of_mass([0, 0, 0])
      
      # Create the atoms
      CUBE.create_atoms()
      
      # Find bonds based on distances between atoms
      bonds = CUBE.find_bonds_from_distances()
      
      # Create the bonds
      CUBE.create_bonds(bonds)
      
      # Calculate and render the isosurface
      isovalue = 0.025
      verts, edges, _, _ = CUBE.calculate_isosurface(isovalue)
      create_isosurface(verts, edges, isovalue, "CH3OH")
      return

   if __name__ == '__main__':
      main()

Geometry optimization
----------------------

Let's say you did a geometry optimization, and for whatever reason you would like to render the output.

I have artificially created a structure of methanol, with the O-H bond elongated.

.. code-block:: python
   
   import os

   from render_molecules.trajectory import Trajectory
   from render_molecules.blender_utils import delete_all_objects
   from render_molecules.blender_utils import get_render_molecules_dir

   def main():
      delete_all_objects()
      
      # Read the xyz file
      render_molecules_dir = get_render_molecules_dir()
      xyz_path = os.path.join(render_molecules_dir, "../examples/animation/CH3OH_OPT_trj.xyz")
      trajectory = Trajectory.from_xyz(xyz_path)
      
      # Set center of mass to origin
      trajectory.set_center_of_mass([0, 0,0])
      
      # Create the atoms
      trajectory.create_animation()
      return 

   if __name__ == '__main__':
      main()

.. video:: _static/CH3OH_optimization.mp4
   :autoplay:
   :loop:
   :muted:
   :playsinline:
   :nocontrols:
   :width: 100%
   :caption: GIF of geometry optimization of methanol

Vibrations
----------

.. code-block:: python

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
      trajectory = Trajectory.from_orca(out_path, use_vibrations=True, vibration_nr = 10, n_frames_per_oscillation=100)
      trajectory.set_center_of_mass([0,0,0])
      
      # Create the animation
      trajectory.create_animation()
      return  

   if __name__ == '__main__':
      main()

.. video:: _static/CH3OH_vibration.mp4
   :autoplay:
   :loop:
   :muted:
   :playsinline:
   :nocontrols:
   :width: 100%
   :caption: GIF of geometry optimization of methanol

.. |CH3OH| replace:: CH\ :sub:`3`\ OH