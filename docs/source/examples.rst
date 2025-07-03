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

.. literalinclude :: ../../examples/isosurface/script.py
   :language: python

Geometry optimization
----------------------

Let's say you did a geometry optimization, and for whatever reason you would like to render the output.

I have artificially created a structure of methanol, with the O-H bond elongated.

.. literalinclude :: ../../examples/animation/script.py
   :language: python

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

.. literalinclude :: ../../examples/vibrations/script.py
   :language: python

.. video:: _static/CH3OH_vibration.mp4
   :autoplay:
   :loop:
   :muted:
   :playsinline:
   :nocontrols:
   :width: 100%
   :caption: GIF of geometry optimization of methanol

.. |CH3OH| replace:: CH\ :sub:`3`\ OH
