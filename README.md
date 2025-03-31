# RenderMolecules
Easily import molecules from various filetypes, and render them in Blender.

## Supported filetypes:
 - XYZ
 - CUBE
 - ...

## Periodic Boundary Conditions

## Rendering volumetric data
RenderMolecules also allows for rendering volumetric data, such as the electron 
density of an orbital. For this, it uses PyTessel, to convert from the volumetric 
data to a .ply file, which can then be imported by Blender.
