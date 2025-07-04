# RenderMolecules
Easily import molecules from various filetypes, and render them in Blender.

## Installation:
RenderMolecules can be installed either from the pip repository
    
    pip install RenderMolecules
or directly from this source code to get the latest version, that can be adjusted to your needs

    pip install -e . 
See below for more info on how to install it such that Blender's python interpreter can access it.

## API documentation:
See online [documentation](https://tobiasdijkhuis.github.io/RenderMolecules)

## Supported filetypes:
### Structure
 - XYZ
 - CUBE
 - XSF (to be tested)
 - ...
### Trajectory
 - XYZ
 - ORCA geometry optimization
 - ORCA normal modes
 - AXSF (not yet)
 - ...

## Periodic Boundary Conditions:
TODO, NOT IMPLEMENTED

## Orbiting the camera around a structure:
Create an animation such that you can view the structure from different angles.
TODO, NOT IMPLEMENTED.
[Tutorial](https://www.blendernation.com/2020/07/08/blender-quick-tip-rotate-orbit-camera-around-object/).
Put in blenderUtils probably.


## Rendering volumetric data:
RenderMolecules also allows for rendering volumetric data, such as the electron 
density of an orbital. For this, there are two options, 
- Without writing and reading from a file, in memory:
    1. Read the volumetric data using CUBEfile.readVolumetricData 
    2. Calculate the isosurface using CUBEfile.calculateIsosurface
    3. Create the isosurface in the Blender scene using blenderUtils.createIsosurface
- With writing and reading from a file:
    1. Read the volumetric data using CUBEfile.readVolumetricData 
    2. Calculate the isosurface, and write it to a .ply file using CUBEfile.writePLY
    3. Read the .ply file using blenderUtils.loadPLY
        > Make this accept an affineMatrix argument, so that it can be displaced as the structure
    
    > This crashed Blender at the moment. I think it has to do with importing PyTessel.
    > Note: With this method you have to be careful that you do the same displacements and rotations as you do to the structure in your scene. Otherwise, the location of the isosurface might be incorrect.

Almost always the first method will be easier. Using the second method straight from Blender (i.e. calculated the isosurface using PyTessel), I had some troubles that on my Windows machine. Blender would crash and exit. It works if I write the .ply file from the commandline directly. On my Linux machine, both work fine.

## Dependencies:
 - PyTessel (optional)
 - scikit-image
 - numpy
 - bpy

## How it works:
### Creating atoms:
The atoms generated by Structure.createAtoms() created using [Vertex Instancing](https://docs.blender.org/manual/en/latest/scene_layout/object/properties/instancing/verts.html). This means that all atoms of a single element (e.g. all hydrogen atoms) are part of a mesh, and only a single hydrogen atom sphere has to be created (positioned at the origin, visible in the viewport but not in the render). This significantly speeds up the creation of the structure, and general working with the structure if you're for example moving around in the scene.

### Creating bonds:
The Structure class has a method Structure.findBondsBasedOnDistance() that will find all bonds in the structure, based on the distances between atoms. The allowed bond lengths are specified in the ElementData.bondLengths dictionary, where all possible bonds should have an key and value. If, for example, your system consists of H, N, C and O atoms, then your dictionary would have at least HH, HN, HC, HO, NN,... pairs. It is essential that the keys are in alphabetical order. If you do not want to create bonds between two hydrogens, you can for example set the value of HH to 0.0, as then no two H atoms will be close enough to eachother to create a bond. 

> If you change anything in one of the files (like a dictionary key-value pair for a bond length), Blender has to be restarted.

The above method returns a list of Bond instances, and sets the Structure._bonds to the found bonds.

> This could also be done using Instancing, although because bonds have a direction, it is a bit more tricky and I have not had time/the need to figure it out yet (also how to generate it automatically from python makes it a bit harder).
> Maybe it could be done like [this](https://www.reddit.com/r/blenderhelp/comments/1esb6nf/geometry_nodes_align_instanced_objects_on_curve/)?

#### Creating bonds with a higher bond order:
Sometimes you might want to create a molecule with double (e.g. in benzene) or triple bonds (e.g. in CO). This can be done using Structure.generateBondOrderBond. Using some linear algebra, it takes a vector, and calculates a displacement vector that is both perpendicular to the bond you want to change, and the vector you input. 

> Any time you use Structure.generateBondOrderBond, the original bond you input into it is removed from the Structure._bond list, and the new bonds are added at the end. This means that the order of bonds changes, and it might take some trial and error to generate the correct higher bondorder bonds.

## Manipulating structures:
Structures and trajectories can be manipulated (i.e. translated or rotated) using a couple of methods. These translations and rotations are also correctly applied to the isosurface, if they happen before the isosurface is calulated.
The Geometry parent class keeps track of the transformations using an [Affine Matrix](https://en.wikipedia.org/wiki/Affine_transformation).

### Translation
Structures can be translated using Structure.setCenterOfMass(), where the center of mass of the structure can be set to a certain position, or by setting the average position of each atom, or by directly giving a translation vector.

### Rotation
Structures can be rotated using a couple of different methods:
 - Structure.rotateAroundX/Y/Z: rotate around the X/Y/Z axis with a certain angle
 - Structure.rotateAroundAxis: rotate around an arbitrary axis

Rotation angle given in degrees, counterclockwise.

> It is important that any manipulation happens before Structure.createAtoms and Structure.createBonds, as otherwise the atoms or bonds might not be in the correct positions.

## 

 
## Materials:
Materials are handled by the manifest in src/RenderMolecules/ElementData.py. If you want to change an atom color, the shiny-ness of atoms, etc, it should be changed here.

## Adding to Blender's Python interpreter
Ideally, you should install Blender without any external tools like Snap, because then it becomes so much harder to install a new module.

If you just installed Blender from the .zip file on Blender's website, do the following (tested on Ubuntu 24.04.02 using Blender 4.2)

    /path/to/blender/version/python/bin/python3.11.py -m pip install RenderMolecules -t /path/to/blender/version/python/lib/site-packages/

or, when in the RenderMolecules directory,

    /path/to/blender/version/python/bin/python3.11.py -m pip install -e . -t /path/to/blender/version/python/lib/site-packages/
