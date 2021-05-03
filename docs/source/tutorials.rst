Tutorials
=========

Getting started
+++++++++++++++++++

In order to sample conformationa space of the flexible molecule you will need the following files:
1. template of the moecule in 3D format for example XYZ, PDB, SDF and so on...
2. "parameters.file" with the configuration of the search
3. ASE calculator file with which you can perform the geometry optimization routines.

To run the structure search:

.. code-block:: sh

    /path_to_GenSec/gensec.py parameters.txt

The rotatable bonds of the molecules can be automatically identified after construction of the connectivity matrix of the template molecule and these bonds can be found in the output file that GenSec produses. If there is ned to modify the number of rotatable bonds they can be set up explicitly in the parameters file. Connectivity is constructed with use of ASE  based on the list of cutoff radii. If the spheres of two atoms that defined by atomic covalent radii overlap, they will be counted bonded atoms. This allows to construct the graph of bonded atoms and identify the torsion angles of the molecules. Then random values of the torsions then applied and the resulting molecule is checked for internal clashes by constructing of the connectivity matrix again and comparing of it with the template one. Check for each molecule also implies periodic boundary conditions if they are specified for the search.
