"""""""""""""""""""""""""""""""
GenSec Tutorial
"""""""""""""""""""""""""""""""
++++++++++++++++++++++++++++++++++
The life is much better if it sampled enough
++++++++++++++++++++++++++++++++++

.. contents:: Overview
   :depth: 2

============
Introduction
============
Structure search for organic/inorganic interfaces is a challenging task due to large amount of degrees of freedom that presented by high flexibility of biomolecules. We develop structure search sampling package intended to explore conformational spaces of flexible molecules also with respect to fixed surroundings. In the package we implement flexible way for preconditioning of geometry optimization BFGS algorithm based on Exp (cite), vdW (cite) and FF (cite) like terms. 

Authors: Dmitrii Maksimov

! This page is under construction !

============
Installation
============
GenSec requires Python 3.8, 3.9, or 3.10.

Download the package::

    git clone https://github.com/sabia-group/gensec.git

Installing of dependencies::

    pip install ase numpy networkx

============================
Manual
============================

For performing of the structure search the parameters.json file is required where
all the settings are specified.

-------------------------
Keywords of parameters.json
-------------------------
– name, default = "ID"
Prefix of the output files.

– number_of_replicas, default = 1
Number of replica of the structure specified ingeometrykeyword to beproduced.

– geometry
Section for specification of the template geometry file.
– filename, default = "template.in"
Template geometry file name to be sampled.
– format, default = "aims"
Format of the geometry file. 
Supporting formats that available inASE.

•fixed_frameSection 
for specification of the surroundings with respect to which thetemplate geometry file will be sampled.
– activate, default = false 
specifies if the surrounding has to be taken into accountor not.3
– filename, default = "none.in"
Name of the surrounding geometry file.
– format, default = "aims"
Format of the surrounding geometry file

•mic
Minimum-image convention specifies the periodic boundary conditions that have to be taken into account for sampling of the template structure. Can be different from the periodic boundary conditionsspecified for the surrounding geometry.
– activate, default = false
Turning on or off periodic boundary conditions for molecule.
– pbc, default = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
Three lattice vectors of periodic boundary conditions to be specified.

•configuration
Section for specification of degrees of freedom for sampling of thetemplate geometry.
– torsions, default = false
   Sampling on torsion degrees of freedom.
   ∗mode, default = false
   Activates sampling on torsion angles.
   ∗same, default = true
   If number of replicas is specified, this flag produces geometrieswith the same set of torsion angles or the produced structuresthat are generated can have different sets of torsion angles.
   ∗list_of_tosrions, default = "auto"
   Specifies particular rotational angles. With auto - all the rotatblebonds will be sampled.
   ∗values, default = "random"
   Specifies range of values for the torsional angles to be sampled in.
– orientations, default = false
   Section for specification of rotational degree of freedom.
   ∗mode, default = false
   Activates rotations of the molecules as new degree of freedom
   ∗same, default = true
   If number of replicas is specified, this flag produces geometrieswith the same orientations or with different ones.
   ∗values, default = "random"
   Specifies if random orientation should be produced. If "restricted" is specified, then geometries with particular orientaitons will beproduced.
   ∗angle, default = [0, 360]
   Specifies range of angles available for rotation of the moleculearound the axis which is eigenvector of the moment of tensor ofinertia molecule with smallest eigenvalue.
   ∗x, default = [0, 0]
   Specifies range of x values to be sampled for the coordinate ofthe axis which is eigenvector of the moment of tensor of inertiamolecule with smallest eigenvalue.
   ∗y, default = [0,0]
   Specifies range of y values to be sampled for the coordinate ofthe axis which is eigenvector of the moment of tensor of inertiamolecule with smallest eigenvalue.
   ∗z, default = [0, 1]
   Specifies range of z values to be sampled for the coordinate ofthe axis which is eigenvector of the moment of tensor of inertiamolecule with smallest eigenvalue.
– coms, default = false
   Section for specification of translational degree of freedom.
   ∗mode, default = false
   activates translational degree of freedom of the molecule
   ∗values, default = "random"
   Specifies if random position of centers of the masses of moleculesshould be produced. If "restricted" is specified, then geometrieswithin particular region will be produced
   ∗x_axis, default = [-10, 10]
   Specifies range of x values to be sampled for the coordinate ofthe axis which is eigenvector of the moment of tensor of inertiamolecule with smallest eigenvalue.
   ∗y_axis, default = [-10, 10]
   Specifies range of y values to be sampled for the coordinate ofthe axis which is eigenvector of the moment of tensor of inertiamolecule with smallest eigenvalue.
   ∗z_axis, default = [-10, 10]
   Specifies range of z values to be sampled for the coordinate ofthe axis which is eigenvector of the moment of tensor of inertiamolecule with smallest eigenvalue.

•calculator
Section related to the calculation settings
– supporting_files_folder, default = "supporting"
Folder that contains supporting files necessary for performing of theenergy evaluation with external codes.
– ase_parameters_file, default = "ase_command.py"
Filename with the settings for the calculator that is used by ASE.– optimize, default = "random"
Specifies the routine for run of GenSec. "generate" produces struc-tures without energy or force evaluations. "single" - preforms geom-etry optimization of the template moleule and surronding. "generate_and_relax" - performs generating and relaxation of the struc-tures.
– fmax, default = 0.01
Maximum residual force for geometry optimization.– preconditionerSpecifies parameters of the preconditioned Hessian matrix for geome-try optimization. Implemented preconditioner schemes: "ID", "Exp","vdW", "Lindh".
   ∗mol, default = "ID"
   Preconditioner scheme applied to molecular part.
   ∗fixed_frame, default = "ID"
   Preconditioner scheme applied to surrounding part.
   ∗mol-mol, default = "ID"
   Preconditioner scheme applied to parts of the Hessian dedicatedto different molecules.
   ∗mol-fixed_frame, default = "ID"
   Preconditioner scheme applied to parts of the Hessian for partsbetween molecules and surroundings.
   ∗rmsd_update, default = 100.1
   Performing of the updating of the Hessian matrix during BFGS algorithm after geometry change exceeds specified RMSD value.
– constraints, default = false
   Applying of the constrains on geometry optimization
   ∗activate, default = false
   Activates constrains on geometry optimization
   ∗symbol, default = "X"
   Atom symbols on which geometry constrains should be applied
   ∗z-coord, default = [-1000, -999.9]
   Range of the z-coordinates within which atoms should be con-strained during geometry optimization.

-------------------------
Module overview
-------------------------
gensec.py - the main module that defines procedure of geometry generation, optimization and search/  

structure.py - module with the collection of the parameters of the template molecule and degrees of freedom

modules.py - collection of diverse help functions such as measurments of degrees of freedom

outputs.py - module includes routines for outputfile of GenSec.

precon.py - module for preconditioning of the Hessian matrix for geometry optimization.

relaxation.py - routines for performing of the geometry optimizations via ASE.

blacklist.py - routines for keeping history of already calculated structures.

