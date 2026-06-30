Welcome to GenSec documentation!
================================================

About
=====

GenSec (Generation and Search) is a molecular structure generation and optimization framework.
It is designed to explore molecular configurations in the presence of fixed environments,
including single-site references (1D), surfaces/interfaces (2D), and fully periodic solids (3D).

The code combines three core ideas:

1. Quasi-random structure generation over selected molecular degrees of freedom.
2. Geometry and connectivity checks to reject unphysical configurations early.
3. Local optimization through the Atomic Simulation Environment (ASE), enabling multiple calculator backends.

GenSec is particularly useful for workflows that require:

- sampling adsorbate or confined-molecule conformations,
- avoiding repeated relaxation of equivalent structures,
- running parallel search campaigns with shared structure databases,
- coupling generation and relaxation in automated screening pipelines.

New modules in this code allow to train machine learning models for energy and force predictions, which can be used to accelerate the search process.

Main Contributors: Dmitrii Maksimov (original author), Mariana Rossi, Philipp Pestlin and Paolo Lazzaroni.

Contents
========

.. toctree::
   :maxdepth: 2

   installation
   tutorials
   modules
   citation
   license
   
   


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
