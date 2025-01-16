"""""""""""""""""""""""""""""""
GenSec (Generation and Search)
"""""""""""""""""""""""""""""""
++++++++++++++++++++++++++++++++++
Will send structures to local minima
++++++++++++++++++++++++++++++++++

.. contents:: Overview
   :depth: 2

============
Introduction
============

GenSec performs a quasi-random global structure search, with the ability to choose different internal degrees of freedom of molecules and sample them with respect to specified fixed surroundings that can be, in general, 1D (e.g. ions), 2D (e.g. surfaces) or 3D (e.g. solids) static references. The efficiency of the random structure search can be increased dramatically first by imposing constraints on the generated structures, avoiding clashes between atoms and keeping the database of previously calculated structures in order to avoid repetitive calculations.  The geometry optimizations are performed by a connection with the ASE environment, which can be connected to many electronic structure and FF packages and offers the choice of a variety of geometry optimization routines, which we have improved. The connection to the ASE database support makes it possible to perform multiple searches in parallel with shared access to the information obtained from all the searches. 

============
Installation
============
Download the package::

    git clone https://github.com/sabia-group/gensec.git

Installing of dependencies::

    pip install ase numpy networkx

============================
Quick start
============================

! This section is under construction !



