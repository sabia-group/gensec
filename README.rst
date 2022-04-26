"""""""""""""""""""""""""""""""
GenSec (Generation and Search)
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
Quick start
============================

For performing of the structure search the parameters.json file is required where
all the settings are specified.



