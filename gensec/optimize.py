"""Structure optimization. """

import sys
import pickle
import time
from math import sqrt
from os.path import isfile

from ase.calculators.calculator import PropertyNotImplementedError
from ase.parallel import world, barrier
from ase.io.trajectory import Trajectory
import collections

from ase.io import read, write
import numpy as np
from itertools import product

from gensec.precon import preconditioned_hessian, Kabsh_rmsd
from gensec.modules import measure_torsion_of_last

from ase.optimize.optimize import Dynamics
from ase.optimize.bfgs import BFGS

from copy import deepcopy


def irun(self):
        
    """Modified Run dynamics algorithm as generator. Checks 
    RMSD difference and initialize the Heesian matrix update routine
    """

    # compute inital structure and log the first step
    self.atoms.get_forces()

    # yield the first time to inspect before logging
    yield False

    if self.nsteps == 0:
        self.log()
        self.call_observers()

    # run the algorithm until converged or max_steps reached
    while not self.converged() and self.nsteps < self.max_steps:

        # compute the next step
        self.step()
        self.nsteps += 1
        # let the user inspect the step and change things before logging
        # and predicting the next step
        yield False
        # log the step
        self.log()
        self.call_observers()
        # Calculate RMSD between current and initial steps:
        # print(self.H)
        if self.initial:
            print(self.atoms.get_potential_energy())
            
            if Kabsh_rmsd(self.atoms, self.initial, self.molindixes) > self.rmsd_dev:
                self.H = preconditioned_hessian(self.structure, 
                                                self.fixed_frame, 
                                                self.parameters,
                                                self.atoms,
                                                self.H,
                                                task="update") 
                a0=self.atoms.copy()
                self.initial=a0
                
        # d = "/home/damaksimovda/Insync/da.maksimov.da@gmail.com/GoogleDrive/PhD/Preconditioner/vdW/Ar/single_vdW/"
        # name = "hessian.hes"
        # import os
        # h = os.path.join(d, name)
        # if not os.path.exists(h):
        #     open(h, 'a').close()
        # f=open(h,'a')
        # np.savetxt(f, self.H)
        # f.write("\n")
        # f.close()
    # finally check if algorithm was converged
    yield self.converged()

Dynamics.irun = irun

class BFGS_mod(BFGS):
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None, maxstep=None, 
                master=None, initial=None, rmsd_dev=1000.0, molindixes=None, structure=None, 
                H0=None, fixed_frame=None, parameters=None, mu=None, A=None, known=None):
        BFGS.__init__(self, atoms, restart=restart, logfile=logfile, trajectory=trajectory, maxstep=0.04, master=None)
        
        # initial hessian
        
        self.H0 = H0 
        self.initial=initial
        self.rmsd_dev=rmsd_dev
        self.molindixes = molindixes
        self.structure = structure
        self.fixed_frame = fixed_frame
        self.parameters = parameters

    def update(self, r, f, r0, f0):
        if self.H is None:
            #self.H = np.eye(3 * len(self.atoms)) * 70.0
            self.H = self.H0
            return
        dr = r - r0

        if np.abs(dr).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return

        df = f - f0
        a = np.dot(dr, df)
        dg = np.dot(self.H, dr)
        b = np.dot(dr, dg)
        self.H -= np.outer(df, df) / a + np.outer(dg, dg) / b