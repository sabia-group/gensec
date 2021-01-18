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

from ase.optimize.bfgs import BFGS

from copy import deepcopy
from numpy.linalg import eigh
import os


print("This is old version")

class BFGS_mod(BFGS):
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None, maxstep=None, 
                master=None, initial=None, rmsd_dev=1000.0, molindixes=None, structure=None, 
                H0=None, fixed_frame=None, parameters=None, mu=None, A=None, known=None):
        BFGS.__init__(self, atoms, restart=restart, logfile=logfile, trajectory=trajectory, maxstep=0.2, master=None)
        
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
            #self.H = np.eye(3 * len(self.atoms)) * 70.0 # This is the change compared to ASE
            self.H = self.H0 # This is the change compared to ASE
            return
        dr = r - r0

        if np.abs(dr).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return
        # Calculate RMSD between current and initial steps:
        if self.initial: # This is the change compared to ASE
            # print(self.atoms.get_potential_energy())
            # Experimental for vdW clusters
            forces = self.atoms.get_forces()
            fmax = sqrt((forces ** 2).sum(axis=1).max())
            print("Force",  fmax)
            if Kabsh_rmsd(self.atoms, self.initial, self.molindixes) > self.rmsd_dev:
                print("Applying update")
                # name = "hessian_progress.hes"
                # h = os.path.join(os.getcwd(), name)
                # if not os.path.exists(h):
                #     open(h, 'a').close()
                # f=open(h,'a')
                # f.write("RMSD is Exceed, the hessian will be updated\n")
                # f.write("Hessian before (GenSec)\n")
                # np.savetxt(f, self.H)
                # f.write("\n")
                
                self.H = preconditioned_hessian(self.structure, 
                                                self.fixed_frame, 
                                                self.parameters,
                                                self.atoms,
                                                self.H,
                                                task="update") 
                # f.write("Hessian after (GenSec)\n")
                # np.savetxt(f, self.H)
                # f.write("\n") 
                # f.close()              
                a0=self.atoms.copy()
                self.initial=a0
            else:
                df = f - f0
                a = np.dot(dr, df)
                dg = np.dot(self.H, dr)
                b = np.dot(dr, dg)
                self.H -= np.outer(df, df) / a + np.outer(dg, dg) / b
        else:
            df = f - f0
            a = np.dot(dr, df)
            dg = np.dot(self.H, dr)
            b = np.dot(dr, dg)
            self.H -= np.outer(df, df) / a + np.outer(dg, dg) / b
#############################################################
        # Prints out Hessian, for developing purposes (GenSec)
        # name = "hessian_progress.hes"
        # h = os.path.join(os.getcwd(), name)
        # if not os.path.exists(h):
        #     open(h, 'a').close()
        # f=open(h,'a')
        # f.write("Hessian after update (GenSec)\n")
        # np.savetxt(f, self.H)
        # f.write("\n")
        # f.close()
#############################################################

    def step(self, f=None):
        atoms = self.atoms

        if f is None:
            f = atoms.get_forces()

        r = atoms.get_positions()
        f = f.reshape(-1)
        self.update(r.flat, f, self.r0, self.f0)
        omega, V = eigh(self.H)
        dr = np.dot(V, np.dot(f, V) / np.fabs(omega)).reshape((-1, 3))
        steplengths = (dr**2).sum(1)**0.5
        dr = self.determine_step(dr, steplengths)
        max_step = round(sqrt((dr ** 2).sum(axis=1).max()), 4)
        print("Step size is {}".format(max_step))
        atoms.set_positions(r + dr)
        self.r0 = r.flat.copy()
        self.f0 = f.copy()
        self.dump((self.H, self.r0, self.f0, self.maxstep))
#############################################################
        # Prints out Hessian, for developing purposes (GenSec)
        # name = "hessian_progress.hes"
        # h = os.path.join(os.getcwd(), name)
        # if not os.path.exists(h):
        #     open(h, 'a').close()
        # f=open(h,'a')
        # f.write("Making step having hessian (GenSec)\n")
        # np.savetxt(f, self.H)
        # f.write("\n")
        # f.close()
#############################################################