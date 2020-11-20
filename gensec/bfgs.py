import warnings

import numpy as np
from numpy.linalg import eigh

from gensec.optimize import Optimizer_mod
from gensec.defaults import defaults
# from ase.optimize.bfgs import BFGS

import sys
class BFGS_mod(Optimizer_mod):

    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 maxstep=None, master=None, 
                 alpha=70.0, initial=None, rmsd_dev=1000.0, molindixes=None, 
                 structure=None, fixed_frame=None, parameters=None, mu=None, A=None,
                 blacklist=None):
        """BFGS optimizer.

        Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        restart: string
            Pickle file used to store hessian matrix. If set, file with
            such a name will be searched and hessian matrix stored will
            be used, if the file exists.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        maxstep: float
            Used to set the maximum distance an atom can move per
            iteration (default value is 0.2 Å).

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.
        """
        if maxstep is not None:
            self.maxstep = maxstep
        else:
            self.maxstep = defaults.maxstep

        if self.maxstep > 1.0:
            warnings.warn('You are using a much too large value for '
                          'the maximum step size: %.1f Å' % maxstep)
        Optimizer_mod.__init__(self, atoms, restart, logfile, trajectory, master)

        # initial hessian
        self.H0 = np.eye(3 * len(self.atoms)) * alpha
        
        self.initial=initial
        self.rmsd_dev=rmsd_dev
        self.molindixes=molindixes
        self.structure = structure
        self.fixed_frame = fixed_frame
        self.parameters = parameters
        self.blacklist = blacklist

    def todict(self):
        d = Optimizer_mod.todict(self)
        if hasattr(self, 'maxstep'):
            d.update(maxstep=self.maxstep)
        return d

    def initialize(self):
        self.H = None
        self.r0 = None
        self.f0 = None

    def read(self):
        self.H, self.r0, self.f0, self.maxstep = self.load()

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
        atoms.set_positions(r + dr)
        self.r0 = r.flat.copy()
        self.f0 = f.copy()
        self.dump((self.H, self.r0, self.f0, self.maxstep))

    def determine_step(self, dr, steplengths):
        """Determine step to take according to maxstep

        Normalize all steps as the largest step. This way
        we still move along the eigendirection.
        """
        maxsteplength = np.max(steplengths)
        if maxsteplength >= self.maxstep:
            dr *= self.maxstep / maxsteplength

        return dr

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

    def replay_trajectory(self, traj):
        """Initialize hessian from old trajectory."""
        if isinstance(traj, str):
            from ase.io.trajectory import Trajectory
            traj = Trajectory(traj, 'r')
        self.H = None
        atoms = traj[0]
        r0 = atoms.get_positions().ravel()
        f0 = atoms.get_forces().ravel()
        for atoms in traj:
            r = atoms.get_positions().ravel()
            f = atoms.get_forces().ravel()
            self.update(r, f, r0, f0)
            r0 = r
            f0 = f

        self.r0 = r0
        self.f0 = f0