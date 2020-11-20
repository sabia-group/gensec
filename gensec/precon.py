""" Make vdW preconditioner. """

import sys
import pickle
import time
from math import sqrt
from os.path import isfile

from ase.calculators.calculator import PropertyNotImplementedError
from ase.parallel import world, barrier
from ase.io.trajectory import Trajectory
import collections

from ase.optimize.precon.neighbors import estimate_nearest_neighbour_distance

import numpy as np
from numpy.linalg import norm
from itertools import product
import operator
import os

def Kabsh_rmsd(atoms, initial, molindixes, removeHs=False):

    coords1 = np.array([atoms.get_positions()[i] for i in molindixes])
    coords2 = np.array([initial.get_positions()[i] for i in molindixes])

    #COM1 = get_centre_of_mass_from_sdf(sdf_string1, removeHs = True)
    #COM2 = get_centre_of_mass_from_sdf(sdf_string2, removeHs = True)
    #"""Return the optimal RMS after aligning two structures."""

    #coords1 = coords_and_masses_from_sdf(sdf_string1, removeHs = True)[:,:3] - COM1
    #coords2 = coords_and_masses_from_sdf(sdf_string2, removeHs = True)[:,:3] - COM2
    #'''Kabsh'''
    A = np.dot(coords1.T, coords2)
    V, S, W = np.linalg.svd(A)
    if np.linalg.det(np.dot(V, W)) < 0.0:
        V[:, -1] = -V[:, -1]
        K = np.dot(V, W)
    else:
        K = np.dot(V, W)

    coords1 = np.dot(coords1, K)
    rmsd_kabsh = 0.0
    for v, w in zip(coords1, coords2):
        rmsd_kabsh += sum([(v[i] - w[i])**2.0 for i in range(len(coords1[0]))])
    return np.sqrt(rmsd_kabsh/len(coords1))


#    VALUES for ALPHA-vdW and C6_vdW:
#    !VVG: The majority of values such as isotropic static polarizability
#    !(in bohr^3), the homo-atomic van der Waals coefficient(in hartree*bohr^6),
#    !and vdW Radii (in bohr) for neutral free atoms are taken from Ref. Chu, X. & Dalgarno,
#    !J. Chem. Phys. 121, 4083 (2004) and Mitroy, et al.
#    !J. Phys. B: At. Mol. Opt. Phys. 43, 202001 (2010)
#    !and for rest of the elements they are calculated using linear response coupled cluster
#    !single double theory with accrate basis. The vdW radii for respective element are
#    !defined as discussed in Tkatchenko, A. & Scheffler, M. Phys. Rev. Lett. 102, 073005 (2009).

BOHR_to_angstr = 0.52917721   # in AA
HARTREE_to_eV = 27.211383  # in eV
HARTREE_to_kcal_mol = 627.509 # in kcal * mol^(-1)

# Ground state polarizabilities α0 (in atomic units) of noble gases and isoelectronic ions. 
# https://iopscience.iop.org/article/10.1088/0953-4075/43/20/202001/pdf
ALPHA_vdW = {'H': 4.5000, 'He': 1.3800, 'Li': 164.2000, 'Be': 38.0000, 'B': 21.0000, 'C': 12.0000,
            'N': 7.4000, 'O': 5.4000, 'F': 3.8000, 'Ne': 2.6700, 'Na': 162.7000, 'Mg': 71.0000, 'Al': 60.0000,
            'Si': 37.0000, 'P': 25.0000, 'S': 19.6000, 'Cl': 15.0000, 'Ar': 11.1000, 'K': 292.9000,
            'Ca': 160.0000,
            'Sc': 120.0000, 'Ti': 98.0000, 'V': 84.0000, 'Cr': 78.0000, 'Mn': 63.0000, 'Fe': 56.0000,
            'Co': 50.0000,
            'Ni': 48.0000, 'Cu': 42.0000, 'Zn': 40.0000, 'Ga': 60.0000, 'Ge': 41.0000, 'As': 29.0000,
            'Se': 25.0000,
            'Br': 20.0000, 'Kr': 16.8000, 'Rb': 319.2000, 'Sr': 199.0000, 'Y': 126.7370, 'Zr': 119.9700,
            'Nb': 101.6030,
            'Mo': 88.4225, 'Tc': 80.0830, 'Ru': 65.8950, 'Rh': 56.1000, 'Pd': 23.6800, 'Ag': 50.6000,
            'Cd': 39.7000,
            'In': 70.2200, 'Sn': 55.9500, 'Sb': 43.6719, 'Te': 37.65, 'I': 35.0000, 'Xe': 27.3000,
            'Cs': 427.12, 'Ba': 275.0,
            'La': 213.70, 'Ce': 204.7, 'Pr': 215.8, 'Nd': 208.4, 'Pm': 200.2, 'Sm': 192.1, 'Eu': 184.2,
            'Gd': 158.3, 'Tb': 169.5,
            'Dy': 164.64, 'Ho': 156.3, 'Er': 150.2, 'Tm': 144.3, 'Yb': 138.9, 'Lu': 137.2, 'Hf': 99.52,
            'Ta': 82.53,
            'W': 71.041, 'Re': 63.04, 'Os': 55.055, 'Ir': 42.51, 'Pt': 39.68, 'Au': 36.5, 'Hg': 33.9,
            'Tl': 69.92,
            'Pb': 61.8, 'Bi': 49.02, 'Po': 45.013, 'At': 38.93, 'Rn': 33.54, 'Fr': 317.8, 'Ra': 246.2,
            'Ac': 203.3,
            'Th': 217.0, 'Pa': 154.4, 'U': 127.8, 'Np': 150.5, 'Pu': 132.2, 'Am': 131.20, 'Cm': 143.6,
            'Bk': 125.3,
            'Cf': 121.5, 'Es': 117.5, 'Fm': 113.4, 'Md': 109.4, 'No': 105.4}


# Ground state polarizabilities α0 (in atomic units) of noble gases and isoelectronic ions. 
# https://iopscience.iop.org/article/10.1088/0953-4075/43/20/202001/pdf
C6_vdW = {'H': 6.5000, 'He': 1.4600, 'Li': 1387.0000, 'Be': 214.0000, 'B': 99.5000, 'C': 46.6000,
        'N': 24.2000, 'O': 15.6000, 'F': 9.5200, 'Ne': 6.3800, 'Na': 1556.0000, 'Mg': 627.0000,
        'Al': 528.0000, 'Si': 305.0000, 'P': 185.0000, 'S': 134.0000, 'Cl': 94.6000, 'Ar': 64.3000,
        'K': 3897.0000, 'Ca': 2221.0000, 'Sc': 1383.0000, 'Ti': 1044.0000, 'V': 832.0000, 'Cr': 602.0000,
        'Mn': 552.0000, 'Fe': 482.0000, 'Co': 408.0000, 'Ni': 373.0000, 'Cu': 253.0000, 'Zn': 284.0000,
        'Ga': 498.0000, 'Ge': 354.0000, 'As': 246.0000, 'Se': 210.0000, 'Br': 162.0000, 'Kr': 129.6000,
        'Rb': 4691.0000, 'Sr': 3170.0000, 'Y': 1968.580, 'Zr': 1677.91, 'Nb': 1263.61, 'Mo': 1028.73,
        'Tc': 1390.87,
        'Ru': 609.754, 'Rh': 469.0, 'Pd': 157.5000, 'Ag': 339.0000, 'Cd': 452.0, 'In': 707.0460,
        'Sn': 587.4170,
        'Sb': 459.322, 'Te': 396.0, 'I': 385.0000, 'Xe': 285.9000, 'Cs': 6582.08, 'Ba': 5727.0, 'La': 3884.5,
        'Ce': 3708.33, 'Pr': 3911.84, 'Nd': 3908.75, 'Pm': 3847.68, 'Sm': 3708.69, 'Eu': 3511.71,
        'Gd': 2781.53, 'Tb': 3124.41, 'Dy': 2984.29, 'Ho': 2839.95, 'Er': 2724.12, 'Tm': 2576.78,
        'Yb': 2387.53, 'Lu': 2371.80, 'Hf': 1274.8, 'Ta': 1019.92, 'W': 847.93, 'Re': 710.2, 'Os': 596.67,
        'Ir': 359.1, 'Pt': 347.1, 'Au': 298.0, 'Hg': 392.0, 'Tl': 717.44, 'Pb': 697.0, 'Bi': 571.0,
        'Po': 530.92, 'At': 457.53, 'Rn': 390.63, 'Fr': 4224.44, 'Ra': 4851.32, 'Ac': 3604.41, 'Th': 4047.54,
        'Pa': 2367.42, 'U': 1877.10, 'Np': 2507.88, 'Pu': 2117.27, 'Am': 2110.98, 'Cm': 2403.22,
        'Bk': 1985.82,
        'Cf': 1891.92, 'Es': 1851.1, 'Fm': 1787.07, 'Md': 1701.0, 'No': 1578.18}

# VdW radii in Bohr
VDW_radii = {'H': 3.1000, 'He': 2.6500, 'Li': 4.1600, 'Be': 4.1700, 'B': 3.8900, 'C': 3.5900,
        'N': 3.3400, 'O': 3.1900, 'F': 3.0400, 'Ne': 2.9100, 'Na': 3.7300, 'Mg': 4.2700,
        'Al': 4.3300, 'Si': 4.2000, 'P': 4.0100, 'S': 3.8600, 'Cl': 3.7100, 'Ar': 3.5500,
        'K': 3.7100, 'Ca': 4.6500, 'Sc': 4.5900, 'Ti': 4.5100, 'V': 4.4400, 'Cr': 3.9900,
        'Mn': 3.9700, 'Fe': 4.2300, 'Co': 4.1800, 'Ni': 3.8200, 'Cu': 3.7600, 'Zn': 4.0200,
        'Ga': 4.1900, 'Ge': 4.1900, 'As': 4.1100, 'Se': 4.0400, 'Br': 3.9300, 'Kr': 3.8200,
        'Rb': 3.7200, 'Sr': 4.5400, 'Y': 4.8151, 'Zr': 4.53, 'Nb': 4.2365, 'Mo': 4.099,
        'Tc': 4.076,
        'Ru': 3.9953, 'Rh': 3.95, 'Pd': 3.6600, 'Ag': 3.8200, 'Cd': 3.99, 'In': 4.2319,
        'Sn': 4.3030,
        'Sb': 4.2760, 'Te': 4.22, 'I': 4.1700, 'Xe': 4.0800, 'Cs': 3.78, 'Ba': 4.77, 'La': 3.14,
        'Ce': 3.26, 'Pr': 3.28, 'Nd': 3.3, 'Pm': 3.27, 'Sm': 3.32, 'Eu': 3.40,
        'Gd': 3.62, 'Tb': 3.42, 'Dy': 3.26, 'Ho': 3.24, 'Er': 3.30, 'Tm': 3.26,
        'Yb': 3.22, 'Lu': 3.20, 'Hf': 4.21, 'Ta': 4.15, 'W': 4.08, 'Re': 4.02, 'Os': 3.84,
        'Ir': 4.00, 'Pt': 3.92, 'Au': 3.86, 'Hg': 3.98, 'Tl': 3.91, 'Pb': 4.31, 'Bi': 4.32,
        'Po': 4.097, 'At': 4.07, 'Rn': 4.23, 'Fr': 3.90, 'Ra': 4.98, 'Ac': 2.75, 'Th': 2.85,
        'Pa': 2.71, 'U': 3.00, 'Np': 3.28, 'Pu': 3.45, 'Am': 3.51, 'Cm': 3.47,
        'Bk': 3.56,
        'Cf': 3.55, 'Es': 3.76, 'Fm': 3.89, 'Md': 3.93, 'No': 3.78}


# Preambule from Lindh.py pthon sctipt
HUGE = 1e10

ABOHR = 0.52917721 # in AA
HARTREE = 27.211383 # in eV

K_BOND    = 0.450 * HARTREE / ABOHR**2
K_BENDING = 0.150 * HARTREE
K_TORSION = 0.005 * HARTREE

ALPHAS = np.array([[1.0000, 0.3949, 0.3949],
                   [0.3949, 0.2800, 0.2800],
                   [0.3949, 0.2800, 0.2800]]) * ABOHR**(-2)
REF_DS = np.array([[1.35, 2.10, 2.53],
                   [2.10, 2.87, 3.40],
                   [2.53, 3.40, 3.40]]) * ABOHR

COVRADS = dict(
    H=0.320, He=0.310,
    Li=1.630, Be=0.900, B=0.820, C=0.770,
    N=0.750, O=0.730, F=0.720, Ne=0.710,
    Na=1.540, Mg=1.360, Al=1.180, Si=1.110,
    P=1.060, S=1.020, Cl=0.990, Ar=0.980,
    K=2.030, Ca=1.740,
    Sc=1.440, Ti=1.320, V=1.220, Cr=1.180, Mn=1.170,
    Fe=1.170, Co=1.160, Ni=1.150, Cu=1.170, Zn=1.250,
    Ga=1.260, Ge=1.220, As=1.200, Se=1.160, Br=1.140, Kr=1.120,
    Rb=2.160, Sr=1.910,
    Y=1.620, Zr=1.450, Nb=1.340, Mo=1.300, Tc=1.270,
    Ru=1.250, Rh=1.250, Pd=1.280, Ag=1.340, Cd=1.480,
    In=1.440, Sn=1.410, Sb=1.400, Te=1.360, I=1.330, Xe=1.310,
    Cs=2.350, Ba=1.980,
    La=1.690, Ce=1.650, Pr=1.650, Nd=1.840, Pm=1.630, Sm=1.620,
    Eu=1.850, Gd=1.610, Tb=1.590, Dy=1.590, Ho=1.580, Er=1.570,
    Tm=1.560, Yb=2.000, Lu=1.560, Hf=1.440, Ta=1.340, W=1.300, Re=1.280,
    Os=1.260, Ir=1.270, Pt=1.300, Au=1.340, Hg=1.490, Tl=1.480, Pb=1.470,
    Bi=1.460, Po=1.460, At=2.000, Rn=2.000, Fr=2.000, Ra=2.000, Ac=2.000,
    Th=1.650, Pa=2.000, U=1.420)


ID = np.identity(3)
ZERO = np.zeros((3, 3))


def vector_separation(cell_h, cell_ih, qi, qj):
    # This file is part of i-PI.
    # i-PI Copyright (C) 2014-2015 i-PI developers
    # See the "licenses" directory for full license information.


    """Calculates the vector separating two atoms.

    Note that minimum image convention is used, so only the image of
    atom j that is the shortest distance from atom i is considered.

    Also note that while this may not work if the simulation
    box is highly skewed from orthorhombic, as
    in this case it is possible to return a distance less than the
    nearest neighbour distance. However, this will not be of
    importance unless the cut-off radius is more than half the
    width of the shortest face-face distance of the simulation box,
    which should never be the case.

    Args:
        cell_h: The simulation box cell vector matrix.
        cell_ih: The inverse of the simulation box cell vector matrix.
        qi: The position vector of atom i.
        qj: The position vectors of one or many atoms j shaped as (N, 3).
    Returns:
        dij: The vectors separating atoms i and {j}.
        rij: The distances between atoms i and {j}.
    """

    sij = np.dot(cell_ih, (qi - qj).T)  # column vectors needed
    sij -= np.rint(sij)

    dij = np.dot(cell_h, sij).T         # back to i-pi shape
    rij = np.linalg.norm(dij, axis=1)

    return dij, rij
    
    

def vdwHessian(atoms):

    def periodic_R(cell_h, cell_ih, qi, qj):
        sij = np.dot(cell_ih, (qi - qj).T)  
        sij -= np.rint(sij)
        dij = np.dot(cell_h, sij).T        
        rij = np.linalg.norm(dij)
        return dij, rij


    N  = len(atoms)
    coordinates = atoms.get_positions()
    atom_names = atoms.get_chemical_symbols()
    cell_h = atoms.get_cell()[:]
    cell_ih = atoms.get_reciprocal_cell()[:]
    hessian = np.zeros(shape = (3 * N, 3 * N))
    atomsRange = list(range(N))
    units = BOHR_to_angstr ** 6 * HARTREE_to_eV

    def C6AB(A, B):

        C6AB = 2. * C6_vdW[A] * C6_vdW[B] / (ALPHA_vdW[B] / ALPHA_vdW[A] * C6_vdW[A] + ALPHA_vdW[A] / ALPHA_vdW[B] * C6_vdW[B]) * units
        return C6AB

    def C12AB(A, B, C6):

        R0AB = (VDW_radii[B] * VDW_radii[A] * BOHR_to_angstr**2 * 2**2)**0.5 
        C12AB = 0.5*C6*(R0AB**6)    
        return C12AB 

    def RAB(cell_h, cell_ih, qi, qj):

        if np.array_equal(cell_h, np.zeros([3, 3])):
            R = np.array(np.linalg.norm(qi-qj))
        else:
            dij, R = periodic_R(cell_h, cell_ih, qi, qj) 
        return R

    def vdW_element(k, l, C6, C12, R, qi, qj):

        if R !=0:
            if k == l:
                return -48*C6*(qi[k]-qj[l])**2/R**10 + 168*C12*(qi[k]-qj[l])**2/R**16 + 6*C6/R**8 - 12*C12/R**14
            else:
                return -48*C6*(-qi[k]+qj[k])*(-qi[l]+qj[l])/R**10 + 168*C12*(-qi[k]+qj[k])*(-qi[l]+qj[l])/R**16
        else:
            return 0

    for i in atomsRange:
        for j in atomsRange:
            for k in range(3):
                for l in range(3):
                    # Calculate C6, C12, rij
                    A = atom_names[i]
                    B = atom_names[j]
                    C6 = C6AB(A, B)
                    C12 = C12AB(A, B, C6)
                    qi = coordinates[i]
                    qj = coordinates[j]
                    R = RAB(cell_h, cell_ih, qi, qj)
                    hessian[3*i+k, 3*j+l] = vdW_element(k, l, C6, C12, R, qi, qj)
       
    for ind in range(len(hessian)):
        hessian[ind, ind] = 0 
        hessian[ind, ind] = -np.sum(hessian[ind]) + 0.005
    
    return hessian


def ExpHessian(atoms, mu=1, A=1):

    N  = len(atoms)
    # A=options.A
    r_NN = estimate_nearest_neighbour_distance(atoms)
    r_cut = 2.0 * r_NN
    coordinates = atoms.get_positions()
    cell_h = atoms.get_cell()[:]
    cell_ih = atoms.get_reciprocal_cell()[:]

    hessian = np.zeros(shape = (3 * N, 3 * N))
    atomsRange = list(range(N))
    for i in atomsRange:
        qi = coordinates[i].reshape(1,3)
        qj = coordinates.reshape(-1,3)

        if np.array_equal(cell_h,np.zeros([3, 3])):
            rij = np.array([np.linalg.norm(qi-Qj) for Qj in qj])
        else:
            dij, rij = vector_separation(cell_h, cell_ih, qi, qj)

        coeff = -mu * np.exp(-A * (rij / r_NN - 1))
        mask = np.array(rij>=r_cut)
        coeff[mask] = 0

        stack = np.hstack([np.identity(3)*coef for coef in coeff])

        hessian[3 * i + 0, :] = stack[0]
        hessian[3 * i + 1, :] = stack[1]
        hessian[3 * i + 2, :] = stack[2]
        
    # hessian = hessian + hessian.T - np.diag(hessian.diagonal())
    for ind in range(len(hessian)):
        hessian[ind, ind] = 0 
        hessian[ind, ind] = -np.sum(hessian[ind]) + 0.005   
    return hessian


def ExpHessian_P(atoms, mu=1, A=1):

    N  = len(atoms)
    # A=options.A
    r_NN = estimate_nearest_neighbour_distance(atoms)
    r_cut = 2.0 * r_NN
    coordinates = atoms.get_positions()
    cell_h = atoms.get_cell()[:]
    cell_ih = atoms.get_reciprocal_cell()[:]

    hessian = np.zeros(shape = ( N, N))
    atomsRange = list(range(N))
    for i in atomsRange:
        qi = coordinates[i].reshape(1,3)
        qj = coordinates.reshape(-1,3)

        if np.array_equal(cell_h,np.zeros([3, 3])):
            rij = np.array([np.linalg.norm(qi-Qj) for Qj in qj])
        else:
            dij, rij = vector_separation(cell_h, cell_ih, qi, qj)

        coeff = -mu * np.exp(-A * (rij / r_NN - 1))
        mask = np.array(rij>=r_cut)
        coeff[mask] = 0

        stack = np.hstack([coef for coef in coeff])

        hessian[i,:] = stack
    # hessian = hessian + hessian.T - np.diag(hessian.diagonal())
    for ind in range(len(hessian)):
        hessian[ind, ind] = -np.sum(hessian[ind])

    return hessian


def _acc_dict(key, d, val):
    """If key in dict, accumulate, otherwise set."""
    if key not in d:
        d[key] = np.zeros_like(val)
    d[key] += val


def isposvec(vec, eps=1e-10, noself=True):
    """Return if vector is in some special sense 'positive'.

    Positiveness is defined by the rightmost non-zero dimension:
    >>> isposvec(np.array([1., 0., -1.]))
    False
    >>> isposvec(np.array([1., 0., 0.]))
    True
    >>> isposvec(np.array([-1., 0., 1.]))
    True
    """
    for x in reversed(vec):
        if x > eps:
            return True
        elif x < -eps:
            return False
    if noself:
        raise ValueError("Self-referencer")
    else:
        return None
    


def canonize(atom_name):
    """Return canonical name of atom_name.

    The canonical name is the first capital of the string with an optional
    minuskel.

    Example:
    >>> print canonize("Ru"), canonize("-H3"), canonize("CT")
    Ru H C
    """
    name = atom_name
    while name and not name[0].isupper():
        name = name[1:]
    if len(name) > 1 and name[1].islower():
        name = name[0:2]
    else:
        name = name[0:1]
    return name

def name2row(atom_name):
    """Return row number of atom type (starting with 0, max 2)."""
    name = canonize(atom_name)
    if name in ('H', 'He'):
        return 0
    elif name in ('Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne'):
        return 1
    else:
        return 2

class Damper(object):
    """Damper interface documentation

    A Damper is an object that can judge the importance of an atom pair from
    the interatomic distance and the atom numbers.
    """
    def exponent(self, AB, i, j):
        """Return exponent for distance AB and atom types i, j.

        What is actually done with AB, and in particular i, j
        is an implementation detail.  Most probably, the initializer
        should be passed some system related information, i.e. at
        least the atom types within the system.
        """
        raise NotImplementedError()
    def Rcut(self, max_exponent):
        """Return the maximum distance leading to max_exponent."""

class SimpleDamper(Damper):
    """Damper for maximum chain lenght (exponent==bondlength)."""
    def __init__(self, atom2any=None):
        self.atom2any = atom2any
    def exponent(self, AB, i, j):
        return norm(AB)
    def Rcut(self, max_exponent):
        """Return the maximum distance leading to max_exponent."""
        return max_exponent

class CovalenceDamper(Damper):
    """Damper class for covalent bonds (exponent in set([0, HUGE]))."""
    def __init__(self, atom2name, covrads=COVRADS):
        self.covrads = covrads
        self.atom2name = atom2name
    def exponent(self, AB, i, j):
        cr_i = self.covrads[canonize(self.atom2name[i])]
        cr_j = self.covrads[canonize(self.atom2name[j])]
        if norm(AB) < 1.3 * (cr_i + cr_j):
            return 0.
        else:
            return HUGE
    def Rcut(self, max_exponent):
        """Return the maximum distance leading to max_exponent."""
        return max(self.covrads[name] for name in list(self.atom2name.values()))


class LindhExponent(Damper):
    """Class of the LINDH object which provides the exponent factors."""
    def __init__(self, atom2row, alphas=ALPHAS, ref_ds=REF_DS):
        self.alphas = alphas
        self.ref_ds = ref_ds
        self.atom2row = atom2row

    def exponent(self, AB, i_atom, j_atom):
        """Return the exponent for distance AB of given types."""
        i_row, j_row = self.atom2row[i_atom], self.atom2row[j_atom]
        alpha = self.alphas[i_row, j_row]
        ref_d = self.ref_ds[i_row, j_row]
        return alpha * (AB**2 - ref_d**2)

    def Rcut(self, max_exponent):
        """Return the maximum distance for given exponent."""
        lr_alpha = np.min(self.alphas)
        lr_ref_d = np.max(self.ref_ds)
        # max_exponent == lr_alpha * (Rcut**2 - lr_ref_d**2)
        # max_exponent / lr_alpha + ref_d**2 == Rcut**2
        Rcut = np.sqrt(max_exponent / lr_alpha + lr_ref_d**2)
        return Rcut


class Bravais(object):
    """Provide tools related to some given Bravais lattice.

    May be initialized by a list of one, two, or three Bravais vectors.
    Provides tools to fold a vector or several vectors into the central
    parallel epipede (into_pe) and to retrieve a list of lattice vectors
    within a given radius (all_within).
    """
    def __init__(self, lattice_vectors):
        """Initializes Bravais object."""
        if lattice_vectors is None:
            lattice_vectors = []
        else:
            lattice_vectors = list(lattice_vectors)
        self.n = len(lattice_vectors)
        if self.n > 0:
            self.bra = np.array(lattice_vectors)
            self.ibra = np.linalg.pinv(self.bra)
            self.rec = 2.*np.pi*np.transpose(self.ibra)
        else:
            self.bra = np.empty((0, 3))
            self.rec = np.empty((0, 3))

    def latvec(self, abc):
        """Return a lattice vector from integer Bravais indices."""
        vec = np.zeros(3)
        a, b, c = abc
        if a != 0: vec += a*self.bra[0,:]
        if b != 0: vec += b*self.bra[1,:]
        if c != 0: vec += c*self.bra[2,:]
        return vec

    def into_pe(self, vecs):
        """Fold vectors (last dimension 3) into parallel epipede.

        Examples:
        >>> lat = Bravais([[1,0,0], [0,2,0]])
        >>> np.allclose(lat.into_pe([3.2, 1.3, 0.5]), [0.2, -0.7, 0.5])
        True
        """
        vecs = np.asarray(vecs, dtype=float)
        shape = vecs.shape
        if shape[-1] != 3: raise ValueError("Last dim should be 3.")
        n_vec = np.product(shape[:-1])
        reslist = []
        for vec in vecs.reshape((n_vec, 3)):
            rcoeff = np.dot(vec, self.ibra)
            icoeff = np.around(rcoeff)
            res = vec - np.dot(icoeff, self.bra)
            reslist.append(res)
        return np.array(reslist).reshape(shape)

    def all_within(self, Rcut, add_base_PE=False):
        r"""Return a list of all lattice vector indices shorter than Rcut.

        If base_PE is True, add one parallel epipede (PE) to the region.

        Examples:
        >>> cos60, sin60 = np.cos(np.pi/3), np.sin(np.pi/3)
        >>> lat = Bravais([[1,0,0], [cos60, sin60, 0], [0,0,1.5]])
        >>> lat.all_within(0.5) == [(0, 0, 0)]
        True
        >>> (1, 0, 0) in lat.all_within(1.1)
        True
        >>> set1 = lat.all_within(1.2)
        >>> len(set1)
        7
        >>> len(lat.all_within(0.2, add_base_PE=True))
        27
        
        The resulting vectors are sorted:
        >>> lat.all_within(2.2)[0] == (0, 0, 0)
        True
        """
        len_rec = np.array([norm(v) for v in self.rec])
        ns = np.floor(len_rec * Rcut / (2*np.pi)) # cross check with diss
        n_cells = np.zeros(3, int)
        n_cells[:self.n] = ns
        abcs = set()
        for a in range(-n_cells[0], n_cells[0]+1):
            for b in range(-n_cells[1], n_cells[1]+1):
                for c in range(-n_cells[2], n_cells[2]+1):
                    vec = self.latvec((a, b, c))
                    if norm(vec) <= Rcut:
                        abcs.add((a, b, c))
        if add_base_PE:
            # add one in each direction
            def _around(d, n): return [-1, 0, 1] if d < n else [0]
            old_abcs = set(abcs)
            for i, j, k in old_abcs:
                for ii in _around(0, self.n):
                    for jj in _around(1, self.n):
                        for kk in _around(2, self.n):
                            abcs.add((i+ii, j+jj, k+kk))

        def _norm_of_abc(abc): return norm(self.latvec(abc))
        return sorted(abcs, key=_norm_of_abc)

def get_pairs(atoms1, atoms2, Rcut, use_scipy=True):
    if use_scipy:
        try:
            import scipy.spatial  # KDTree
            have_scipy = True
        except ImportError:
            have_scipy = False
    else:
        have_scipy = False

    if have_scipy:
        tree1 = scipy.spatial.KDTree(atoms1)
        tree2 = scipy.spatial.KDTree(atoms2)
        pairs = tree1.query_ball_tree(tree2, Rcut)
    else:
        sys.stderr.write("No scipy found; using fallback.\n")
        pairs = []
        for i, atom1 in enumerate(atoms1):
            this_pairs = []
            for j, atom2 in enumerate(atoms2):
                dist = norm(atom2 - atom1)
                if dist < Rcut:
                    this_pairs.append(j)
            pairs.append(this_pairs)
        sys.stderr.write("Searching done.\n")
    return pairs



class Pairs(object):
    """Find chains (pairs, triples, ...) of atoms.

    Example:
    # Chain of pairs, one at zero, one slightly distorted.
    >>> bra = [[1, 0, 0]]
    >>> atom = [[0, 0, 0], [0.01, 0.25, 0]]
    >>> pairs = Pairs(bra, atom, SimpleDamper(), 1.5)
    >>> bonds = list(pairs.chains(2, 1.01))   # all bonds up to length 1.
    >>> for damp, atlist in bonds:  # only intracell and its own images.
    ...     assert len(atlist) == 2
    ...     print "%6.4f %s" % (damp, atlist)
    0.2502 [(0, (0, 0, 0)), (1, (0, 0, 0))]
    1.0000 [(0, (0, 0, 0)), (0, (1, 0, 0))]
    1.0000 [(1, (0, 0, 0)), (1, (1, 0, 0))]
    >>> bendings = list(pairs.chains(3, 1.251))   # 1.251: one short 1->2 bond.
    >>> for damp, atlist in bendings:
    ...     assert len(atlist) == 3
    ...     print "%6.4f %s" % (damp, atlist)
    1.2502 [(0, (0, 0, 0)), (0, (-1, 0, 0)), (1, (-1, 0, 0))]
    1.2502 [(0, (0, 0, 0)), (0, (1, 0, 0)), (1, (1, 0, 0))]
    1.2502 [(0, (0, 0, 0)), (1, (0, 0, 0)), (1, (-1, 0, 0))]
    1.2502 [(0, (0, 0, 0)), (1, (0, 0, 0)), (1, (1, 0, 0))]
    """
    def __init__(self, bra, atom, damper, max_sing_thres):
        """Initialize Pairs object.

        Returns a Pairs object containing all pairs which the damper gives
        a value smaller than max_sing_thres
        """
        # save parameters
        self.lat = Bravais(bra)
        self.atom = np.array(atom, dtype=float)
        self.n_atom = len(self.atom)
        if self.atom.shape != (self.n_atom, 3):
            raise ValueError("Invalid atom shape")
        self.damper = damper
        self.max_sing_thres = max_sing_thres
        Rcut = self.damper.Rcut(max_sing_thres)

        # get pairs
        self.abcs = self.lat.all_within(Rcut, add_base_PE=True)
        per_atom = []
        for abc in self.abcs:
            latvec = self.lat.latvec(abc)
            for atomvec in self.atom:
                per_atom.append(latvec + atomvec)
        self.per_atom = np.array(per_atom)
        pairs = get_pairs(self.atom, self.per_atom, Rcut)
        assert len(pairs) == self.n_atom

        # sort pairs
        self.pairs = []
        for i, partners in enumerate(pairs):
            proc_partners = []
            for jj in partners:
                if jj == i: continue
                Avec = self.atom[i]
                Bvec = self.per_atom[jj]
                vec = Bvec - Avec
                j = jj % self.n_atom    # original vector
                # Can be the conversion problem, Must be integer
                a = jj // self.n_atom
                abc = self.abcs[a]
                Rvec = self.lat.latvec(abc)
                assert np.allclose(vec, Rvec + self.atom[j] - self.atom[i])
                damp = damper.exponent(norm(vec), i, j)
                proc_partners.append((j, abc, damp))
            proc_partners.sort(key=operator.itemgetter(2)) # sort by damp
            self.pairs.append(proc_partners)

    def getvec(self, iabc):
        i, abc = iabc
        return self.lat.latvec(abc) + self.atom[i]

    def chains(self, n, thres, directed=True):
        """Return a list of (damp, [(i, abc), ...]) tuples of n-chains.

        This is the main workhorse and returns a weight-sorted list of
        bonds (n=2), bendings (n=3), or torsions (n=3).
        """
        res = []
        for i in range(self.n_atom):
            res.extend(self._chains_i(i, n, thres))
        if directed:
            final = []
            for damp, atlist in res:
                Avec = self.getvec(atlist[0])
                Dvec = self.getvec(atlist[-1])
                if isposvec(Dvec - Avec):
                    final.append((damp, atlist))
        final.sort()
        return final

    def _chains_i(self, i, n, thres):
        """Get all chains of length n from atom i."""
        if n < 2: raise ValueError("n should be at least 2.")
        res = []
        Avec = self.atom[i]
        for j, abc, damp in self.pairs[i]:
            if damp > thres: break    # they should be sorted
            if n == 2:
                # just pairs
                Bvec = self.lat.latvec(abc) + self.atom[j]
                tot_chain_damp = damp
                res.append((tot_chain_damp, [(i, (0,0,0)), (j, abc)]))
            else:
                # recursion
                rest_thres = thres - damp
                for chain_damp, atlist in self._chains_i(j, n-1, rest_thres):
                    shifted_atlist = [(i, (0,0,0))]
                    for (k, kabc) in atlist:
                        kabc = tuple([ai+ak for (ai, ak) in zip(abc, kabc)])
                        if i == k and kabc == (0, 0, 0):
                            break   # self reference
                        shifted_atlist.append((k, kabc))
                    else:
                        tot_chain_damp = damp + chain_damp
                        res.append((tot_chain_damp, shifted_atlist))
        return sorted(res)

class HessianBuilder(object):
    """Builder object for Hessians by rank-one additions.

    For a Hessian which is built successively as a sum of rank-one updates:
       H_{nu i, mu j} = \sum_k fac v_{nu i} v_{mu j}
    where each update vector is assumed to be sparse wrt the number of
    associated atoms.  The rank-one update is done by:
      HB.add_rank1(fac, vec)
    where vec == {nu: [x,y,z], ...}

    >>> HB = HessianBuilder(2, 0)
    >>> HB.add_rank1(1.0, {0: [1., 0., 0.]})
    >>> HB.add_rank1(0.5, {1: [0., 0., 1.]})
    >>> HD = np.zeros((2, 3, 2, 3))
    >>> HD[0,0,0,0] = 1.0
    >>> HD[1,2,1,2] = 0.5
    >>> np.allclose(HB.to_array(), HD)
    True
    """
    def __init__(self, n_atom, n_dyn_periodic):
        self.n_atom = n_atom
        self.n_dyn_periodic = n_dyn_periodic
        self.n_vec = n_atom + n_dyn_periodic
        self.Hdict = dict()

    def add_rank1(self, fac, vec):
        """Add rank-one term vec * vec * vec^T.

        Here, vec = {i_atom: np.array([xi, yi, zi]), ...}.
        """
        # Make sure that we have np.ndarrays
        for i_atom in vec:
            vec[i_atom] = np.asarray(vec[i_atom])
        # Perform dyadic product
        for i_atom, ivec in vec.items():
            for j_atom, jvec in vec.items():
                blk = fac * ivec[:,np.newaxis] * jvec[np.newaxis,:]
                _acc_dict((i_atom, j_atom), self.Hdict, blk)

    def add_rank1_from_atlist(self, fac, dq_datom, atlist):
        """Add rank-one term vec * vec * vec^T.

        Here, dq_atom = [np.array([xi, yi, zi]), ...], and
        atlist = [(i_tau, (a, b, c)), ...].
        """
        vecdict = dict()
        for (dqi, (i_atom, abc)) in zip(dq_datom, atlist):
            # Force term:
            _acc_dict(i_atom, vecdict, dqi)
            # Stress term:
            for i_bra, a_bra in enumerate(abc):
                i_vec = self.n_atom + i_bra
                if abs(a_bra) > 0 and i_bra < self.n_dyn_periodic:
                    _acc_dict(i_vec, vecdict, a_bra * dqi)
        self.add_rank1(fac, vecdict)

    def add_unity(self, fac):
        """Add multiple of unity."""
        blk = fac * np.identity(3)
        for i_vec in range(self.n_vec):
            _acc_dict((i_vec, i_vec), self.Hdict, blk)

    def to_array(self):
        """Construct full np.ndarray (only atomic coordinates, no stress)."""
        H = np.zeros((self.n_vec, 3, self.n_vec, 3))
        for i_atom, j_atom in self.Hdict:
            H[i_atom, :, j_atom, :] = self.Hdict[(i_atom, j_atom)]
        return H

def format_atlist(atlist, is_periodic):
    """Nicely format an atom list (atlist).

    Additionally adds 1 to atom numbers (-> start with 1):
    >>> print format_atlist([(0, (0, 0, 0)), (1, (0, -1, 0))], True)
      1( 0, 0, 0) --  2( 0,-1, 0)
    """
    if is_periodic:
        return " --".join(["%3i(%2i,%2i,%2i)" % ((i_atom+1,) + abc)
                           for (i_atom, abc) in atlist])
    else:
        return " --".join(["%3i" % (i_atom+1) for (i_atom, abc) in atlist])


def makeorthvec(orth):
    """Construct a (3 component) vector orthogonal to orth.

    >>> import numpy.random
    >>> vec = numpy.random.random(3)
    >>> assert np.dot(vec, makeorthvec(vec)) < 1e-12
    """
    orth /= norm(orth)
    vec = np.cross(orth, np.array([0., 0., 1.]))
    if (norm(vec) < 0.33):
        vec = np.cross(orth, np.array([1., 0., 0.]))
    return vec / norm(vec)


def model_matrix(bra, atom, builder, damper, thres, logfile=None):
    """Construct model Hessian.  Returns the HessianBuilder object.

    >>> builder = HessianBuilder(2, 0)
    >>> damper = LindhExponent(atom2row=[0, 0])    # Say, two hydrogens.
    >>> atom = np.array([[1., 0., 0.], [0., 0., 0.]])
    >>> HB = model_matrix(None, atom, builder, damper, 10.)
    >>> assert HB is builder
    >>> H = builder.to_array().reshape((3*2, 3*2))
    >>> assert np.allclose(H, H.T)
    >>> assert np.allclose(H[:,1:3], 0.)
    >>> assert np.allclose(H[:,4:6], 0.)
    >>> assert not np.allclose(H[0,0], 0.)
    >>> assert np.allclose(H[0,0], H[3,3])
    >>> assert np.allclose(H[0,3], -H[0,0])
    """
    thres_fac = np.exp(- thres) * K_BOND
    if logfile is not None:
        logfile.write("# Neglecting anything with a prefac < %8.3g "
                      "eV[/A^2]\n\n" % thres_fac)

    is_per = bra is not None and len(bra) > 0
    pairs = Pairs(bra, atom, damper, thres)

    # bonds:
    for damp, atlist in pairs.chains(2, thres):
        fac = np.exp(- damp) * K_BOND
        if fac < thres_fac:
            continue
        vecs = [pairs.getvec(at) for at in atlist]
        q, dq = q_bond(*vecs)
        if logfile is not None:
            logfile.write("# bond: %4.2f A %s  "
                          "[damp: %8.3g; prefac: %8.3g eV/A^2]\n" %
                          (q, format_atlist(atlist, is_per), damp, fac))
        builder.add_rank1_from_atlist(fac, dq, atlist)
    if logfile is not None:
        logfile.write("\n")

    # bendings:
    for damp, atlist in pairs.chains(3, thres):
        fac = np.exp(- damp) * K_BENDING
        if fac < thres_fac:
            continue
        vecs = [pairs.getvec(at) for at in atlist]
        q, dq = q_bending(*vecs)
        if logfile is not None:
            logfile.write("# angle: %4.0f deg %s  "
                          "[damp: %8.3g; prefac: %8.3g eV]\n" %
                          (np.rad2deg(q), format_atlist(atlist, is_per),
                           damp, fac))
        if 0.05*np.pi < q < 0.95*np.pi:
            builder.add_rank1_from_atlist(fac, dq, atlist)
        else:
            Avec, Bvec, Cvec = vecs
            wvec = makeorthvec(Cvec - Avec)
            q, dq = q_bending(Avec, Bvec, Cvec, direction=wvec)
            builder.add_rank1_from_atlist(fac, dq, atlist)
            wvec = np.cross(Bvec - Avec, wvec)
            wvec /= norm(wvec)
            q, dq = q_bending(Avec, Bvec, Cvec, direction=wvec)
            builder.add_rank1_from_atlist(fac, dq, atlist)
    if logfile is not None:
        logfile.write("\n")

    # torsions
    for damp, atlist in pairs.chains(4, thres):
        fac = np.exp(- damp) * K_TORSION
        if fac < thres_fac:
            continue
        vecs = [pairs.getvec(at) for at in atlist]
        try:
            q, dq = q_torsion(*vecs)
            if logfile is not None:
                logfile.write("# torsion: %4.0f deg %s  "
                              "[damp: %8.3g; prefac: %8.3g eV]\n" %
                              (np.rad2deg(q), format_atlist(atlist, is_per),
                               damp, fac))
            builder.add_rank1_from_atlist(fac, dq, atlist)
        except ValueError:
            if logfile is not None:
                logfile.write("# torsion: ---- deg %s "
                              "[damp: %8.3g; prefac: %8.3g eV]\n" %
                              (format_atlist(atlist, is_per), damp, fac))
    if logfile is not None:
        logfile.write("\n")

    
    return builder


##############################################################################
##################################### dd #####################################
##############################################################################

# The "dd" functions generally share the same interface.  The arguments are
# 2-tuples where the first object contains the actual parameter and the second
# object contains the derivatives of this parameter with respect to the
# original input parameters.  From this, the output value is calculated.
# Additionally, the derivatives of this value with respect to the original
# input parameters are evaluated by the chain rule.  The return value is a
# tuple of the result value and its derivative.

# Obviously, this is not the most efficient way to do it.  But at least,
# it works...

def _dd_matmat(val_shape, dval_di, di_dvar):
    val_rank = len(val_shape)
    assert val_shape == np.shape(dval_di)[:val_rank]
    di_shape = np.shape(dval_di)[val_rank:]
    di_rank = len(di_shape)
    assert di_shape == np.shape(di_dvar)[:di_rank]
    axes1 = list(range(val_rank, val_rank+di_rank))
    return np.tensordot(dval_di, di_dvar, (axes1, list(range(di_rank))))

def _dd_broadcast(val, dval):
    val_rank = len(np.shape(val))
    assert np.shape(val) == np.shape(dval)[:val_rank]
    dval_rank = len(np.shape(dval))
    newshape = np.shape(val) + (dval_rank-val_rank)*(1,)
    return np.reshape(val, newshape)

def dd_sum(*arg_ds):
    shape = np.shape(arg_ds[0][1])
    res = np.float(0.)
    dres = np.zeros(shape)
    for arg, darg in arg_ds:
        res += np.asarray(arg)
        dres += np.asarray(darg)
    return res, dres

def dd_mult(vec_d, fac):
    vec, dvec = vec_d
    return fac*vec, fac*dvec

def dd_prod(*arg_ds):
    shape = np.shape(arg_ds[0][1])
    res = np.float(1.)
    dres = np.zeros(shape)
    for arg, darg in arg_ds:
        dres *= arg                     # update previous derivs
        dres += np.asarray(darg) * res  # update with previous factors
        res *= arg                      # update value
    return res, dres

def dd_power(var_d, n):
    var, dvar = var_d
    val = var**n
    dval = n*(var**(n-1)) * dvar
    return val, dval

def dd_dot(vec1_d, vec2_d):
    vec1, dvec1 = vec1_d
    vec2, dvec2 = vec2_d
    res = np.dot(vec1, vec2)
    dres = (np.tensordot(vec1, dvec2, (-1, 0)) +
            np.tensordot(vec2, dvec1, (-1, 0)))
    return res, dres

def dd_cross(vec1_d, vec2_d):
    vec1, dvec1 = vec1_d
    vec2, dvec2 = vec2_d
    assert np.shape(vec1) == np.shape(vec2) == (3,)   # otherwise...
    res = np.cross(vec1, vec2)
    dres = - np.cross(vec2, dvec1, axisb=0).T + np.cross(vec1, dvec2, axisb=0).T
    return res, dres

def dd_norm(vec_d):
    return dd_power(dd_dot(vec_d, vec_d), 0.5)

def dd_normalized(vec_d):
    vec, dvec = vec_d
    fac, dfac = dd_power(dd_norm(vec_d), -1.)
    res = fac*vec
    dres = fac*dvec + vec[:,np.newaxis]*dfac[np.newaxis,:]
    return res, dres

def dd_cosv1v2(vec1_d, vec2_d):
    return dd_prod(dd_dot(vec1_d, vec2_d),
                   dd_power(dd_norm(vec1_d), -1.),
                   dd_power(dd_norm(vec2_d), -1.))

def dd_arccos(val_d):
    val, dval = val_d
    if 1. < abs(val) < 1. + 1e-10:
        val = np.sign(val)
    res = np.arccos(val)
    vval = _dd_broadcast(val, dval)
    dres = - 1. / np.sqrt(1. - vval**2) * dval
    return res, dres

def dd_arcsin(val_d):
    val, dval = val_d
    if 1. < abs(val) < 1. + 1e-10:
        val = np.sign(val)
    res = np.arcsin(val)
    vval = _dd_broadcast(val, dval)
    dres = 1. / np.sqrt(1. - vval**2) * dval
    return res, dres

def dd_angle(vec1_d, vec2_d):
    return dd_arccos(dd_cosv1v2(vec1_d, vec2_d))

def dd_bondlength(pos1_d, pos2_d):
    AB_d = dd_sum(pos2_d, dd_mult(pos1_d, -1.))
    return dd_norm(AB_d)

def dd_bondangle(pos1_d, pos2_d, pos3_d):
    BA_d = dd_sum(pos2_d, dd_mult(pos1_d, -1.))
    BC_d = dd_sum(pos2_d, dd_mult(pos3_d, -1.))
    return dd_angle(BA_d, BC_d)

def dd_bondangle_directed(pos1_d, pos2_d, pos3_d, dir_d):
    BA_d = dd_sum(pos2_d, dd_mult(pos1_d, -1.))
    BC_d = dd_sum(pos2_d, dd_mult(pos3_d, -1.))
    return dd_directed_angle(BA_d, BC_d, dir_d)

def dd_arctan2(y_d, x_d):
    y, dy = y_d
    x, dx = x_d
    phi = np.arctan2(y, x)
    tan, dtan = dd_prod(x_d, dd_power(y_d, -1.))
    tan = _dd_broadcast(tan, dtan)
    dphi = (1. + tan**2) * dtan
    return phi, dphi

def dd_directed_angle(vec1_d, vec2_d, dir_d):
    ndir_d = dd_normalized(dir_d)
    vv1_d = dd_cross(vec1_d, ndir_d)
    vv2_d = dd_cross(vec2_d, ndir_d)
    if (norm(vv1_d[0]) < 1e-7 or
        norm(vv2_d[0]) < 1e-7):
        return 0., np.zeros(np.shape(vec1_d[1])[1:])
    vv1_d = dd_normalized(vv1_d)
    vv2_d = dd_normalized(vv2_d)
    cosphi_d = dd_dot(vv1_d, vv2_d)
    vvv_d = dd_cross(vv1_d, vv2_d)
    sinphi_d = dd_dot(vvv_d, ndir_d)
    # phi_d = dd_arctan2(sinphi_d, cosphi_d)
    if (abs(cosphi_d[0]) < np.sqrt(0.5)):
        phi, dphi = dd_arccos(cosphi_d)
        if sinphi_d[0] < 0.:
            phi *= -1.
            dphi *= -1.
    else:
        phi, dphi = dd_arcsin(sinphi_d)
        if cosphi_d[0] < 0.:
            phi = - np.pi - phi
            if phi < np.pi: phi += 2*np.pi
            dphi *= -1.
    return phi, dphi

def dd_bondtorsion(pos1_d, pos2_d, pos3_d, pos4_d):
    BA_d = dd_sum(pos2_d, dd_mult(pos1_d, -1.))
    BC_d = dd_sum(pos2_d, dd_mult(pos3_d, -1.))
    CD_d = dd_sum(pos3_d, dd_mult(pos4_d, -1.))
    return dd_directed_angle(BA_d, CD_d, BC_d)

##############################################################################
###################################### q #####################################
##############################################################################

def q_bond(Avec, Bvec):
    """Bond length and derivative wrt vector AB.

    Test:
    >>> np.allclose(q_bond([0., 0., 0.], [1., 1., 1.])[0], np.sqrt(3.))
    True
    >>> assert _test_qgrad(q_bond, 2) < 1e-5
    """
    Avec_d = (np.asarray(Avec), np.c_[ID, ZERO])
    Bvec_d = (np.asarray(Bvec), np.c_[ZERO, ID])
    q, dq = dd_bondlength(Avec_d, Bvec_d)
    return q, dq.reshape((2, 3))

def q_bending(Avec, Bvec, Cvec, direction=None):
    """Bond angle and derivative wrt vectors AB and BC.

    Test:
    >>> A = np.array([ 1, 1, 1])
    >>> B = np.zeros(3)
    >>> C = np.array([-1,-1, 1])
    >>> print round(np.rad2deg(q_bending(A, B, C)[0]), 1)
    109.5
    >>> assert _test_qgrad(q_bending, 3) < 1e-5
    """
    Avec_d = (np.asarray(Avec), np.c_[ID, ZERO, ZERO])
    Bvec_d = (np.asarray(Bvec), np.c_[ZERO, ID, ZERO])
    Cvec_d = (np.asarray(Cvec), np.c_[ZERO, ZERO, ID])
    if direction is None:
        q, dq = dd_bondangle(Avec_d, Bvec_d, Cvec_d)
    else:
        dir_d = (direction, np.c_[ZERO, ZERO, ZERO])
        q, dq = dd_bondangle_directed(Avec_d, Bvec_d, Cvec_d, dir_d)
    return q, dq.reshape((3, 3))

def q_torsion(Avec, Bvec, Cvec, Dvec):
    """Bond torsion and derivative wrt vectors AB, BC, and CD.

    Test:
    >>> A = np.array([0., 0., 1.])
    >>> B = np.array([0., 0., 0.])
    >>> C = np.array([1., 0., 0.])
    >>> D = np.array([1., 1., 0.])
    >>> print round(np.rad2deg(q_torsion(A, B, C, D)[0]), 5)
    90.0
    >>> try:
    ...    assert _test_qgrad(q_torsion, 4) < 1e-5
    ... except ValueError:   # May happen with bad luck.
    ...    pass
    """
    ABvec = Bvec - Avec
    BCvec = Cvec - Bvec
    CDvec = Dvec - Cvec
    cosABC = np.dot(ABvec, BCvec) / (norm(ABvec) * norm(BCvec))
    cosBCD = np.dot(BCvec, CDvec) / (norm(BCvec) * norm(CDvec))
    if max(abs(cosABC), abs(cosBCD)) > 0.99:   # nearly linear angle
        raise ValueError("Nearly linear angle")
    else:
        Avec_d = (np.asarray(Avec), np.c_[ID, ZERO, ZERO, ZERO])
        Bvec_d = (np.asarray(Bvec), np.c_[ZERO, ID, ZERO, ZERO])
        Cvec_d = (np.asarray(Cvec), np.c_[ZERO, ZERO, ID, ZERO])
        Dvec_d = (np.asarray(Dvec), np.c_[ZERO, ZERO, ZERO, ID])
        q, dq = dd_bondtorsion(Avec_d, Bvec_d, Cvec_d, Dvec_d)
        return q, dq.reshape(4, 3)

##############################################################################
############################# unit test utilities ############################
##############################################################################

def _test_qgrad(q_func, n):
    import scipy.optimize
    import numpy.random
    x0 = np.random.standard_normal(3*n)
    def func(x):
        vecs = np.asarray(x).reshape((n,3))
        q, dq = q_func(*vecs)
        return q
    def grad(x):
        vecs = np.asarray(x).reshape((n,3))
        q, dq = q_func(*vecs)
        return np.reshape(dq, (3*n))
    return scipy.optimize.check_grad(func, grad, x0)

def testmod():
    import doctest
    doctest.testmod(raise_on_error=False)



##############################################################################
##################################### main ###################################
##############################################################################

def LindhHessian(atoms):

    cutoff = 15.
    add_unity = 0.005

    atom = atoms.get_positions()
    atom_name = atoms.get_chemical_symbols()
    bra = atoms.get_cell()[:]
    if np.array_equal(bra, np.zeros(9).reshape(3, 3)):
        bra = []

    n_atom = len(atom_name)

    n_dyn_periodic = 0
    n_vec = n_atom + n_dyn_periodic
    builder = HessianBuilder(n_atom, n_dyn_periodic)
    damper = LindhExponent([name2row(name) for name in atom_name])
    logfile = None
    model_matrix(bra, atom, builder, damper, cutoff, logfile=logfile)
    # builder.add_unity(add_unity)

    hessian = builder.to_array().reshape((3*n_vec, 3*n_vec))
    return hessian

def preconditioned_hessian(structure, fixed_frame, parameters, atoms_current, H, task="update"):

    if len(structure.molecules) > 1:
        a0 = structure.molecules[0].copy()
        for i in range(1, len(structure.molecules)):
            a0+=structure.molecules[i]
    else:
        a0 = structure.molecules[0]

    if hasattr(fixed_frame, "fixed_frame"):
        all_atoms = a0 + fixed_frame.fixed_frame
    else:
        all_atoms = a0
    symbol = all_atoms.get_chemical_symbols()[0]  
    atoms = all_atoms.copy()
    atoms.set_positions(atoms_current.get_positions())


    ### Preconditioner part

    # Isolate indices
    hessian_indices = []
    for i in range(len(structure.molecules)):
        hessian_indices.append(["mol{}".format(i) for k in range(3*len(structure.molecules[i]))])
    if hasattr(fixed_frame, "fixed_frame"):
        hessian_indices.append(["fixed_frame" for k in range(3*len(fixed_frame.fixed_frame))])
    hessian_indices = sum(hessian_indices, [])

    # Genrate all nececcary hessians
    precons = {}
    precon_names = []

    precons_parameters = {
        "mol" : parameters["calculator"]["preconditioner"]["mol"]["precon"],
        "fixed_frame" : parameters["calculator"]["preconditioner"]["fixed_frame"]["precon"], 
        "mol-mol" : parameters["calculator"]["preconditioner"]["mol-mol"]["precon"],
        "mol-fixed_frame" : parameters["calculator"]["preconditioner"]["mol-fixed_frame"]["precon"]
    }

    routine = {
        "mol" : parameters["calculator"]["preconditioner"]["mol"][task],
        "fixed_frame" : parameters["calculator"]["preconditioner"]["fixed_frame"][task], 
        "mol-mol" : parameters["calculator"]["preconditioner"]["mol-mol"][task],
        "mol-fixed_frame" : parameters["calculator"]["preconditioner"]["mol-fixed_frame"][task]
    }
    print(routine)
    precon_names = [list(precons_parameters.values())[i] for i in range(len(routine)) if list(routine.values())[i]]

    if "Lindh" in precon_names:
        precons["Lindh"] = LindhHessian(atoms)
    if "Exp" in precon_names:
        precons["Exp"] = ExpHessian(atoms, mu=structure.mu, A=structure.A)
    if "vdW" in precon_names:
        precons["vdW"] = vdwHessian(atoms)
    if "ID" in precon_names:
        precons["ID"] = np.eye(3 * len(atoms)) * 70

    # Combine hessians into hessian
    N = len(all_atoms)
    preconditioned_hessian = H
    # for ind in range(len(preconditioned_hessian)):
        # print("old")
        # print(preconditioned_hessian[ind, ind])


    # for i in range(3 * len(all_atoms)):
    #     for j in range(3 * len(all_atoms)):
    #         if hessian_indices[i] == hessian_indices[j]:
    #             if "fixed_frame" in hessian_indices[j] and routine["fixed_frame"]:
    #                 p = precons_parameters["fixed_frame"]
    #                 preconditioned_hessian[i,j] = precons[p][i,j]
    #             elif "mol" in hessian_indices[j] and routine["mol"]:
    #                 p = precons_parameters["mol"]
    #                 preconditioned_hessian[i,j] = precons[p][i,j]
    #         else:
    #             if "fixed_frame" not in [hessian_indices[i], hessian_indices[j]] and routine["mol-mol"]:
    #                 p = precons_parameters["mol-mol"]
    #                 preconditioned_hessian[i,j] = precons[p][i,j]
    #             elif routine["mol-fixed_frame"]:               
    #                 p = precons_parameters["mol-fixed_frame"]
    #                 preconditioned_hessian[i,j] = precons[p][i,j]
    
    if task == "update":
        if not any(a ==  False for a in routine.values()):
            return preconditioned_hessian
        else:
            for i in range(3 * len(all_atoms)):
                for j in range(3 * len(all_atoms)):
                    if hessian_indices[i] == hessian_indices[j]:
                        if "fixed_frame" in hessian_indices[j] and routine["fixed_frame"]:
                            p = precons_parameters["fixed_frame"]
                            preconditioned_hessian[i,j] = precons[p][i,j]
                        elif "mol" in hessian_indices[j] and routine["mol"]:
                            p = precons_parameters["mol"]
                            preconditioned_hessian[i,j] = precons[p][i,j]
                    else:
                        if "fixed_frame" not in [hessian_indices[i], hessian_indices[j]] and routine["mol-mol"]:
                            p = precons_parameters["mol-mol"]
                            preconditioned_hessian[i,j] = precons[p][i,j]
                        elif routine["mol-fixed_frame"]:               
                            p = precons_parameters["mol-fixed_frame"]
                            preconditioned_hessian[i,j] = precons[p][i,j]
            for ind in range(len(preconditioned_hessian)):
                preconditioned_hessian[ind, ind] = 0
                preconditioned_hessian[ind, ind] = -np.sum(preconditioned_hessian[ind]) + 0.005
                if preconditioned_hessian[ind, ind] == 0.005:
                    preconditioned_hessian[ind, ind] = 70
            return preconditioned_hessian


    if task == "initial":
        for i in range(3 * len(all_atoms)):
            for j in range(3 * len(all_atoms)):
                if hessian_indices[i] == hessian_indices[j]:
                    if "fixed_frame" in hessian_indices[j] and routine["fixed_frame"]:
                        p = precons_parameters["fixed_frame"]
                        preconditioned_hessian[i,j] = precons[p][i,j]
                    elif "mol" in hessian_indices[j] and routine["mol"]:
                        p = precons_parameters["mol"]
                        preconditioned_hessian[i,j] = precons[p][i,j]
                else:
                    if "fixed_frame" not in [hessian_indices[i], hessian_indices[j]] and routine["mol-mol"]:
                        p = precons_parameters["mol-mol"]
                        preconditioned_hessian[i,j] = precons[p][i,j]
                    elif routine["mol-fixed_frame"]:               
                        p = precons_parameters["mol-fixed_frame"]
                        preconditioned_hessian[i,j] = precons[p][i,j]
        for ind in range(len(preconditioned_hessian)):
            preconditioned_hessian[ind, ind] = 0
            preconditioned_hessian[ind, ind] = -np.sum(preconditioned_hessian[ind]) + 0.005
            if preconditioned_hessian[ind, ind] == 0.005:
                preconditioned_hessian[ind, ind] = 70
        return preconditioned_hessian




