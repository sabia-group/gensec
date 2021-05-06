"""Create the preconditioner based on the geometry.

Attributes:
    abohr (float): convert Bohr to Angstrom
    hartree (float): convert Hartree to eV
    ID (TYPE): Description
    k_bending (TYPE): Description
    k_bond (TYPE): Description
    k_torsion (TYPE): Description
    ref_ds (TYPE): Description
    ZERO (TYPE): Description
"""

import sys
from ase.optimize.precon import Exp
import numpy as np
from numpy.linalg import norm
import operator
from ase.constraints import FixAtoms
from gensec.coefficients import (
    COVRADS,
    VDW_radii,
    C6_vdW,
    ALPHA_vdW,
    alphas_vdW,
)


def set_constrains(atoms, parameters):
    """Set the constrains

    Adds the constrains for geometry optimizations to the Atoms object.
    During geometry optimization the forces on the constrained atoms
    will be always zero.

    Args:
        atoms (ASE Atoms object): Geometry to optimize
        parameters (file): file with parameters for constrainings
    """

    z = parameters["calculator"]["constraints"]["z-coord"]
    c = FixAtoms(
        indices=[atom.index for atom in atoms if atom.position[2] <= z[-1]]
    )
    atoms.set_constraint(c)


def Kabsh_rmsd(atoms, initial, molindixes, removeHs=False):
    """Root-mean-square deviation (RMSD) between tructures

    Finds the optimal rotation for alignment of two structures
    with use of Kabsh algorithm and then calculates RMSD
    between all the corresponding atoms.

    Args:
        atoms (ASE Atoms object): The structure to calculate RMSD with the
        reference geometry.
        initial (ASE Atoms object): The reference geometry
        molindixes (list): list of indexes of the atoms object that
        should be taken into consideration while calculation of RMSD value
        removeHs (bool, optional): True if Hydrogens should be taken into
        consideration

    Returns:
        float: Root-mean-square deviation value between two structures
    """

    # Extract the necessary atoms
    mol = atoms[[atom.index for atom in atoms if atom.index in molindixes]]
    ref = initial[[atom.index for atom in initial if atom.index in molindixes]]
    if removeHs:
        mol = mol[[atom.index for atom in mol if atom.index != "H"]]
        ref = ref[[atom.index for atom in ref if atom.index != "H"]]
    # Center both molecules:
    mol.set_positions(mol.get_positions() - mol.get_center_of_mass())
    ref.set_positions(ref.get_positions() - ref.get_center_of_mass())

    coords1 = mol.get_positions()
    coords2 = ref.get_positions()

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
        rmsd_kabsh += sum(
            [(v[i] - w[i]) ** 2.0 for i in range(len(coords1[0]))]
        )
    return np.sqrt(rmsd_kabsh / len(coords1))


def ASR(hessian):
    """Acoustic sum rule (ASR)

    ASR comes from the continous translational invariance of the periodic
    system. With translation of the whole system by a uniform
    displacement, there should be no forces acting on the atoms.
    This function restores the ASR for the given Hessian matrix.

    Arguments:
        hessian (matrix): 3Nx3N where N is lenght of atoms

    Returns:
        matrix: Hessian matrix with correct ASR

    """
    x_range = [3 * ind for ind in range(int(len(hessian) / 3))]
    for ind in range(len(x_range)):
        elements = np.delete(x_range, ind)
        x, y, z = x_range[ind], x_range[ind] + 1, x_range[ind] + 2
        xs, ys, zs = elements, elements + 1, elements + 2
        # (0,0),  (1,1), (2,2) Block elements
        hessian[x, x] = -np.sum(hessian[x, xs])
        hessian[y, y] = -np.sum(hessian[y, ys])
        hessian[z, z] = -np.sum(hessian[z, zs])
        # (1,0),  (2,0), (2,1) Block elements Upper triangle
        hessian[x, y] = -np.sum(hessian[x, ys])
        hessian[x, z] = -np.sum(hessian[x, zs])
        hessian[y, z] = -np.sum(hessian[y, zs])
        # (0,1),  (0,2), (1,2) Block elements Lower Triangle
        hessian[y, x] = hessian[x, y]
        hessian[z, x] = hessian[x, z]
        hessian[z, y] = hessian[y, z]
    return hessian


def add_jitter(hessian, jitter):
    """Add jitter to the diagonal

    For the reasons of numerical stability adding jitter parameter to the
    diagonal of the matrix allows inverting of the matrix without problems.

    Arguments:
        hessian (matrix): 3Nx3N where N is lenght of atoms
        jitter (float): small value that is added to diagonal

    Returns:
        matrix: Hessian with jitter value added to the diagonal
    """

    return hessian + np.eye(len(hessian)) * jitter


def check_positive_symmetric(hessian):
    """Check properties of Hessian matrix

    The function in quazi-Newton's method has to be strongly convex
    which is true if Hessian is positive definite and symmetric matrix.
    Symmetric if all the values of the Hessian after substracting
    of it's transpose are smaller than 1e-10.
    Positive definite if all the eigenvalues are positive.

    Arguments:
        hessian (matrix): 3Nx3N where N is lenght of atoms

    Returns:
        boolean, boolean: True and True if Hessian symmetric
        and positive definite
    """

    symmetric = np.all(np.abs(hessian - hessian.T) < 1e-10)
    d, w = np.linalg.eigh(hessian)
    positive = np.all(d > 0)
    return symmetric, positive


# Preambule from Lindh.py pthon sctipt

abohr = 0.52917721  # in AA
hartree = 27.211383  # in eV

k_bond = 0.450 * hartree / abohr ** 2
k_bending = 0.150 * hartree
k_torsion = 0.005 * hartree

alphas = (
    np.array(
        [
            [1.0000, 0.3949, 0.3949],
            [0.3949, 0.2800, 0.2800],
            [0.3949, 0.2800, 0.2800],
        ]
    )
    * abohr ** (-2)
)
ref_ds = (
    np.array([[1.35, 2.10, 2.53], [2.10, 2.87, 3.40], [2.53, 3.40, 3.40]])
    * abohr
)

ID = np.identity(3)
ZERO = np.zeros((3, 3))


def vector_separation(cell_h, cell_ih, qi, qj):
    """Summary

    Args:
        cell_h (TYPE): Description
        cell_ih (TYPE): Description
        qi (TYPE): Description
        qj (TYPE): Description

    Returns:
        TYPE: Description
    """
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

    dij = np.dot(cell_h, sij).T  # back to i-pi shape
    rij = np.linalg.norm(dij, axis=1)

    return dij, rij


def vdwHessian(atoms):
    """Summary

    Args:
        atoms (TYPE): Description

    Returns:
        TYPE: Description
    """

    def periodic_R(cell_h, cell_ih, qi, qj):
        """Summary

        Args:
            cell_h (TYPE): Description
            cell_ih (TYPE): Description
            qi (TYPE): Description
            qj (TYPE): Description

        Returns:
            TYPE: Description
        """
        sij = np.dot(cell_ih, (qi - qj).T)
        sij -= np.rint(sij)
        dij = np.dot(cell_h, sij).T
        rij = np.linalg.norm(dij)
        return dij, rij

    N = len(atoms)
    coordinates = atoms.get_positions()
    atom_names = atoms.get_chemical_symbols()
    cell_h = atoms.get_cell()[:]
    cell_ih = atoms.get_reciprocal_cell()[:]
    hessian = np.zeros(shape=(3 * N, 3 * N))
    atomsRange = list(range(N))
    units = (abohr ** 6) * hartree

    def C6AB(A, B):
        """Summary

        Args:
            A (TYPE): Description
            B (TYPE): Description

        Returns:
            TYPE: Description
        """
        C6AB = (
            2.0
            * C6_vdW[A]
            * C6_vdW[B]
            / (
                ALPHA_vdW[B] / ALPHA_vdW[A] * C6_vdW[A]
                + ALPHA_vdW[A] / ALPHA_vdW[B] * C6_vdW[B]
            )
        )
        # C6AB = 0.5 * (C6_vdW[A] + C6_vdW[B])

        return C6AB * units  # in eV*Angstr^6

    def get_R0AB(A, B):
        """Summary

        Args:
            A (TYPE): Description
            B (TYPE): Description

        Returns:
            TYPE: Description
        """
        return (VDW_radii[B] + VDW_radii[A]) * 0.5 * abohr  # in Angstroms

    def rho_ij(A, B, R):
        """Summary

        Args:
            A (TYPE): Description
            B (TYPE): Description
            R (TYPE): Description

        Returns:
            TYPE: Description
        """

        def name2row(atom_name):
            """Return row number of atom type (starting with 0, max 2).

            Args:
                atom_name (TYPE): Description

            Returns:
                TYPE: Description
            """
            name = canonize(atom_name)
            if name in ("H", "He"):
                return 0
            elif name in ("Li", "Be", "B", "C", "N", "O", "F", "Ne"):
                return 1
            else:
                return 2

        R0_vdW_AB = get_R0AB(A, B)
        alpha = alphas_vdW[A][B]
        return np.exp(alpha * (R0_vdW_AB ** 2 - R ** 2))

    def C12AB(A, B, C6):
        """Summary

        Args:
            A (TYPE): Description
            B (TYPE): Description
            C6 (TYPE): Description

        Returns:
            TYPE: Description
        """
        R0AB = get_R0AB(A, B)
        C12AB = 0.5 * C6 * (R0AB ** 6)
        return C12AB  # in eV*Angstr^6

    def RAB(cell_h, cell_ih, qi, qj):
        """Summary

        Args:
            cell_h (TYPE): Description
            cell_ih (TYPE): Description
            qi (TYPE): Description
            qj (TYPE): Description

        Returns:
            TYPE: Description
        """
        if np.array_equal(cell_h, np.zeros([3, 3])):
            R = np.array(np.linalg.norm(qi - qj))
        else:
            dij, R = periodic_R(cell_h, cell_ih, qi, qj)
        return R  # in Angstroms

    def vdW_element(k, l, C6, C12, R0, R, qi, qj):
        """Summary

        Args:
            k (TYPE): Description
            l (TYPE): Description
            C6 (TYPE): Description
            C12 (TYPE): Description
            R0 (TYPE): Description
            R (TYPE): Description
            qi (TYPE): Description
            qj (TYPE): Description

        Returns:
            TYPE: Description
        """
        norm = (R0 / R) ** 2
        a1 = 48 * C6 * (qi[k] - qj[k]) * (qi[l] - qj[l]) * norm / R0 ** 10
        a2 = -168 * C12 * (qi[k] - qj[k]) * (qi[l] - qj[l]) * norm / R0 ** 16
        if k == l:
            a3 = -6 * C6 / R0 ** 8
            a4 = 12 * C12 / R0 ** 14
        else:
            a3 = 0
            a4 = 0
        return a1 + a2 + a3 + a4

    def vdW_element_exact(k, l, C6, C12, R0, R, qi, qj):
        """Summary

        Args:
            k (TYPE): Description
            l (TYPE): Description
            C6 (TYPE): Description
            C12 (TYPE): Description
            R0 (TYPE): Description
            R (TYPE): Description
            qi (TYPE): Description
            qj (TYPE): Description

        Returns:
            TYPE: Description
        """
        a1 = 48 * C6 * (qi[k] - qj[k]) * (qi[l] - qj[l]) / R ** 10
        a2 = -168 * C12 * (qi[k] - qj[k]) * (qi[l] - qj[l]) / R ** 16
        if k == l:
            a3 = -6 * C6 / R ** 8
            a4 = 12 * C12 / R ** 14
        else:
            a3 = 0
            a4 = 0
        return a1 + a2 + a3 + a4

    for i in atomsRange:
        for j in atomsRange:
            if j > i:
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
                        R0AB = get_R0AB(A, B)
                        # print("R ,",  R)
                        # print("rho, ",rho_ij(A, B, R))
                        # print(rho_ij(A, B, R))
                        # print(vdW_element(k, l, C6, C12, R0AB, R, qi, qj))
                        hessian[3 * i + k, 3 * j + l] = vdW_element(
                            k, l, C6, C12, R0AB, R, qi, qj
                        ) * rho_ij(A, B, R)
                        # hessian[3*i+k, 3*j+l] = vdW_element_exact(k, l, C6, C12, R0AB, R, qi, qj)

    # Fill the down triangle
    hessian = hessian + hessian.T
    # Calculate Acoustic sum rule
    hessian = ASR(hessian)
    # Add stabilization to the diagonal
    jitter = 0.005
    hessian = add_jitter(hessian, jitter)
    # Check if positive and symmetric:
    symmetric, positive = check_positive_symmetric(hessian)
    if not symmetric:
        print(
            "Hessian is not symmetric! Will give troubles during optimization!"
        )
        sys.exit(0)
    if not positive:
        print(
            "Hessian is not positive definite! Will give troubles during optimization!"
        )
        sys.exit(0)
    return hessian


def ExpHessian(atoms, mu=1, A=3.0, recalc_mu=False):
    """Summary

    Args:
        atoms (TYPE): Description
        mu (int, optional): Description
        A (float, optional): Description
        recalc_mu (bool, optional): Description

    Returns:
        TYPE: Description
    """
    # If needed mu is estimated in the beginning and never changed
    precon = Exp(mu=mu, A=3.0, recalc_mu=False)
    precon.make_precon(atoms)
    return precon.P.todense()


"""
Contribution from Jürgen Wieferink regarding the Lindh preconditioner
"""


def _acc_dict(key, d, val):
    """If key in dict, accumulate, otherwise set.

    Args:
        key (TYPE): Description
        d (TYPE): Description
        val (TYPE): Description
    """
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

    Args:
        vec (TYPE): Description
        eps (float, optional): Description
        noself (bool, optional): Description

    Returns:
        TYPE: Description

    Raises:
        ValueError: Description
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

    Args:
        atom_name (TYPE): Description

    Returns:
        TYPE: Description
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
    """Return row number of atom type (starting with 0, max 2).

    Args:
        atom_name (TYPE): Description

    Returns:
        TYPE: Description
    """
    name = canonize(atom_name)
    if name in ("H", "He"):
        return 0
    elif name in ("Li", "Be", "B", "C", "N", "O", "F", "Ne"):
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

        Args:
            AB (TYPE): Description
            i (TYPE): Description
            j (TYPE): Description

        Raises:
            NotImplementedError: Description
        """
        raise NotImplementedError()

    def Rcut(self, max_exponent):
        """Return the maximum distance leading to max_exponent.

        Args:
            max_exponent (TYPE): Description
        """


class SimpleDamper(Damper):
    """Damper for maximum chain lenght (exponent==bondlength).

    Attributes:
        atom2any (TYPE): Description
    """

    def __init__(self, atom2any=None):
        """Summary

        Args:
            atom2any (None, optional): Description
        """
        self.atom2any = atom2any

    def exponent(self, AB, i, j):
        """Summary

        Args:
            AB (TYPE): Description
            i (TYPE): Description
            j (TYPE): Description

        Returns:
            TYPE: Description
        """
        return norm(AB)

    def Rcut(self, max_exponent):
        """Return the maximum distance leading to max_exponent.

        Args:
            max_exponent (TYPE): Description

        Returns:
            TYPE: Description
        """
        return max_exponent


class CovalenceDamper(Damper):
    """Damper class for covalent bonds (exponent in set([0, HUGE])).

    Attributes:
        atom2name (TYPE): Description
        covrads (TYPE): Description
    """

    def __init__(self, atom2name, covrads=COVRADS):
        """Summary

        Args:
            atom2name (TYPE): Description
            covrads (TYPE, optional): Description
        """
        self.covrads = covrads
        self.atom2name = atom2name

    def exponent(self, AB, i, j):
        """Summary

        Args:
            AB (TYPE): Description
            i (TYPE): Description
            j (TYPE): Description

        Returns:
            TYPE: Description
        """
        cr_i = self.covrads[canonize(self.atom2name[i])]
        cr_j = self.covrads[canonize(self.atom2name[j])]
        if norm(AB) < 1.3 * (cr_i + cr_j):
            return 0.0
        else:
            return 1e10

    def Rcut(self, max_exponent):
        """Return the maximum distance leading to max_exponent.

        Args:
            max_exponent (TYPE): Description

        Returns:
            TYPE: Description
        """
        return max(self.covrads[name] for name in list(self.atom2name.values()))


class LindhExponent(Damper):
    """Class of the LINDH object which provides the exponent factors.

    Attributes:
        alphas (TYPE): Description
        atom2row (TYPE): Description
        ref_ds (TYPE): Description
    """

    def __init__(self, atom2row, alphas=alphas, ref_ds=ref_ds):
        """Summary

        Args:
            atom2row (TYPE): Description
            alphas (TYPE, optional): Description
            ref_ds (TYPE, optional): Description
        """
        self.alphas = alphas
        self.ref_ds = ref_ds
        self.atom2row = atom2row

    def exponent(self, AB, i_atom, j_atom):
        """Return the exponent for distance AB of given types.

        Args:
            AB (TYPE): Description
            i_atom (TYPE): Description
            j_atom (TYPE): Description

        Returns:
            TYPE: Description
        """
        i_row, j_row = self.atom2row[i_atom], self.atom2row[j_atom]
        alpha = self.alphas[i_row, j_row]
        ref_d = self.ref_ds[i_row, j_row]
        return alpha * (AB ** 2 - ref_d ** 2)

    def Rcut(self, max_exponent):
        """Return the maximum distance for given exponent.

        Args:
            max_exponent (TYPE): Description

        Returns:
            TYPE: Description
        """
        lr_alpha = np.min(self.alphas)
        lr_ref_d = np.max(self.ref_ds)
        # max_exponent == lr_alpha * (Rcut**2 - lr_ref_d**2)
        # max_exponent / lr_alpha + ref_d**2 == Rcut**2
        Rcut = np.sqrt(max_exponent / lr_alpha + lr_ref_d ** 2)
        return Rcut


class Bravais(object):
    """Provide tools related to some given Bravais lattice.

    May be initialized by a list of one, two, or three Bravais vectors.
    Provides tools to fold a vector or several vectors into the central
    parallel epipede (into_pe) and to retrieve a list of lattice vectors
    within a given radius (all_within).

    Attributes:
        bra (TYPE): Description
        ibra (TYPE): Description
        n (TYPE): Description
        rec (TYPE): Description
    """

    def __init__(self, lattice_vectors):
        """Initializes Bravais object.

        Args:
            lattice_vectors (TYPE): Description
        """
        if lattice_vectors is None:
            lattice_vectors = []
        else:
            lattice_vectors = list(lattice_vectors)
        self.n = len(lattice_vectors)
        if self.n > 0:
            self.bra = np.array(lattice_vectors)
            self.ibra = np.linalg.pinv(self.bra)
            self.rec = 2.0 * np.pi * np.transpose(self.ibra)
        else:
            self.bra = np.empty((0, 3))
            self.rec = np.empty((0, 3))

    def latvec(self, abc):
        """Return a lattice vector from integer Bravais indices.

        Args:
            abc (TYPE): Description

        Returns:
            TYPE: Description
        """
        vec = np.zeros(3)
        a, b, c = abc
        if a != 0:
            vec += a * self.bra[0, :]
        if b != 0:
            vec += b * self.bra[1, :]
        if c != 0:
            vec += c * self.bra[2, :]
        return vec

    def into_pe(self, vecs):
        """Fold vectors (last dimension 3) into parallel epipede.

        Examples:
            >>> lat = Bravais([[1,0,0], [0,2,0]])
            >>> np.allclose(lat.into_pe([3.2, 1.3, 0.5]), [0.2, -0.7, 0.5])
            True

        Args:
            vecs (TYPE): Description

        Returns:
            TYPE: Description

        Raises:
            ValueError: Description
        """
        vecs = np.asarray(vecs, dtype=float)
        shape = vecs.shape
        if shape[-1] != 3:
            raise ValueError("Last dim should be 3.")
        n_vec = np.product(shape[:-1])
        reslist = []
        for vec in vecs.reshape((n_vec, 3)):
            rcoeff = np.dot(vec, self.ibra)
            icoeff = np.around(rcoeff)
            res = vec - np.dot(icoeff, self.bra)
            reslist.append(res)
        return np.array(reslist).reshape(shape)

    def all_within(self, Rcut, add_base_PE=False):
        """Return a list of all lattice vector indices shorter than Rcut.

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

        Args:
            Rcut (TYPE): Description
            add_base_PE (bool, optional): Description

        Returns:
            TYPE: Description
        """
        len_rec = np.array([norm(v) for v in self.rec])
        ns = np.floor(len_rec * Rcut / (2 * np.pi))  # cross check with diss
        n_cells = np.zeros(3, int)
        n_cells[: self.n] = ns
        abcs = set()
        for a in range(-n_cells[0], n_cells[0] + 1):
            for b in range(-n_cells[1], n_cells[1] + 1):
                for c in range(-n_cells[2], n_cells[2] + 1):
                    vec = self.latvec((a, b, c))
                    if norm(vec) <= Rcut:
                        abcs.add((a, b, c))
        if add_base_PE:
            # add one in each direction
            def _around(d, n):
                """Summary

                Args:
                    d (TYPE): Description
                    n (TYPE): Description

                Returns:
                    TYPE: Description
                """
                return [-1, 0, 1] if d < n else [0]

            old_abcs = set(abcs)
            for i, j, k in old_abcs:
                for ii in _around(0, self.n):
                    for jj in _around(1, self.n):
                        for kk in _around(2, self.n):
                            abcs.add((i + ii, j + jj, k + kk))

        def _norm_of_abc(abc):
            """Summary

            Args:
                abc (TYPE): Description

            Returns:
                TYPE: Description
            """
            return norm(self.latvec(abc))

        return sorted(abcs, key=_norm_of_abc)


def get_pairs(atoms1, atoms2, Rcut, use_scipy=True):
    """Summary

    Args:
        atoms1 (TYPE): Description
        atoms2 (TYPE): Description
        Rcut (TYPE): Description
        use_scipy (bool, optional): Description

    Returns:
        TYPE: Description
    """
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

    Attributes:
        abcs (TYPE): Description
        atom (TYPE): Description
        damper (TYPE): Description
        lat (TYPE): Description
        max_sing_thres (TYPE): Description
        n_atom (TYPE): Description
        pairs (list): Description
        per_atom (TYPE): Description
    """

    def __init__(self, bra, atom, damper, max_sing_thres):
        """Initialize Pairs object.

        Returns a Pairs object containing all pairs which the damper gives
        a value smaller than max_sing_thres

        Args:
            bra (TYPE): Description
            atom (TYPE): Description
            damper (TYPE): Description
            max_sing_thres (TYPE): Description

        Raises:
            ValueError: Description
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
                if jj == i:
                    continue
                Avec = self.atom[i]
                Bvec = self.per_atom[jj]
                vec = Bvec - Avec
                j = jj % self.n_atom  # original vector
                # Can be the conversion problem, Must be integer
                a = jj // self.n_atom
                abc = self.abcs[a]
                Rvec = self.lat.latvec(abc)
                assert np.allclose(vec, Rvec + self.atom[j] - self.atom[i])
                damp = damper.exponent(norm(vec), i, j)
                proc_partners.append((j, abc, damp))
            proc_partners.sort(key=operator.itemgetter(2))  # sort by damp
            self.pairs.append(proc_partners)

    def getvec(self, iabc):
        """Summary

        Args:
            iabc (TYPE): Description

        Returns:
            TYPE: Description
        """
        i, abc = iabc
        return self.lat.latvec(abc) + self.atom[i]

    def chains(self, n, thres, directed=True):
        """Return a list of (damp, [(i, abc), ...]) tuples of n-chains.

        This is the main workhorse and returns a weight-sorted list of
        bonds (n=2), bendings (n=3), or torsions (n=3).

        Args:
            n (TYPE): Description
            thres (TYPE): Description
            directed (bool, optional): Description

        Returns:
            TYPE: Description
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
        """Get all chains of length n from atom i.

        Args:
            i (TYPE): Description
            n (TYPE): Description
            thres (TYPE): Description

        Returns:
            TYPE: Description

        Raises:
            ValueError: Description
        """
        if n < 2:
            raise ValueError("n should be at least 2.")
        res = []
        Avec = self.atom[i]
        for j, abc, damp in self.pairs[i]:
            if damp > thres:
                break  # they should be sorted
            if n == 2:
                # just pairs
                Bvec = self.lat.latvec(abc) + self.atom[j]
                tot_chain_damp = damp
                res.append((tot_chain_damp, [(i, (0, 0, 0)), (j, abc)]))
            else:
                # recursion
                rest_thres = thres - damp
                for chain_damp, atlist in self._chains_i(j, n - 1, rest_thres):
                    shifted_atlist = [(i, (0, 0, 0))]
                    for (k, kabc) in atlist:
                        kabc = tuple([ai + ak for (ai, ak) in zip(abc, kabc)])
                        if i == k and kabc == (0, 0, 0):
                            break  # self reference
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

    Attributes:
        Hdict (TYPE): Description
        n_atom (TYPE): Description
        n_dyn_periodic (TYPE): Description
        n_vec (TYPE): Description
    """

    def __init__(self, n_atom, n_dyn_periodic):
        """Summary

        Args:
            n_atom (TYPE): Description
            n_dyn_periodic (TYPE): Description
        """
        self.n_atom = n_atom
        self.n_dyn_periodic = n_dyn_periodic
        self.n_vec = n_atom + n_dyn_periodic
        self.Hdict = dict()

    def add_rank1(self, fac, vec):
        """Add rank-one term vec * vec * vec^T.

        Here, vec = {i_atom: np.array([xi, yi, zi]), ...}.

        Args:
            fac (TYPE): Description
            vec (TYPE): Description
        """
        # Make sure that we have np.ndarrays
        for i_atom in vec:
            vec[i_atom] = np.asarray(vec[i_atom])
        # Perform dyadic product
        for i_atom, ivec in vec.items():
            for j_atom, jvec in vec.items():
                blk = fac * ivec[:, np.newaxis] * jvec[np.newaxis, :]
                _acc_dict((i_atom, j_atom), self.Hdict, blk)

    def add_rank1_from_atlist(self, fac, dq_datom, atlist):
        """Add rank-one term vec * vec * vec^T.

        Here, dq_atom = [np.array([xi, yi, zi]), ...], and
        atlist = [(i_tau, (a, b, c)), ...].

        Args:
            fac (TYPE): Description
            dq_datom (TYPE): Description
            atlist (TYPE): Description
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
        """Add multiple of unity.

        Args:
            fac (TYPE): Description
        """
        blk = fac * np.identity(3)
        for i_vec in range(self.n_vec):
            _acc_dict((i_vec, i_vec), self.Hdict, blk)

    def to_array(self):
        """Construct full np.ndarray (only atomic coordinates, no stress).

        Returns:
            TYPE: Description
        """
        H = np.zeros((self.n_vec, 3, self.n_vec, 3))
        for i_atom, j_atom in self.Hdict:
            H[i_atom, :, j_atom, :] = self.Hdict[(i_atom, j_atom)]
        return H


def format_atlist(atlist, is_periodic):
    """Nicely format an atom list (atlist).

    Additionally adds 1 to atom numbers (-> start with 1):
    >>> print format_atlist([(0, (0, 0, 0)), (1, (0, -1, 0))], True)
      1( 0, 0, 0) --  2( 0,-1, 0)

    Args:
        atlist (TYPE): Description
        is_periodic (TYPE): Description

    Returns:
        TYPE: Description
    """
    if is_periodic:
        return " --".join(
            [
                "%3i(%2i,%2i,%2i)" % ((i_atom + 1,) + abc)
                for (i_atom, abc) in atlist
            ]
        )
    else:
        return " --".join(["%3i" % (i_atom + 1) for (i_atom, abc) in atlist])


def makeorthvec(orth):
    """Construct a (3 component) vector orthogonal to orth.

    >>> import numpy.random
    >>> vec = numpy.random.random(3)
    >>> assert np.dot(vec, makeorthvec(vec)) < 1e-12

    Args:
        orth (TYPE): Description

    Returns:
        TYPE: Description
    """
    orth /= norm(orth)
    vec = np.cross(orth, np.array([0.0, 0.0, 1.0]))
    if norm(vec) < 0.33:
        vec = np.cross(orth, np.array([1.0, 0.0, 0.0]))
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

    Args:
        bra (TYPE): Description
        atom (TYPE): Description
        builder (TYPE): Description
        damper (TYPE): Description
        thres (TYPE): Description
        logfile (None, optional): Description

    Returns:
        TYPE: Description
    """
    thres_fac = np.exp(-thres) * k_bond
    if logfile is not None:
        logfile.write(
            "# Neglecting anything with a prefac < %8.3g "
            "eV[/A^2]\n\n" % thres_fac
        )

    is_per = bra is not None and len(bra) > 0
    pairs = Pairs(bra, atom, damper, thres)

    # bonds:
    for damp, atlist in pairs.chains(2, thres):
        fac = np.exp(-damp) * k_bond
        if fac < thres_fac:
            continue
        vecs = [pairs.getvec(at) for at in atlist]
        q, dq = q_bond(*vecs)
        if logfile is not None:
            logfile.write(
                "# bond: %4.2f A %s  "
                "[damp: %8.3g; prefac: %8.3g eV/A^2]\n"
                % (q, format_atlist(atlist, is_per), damp, fac)
            )
        builder.add_rank1_from_atlist(fac, dq, atlist)
    if logfile is not None:
        logfile.write("\n")

    # bendings:
    for damp, atlist in pairs.chains(3, thres):
        fac = np.exp(-damp) * k_bending
        if fac < thres_fac:
            continue
        vecs = [pairs.getvec(at) for at in atlist]
        q, dq = q_bending(*vecs)
        if logfile is not None:
            logfile.write(
                "# angle: %4.0f deg %s  "
                "[damp: %8.3g; prefac: %8.3g eV]\n"
                % (np.rad2deg(q), format_atlist(atlist, is_per), damp, fac)
            )
        if 0.05 * np.pi < q < 0.95 * np.pi:
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
        fac = np.exp(-damp) * k_torsion
        if fac < thres_fac:
            continue
        vecs = [pairs.getvec(at) for at in atlist]
        try:
            q, dq = q_torsion(*vecs)
            if logfile is not None:
                logfile.write(
                    "# torsion: %4.0f deg %s  "
                    "[damp: %8.3g; prefac: %8.3g eV]\n"
                    % (np.rad2deg(q), format_atlist(atlist, is_per), damp, fac)
                )
            builder.add_rank1_from_atlist(fac, dq, atlist)
        except ValueError:
            if logfile is not None:
                logfile.write(
                    "# torsion: ---- deg %s "
                    "[damp: %8.3g; prefac: %8.3g eV]\n"
                    % (format_atlist(atlist, is_per), damp, fac)
                )
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
    """Summary

    Args:
        val_shape (TYPE): Description
        dval_di (TYPE): Description
        di_dvar (TYPE): Description

    Returns:
        TYPE: Description
    """
    val_rank = len(val_shape)
    assert val_shape == np.shape(dval_di)[:val_rank]
    di_shape = np.shape(dval_di)[val_rank:]
    di_rank = len(di_shape)
    assert di_shape == np.shape(di_dvar)[:di_rank]
    axes1 = list(range(val_rank, val_rank + di_rank))
    return np.tensordot(dval_di, di_dvar, (axes1, list(range(di_rank))))


def _dd_broadcast(val, dval):
    """Summary

    Args:
        val (TYPE): Description
        dval (TYPE): Description

    Returns:
        TYPE: Description
    """
    val_rank = len(np.shape(val))
    assert np.shape(val) == np.shape(dval)[:val_rank]
    dval_rank = len(np.shape(dval))
    newshape = np.shape(val) + (dval_rank - val_rank) * (1,)
    return np.reshape(val, newshape)


def dd_sum(*arg_ds):
    """Summary

    Args:
        *arg_ds: Description

    Returns:
        TYPE: Description
    """
    shape = np.shape(arg_ds[0][1])
    res = np.float(0.0)
    dres = np.zeros(shape)
    for arg, darg in arg_ds:
        res += np.asarray(arg)
        dres += np.asarray(darg)
    return res, dres


def dd_mult(vec_d, fac):
    """Summary

    Args:
        vec_d (TYPE): Description
        fac (TYPE): Description

    Returns:
        TYPE: Description
    """
    vec, dvec = vec_d
    return fac * vec, fac * dvec


def dd_prod(*arg_ds):
    """Summary

    Args:
        *arg_ds: Description

    Returns:
        TYPE: Description
    """
    shape = np.shape(arg_ds[0][1])
    res = np.float(1.0)
    dres = np.zeros(shape)
    for arg, darg in arg_ds:
        dres *= arg  # update previous derivs
        dres += np.asarray(darg) * res  # update with previous factors
        res *= arg  # update value
    return res, dres


def dd_power(var_d, n):
    """Summary

    Args:
        var_d (TYPE): Description
        n (TYPE): Description

    Returns:
        TYPE: Description
    """
    var, dvar = var_d
    val = var ** n
    dval = n * (var ** (n - 1)) * dvar
    return val, dval


def dd_dot(vec1_d, vec2_d):
    """Summary

    Args:
        vec1_d (TYPE): Description
        vec2_d (TYPE): Description

    Returns:
        TYPE: Description
    """
    vec1, dvec1 = vec1_d
    vec2, dvec2 = vec2_d
    res = np.dot(vec1, vec2)
    dres = np.tensordot(vec1, dvec2, (-1, 0)) + np.tensordot(
        vec2, dvec1, (-1, 0)
    )
    return res, dres


def dd_cross(vec1_d, vec2_d):
    """Summary

    Args:
        vec1_d (TYPE): Description
        vec2_d (TYPE): Description

    Returns:
        TYPE: Description
    """
    vec1, dvec1 = vec1_d
    vec2, dvec2 = vec2_d
    assert np.shape(vec1) == np.shape(vec2) == (3,)  # otherwise...
    res = np.cross(vec1, vec2)
    dres = -np.cross(vec2, dvec1, axisb=0).T + np.cross(vec1, dvec2, axisb=0).T
    return res, dres


def dd_norm(vec_d):
    """Summary

    Args:
        vec_d (TYPE): Description

    Returns:
        TYPE: Description
    """
    return dd_power(dd_dot(vec_d, vec_d), 0.5)


def dd_normalized(vec_d):
    """Summary

    Args:
        vec_d (TYPE): Description

    Returns:
        TYPE: Description
    """
    vec, dvec = vec_d
    fac, dfac = dd_power(dd_norm(vec_d), -1.0)
    res = fac * vec
    dres = fac * dvec + vec[:, np.newaxis] * dfac[np.newaxis, :]
    return res, dres


def dd_cosv1v2(vec1_d, vec2_d):
    """Summary

    Args:
        vec1_d (TYPE): Description
        vec2_d (TYPE): Description

    Returns:
        TYPE: Description
    """
    return dd_prod(
        dd_dot(vec1_d, vec2_d),
        dd_power(dd_norm(vec1_d), -1.0),
        dd_power(dd_norm(vec2_d), -1.0),
    )


def dd_arccos(val_d):
    """Summary

    Args:
        val_d (TYPE): Description

    Returns:
        TYPE: Description
    """
    val, dval = val_d
    if 1.0 < abs(val) < 1.0 + 1e-10:
        val = np.sign(val)
    res = np.arccos(val)
    vval = _dd_broadcast(val, dval)
    dres = -1.0 / np.sqrt(1.0 - vval ** 2) * dval
    return res, dres


def dd_arcsin(val_d):
    """Summary

    Args:
        val_d (TYPE): Description

    Returns:
        TYPE: Description
    """
    val, dval = val_d
    if 1.0 < abs(val) < 1.0 + 1e-10:
        val = np.sign(val)
    res = np.arcsin(val)
    vval = _dd_broadcast(val, dval)
    dres = 1.0 / np.sqrt(1.0 - vval ** 2) * dval
    return res, dres


def dd_angle(vec1_d, vec2_d):
    """Summary

    Args:
        vec1_d (TYPE): Description
        vec2_d (TYPE): Description

    Returns:
        TYPE: Description
    """
    return dd_arccos(dd_cosv1v2(vec1_d, vec2_d))


def dd_bondlength(pos1_d, pos2_d):
    """Summary

    Args:
        pos1_d (TYPE): Description
        pos2_d (TYPE): Description

    Returns:
        TYPE: Description
    """
    AB_d = dd_sum(pos2_d, dd_mult(pos1_d, -1.0))
    return dd_norm(AB_d)


def dd_bondangle(pos1_d, pos2_d, pos3_d):
    """Summary

    Args:
        pos1_d (TYPE): Description
        pos2_d (TYPE): Description
        pos3_d (TYPE): Description

    Returns:
        TYPE: Description
    """
    BA_d = dd_sum(pos2_d, dd_mult(pos1_d, -1.0))
    BC_d = dd_sum(pos2_d, dd_mult(pos3_d, -1.0))
    return dd_angle(BA_d, BC_d)


def dd_bondangle_directed(pos1_d, pos2_d, pos3_d, dir_d):
    """Summary

    Args:
        pos1_d (TYPE): Description
        pos2_d (TYPE): Description
        pos3_d (TYPE): Description
        dir_d (TYPE): Description

    Returns:
        TYPE: Description
    """
    BA_d = dd_sum(pos2_d, dd_mult(pos1_d, -1.0))
    BC_d = dd_sum(pos2_d, dd_mult(pos3_d, -1.0))
    return dd_directed_angle(BA_d, BC_d, dir_d)


def dd_arctan2(y_d, x_d):
    """Summary

    Args:
        y_d (TYPE): Description
        x_d (TYPE): Description

    Returns:
        TYPE: Description
    """
    y, dy = y_d
    x, dx = x_d
    phi = np.arctan2(y, x)
    tan, dtan = dd_prod(x_d, dd_power(y_d, -1.0))
    tan = _dd_broadcast(tan, dtan)
    dphi = (1.0 + tan ** 2) * dtan
    return phi, dphi


def dd_directed_angle(vec1_d, vec2_d, dir_d):
    """Summary

    Args:
        vec1_d (TYPE): Description
        vec2_d (TYPE): Description
        dir_d (TYPE): Description

    Returns:
        TYPE: Description
    """
    ndir_d = dd_normalized(dir_d)
    vv1_d = dd_cross(vec1_d, ndir_d)
    vv2_d = dd_cross(vec2_d, ndir_d)
    if norm(vv1_d[0]) < 1e-7 or norm(vv2_d[0]) < 1e-7:
        return 0.0, np.zeros(np.shape(vec1_d[1])[1:])
    vv1_d = dd_normalized(vv1_d)
    vv2_d = dd_normalized(vv2_d)
    cosphi_d = dd_dot(vv1_d, vv2_d)
    vvv_d = dd_cross(vv1_d, vv2_d)
    sinphi_d = dd_dot(vvv_d, ndir_d)
    # phi_d = dd_arctan2(sinphi_d, cosphi_d)
    if abs(cosphi_d[0]) < np.sqrt(0.5):
        phi, dphi = dd_arccos(cosphi_d)
        if sinphi_d[0] < 0.0:
            phi *= -1.0
            dphi *= -1.0
    else:
        phi, dphi = dd_arcsin(sinphi_d)
        if cosphi_d[0] < 0.0:
            phi = -np.pi - phi
            if phi < np.pi:
                phi += 2 * np.pi
            dphi *= -1.0
    return phi, dphi


def dd_bondtorsion(pos1_d, pos2_d, pos3_d, pos4_d):
    """Summary

    Args:
        pos1_d (TYPE): Description
        pos2_d (TYPE): Description
        pos3_d (TYPE): Description
        pos4_d (TYPE): Description

    Returns:
        TYPE: Description
    """
    BA_d = dd_sum(pos2_d, dd_mult(pos1_d, -1.0))
    BC_d = dd_sum(pos2_d, dd_mult(pos3_d, -1.0))
    CD_d = dd_sum(pos3_d, dd_mult(pos4_d, -1.0))
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

    Args:
        Avec (TYPE): Description
        Bvec (TYPE): Description

    Returns:
        TYPE: Description
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

    Args:
        Avec (TYPE): Description
        Bvec (TYPE): Description
        Cvec (TYPE): Description
        direction (None, optional): Description

    Returns:
        TYPE: Description
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

    Args:
        Avec (TYPE): Description
        Bvec (TYPE): Description
        Cvec (TYPE): Description
        Dvec (TYPE): Description

    Returns:
        TYPE: Description

    Raises:
        ValueError: Description
    """
    ABvec = Bvec - Avec
    BCvec = Cvec - Bvec
    CDvec = Dvec - Cvec
    cosABC = np.dot(ABvec, BCvec) / (norm(ABvec) * norm(BCvec))
    cosBCD = np.dot(BCvec, CDvec) / (norm(BCvec) * norm(CDvec))
    if max(abs(cosABC), abs(cosBCD)) > 0.99:  # nearly linear angle
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
    """Summary

    Args:
        q_func (TYPE): Description
        n (TYPE): Description

    Returns:
        TYPE: Description
    """
    import scipy.optimize
    import numpy.random

    x0 = np.random.standard_normal(3 * n)

    def func(x):
        """Summary

        Args:
            x (TYPE): Description

        Returns:
            TYPE: Description
        """
        vecs = np.asarray(x).reshape((n, 3))
        q, dq = q_func(*vecs)
        return q

    def grad(x):
        """Summary

        Args:
            x (TYPE): Description

        Returns:
            TYPE: Description
        """
        vecs = np.asarray(x).reshape((n, 3))
        q, dq = q_func(*vecs)
        return np.reshape(dq, (3 * n))

    return scipy.optimize.check_grad(func, grad, x0)


def testmod():
    """Summary"""
    import doctest

    doctest.testmod(raise_on_error=False)


##############################################################################
##################################### main ###################################
##############################################################################


def LindhHessian(atoms):
    """Summary

    Args:
        atoms (TYPE): Description

    Returns:
        TYPE: Description
    """
    cutoff = 15.0

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

    hessian = builder.to_array().reshape((3 * n_vec, 3 * n_vec))

    # Proper ASR
    jitter = 0.005
    # x_range = [3*ind for ind in range(len(atoms))]
    # y_range = [3*ind+1 for ind in range(len(atoms))]
    # z_range = [3*ind+2 for ind in range(len(atoms))]

    # for ind in range(len(x_range)):
    #     to_sum  = np.delete(x_range, ind)
    #     hessian[x_range[ind], x_range[ind]] = -np.sum(hessian[x_range[ind],to_sum])
    #     hessian[x_range[ind]+1, x_range[ind]] = -np.sum(hessian[x_range[ind]+1,to_sum])
    #     hessian[x_range[ind]+2, x_range[ind]] = -np.sum(hessian[x_range[ind]+2,to_sum])

    # for ind in range(len(y_range)):
    #     to_sum  = np.delete(y_range, ind)
    #     hessian[x_range[ind], x_range[ind]+1] = -np.sum(hessian[x_range[ind],to_sum])
    #     hessian[x_range[ind]+1, x_range[ind]+1] = -np.sum(hessian[x_range[ind]+1,to_sum])
    #     hessian[x_range[ind]+2, x_range[ind]+1] = -np.sum(hessian[x_range[ind]+2,to_sum])

    # for ind in range(len(y_range)):
    #     to_sum  = np.delete(z_range, ind)
    #     hessian[x_range[ind], x_range[ind]+2] = -np.sum(hessian[x_range[ind],to_sum])
    #     hessian[x_range[ind]+1, x_range[ind]+2] = -np.sum(hessian[x_range[ind]+1,to_sum])
    #     hessian[x_range[ind]+2, x_range[ind]+2] = -np.sum(hessian[x_range[ind]+2,to_sum])

    for ind in range(len(hessian)):
        hessian[ind, ind] += jitter
    return hessian


def preconditioned_hessian(
    structure, fixed_frame, parameters, atoms_current, H, task="update"
):
    """Summary

    Args:
        structure (TYPE): Description
        fixed_frame (TYPE): Description
        parameters (TYPE): Description
        atoms_current (TYPE): Description
        H (TYPE): Description
        task (str, optional): Description
    """
    if len(structure.molecules) > 1:
        a0 = structure.molecules[0].copy()
        for i in range(1, len(structure.molecules)):
            a0 += structure.molecules[i]
    else:
        a0 = structure.molecules[0]

    if hasattr(fixed_frame, "fixed_frame"):
        all_atoms = a0 + fixed_frame.fixed_frame
    else:
        all_atoms = a0
    symbol = all_atoms.get_chemical_symbols()[0]
    atoms = all_atoms.copy()
    atoms.set_positions(atoms_current.get_positions())
    set_constrains(atoms, parameters)

    ### Preconditioner part

    # Isolate indices
    hessian_indices = []
    for i in range(len(structure.molecules)):
        hessian_indices.append(
            ["mol{}".format(i) for k in range(3 * len(structure.molecules[i]))]
        )
    if hasattr(fixed_frame, "fixed_frame"):
        hessian_indices.append(
            ["fixed_frame" for k in range(3 * len(fixed_frame.fixed_frame))]
        )
    hessian_indices = sum(hessian_indices, [])

    # Genrate all nececcary hessians
    precons = {}
    precon_names = []

    precons_parameters = {
        "mol": parameters["calculator"]["preconditioner"]["mol"]["precon"],
        "fixed_frame": parameters["calculator"]["preconditioner"][
            "fixed_frame"
        ]["precon"],
        "mol-mol": parameters["calculator"]["preconditioner"]["mol-mol"][
            "precon"
        ],
        "mol-fixed_frame": parameters["calculator"]["preconditioner"][
            "mol-fixed_frame"
        ]["precon"],
    }

    routine = {
        "mol": parameters["calculator"]["preconditioner"]["mol"][task],
        "fixed_frame": parameters["calculator"]["preconditioner"][
            "fixed_frame"
        ][task],
        "mol-mol": parameters["calculator"]["preconditioner"]["mol-mol"][task],
        "mol-fixed_frame": parameters["calculator"]["preconditioner"][
            "mol-fixed_frame"
        ][task],
    }
    precon_names = [
        list(precons_parameters.values())[i]
        for i in range(len(routine))
        if list(routine.values())[i]
    ]

    # find for which atoms Lindh should be calculated
    # find for which atoms vdW should be calculated
    # find for which atoms Exp should be calculated
    # find for which atoms ID should be calculated

    # take a 'Cholesky' decomposition:
    # chol_A = np.linalg.cholesky(preconditioned_hessian)
    from numpy import linalg as la

    def nearestPD(A):
        """Find the nearest positive-definite matrix to input

        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
        credits [2].

        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6

        Args:
            A (TYPE): Description

        Returns:
            TYPE: Description
        """

        def isPD(B):
            """Returns true when input is positive-definite, via Cholesky

            Args:
                B (TYPE): Description

            Returns:
                TYPE: Description
            """
            try:
                _ = la.cholesky(B)
                return True
            except la.LinAlgError:
                return False

        B = (A + A.T) / 2
        _, s, V = la.svd(B)

        H = np.dot(V.T, np.dot(np.diag(s), V))

        A2 = (B + H) / 2

        A3 = (A2 + A2.T) / 2

        if isPD(A3):
            return A3

        spacing = np.spacing(la.norm(A))
        # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
        # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
        # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
        # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
        # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
        # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
        # `spacing` will, for Gaussian random matrixes of small dimension, be on
        # othe order of 1e-16. In practice, both ways converge, as the unit test
        # below suggests.
        I = np.eye(A.shape[0])
        k = 1
        while not isPD(A3):
            mineig = np.min(np.real(la.eigvals(A3)))
            A3 += I * (-mineig * k ** 2 + spacing)
            k += 1

        return A3

    if "Lindh" in precon_names:
        precons["Lindh"] = LindhHessian(atoms)
    if "Exp" in precon_names:
        precons["Exp"] = ExpHessian(
            atoms, mu=structure.mu, A=3.0, recalc_mu=False
        )
    if "vdW" in precon_names:
        precons["vdW"] = vdwHessian(atoms)
    if "ID" in precon_names:
        precons["ID"] = np.eye(3 * len(atoms)) * 70

    # Combine hessians into hessian
    N = len(all_atoms)
    if task == "update":
        if all(a == False for a in routine.values()):
            # Nothing to update
            return H
        else:
            preconditioned_hessian = H.copy()
            for i in range(3 * len(all_atoms)):
                for j in range(3 * len(all_atoms)):
                    if hessian_indices[i] == hessian_indices[j]:
                        if (
                            "fixed_frame" in hessian_indices[j]
                            and routine["fixed_frame"]
                        ):
                            p = precons_parameters["fixed_frame"]
                            preconditioned_hessian[i, j] = precons[p][i, j]
                        elif "mol" in hessian_indices[j] and routine["mol"]:
                            p = precons_parameters["mol"]
                            preconditioned_hessian[i, j] = precons[p][i, j]
                    else:
                        if (
                            "fixed_frame"
                            not in [hessian_indices[i], hessian_indices[j]]
                            and routine["mol-mol"]
                        ):
                            p = precons_parameters["mol-mol"]
                            preconditioned_hessian[i, j] = precons[p][i, j]
                        elif routine["mol-fixed_frame"]:
                            p = precons_parameters["mol-fixed_frame"]
                            preconditioned_hessian[i, j] = precons[p][i, j]

            if np.array_equal(
                preconditioned_hessian, np.eye(3 * len(atoms)) * 70
            ):
                return preconditioned_hessian
            else:
                # Fill the down triangle
                # preconditioned_hessian = preconditioned_hessian + preconditioned_hessian.T
                # Calculate Acoustic sum rule
                preconditioned_hessian = ASR(preconditioned_hessian)
                # Add stabilization to the diagonal
                jitter = 0.005
                preconditioned_hessian = add_jitter(
                    preconditioned_hessian, jitter
                )
                # Check if positive and symmetric:
                symmetric, positive = check_positive_symmetric(
                    preconditioned_hessian
                )

                if not positive:
                    p = preconditioned_hessian.copy()
                    preconditioned_hessian = nearestPD(preconditioned_hessian)
                    preconditioned_hessian = add_jitter(
                        preconditioned_hessian, jitter
                    )

                symmetric, positive = check_positive_symmetric(
                    preconditioned_hessian
                )

                if not symmetric:
                    print(
                        "Hessian is not symmetric! Will give troubles during optimization!"
                    )
                    sys.exit(0)
                if not positive:
                    print(
                        "Hessian is not positive definite! Will give troubles during optimization!"
                    )
                    sys.exit(0)
                if symmetric and positive:
                    print("Hessian is symmetric and positive definite")
                    return preconditioned_hessian

    if task == "initial":
        preconditioned_hessian = np.eye(3 * N) * 70
        for i in range(3 * len(all_atoms)):
            for j in range(3 * len(all_atoms)):
                if j > i:
                    if hessian_indices[i] == hessian_indices[j]:
                        if (
                            "fixed_frame" in hessian_indices[j]
                            and routine["fixed_frame"]
                        ):
                            p = precons_parameters["fixed_frame"]
                            preconditioned_hessian[i, j] = precons[p][i, j]
                        elif "mol" in hessian_indices[j] and routine["mol"]:
                            p = precons_parameters["mol"]
                            preconditioned_hessian[i, j] = precons[p][i, j]
                    else:
                        if (
                            "fixed_frame"
                            not in [hessian_indices[i], hessian_indices[j]]
                            and routine["mol-mol"]
                        ):
                            p = precons_parameters["mol-mol"]
                            preconditioned_hessian[i, j] = precons[p][i, j]
                        elif routine["mol-fixed_frame"]:
                            p = precons_parameters["mol-fixed_frame"]
                            preconditioned_hessian[i, j] = precons[p][i, j]

        if np.array_equal(preconditioned_hessian, np.eye(3 * len(atoms)) * 70):
            return preconditioned_hessian
        else:
            # Fill the down triangle
            preconditioned_hessian = (
                preconditioned_hessian + preconditioned_hessian.T
            )
            # Calculate Acoustic sum rule
            preconditioned_hessian = ASR(preconditioned_hessian)
            # Add stabilization to the diagonal
            jitter = 0.005
            preconditioned_hessian = add_jitter(preconditioned_hessian, jitter)
            # Check if positive and symmetric:
            symmetric, positive = check_positive_symmetric(
                preconditioned_hessian
            )

            p = preconditioned_hessian.copy()
            preconditioned_hessian = nearestPD(preconditioned_hessian)
            # preconditioned_hessian = add_jitter(preconditioned_hessian, jitter)

            # print(preconditioned_hessian - p)

            matplotlib.use("tkagg")
            z = preconditioned_hessian - p
            fig, ax = plt.subplots()
            im = ax.imshow(z)
            plt.colorbar(im)
            plt.show()

            if not symmetric:
                print(
                    "Hessian is not symmetric! Will give troubles during optimization!"
                )
                sys.exit(0)
            if not positive:
                print(
                    "Hessian is not positive definite! Will give troubles during optimization!"
                )
                sys.exit(0)
            if symmetric and positive:
                # print("Hessian is symmetric and positive definite")
                return preconditioned_hessian
