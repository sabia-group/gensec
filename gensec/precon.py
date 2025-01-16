"""Create the preconditioner based on the geometry.

Attributes:
    abohr (float): convert Bohr to Angstrom
    hartree (float): convert Hartree to eV
    k_bending (TYPE): Description
    k_bond (TYPE): Description
    k_torsion (TYPE): Description
    ref_ds (TYPE): Description
"""

import sys
from ase.optimize.precon import Exp
import numpy as np
from numpy.linalg import norm
import operator
from ase.constraints import FixAtoms
from numpy import linalg as la
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
    will be always zero. Multiple variants for setting of constrains
    are possible  for setting of the constrains.

    Args:
        atoms (ASE Atoms object): Geometry to optimize
        parameters (file): file with parameters for constrainings
    """

    z = parameters["calculator"]["constraints"]["fix_atoms"]
    c = FixAtoms(
        indices=[atom.index for atom in atoms if atom.position[2] <= z[-1]]
    )
    atoms.set_constraint(c)


def Kabsh_rmsd(atoms, initial, molindixes, removeHs=False):
    """Root-mean-square deviation (RMSD) between tructures

    Finds the optimal rotation for alignment of two structures
    with use of Kabsh algorithm and then calculates RMSD
    between all the corresponding atoms.
    Takes the atoms of structure and the reference structure by
    selecting all the atoms from inixes stored in molindixes.
    If necessary removes all the Hydrogen atoms from structure.
    Aligns centers of masses of the structures.
    Calculates the optimal rotation for aligning.
    Aligns structures and calculates deviation between positions of the
    corresponding atoms in both structures.
    Takes the square root of the sum square of those deviations.


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
    Calculates the block-diagonal elements as sum of the corresponding
    x y or z elements of the other elements in the row.

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
    creates the Identity matrix with scale of jitter and adds
    it to the input hessian.

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
units = (abohr ** 6) * hartree

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


def RAB(cell_h, cell_ih, qi, qj):
    """Calculates the vector separating two atoms.

    This file is part of i-PI.
    i-PI Copyright (C) 2014-2015 i-PI developers

    Note that minimum image convention is used, so only the image of
    atom j that is the shortest distance from atom i is considered.

    Also note that while this may not work if the simulation
    box is highly skewed from orthorhombic, as
    in this case it is possible to return a distance less than the
    nearest neighbour distance. However, this will not be of
    importance unless the cut-off radius is more than half the
    width of the shortest face-face distance of the simulation box,
    which should never be the case.
    rij is in Angstroms

    Args:
        cell_h: The simulation box cell vector matrix.
        cell_ih: The inverse of the simulation box cell vector matrix.
        qi: The position vector of atom i.
        qj: The position vectors of one or many atoms j shaped as (N, 3).
    Returns:
        dij: The vectors separating atoms i and {j}.
        rij: The distances between atoms i and {j}.
    """

    sij = np.dot(cell_ih, (qi - qj).T)
    sij -= np.rint(sij)
    dij = np.dot(cell_h, sij).T
    rij = np.linalg.norm(dij)

    if np.array_equal(cell_h, np.zeros([3, 3])):
        rij = np.array(np.linalg.norm(qi - qj))

    return rij


def C6AB(A, B):
    """Calculates C6 coefficients

    C6 coefficients calculated based on the static
    dipole polarizabilites. Tkatchenko-Scheffler scheme.
    The units are converted to eV*Angstr^6.

    Args:
        A (atom type): Chemical symbol of the atom A
        B (atom type): type): Chemical symbol of the atom B

    Returns:
        float: C6 coefficient
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
    return C6AB * (abohr ** 6) * hartree


def get_R0AB(A, B):
    """Get the average of the two vdW radi of atoms.

    Takes the vdW radii of atom A and B and calculates their
    average. The units converted to Angstroms

    Args:
        A (atom type): Chemical symbol of the atom A
        B (atom type): type): Chemical symbol of the atom B

    Returns:
        float: average vdW radii of two atoms.
    """
    return (VDW_radii[B] + VDW_radii[A]) * 0.5 * abohr


def C12AB(A, B, C6):
    """C12 coefficients between A and B

    Based on the C6 coefficients calculate the C12
    C12 coefficients are in eV*Angstr^6.
    Args:
        A (str): chemical symbol of A
        B (str): chemical symbol of B
        C6 (float): C6 coefficients between A and B

    Returns:
        float: C12 coefficient between A and B
    """

    R0AB = get_R0AB(A, B)
    C12AB = 0.5 * C6 * (R0AB ** 6)
    return C12AB


def vdW_element(k, l, C6, C12, R0, R, qi, qj):
    """VdW element

    Args:
        k (int): index of cartesian coordinates 0, 1 or 2
        l (int): index of cartesian coordinates 0, 1 or 2
        C6 (float): C6 coefficients between A and B
        C12 (float): C12 coefficients between A and B
        R0 (float): average vdW radii of atoms A and B
        R (float): distance between atoms A and B
        qi (float): Cartesian coordinates of atom A
        qj (float): Cartesian coordinates of atom B

    Returns:
        vdW element (float): vdW element of k and l
    """
    norm = (R0 / R) ** 2
    a1 = 48 * C6 * (qi[k] - qj[k]) * (qi[l] - qj[l]) * norm / (R0 ** 10)
    a2 = -168 * C12 * (qi[k] - qj[k]) * (qi[l] - qj[l]) * norm / (R0 ** 16)
    if k == l:
        a3 = -6 * C6 / (R0 ** 8)
        a4 = 12 * C12 / (R0 ** 14)
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


def rho_ij(A, B, R):
    """Calculates rho between A and B atoms

    Args:
        A (str): chemical symbol of A
        B (str): chemical symbol of B
        R (float): Distance between atoms

    Returns:
        float: rho between A and B
    """

    R0_vdW_AB = get_R0AB(A, B)
    alpha = alphas_vdW[A][B]
    return np.exp(alpha * (R0_vdW_AB ** 2 - R ** 2))


def vdwHessian(atoms):
    """vdW Preconditioning scheme

    Calculates Hessian matrix with use of vdW preconditioning scheme.

    Args:
        atoms (ASE atoms object): atoms for which vdW Hessian is calculated

    Returns:
        matrix: Hessian matrix obtained with vdW preconditioning scheme.
    """

    N = len(atoms)
    coordinates = atoms.get_positions()
    atom_names = atoms.get_chemical_symbols()
    cell_h = atoms.get_cell()[:]
    cell_ih = atoms.get_reciprocal_cell()[:]
    hessian = np.zeros(shape=(3 * N, 3 * N))
    atomsRange = list(range(N))

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
                        hessian[3 * i + k, 3 * j + l] = vdW_element(
                            k, l, C6, C12, R0AB, R, qi, qj
                        ) * rho_ij(A, B, R)

    # Fill the down triangle
    hessian = hessian + hessian.T
    # Calculate Acoustic sum rule
    hessian = ASR(hessian)
    # Add stabilization to the diagonal
    hessian = add_jitter(hessian, jitter=0.001)
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
    """Callls Exp Hessian implemented in ASE

    Args:
        atoms (ASE atoms object): atoms
        mu (int, optional): scaling coefficient mu
        A (float, optional): user specific value
        recalc_mu (bool, optional): recalculates mu if True

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
    """Adding to dictionary

    If key in dict, accumulate, otherwise set.

    Args:
        key (str): key to check in dictionary
        d (dict): dictionary
        val (value): value
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
        vec (vector): vector
        eps (float, optional): threshhold
        noself (bool, optional): Description

    Returns:
        TYPE: True if the vector is positive definite

    Raises:
        ValueError: If vector is zero
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
        atom_name (str): Symbol of the element

    Returns:
        TYPE: Canonical representation of Chemical symbol
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
    """Name to row

    Returns row number of atom type (starting with 0, max 2).

    Args:
        atom_name (str): Symbol of the element

    Returns:
        (int): Number of row of element
    """
    name = canonize(atom_name)
    if name in ("H", "He"):
        return 0
    elif name in ("Li", "Be", "B", "C", "N", "O", "F", "Ne"):
        return 1
    else:
        return 2


class LindhExponent:
    """Class of the LINDH object which provides the exponent factors.

    A Lindh Damper is an object that can judge the importance of an atom pair
    from the interatomic distance and the atom numbers.

    Attributes:
        alphas (TYPE): reference alpha values  for rows i and j used
        as the dumping coefficient in the exponent.
        atom2row (TYPE): list of numbers where for each atom
        in the structure the row in periodic table is associated.
        ref_ds (matrix): reference  distance values rij from the paper
        for more details:
        R. Lindh et al. /  Chemical Physics Letters 241 (199s) 423-428
    """

    def __init__(self, atom2row, alphas=alphas, ref_ds=ref_ds):
        """Summary

        Args:
            alphas (TYPE): reference alpha values  for rows i and j used
            as the dumping coefficient in the exponent.
            atom2row (TYPE): list of numbers where for each atom
            in the structure the row in periodic table is associated.
            ref_ds (matrix): reference  distance values rij from the paper
            for more details:
            R. Lindh et al. /  Chemical Physics Letters 241 (199s) 423-428
        """
        self.alphas = alphas
        self.ref_ds = ref_ds
        self.atom2row = atom2row

    def exponent(self, AB, i_atom, j_atom):
        """Return the exponent for distance AB of given types.

        Args:
            AB (float): Distance between A and B
            i_atom (int): Number of the atom i in the structure
            j_atom (int): Number of the atom j in the structure

        Returns:
            float: Lindh Exponent
        """
        i_row, j_row = self.atom2row[i_atom], self.atom2row[j_atom]
        alpha = self.alphas[i_row, j_row]
        ref_d = self.ref_ds[i_row, j_row]
        return alpha * (AB ** 2 - ref_d ** 2)

    def Rcut(self, max_exponent):
        """Return the maximum distance for given exponent.

        Args:
            max_exponent (float): Description

        Returns:
            float: cutoff distance
        """
        lr_alpha = np.min(self.alphas)
        lr_ref_d = np.max(self.ref_ds)
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
            lattice_vectors (array): lattice vectors of the system
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
            abc (tuple): three Bravais indices

        Returns:
            list: lattice vector
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

    def all_within(self, Rcut, add_base_PE=False):
        """Return a list of all lattice vector indices shorter than Rcut.

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
            Rcut (float): Cutoff distance for the exponent
            add_base_PE (bool, optional): if True,
            add one parallel epipede (PE) to the region.


        Returns:
            list: all lattice vector indices shorter than Rcut
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
                """Reflects if structure periodic or not

                Args:
                    d (int): lattice vector number
                    n (int): number of lattice vectors

                Returns:
                    list: [-1, 0, 1] if d<n and [0] otherwise
                """
                return [-1, 0, 1] if d < n else [0]

            old_abcs = set(abcs)
            for i, j, k in old_abcs:
                for ii in _around(0, self.n):
                    for jj in _around(1, self.n):
                        for kk in _around(2, self.n):
                            abcs.add((i + ii, j + jj, k + kk))

        def _norm_of_abc(abc):
            """Norm of lattice vectors

            Args:
                abc (matrix): lattice vectors

            Returns:
                matrix: normalized lattice vectors
            """
            return norm(self.latvec(abc))

        return sorted(abcs, key=_norm_of_abc)


def get_pairs(atoms1, atoms2, Rcut, use_scipy=True):
    """Feor eac atom get pairs within cutoff distance

    Args:
        atoms1 (array): coordinates of atoms1
        atoms2 (array): coordinates of atoms2
        Rcut (float): cutoff distance
        use_scipy (bool, optional): Use scipyif available

    Returns:
        list: lists of atoms for each atom a list that is within
        cutoff distance.
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
        >>> bendings = list(pairs.chains(3, 1.251))   # 1.251:
        one short 1->2 bond.
        >>> for damp, atlist in bendings:
        ...     assert len(atlist) == 3
        ...     print "%6.4f %s" % (damp, atlist)
        1.2502 [(0, (0, 0, 0)), (0, (-1, 0, 0)), (1, (-1, 0, 0))]
        1.2502 [(0, (0, 0, 0)), (0, (1, 0, 0)), (1, (1, 0, 0))]
        1.2502 [(0, (0, 0, 0)), (1, (0, 0, 0)), (1, (-1, 0, 0))]
        1.2502 [(0, (0, 0, 0)), (1, (0, 0, 0)), (1, (1, 0, 0))]

    Attributes:
        abcs (array of tuples): bravais lattice multipliers
        atom (array): coordinates of atoms
        damper (object): Lindh exponent
        lat (array): lattice vectors
        max_sing_thres (float): threshold for dumber
        n_atom (int): Number of atoms
        pairs (list): for each atom corresponding with
        list of atoms that are within cutoff distance
        periodicatoms (array): coordinates of atoms within lattice vectors
    """

    def __init__(self, bra, atom, damper, max_sing_thres):
        """Initialize Pairs object.

        Returns a Pairs object containing all pairs which the damper gives
        a value smaller than max_sing_thres

        Args:
            bra (array): Bravais lattice vectors
            atom (array): coordinates of atoms
            damper (object): Lindh exponent
            max_sing_thres (float): threshold for dumber

        Raises:
            ValueError: If not correct shape of atoms
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
        periodicatoms = []
        for abc in self.abcs:
            latvec = self.lat.latvec(abc)
            for atomvec in self.atom:
                periodicatoms.append(latvec + atomvec)
        self.periodicatoms = np.array(periodicatoms)
        pairs = get_pairs(self.atom, self.periodicatoms, Rcut)
        assert len(pairs) == self.n_atom

        # sort pairs
        self.pairs = []
        for i, partners in enumerate(pairs):
            proc_partners = []
            for jj in partners:
                if jj == i:
                    continue
                Avec = self.atom[i]
                Bvec = self.periodicatoms[jj]
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
            iabc (int, tuple): atom index and Bravais multipliers

        Returns:
            array: coordinate of atom multiplied by lattice vector
            and Bravais multiplier
        """
        i, abc = iabc
        return self.lat.latvec(abc) + self.atom[i]

    def chains(self, n, thres, directed=True):
        """Return a list of (damp, [(i, abc), ...]) tuples of n-chains.

        This is the main workhorse and returns a weight-sorted list of
        bonds (n=2), bendings (n=3), or torsions (n=3).

        Args:
            n (int): number of n-chain
            thres (float): threshhold, default=15.0
            directed (bool, optional): Specifies the direction
            in Bravais lattice

        Returns:
            list: list of (damp, [(i, abc), ...]) tuples of n-chains
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
            i (int): atom index
            n (int): number of n-chain
            thres (float): threshhold, default=15.0

        Returns:
            TYPE: all chains of length n from atom i.

        Raises:
            ValueError: n Should be at least 2
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
        Hdict (dict): Hessian matrix in dictionary representation.
        n_atom (int): Number of atoms
        n_dyn_periodic (int): number of periodic cells
        n_vec (int): number of atoms with periodic Bravais lattice
    """

    def __init__(self, n_atom, n_dyn_periodic):
        """Initialize values

        Args:
            n_atom (int): Number of atoms
            n_dyn_periodic (int): number of periodic cells
        """
        self.n_atom = n_atom
        self.n_dyn_periodic = n_dyn_periodic
        self.n_vec = n_atom + n_dyn_periodic
        self.Hdict = dict()

    def add_rank1(self, fac, vec):
        """Add rank-one term vec * vec * vec^T.

        Args:
            fac (float): prefactor calculated from dumping object
            that enters in the formula like fac * k_bond
            vec (dict): {i_atom: np.array([xi, yi, zi]), ...}
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

        Args:
            fac (float): prefactor calculated from dumping object
            that enters in the formula like fac * k_bond
            dq_datom (list): [np.array([xi, yi, zi]), ...]
            atlist (list): [(i_tau, (a, b, c)), ...]
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
            fac (float): prefactor calculated from dumping object
            that enters in the formula like fac * k_bond
        """
        blk = fac * np.identity(3)
        for i_vec in range(self.n_vec):
            _acc_dict((i_vec, i_vec), self.Hdict, blk)

    def to_array(self):
        """Construct full np.ndarray (only atomic coordinates, no stress).

        Returns:
            matrix: Hessian matrix
        """
        H = np.zeros((self.n_vec, 3, self.n_vec, 3))
        for i_atom, j_atom in self.Hdict:
            H[i_atom, :, j_atom, :] = self.Hdict[(i_atom, j_atom)]
        return H


def makeorthvec(orth):
    """Construct a (3 component) vector orthogonal to orth.

    Args:
        orth (vector): vector

    Returns:
        vector: Orthonormal to the input vector
    """
    orth /= norm(orth)
    vec = np.cross(orth, np.array([0.0, 0.0, 1.0]))
    if norm(vec) < 0.33:
        vec = np.cross(orth, np.array([1.0, 0.0, 0.0]))
    return vec / norm(vec)


def model_matrix(bra, atom, builder, damper, thres, logfile=None):
    """Construct model Hessian.  Returns the HessianBuilder object.

    Implementation of the Lindh Hessian preconditioner routine

    Args:
        bra (matrix): bravais lattice vectors
        atom (array): Coordinates of atoms
        builder (object): Hessian builder
        damper (object): Lindh damping function LindhExponent(atom2row=[0, 0])
        two Hydrogens in this case
        thres (float): cutoff distance for atom pairs and so on
        logfile (None, optional): Logfile for output. Disabled
        in the module version of the implementation.

    Returns:
        object: Builder object that produces Hessian matrix
    """
    thres_fac = np.exp(-thres) * k_bond
    pairs = Pairs(bra, atom, damper, thres)

    # bonds:
    for damp, atlist in pairs.chains(2, thres):
        fac = np.exp(-damp) * k_bond
        if fac < thres_fac:
            continue
        vecs = [pairs.getvec(at) for at in atlist]
        q, dq = q_bond(*vecs)
        builder.add_rank1_from_atlist(fac, dq, atlist)

    # bendings:
    for damp, atlist in pairs.chains(3, thres):
        fac = np.exp(-damp) * k_bending
        if fac < thres_fac:
            continue
        vecs = [pairs.getvec(at) for at in atlist]
        q, dq = q_bending(*vecs)
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

    # torsions
    for damp, atlist in pairs.chains(4, thres):
        fac = np.exp(-damp) * k_torsion
        if fac < thres_fac:
            continue
        vecs = [pairs.getvec(at) for at in atlist]
        try:
            q, dq = q_torsion(*vecs)
            builder.add_rank1_from_atlist(fac, dq, atlist)
        except ValueError:
            pass  # Bad luck
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


def _dd_broadcast(val, dval):
    """Reshape parameter

    Args:
        val (float or array): value
        dval (array): derivative of the value

    Returns:
        float or array: reshaped value
    """
    val_rank = len(np.shape(val))
    assert np.shape(val) == np.shape(dval)[:val_rank]
    dval_rank = len(np.shape(dval))
    newshape = np.shape(val) + (dval_rank - val_rank) * (1,)
    return np.reshape(val, newshape)


def dd_sum(*arg_ds):
    """Summation of the value and derivatives

    Args:
        arg_ds: tuple that contains all the value and derivatives

    Returns:
        (array, array): summ of the value and derivatives
    """
    shape = np.shape(arg_ds[0][1])
    res = float(0.0)
    dres = np.zeros(shape)
    for arg, darg in arg_ds:
        res += np.asarray(arg)
        dres += np.asarray(darg)
    return res, dres


def dd_mult(vec_d, fac):
    """Multiply value and derivative by the factor

    Args:
        vec_d (tuple): value and derivative
        fac (float): multiplication factor

    Returns:
        (array, array): value*factor, derivative*factor
    """
    vec, dvec = vec_d
    return fac * vec, fac * dvec


def dd_prod(*arg_ds):
    """Product of the value and derivatives

    Args:
        arg_ds: tuple that contains all the value and derivatives

    Returns:
        (array, array): product of the value and derivatives
    """
    shape = np.shape(arg_ds[0][1])
    res = float(1.0)
    dres = np.zeros(shape)
    for arg, darg in arg_ds:
        dres *= arg  # update previous derivs
        dres += np.asarray(darg) * res  # update with previous factors
        res *= arg  # update value
    return res, dres


def dd_power(var_d, n):
    """Power value and derivative to n

    Args:
        var_d (tuple): value and derivative
        n (float): power

    Returns:
        (array, array): value**n, n * (var ** (n - 1)) * dvar
    """
    var, dvar = var_d
    val = var ** n
    dval = n * (var ** (n - 1)) * dvar
    return val, dval


def dd_dot(vec1_d, vec2_d):
    """Dot product of the value and derivatives

    Args:
        vec1_d (tuple): value1 and derivative1
        vec2_d (tuple): value2 and derivative2

    Returns:
        (array, array): dot product of value1 and value2 and
        sum of the tensorproducts corresponding value and derivatives
    """
    vec1, dvec1 = vec1_d
    vec2, dvec2 = vec2_d
    res = np.dot(vec1, vec2)
    dres = np.tensordot(vec1, dvec2, (-1, 0)) + np.tensordot(
        vec2, dvec1, (-1, 0)
    )
    return res, dres


def dd_cross(vec1_d, vec2_d):
    """Cross product of the value and derivatives

    Args:
        vec1_d (tuple): value1 and derivative1
        vec2_d (tuple): value2 and derivative2

    Returns:
        (array, array): cross product of value1 and value2 and
        sum of the cross products of corresponding value and derivatives
    """
    vec1, dvec1 = vec1_d
    vec2, dvec2 = vec2_d
    assert np.shape(vec1) == np.shape(vec2) == (3,)  # otherwise...
    res = np.cross(vec1, vec2)
    dres = -np.cross(vec2, dvec1, axisb=0).T + np.cross(vec1, dvec2, axisb=0).T
    return res, dres


def dd_norm(vec_d):
    """Norm of the parameter and derivative

    Args:
        vec_d (tuple): parameter and derivative

    Returns:
        tuple: Normalized parameter and derivative
    """
    return dd_power(dd_dot(vec_d, vec_d), 0.5)


def dd_normalized(vec_d):
    """Norm of the parameter and derivative

    Args:
        vec_d (tuple): parameter and derivative

    Returns:
        tuple: value divided by it's norm and derivative  and
        sum of the products : norm of parameter*derivative and
        parameter* norm of derivative.
    """
    vec, dvec = vec_d
    fac, dfac = dd_power(dd_norm(vec_d), -1.0)
    res = fac * vec
    dres = fac * dvec + vec[:, np.newaxis] * dfac[np.newaxis, :]
    return res, dres


def dd_cosv1v2(vec1_d, vec2_d):
    """Cosine between parameter and derivatives

    Args:
        vec1_d (tuple): parameter1 and derivatives1
        vec2_d (tuple): parameter2 and derivatives2

    Returns:
        tuple: angle between parameter and derivatives
    """

    cos = dd_prod(
        dd_dot(vec1_d, vec2_d),
        dd_power(dd_norm(vec1_d), -1.0),
        dd_power(dd_norm(vec2_d), -1.0),
    )
    return cos


def dd_arccos(val_d):
    """Arccosine between parameter and derivatives

    Args:
        vec1_d (tuple): parameter1 and derivatives1
        vec2_d (tuple): parameter2 and derivatives2

    Returns:
        tuple: angle between parameter and derivatives
    """
    val, dval = val_d
    if 1.0 < abs(val) < 1.0 + 1e-10:
        val = np.sign(val)
    res = np.arccos(val)
    vval = _dd_broadcast(val, dval)
    dres = -1.0 / np.sqrt(1.0 - vval ** 2) * dval
    return res, dres


def dd_arcsin(val_d):
    """Arcsine angle between parameter and derivatives

    Args:
        vec1_d (tuple): parameter1 and derivatives1
        vec2_d (tuple): parameter2 and derivatives2

    Returns:
        tuple: Arcsine of the angle between parameter and derivatives
    """
    val, dval = val_d
    if 1.0 < abs(val) < 1.0 + 1e-10:
        val = np.sign(val)
    res = np.arcsin(val)
    vval = _dd_broadcast(val, dval)
    dres = 1.0 / np.sqrt(1.0 - vval ** 2) * dval
    return res, dres


def dd_directed_angle(vec1_d, vec2_d, dir_d):
    """Angle and derivative of the parameter

    Args:
        vec1_d (tuple): parameter1 and derivative1
        vec2_d (tuple): parameter2 and derivative2
        dir_d (tuple): (direction,
        np.c_[np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))])

    Returns:
        tuple: Angle and derivative of the parameter
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


def q_bond(Avec, Bvec):
    """Bond length and derivative with respect to vector AB.

    Test:
        >>> np.allclose(q_bond([0., 0., 0.], [1., 1., 1.])[0], np.sqrt(3.))
        True
        >>> assert _test_qgrad(q_bond, 2) < 1e-5

    Args:
        Avec (array): coordinate A
        Bvec (array): coordinate B

    Returns:
        (float, array): Bond length and derivative with respect to vector AB
    """
    Avec_d = (np.asarray(Avec), np.c_[np.identity(3), np.zeros((3, 3))])
    Bvec_d = (np.asarray(Bvec), np.c_[np.zeros((3, 3)), np.identity(3)])
    AB_d = dd_sum(Bvec_d, dd_mult(Avec_d, -1.0))
    q, dq = dd_norm(AB_d)
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
        Avec (array): coordinate A
        Bvec (array): coordinate B
        Cvec (array): coordinate C
        direction (None, optional): order of atoms A, B and C

    Returns:
        (float, array): Bond angle and derivative
        with respect to vectors Ab and BC
    """
    Avec_d = (
        np.asarray(Avec),
        np.c_[np.identity(3), np.zeros((3, 3)), np.zeros((3, 3))],
    )
    Bvec_d = (
        np.asarray(Bvec),
        np.c_[np.zeros((3, 3)), np.identity(3), np.zeros((3, 3))],
    )
    Cvec_d = (
        np.asarray(Cvec),
        np.c_[np.zeros((3, 3)), np.zeros((3, 3)), np.identity(3)],
    )
    if direction is None:
        BA_d = dd_sum(Bvec_d, dd_mult(Avec_d, -1.0))
        BC_d = dd_sum(Bvec_d, dd_mult(Cvec_d, -1.0))
        q, dq = dd_arccos(dd_cosv1v2(BA_d, BC_d))  # dd_angle
    else:
        dir_d = (
            direction,
            np.c_[np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
        )
        BA_d = dd_sum(Bvec_d, dd_mult(Avec_d, -1.0))
        BC_d = dd_sum(Bvec_d, dd_mult(Cvec_d, -1.0))
        q, dq = dd_directed_angle(BA_d, BC_d, dir_d)
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
        Avec (array): coordinate A
        Bvec (array): coordinate B
        Cvec (array): coordinate C
        Dvec (array): coordinate D

    Returns:
        (float, array): Bond torsion and derivative
        with respect to vectors AB, BC, and CD.
    """
    ABvec = Bvec - Avec
    BCvec = Cvec - Bvec
    CDvec = Dvec - Cvec
    cosABC = np.dot(ABvec, BCvec) / (norm(ABvec) * norm(BCvec))
    cosBCD = np.dot(BCvec, CDvec) / (norm(BCvec) * norm(CDvec))
    if max(abs(cosABC), abs(cosBCD)) > 0.99:  # nearly linear angle
        raise ValueError("Nearly linear angle")
    else:
        Avec_d = (
            np.asarray(Avec),
            np.c_[
                np.identity(3),
                np.zeros((3, 3)),
                np.zeros((3, 3)),
                np.zeros((3, 3)),
            ],
        )
        Bvec_d = (
            np.asarray(Bvec),
            np.c_[
                np.zeros((3, 3)),
                np.identity(3),
                np.zeros((3, 3)),
                np.zeros((3, 3)),
            ],
        )
        Cvec_d = (
            np.asarray(Cvec),
            np.c_[
                np.zeros((3, 3)),
                np.zeros((3, 3)),
                np.identity(3),
                np.zeros((3, 3)),
            ],
        )
        Dvec_d = (
            np.asarray(Dvec),
            np.c_[
                np.zeros((3, 3)),
                np.zeros((3, 3)),
                np.zeros((3, 3)),
                np.identity(3),
            ],
        )
        BA_d = dd_sum(Bvec_d, dd_mult(Avec_d, -1.0))
        BC_d = dd_sum(Bvec_d, dd_mult(Cvec_d, -1.0))
        CD_d = dd_sum(Cvec_d, dd_mult(Dvec_d, -1.0))
        q, dq = dd_directed_angle(BA_d, CD_d, BC_d)
        return q, dq.reshape(4, 3)


def LindhHessian(atoms):
    """Construction of Lindh preconditioned Hessian matrix

    Args:
        atoms (ASE atoms object): atoms of the system

    Returns:
        matrix: Lindh Hesian
    """
    cutoff = 15.0
    atom = atoms.get_positions()
    atom_name = atoms.get_chemical_symbols()
    bra = atoms.get_cell()[:]
    if np.array_equal(bra, np.zeros(9).reshape(3, 3)):
        bra = []
    n_atom = len(atom_name)
    builder = HessianBuilder(n_atom, n_dyn_periodic=0)
    damper = LindhExponent([name2row(name) for name in atom_name])
    logfile = None
    model_matrix(bra, atom, builder, damper, cutoff, logfile=logfile)
    hessian = builder.to_array().reshape((3 * n_atom, 3 * n_atom))
    hessian = add_jitter(hessian, jitter=0.005)
    return hessian


def preconditioned_hessian(
    structure, fixed_frame, parameters, atoms_current, H, task="update"
):
    """Summary

    Args:
        structure (ASE Atoms object): template structure object
        fixed_frame (ASE Atoms object): atoms in the fixed frame
        parameters (dict): parameters loaded from parameters file
        atoms_current (ASE Atoms object): atoms
        H (matrix): Current Hessian matrix
        task (str, optional): "update" or "initial"
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
    atoms = all_atoms.copy()
    atoms.set_positions(atoms_current.get_positions())
    set_constrains(atoms, parameters)

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
        k = 1
        while not isPD(A3):
            mineig = np.min(np.real(la.eigvals(A3)))
            A3 += np.eye(A.shape[0]) * (-mineig * k ** 2 + spacing)
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
        if all(a is False for a in routine.values()):
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
            preconditioned_hessian = add_jitter(
                preconditioned_hessian, jitter=0.005
            )
            # Check if positive and symmetric:
            symmetric, positive = check_positive_symmetric(
                preconditioned_hessian
            )

            if not positive:
                p = preconditioned_hessian.copy()
                preconditioned_hessian = nearestPD(preconditioned_hessian)
                preconditioned_hessian = add_jitter(
                    preconditioned_hessian, jitter=0.005
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
                # print("Hessian is symmetric and positive definite")
                return preconditioned_hessian
