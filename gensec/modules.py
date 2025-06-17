"""Summary
"""
from ase import neighborlist
import numpy as np
import operator

import timeout_decorator


def create_connectivity_matrix(atoms, bothways):
    """
    Creates a connectivity matrix for a given set of atoms.

    This function calculates the natural cutoffs for all atoms (vdW radii) in the molecule, 
    which are used to determine whether two atoms are bonded. 
    It then constructs a neighbor list based on these cutoffs. If the `bothways` parameter is True, 
    it considers the periodic boundary conditions, meaning each bond is counted twice (once for each direction).
    From this neighbor list, it creates a connectivity matrix where each non-zero entry indicates a bond between two atoms.

    Args:
        atoms (ase.Atoms): An ASE Atoms object representing the molecule.
        bothways (bool): If True, considers the periodic boundary conditions, each bond is counted twice (once for each direction).

    Returns:
        scipy.sparse.csr_matrix: A sparse matrix where each non-zero entry indicates a bond between two atoms.
    """

    cutOff = neighborlist.natural_cutoffs(atoms)
    neighborList = neighborlist.NeighborList(
        cutOff, self_interaction=False, bothways=bothways
    )
    neighborList.update(atoms)
    connectivity_matrix = neighborList.get_connectivity_matrix()
    return connectivity_matrix


def construct_graph(connectivity_matrix):
    """
    Constructs a graph from a given connectivity matrix.

    This function iterates over the keys of the connectivity matrix, which represent bonded atom pairs in a molecule.
    For each pair of atoms, it adds an entry to the graph dictionary where the key is the atom index and the value is a list of indices of atoms it is bonded to.
    The function does this for both atoms in each pair, ensuring that all bonds are represented in both directions.

    Args:
        connectivity_matrix (dict): A dictionary representing the connectivity matrix of the molecule. 
                                    The keys are tuples of two integers representing the indices of bonded atoms, 
                                    and the values are the bond orders.

    Returns:
        dict: A dictionary representing the graph of connected atoms. The keys are atom indices, 
              and the values are lists of indices of connected atoms.
    """

    graph = {}
    for i in connectivity_matrix.keys():
        if i[0] not in graph:
            graph[i[0]] = [i[1]]
        else:
            graph[i[0]].append(i[1])
    for i in connectivity_matrix.keys():
        if i[1] not in graph:
            graph[i[1]] = [i[0]]
        else:
            graph[i[1]].append(i[0])
    return graph


def create_torsion_list(bond, graph, atoms):
    """
    Generates a list of torsions for a given bond.

    A torsion is defined by four atoms. The first and last atoms are connected to one of the atoms in the bond but are not hydrogen. 
    and also must be connected to at least one other atom. The middle two atoms are the atoms in the bond.
    This function iterates over the atoms connected to the atoms in the bond, checks the constraints, and if they are met, 
    adds the indices of the four atoms to the torsion list.

    Args:
        bond (tuple): A tuple of two integers representing the indices of the bonded atoms.
        graph (dict): A dictionary representing the connectivity graph of the molecule. 
                      The keys are atom indices, and the values are lists of indices of connected atoms.
        atoms (ASE Atoms object): An ASE Atoms object representing the molecule.

    Returns:
        list: A list of four integers representing the indices of the atoms in the torsion. 
              If no suitable torsion can be found (i.e., the constraints are not met), returns None.
    """ 
 
    symbols = atoms.get_chemical_symbols()
    append = True
    torsions = None
    t1 = [
        i
        for i in graph[bond[0]]
        if i != bond[1] and symbols[i] != "H" and len(graph[i]) > 1
    ]
    t4 = [
        i
        for i in graph[bond[1]]
        if i != bond[0] and symbols[i] != "H" and len(graph[i]) > 1
    ]
    if len(t1) > 0 and len(t4) > 0:
        torsions = [t1[0], bond[0], bond[1], t4[0]]
    else:
        append = False

    return append, torsions


def detect_rotatble(connectivity_matrix, atoms):
    """
    Detects all rotatable bonds in a molecule.

    This function constructs a graph from the connectivity matrix, and then identifies all non-terminal atoms in the molecule.
    It then iterates over these non-terminal atoms, and for each atom that has exactly four connections, 
    it checks if three of these connections are to terminal atoms. 
    If this is the case, the bond is considered rotatable. Additionally, the function checks for cycles in the molecule. 
    Bonds that are part of a cycle are not considered rotatable.

    Args:
        connectivity_matrix (dict): A dictionary representing the connectivity matrix of the molecule. 
                                    The keys are tuples of two integers representing the indices of bonded atoms, 
                                    and the values are the bond orders.
        atoms (ase.Atoms): An ASE Atoms object representing the molecule.

    Returns:
        list: A list of tuples, where each tuple represents a rotatable bond and contains the indices of the bonded atoms.
    """
    
    graph = construct_graph(connectivity_matrix)
    indx_not_terminal = [i for i in graph if len(graph[i]) > 1]
    # Additional checks:
    for i in indx_not_terminal:
        # Removing atoms like CH3 from search
        # Finding all atoms that have exactly 4 connections
        if len(graph[i]) == 4:
            # Three of the atoms are terminal
            if [len(graph[k]) for k in graph[i]].count(1) == 3:
                indx_not_terminal.remove(i)
            else:
                pass

    for i in indx_not_terminal:
        # Removing atoms like NH2 from search
        # Finding all atoms that have exactly 3 connections
        if len(graph[i]) == 3:
            # Two of the atoms are terminal
            if [len(graph[k]) for k in graph[i]].count(1) == 2 and [
                atoms.get_chemical_symbols()[k] for k in graph[i]
            ].count("H") == 2:
                indx_not_terminal.remove(i)
            else:
                pass

    for i in indx_not_terminal:
        # Removing atoms with capping Hydrogen from search
        # Finding all atoms that have exactly 2 connections
        if len(graph[i]) == 2:
            # One Hydrogen and it is terminal
            if [len(graph[k]) for k in graph[i]].count(1) == 1 and [
                atoms.get_chemical_symbols()[k] for k in graph[i]
            ].count("H") == 1:
                indx_not_terminal.remove(i)
            else:
                pass

    conn = [
        i
        for i in connectivity_matrix.keys()
        if all(k in indx_not_terminal for k in i)
    ]
    list_of_torsions = []
    # If no cycles in the molecule
    # if not cycle_exists(graph):
    for bond in conn:
        # Check for the index order
        append, torsions = create_torsion_list(bond, graph, atoms)
        if append:
            list_of_torsions.append(torsions)

    return list_of_torsions


def detect_cycles(connectivity_matrix):
    """
    Detects all cycles in a molecule.

    This function constructs a graph from the connectivity matrix and uses the NetworkX library to find all simple cycles in the graph.
    A simple cycle is a closed path where no node appears more than once, and the minimum length of the cycle is 3.
    It then checks for overlapping cycles and adds them to the list of all cycles.

    Args:
        connectivity_matrix (dict): A dictionary representing the connectivity matrix of the molecule. 
                                    The keys are tuples of two integers representing the indices of bonded atoms, 
                                    and the values are the bond orders.

    Returns:
        list: A list of lists, where each inner list represents a cycle and contains the indices of the atoms in the cycle.
    """
    import networkx as nx
    from itertools import combinations

    all_cycles = []
    graph = construct_graph(connectivity_matrix)
    G = nx.DiGraph(graph)
    cycles = [i for i in list(nx.simple_cycles(G)) if len(i) > 2]
    comb = list(combinations(range(len(cycles)), 2))
    for i in comb:
        if set(cycles[i[0]]) & set(cycles[i[1]]) != set():
            all_cycles.append(cycles[i[0]])
    return all_cycles


def exclude_rotatable_from_cycles(list_of_torsions, cycles):
    """
    Excludes rotatable bonds that are part of a cycle from a list of torsions.

    This function iterates over the list of torsions and checks if any of the bonds in a torsion are part of a cycle.
    If a bond is part of a cycle, the torsion is removed from the list.

    Args:
        list_of_torsions (list): A list of tuples, where each tuple represents a torsion and contains the indices of the atoms in the torsion.
        cycles (list): A list of lists, where each inner list represents a cycle and contains the indices of the atoms in the cycle.

    Returns:
        list: A list of tuples, where each tuple represents a torsion and contains the indices of the atoms in the torsion. 
              Torsions that contain bonds that are part of a cycle are excluded.
    """
    rotatable = []
    for torsion in list_of_torsions:
        found = False
        for cycle in cycles:
            if torsion[1] in cycle and torsion[2] in cycle:
                found = True
                if found:
                    break
        if not found:
            rotatable.append(torsion)
    return rotatable


def set_centre_of_mass(atoms, new_com):
    """
    Sets the centre of mass of a given molecule to a specified position. 

    This function translates all atoms in the molecule so that the centre of mass is at the specified position.
    The centre of mass is calculated using the positions and masses of all atoms in the molecule. 
    The function then cawrlculates the translation vector needed to move the current centre of mass to the new position,
    and applies this translation to all atoms in the molecule.

    Args:
        atoms (ase.Atoms): An ASE Atoms object representing the molecule.
        new_com (array-like): A 3-element array-like object specifying the desired position 
                              of the centre of mass in Cartesian coordinates.

    Returns:
        None. The function modifies the atoms object in-place.
    """

    old_positions = atoms.get_positions()
    old_com = atoms.get_center_of_mass()
    atoms.set_positions(old_positions - old_com + new_com)


def make_canonical_pyranosering(atoms, cycle):
    """
    Adjusts the order of atoms in a pyranose ring to a canonical form.

    This function takes a list of atom indices representing a pyranose ring and a set of atoms.
    It then rolls the list of atom indices until the order of atom types matches the canonical form ("C", "C", "C", "C", "C", "O").
    The function continues this process until the order of atom types in the cycle matches the canonical form.

    Args:
        atoms (ase.Atoms): An ASE Atoms object representing the molecule.
        cycle (list): A list of integers representing the indices of atoms in a pyranose ring.

    Returns:
        list: A list of integers representing the indices of atoms in the canonical form of the pyranose ring.
    """

    pattern = ["C", "C", "C", "C", "C", "O"]
    while True:
        cycle = np.roll(cycle, 1)
        atom_names = [atoms.get_chemical_symbols()[i] for i in cycle]
        if atom_names == pattern:
            return cycle


def getroots(aNeigh):
    """
    Finds the root nodes in a graph.

    This function takes a dictionary representing a graph where the keys are nodes and the values are lists of connected nodes.
    It then finds the root nodes in the graph, which are nodes that are not descendants of any other nodes.

    Args:
        aNeigh (dict): A dictionary representing a graph. The keys are nodes and the values are lists of connected nodes.

    Returns:
        list: A list of nodes that are root nodes in the graph.
    """

    #    source: https://stackoverflow.com/questions/10301000/python-connected-components
    def findroot(aNode, aRoot):
        """Summary

        Args:
            aNode (TYPE): Description
            aRoot (TYPE): Description

        Returns:
            TYPE: Description
        """
        while aNode != aRoot[aNode][0]:
            aNode = aRoot[aNode][0]
        return aNode, aRoot[aNode][1]

    myRoot = {}
    for myNode in aNeigh.keys():
        myRoot[myNode] = (myNode, 0)
    for myI in aNeigh:
        for myJ in aNeigh[myI]:
            (myRoot_myI, myDepthMyI) = findroot(myI, myRoot)
            (myRoot_myJ, myDepthMyJ) = findroot(myJ, myRoot)
            if myRoot_myI != myRoot_myJ:
                myMin = myRoot_myI
                myMax = myRoot_myJ
                if myDepthMyI > myDepthMyJ:
                    myMin = myRoot_myJ
                    myMax = myRoot_myI
                myRoot[myMax] = (
                    myMax,
                    max(myRoot[myMin][1] + 1, myRoot[myMax][1]),
                )
                myRoot[myMin] = (myRoot[myMax][0], -1)
    mytoret = {}
    for myI in aNeigh:
        if myRoot[myI][0] == myI:
            mytoret[myI] = []
    for myI in aNeigh:
        mytoret[findroot(myI, myRoot)[0]].append(myI)
    return mytoret


def insertbreak(graph, atom1, atom2):
    """Insert the break in the graph

    Creates two disconnected graphs from one connected graph

    Arguments:
        graph (TYPE): Description
        atom1 (TYPE): Description
        atom2 (TYPE): Description
        graph {graph} -- Graph representation of the system
        atom1 {atom number} -- Terminate atom fro the first graph
        atom2 {atom number} -- Terminate atom for the second graph

    Returns:
        Graph -- Now the graph will not have connection between atom1 and atom2
    """

    graph[atom1].pop(graph[atom1].index(atom2))
    graph[atom2].pop(graph[atom2].index(atom1))

    return graph


def carried_atoms(connectivity_matrix_isolated, positions):
    """Returns list of carried atoms

    Args:
        connectivity_matrix_isolated (TYPE): Description
        positions (TYPE): Description

    Returns:
        TYPE: Description
    """
    graph = construct_graph(connectivity_matrix_isolated)
    graph_with_break = insertbreak(graph, positions[1], positions[2])
    if positions[2] in list(getroots(graph_with_break).values())[0]:
        return list(getroots(graph_with_break).values())[0]
    else:
        return list(getroots(graph_with_break).values())[1]


def unit_vector(vector):
    """Returns the unit vector of the vector.

    Args:
        vector (TYPE): Description

    Returns:
        TYPE: Description
    """
    if np.linalg.norm(vector) == 0.0:
        vector = np.array(
            [0.0, 0.0, 0.0000000001]
        )  #       Not to forget to check again this section
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns angle between two vectors

    Args:
        v1 (TYPE): Description
        v2 (TYPE): Description

    Returns:
        TYPE: Description
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.dot(v1_u, v2_u)) * 180.0 / np.pi


def translate(point, coord):
    """Summary

    Args:
        point (TYPE): Description
        coord (TYPE): Description

    Returns:
        TYPE: Description
    """
    translated = coord[:] - point[:]
    return translated


def translate_back(point, coord):
    """Summary

    Args:
        point (TYPE): Description
        coord (TYPE): Description

    Returns:
        TYPE: Description
    """
    translated = coord[:] + point[:]
    return translated


def mult_quats(q_1, q_2):
    """Summary

    Args:
        q_1 (TYPE): Description
        q_2 (TYPE): Description

    Returns:
        TYPE: Description
    """
    Q_q_2 = np.array(
        [
            [q_2[0], q_2[1], q_2[2], q_2[3]],
            [-q_2[1], q_2[0], -q_2[3], q_2[2]],
            [-q_2[2], q_2[3], q_2[0], -q_2[1]],
            [-q_2[3], -q_2[2], q_2[1], q_2[0]],
        ]
    )
    q_3 = np.dot(q_1, Q_q_2)
    return q_3


def unit_quaternion(q):
    """Summary

    Args:
        q (TYPE): Description

    Returns:
        TYPE: Description
    """
    ones = np.ones((1, 4))
    ones[:, 0] = np.cos(q[0] * np.pi / 180 / 2)
    vec = np.array([q[1], q[2], q[3]])
    vec = unit_vector(vec)
    ones[:, 1:] = vec * np.sin(q[0] * np.pi / 180 / 2)
    quaternion = ones[0]
    return quaternion


def rotation_quat(coord, q):
    """Summary

    Args:
        coord (TYPE): Description
        q (TYPE): Description

    Returns:
        TYPE: Description
    """
    q = unit_quaternion(q)
    R_q = np.array(
        [
            [
                1 - 2 * q[2] ** 2 - 2 * q[3] ** 2,
                2 * q[1] * q[2] - 2 * q[0] * q[3],
                2 * q[1] * q[3] + 2 * q[0] * q[2],
            ],
            [
                2 * q[2] * q[1] + 2 * q[0] * q[3],
                1 - 2 * q[3] ** 2 - 2 * q[1] ** 2,
                2 * q[2] * q[3] - 2 * q[0] * q[1],
            ],
            [
                2 * q[3] * q[1] - 2 * q[0] * q[2],
                2 * q[3] * q[2] + 2 * q[0] * q[1],
                1 - 2 * q[1] ** 2 - 2 * q[2] ** 2,
            ],
        ]
    )
    rotated = np.dot(R_q, coord.transpose())
    return rotated.transpose()


def Rotation(coord, point, quaternion):
    """Summary

    Args:
        coord (TYPE): Description
        point (TYPE): Description
        quaternion (TYPE): Description

    Returns:
        TYPE: Description
    """
    trans = translate(point, coord)
    rotate = rotation_quat(trans, quaternion)
    final = translate_back(point, rotate)
    return np.array(final)


def produce_quaternion(angle, vector):
    """Summary

    Args:
        angle (TYPE): Description
        vector (TYPE): Description

    Returns:
        TYPE: Description
    """
    ones = np.ones((1, 4))
    ones[:, 0] = angle
    ones[:, 1:] = unit_vector(vector[:])
    quaternion = ones[0]
    return quaternion


def measure_quaternion(atoms, atom_1_indx, atom_2_indx):
    # TODO: This assumes a configuration of the molecule at generation. Needs comparison to initial orientation
    """Summary

    Args:
        atoms (TYPE): Description
        atom_1_indx (TYPE): Description
        atom_2_indx (TYPE): Description

    Returns:
        TYPE: Description
    """
    # To revise and test
    coords = atoms.get_positions()
    orient_vec = unit_vector(coords[atom_2_indx] - coords[atom_1_indx])
    x_axis = np.array([1, 0, 0])
    z_axis = np.array([0, 0, 1])
    center = atoms.get_center_of_mass()
    try:
        inertia_tensor = atoms.get_moments_of_inertia(vectors=True)
    except:
        print(coords)
        inertia_tensor = atoms.get_moments_of_inertia(vectors=True)
    eigvals = inertia_tensor[0]
    eigvecs = inertia_tensor[1]
    z_index = np.argmax(eigvals)
    x_index = np.argmin(eigvals)
    if np.dot(unit_vector(eigvecs[z_index]), orient_vec) < 0:
        eigvecs[z_index] = -eigvecs[z_index]
    ang_1 = angle_between(eigvecs[z_index], z_axis)
    vec_1 = np.cross(eigvecs[z_index], z_axis)
    quat_1 = produce_quaternion(ang_1, vec_1)
    rotated_1 = Rotation(coords, center, quat_1)
    atoms.set_positions(rotated_1)
    orient_vec_2 = unit_vector(rotated_1[atom_2_indx] - rotated_1[atom_1_indx])
    eigs_after = atoms.get_moments_of_inertia(vectors=True)[1]
    if np.dot(unit_vector(eigs_after[x_index]), orient_vec_2) < 0:
        eigs_after[x_index] = -eigs_after[x_index]
    angle_x = angle_between(eigs_after[x_index], x_axis)
    if np.dot(np.cross(unit_vector(eigs_after[x_index]), x_axis), z_axis) > 0:
        angle_x = -angle_x
    quaternion_of_the_molecule = np.array(
        [angle_x, eigvecs[z_index, 0], eigvecs[z_index, 1], eigvecs[z_index, 2]]
    )
    return quaternion_of_the_molecule


def align_to_axes(atoms, atom_1_indx, atom_2_indx):
    """Summary

    Args:
        atoms (TYPE): Description
        atom_1_indx (TYPE): Description
        atom_2_indx (TYPE): Description

    Returns:
        TYPE: Description
    """
    coords = atoms.get_positions()
    center = atoms.get_center_of_mass()
    quaternion = measure_quaternion(atoms, atom_1_indx, atom_2_indx)
    vec = np.cross(quaternion[1:], np.array([0, 0, 1]))
    angle = angle_between(quaternion[1:], np.array([0, 0, 1]))
    quat_1 = produce_quaternion(angle, vec)
    rotation_1 = Rotation(coords, center, quat_1)
    angle_2 = -quaternion[0]
    quat_2 = produce_quaternion(angle_2, np.array([0, 0, 1]))
    rotation_2 = Rotation(rotation_1, center, quat_2)
    return atoms.set_positions(rotation_2)


def quaternion_set(atoms, quaternion, atom_1_indx, atom_2_indx):
    """
    Sets the positions of atoms based on a quaternion rotation.

    This function takes an ASE Atoms object, a quaternion representing a rotation, and the indices of two atoms.
    It first aligns the atoms to the axes based on the two given atom indices.
    Then it produces a quaternion for the first rotation and applies this rotation to the atoms.
    It calculates the angle and vector for the second rotation based on the remaining three components of the input quaternion,
    produces a quaternion for this rotation, and applies this rotation to the atoms.
    The function returns the atoms with their positions set based on the quaternion rotation.

    Args:
        atoms (ase.Atoms): An ASE Atoms object representing the molecule.
        quaternion (np.array): A numpy array representing a quaternion.
        atom_1_indx (int): The index of the first atom.
        atom_2_indx (int): The index of the second atom.

    Returns:
        ase.Atoms: The Atoms object with positions set based on the quaternion rotation.
    """
    coords = atoms.get_positions()
    center = atoms.get_center_of_mass()
    align_to_axes(atoms, atom_1_indx, atom_2_indx)
    first_rot = produce_quaternion(quaternion[0], np.array([0, 0, 1]))
    rotation_1 = Rotation(atoms.get_positions(), center, first_rot)
    angle_2 = angle_between(np.array([0, 0, 1]), quaternion[1:])
    vec_2 = np.cross(np.array([0, 0, 1]), quaternion[1:])
    quat_2 = produce_quaternion(angle_2, vec_2)
    rotation_2 = Rotation(rotation_1, center, quat_2)
    return atoms.set_positions(rotation_2)


def internal_clashes(structure):
    """
    Checks for internal clashes within a molecule.

    This function takes a list of molecules and checks for any internal clashes within each molecule.
    It iterates over the molecules and compares their connectivity matrices with a template's one.
    The function takes into account periodic boundary conditions with the use of the flag "bothways = True".
    If no clashes are found, the function returns False.

    Args:
        structure (list): A list of the molecules.

    Returns:
        bool: False if no clashes found.
    """
    clashes = False
    for i in range(len(structure.molecules)):
        a = sorted(
            create_connectivity_matrix(
                structure.molecules[i], bothways=True
            ).keys(),
            key=lambda element: (element[1:]),
        )
        b = sorted(
            structure.connectivity_matrix_full.keys(),
            key=lambda element: (element[1:]),
        )
        if operator.eq(set(a), set(b)):
            pass
        else:
            clashes = True

    return clashes


def intramolecular_clashes(structure):
    """Checks for intermolecular clashes

    Claculates distances between all atoms that
    belong to the different molecules. Passed if all the
    distances are greater than 1.4 A. Periodic boundary
    conditions are taken into account with use of the
    mic=structure.mic

    Arguments:
        structure (TYPE): Description
        structure {list} -- list of the molecules

    Returns:
        boolean -- False if no cllashes found
    """
    all_atoms = structure.molecules[0].copy()
    for molecule in structure.molecules[1:]:
        all_atoms.extend(molecule)

    # Distances between all the atoms with periodic boundary conditions
    if hasattr(structure, "mic"):
        distances = all_atoms.get_all_distances(mic=structure.mic).reshape(
            len(all_atoms), len(all_atoms)
        )
    else:
        distances = all_atoms.get_all_distances().reshape(
            len(all_atoms), len(all_atoms)
        )
    distances_no_pbc = all_atoms.get_all_distances().reshape(len(all_atoms), len(all_atoms))

    # Excluding check within each molecule ONLY IF THEY ARE CLOSE TO EACH OTHER WITHOUT PBC
    for i in range(len(structure.molecules)):
        # values = (
        #     np.ones(len(structure.molecules[i]) ** 2).reshape(
        #         len(structure.molecules[i]), len(structure.molecules[i])
        #     )
        #     * 100
        # )
        # TODO: Think about how we can efficiently include vdw radii here maybe.
        values = np.where(distances_no_pbc[len(structure.molecules[i]) * i : len(structure.molecules[i]) * i
            + len(structure.molecules[i]),
            len(structure.molecules[i]) * i : len(structure.molecules[i]) * i
            + len(structure.molecules[i]),
        ] < structure.clashes_intramolecular + 1e-7, 100, 0)
        
        distances[
            len(structure.molecules[i]) * i : len(structure.molecules[i]) * i
            + len(structure.molecules[i]),
            len(structure.molecules[i]) * i : len(structure.molecules[i]) * i
            + len(structure.molecules[i]),
        ] += values

    return not all(
        i >= structure.clashes_intramolecular for i in distances.flatten()
    )


def adsorption_point(structure, fixed_frame):
    from ase import Atoms
    import sys

    """Checks for distance between molecules point

    Claculates distances between all atoms in all molecules
    with selected point. Passed if at least one of the
    closest atom is within the specified range.

    Arguments:
        structure {list} -- list of the molecules

    Returns:
        boolean -- False if at lesat one of the closest molecular atom
        is within the specified range
    """
    mols = structure.molecules[0].copy()
    for molecule in structure.molecules[1:]:
        mols.extend(molecule)

    all_atoms = mols + Atoms("X", positions=[structure.adsorption_point])
    # print(all_atoms)
    # print(all_atoms.index)
    d = all_atoms.get_distances(
        a=-1, indices=range(len(all_atoms) - 1), mic=fixed_frame.mic
    )

    # closest_ind = list(d).index(min(d))
    closest_distance = min(d)

    print(closest_distance)

    return (
        structure.adsorption_range[0]
        < closest_distance
        < structure.adsorption_range[-1]
    )


def adsorption_surface(structure, fixed_frame):
    from ase import Atoms
    import sys

    """Checks for distance between molecules point

    Claculates distances between all atoms in all molecules
    with selected point. Passed if at least one of the
    closest atom is within the specified range.

    Arguments:
        structure {list} -- list of the molecules

    Returns:
        boolean -- False if at lesat one of the closest molecular atom
        is within the specified range
    """

    zz = structure.adsorption_surface_Z
    rr = structure.adsorption_range

    if structure.adsorption_surface_mols == "all":
        ready = True
        for molecule in structure.molecules:
            distances = molecule.get_positions()[:, -1] - zz
            # print(distances)
            if all(z > 0 for z in distances):
                check = [x for x in distances if rr[0] <= x <= rr[1]]
                # print(check)
                if len(check) < 1:
                    ready = False
                    return False
            else:
                return False

        return ready
    elif structure.adsorption_surface_mols == "one":
        ready = False
        for molecule in structure.molecules:
            distances = molecule.get_positions()[:, -1] - zz
            # print(distances)
            check = [x for x in distances if rr[0] <= x <= rr[1]]
            # print(check)
            if len(check) < 1:
                ready = True
                return True


def clashes_with_fixed_frame(structure, fixed_frame):
    """
    Checks for clashes between a structure and a fixed frame.

    This function takes a structure and a fixed frame, and checks for any clashes between them.
    It first combines all atoms from the structure and the fixed frame into a single list.
    It then calculates the distances between all pairs of atoms, taking into account the minimum image convention if necessary.
    The function sets the distances between atoms in the same molecule or the same fixed frame to a large value to ignore them.
    Finally, it checks if all distances are greater than the clash distance defined in the structure.
    If any distance is smaller than the clash distance, the function returns False, indicating a clash.

    Args:
        structure (GenSec structure): A structure object.
        fixed_frame (GenSec fixed frame): A fixed frame object.

    Returns:
        bool: True if there are no clashes between the structure and the fixed frame, False otherwise.
    """

    mols = structure.molecules[0].copy()
    for molecule in structure.molecules[1:]:
        mols.extend(molecule)
    all_atoms = mols + fixed_frame.fixed_frame
    distances = all_atoms.get_all_distances(mic=fixed_frame.mic).reshape(
        len(all_atoms), len(all_atoms)
    )
    values_mol = np.ones(len(mols) ** 2).reshape(len(mols), len(mols)) * 100
    distances[0 : len(mols), 0 : len(mols)] = values_mol
    values_fixed = (
        np.ones(fixed_frame.get_len() ** 2).reshape(
            fixed_frame.get_len(), fixed_frame.get_len()
        )
        * 100
    )
    distances[
        len(mols) : len(mols) + fixed_frame.get_len(),
        len(mols) : len(mols) + fixed_frame.get_len(),
    ] = values_fixed

    return not all(
        i >= structure.clashes_with_fixed_frame for i in distances.flatten()
    )

def z_min_max_clashes(structure):
    """_summary_
    Checks if all atoms of the structure are within the specified range.
    
    """
    
    if "z_min_max" in structure.parameters["configuration"]:
        z_min_max = structure.parameters["configuration"]["z_min_max"]
        mols = structure.molecules[0].copy()
        for molecule in structure.molecules[1:]:
            mols.extend(molecule)
        z_atoms = mols.get_positions()[:, 2]
        
        return np.any((z_atoms < z_min_max[0]) | (z_atoms > z_min_max[1]))
        
    else:
        return False

# TODO: Check if this is still up to date with supercell finder. For example no check for intermolecular clashes/ clashes du to pbc

def all_right(structure, fixed_frame):
    """
    Checks if a structure is ready to be added to a fixed frame.

    This function takes a structure and a fixed frame, and checks if the structure is ready to be added to the fixed frame.
    The exact conditions for a structure to be ready are not specified in the code excerpt provided.
    The function returns a boolean value indicating whether the structure is ready or not.

    Args:
        structure (TYPE): A structure object.
        fixed_frame (TYPE): A fixed frame object.

    Returns:
        bool: True if the structure is ready to be added to the fixed frame, False otherwise.
    """

    ready = False

    # TODO: Add a check for fixed frame which makes sure all atoms have the same number of neighbors
    # Numerical instability can lead to atoms missing/ too many being added. This will cause holes or overlaping
    # atoms. If the neighbours of equivalent atoms differ, it indicates one of the above (assuming mic)
    # 'inside' in function generate_supercell_points is causing to this problem
    
    if not internal_clashes(structure):
        if not z_min_max_clashes(structure):
            if len(structure.molecules) > 1:
                if not intramolecular_clashes(structure):
                    if hasattr(fixed_frame, "fixed_frame"):
                        if not clashes_with_fixed_frame(structure, fixed_frame):
                            if (
                                hasattr(structure, "adsorption_surface")
                                and structure.adsorption_surface
                            ):
                                if adsorption_surface(structure, fixed_frame):
                                    ready = True
                            else:
                                ready = True
                    else:
                        ready = True
            else:
                if hasattr(fixed_frame, "fixed_frame"):
                    if not clashes_with_fixed_frame(structure, fixed_frame):
                        if (
                            hasattr(structure, "adsorption")
                            and not structure.adsorption_surface
                        ):
                            if adsorption_point(structure, fixed_frame):
                                ready = True
                        elif (
                                hasattr(structure, "adsorption_surface")
                                and structure.adsorption_surface
                            ):
                                if adsorption_surface(structure, fixed_frame):
                                    ready = True
                        else:
                            ready = True
                else:
                    ready = True

    return ready


def measure_torsion_of_last(atoms, list_of_torsions):
    """
    Calculates the dihedral angles for a list of torsions in a molecule.

    This function takes a list of torsions and an ASE Atoms object representing a molecule.
    For each torsion in the list, it calculates the dihedral angle using the `get_dihedral` method of the Atoms object.
    The dihedral angle is defined as the angle between two planes formed by three atoms each.
    The function appends each calculated angle to a list and returns this list.

    Args:
        atoms (ase.Atoms): An ASE Atoms object representing the molecule.
        list_of_torsions (list): A list of tuples, where each tuple represents a torsion and contains the indices of the atoms in the torsion.

    Returns:
        list: A list of floats representing the dihedral angles for each torsion in the list of torsions.
    """

    torsions = []
    for torsion in list_of_torsions:
        torsions.append(
            atoms.get_dihedral(
                a0=torsion[0], a1=torsion[1], a2=torsion[2], a3=torsion[3]
            )
        )
    return torsions


def merge_together(structure, fixed_frame):
    """Merging together structure and fixed frame

    Used for convenient way for output of resulting structure.

    Args:
        structure (GenSec structure): Structure object
        fixed_frame (GenSec fixed frame): Fixed Frame object

    Returns:
        ASE Atoms: All atoms together
    """
    ensemble = structure.atoms.copy()
    del ensemble[[atom.index for atom in structure.atoms]]
    for molecule in structure.molecules:
        ensemble += molecule
    if hasattr(fixed_frame, "fixed_frame"):
        ensemble += fixed_frame.fixed_frame
    return ensemble

def run_with_timeout_decorator(func1, func2, timeout=10, *args, **kwargs):
    """
    Runs func1 with a timeout using timeout-decorator. If func1 times out,
    calls func2.
    
    :param func1: The long-running function.
    :param func2: The backup function.
    :param timeout: Time limit for func1.
    :param args: Positional arguments for func1.
    :param kwargs: Keyword arguments for func1.
    """
    # Wrap func1 with a timeout-decorator.
    timed_func1 = timeout_decorator.timeout(timeout, use_signals=True)(func1)
    try:
        result = timed_func1(*args, **kwargs)
        return result
    except timeout_decorator.TimeoutError:
        return func2(*args, **kwargs)

def return_inf():
    print("Timeout occurred")
    return np.inf