"""Summary
"""
from ase import neighborlist
import numpy as np
import operator


def construct_graph(connectivity_matrix):
    """Construct the graph from connectivity matrix

    Args:
        connectivity_matrix {matrix}: ASE connectivity matrix

    Returns:
        Dictionary: graph of connected atoms
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
    """Summary

    Args:
        bond (TYPE): Description
        graph (TYPE): Description
        atoms (TYPE): Description

    Returns:
        TYPE: Description
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


def set_centre_of_mass(atoms, new_com):
    """Summary

    Args:
        atoms (TYPE): Description
        new_com (TYPE): Description
    """
    old_positions = atoms.get_positions()
    old_com = atoms.get_center_of_mass()
    atoms.set_positions(old_positions - old_com + new_com)


def create_connectivity_matrix(atoms, bothways):
    """Summary

    Args:
        atoms (TYPE): Description
        bothways (TYPE): Description

    Returns:
        TYPE: Description
    """
    cutOff = neighborlist.natural_cutoffs(atoms)
    neighborList = neighborlist.NeighborList(
        cutOff, self_interaction=False, bothways=bothways
    )
    neighborList.update(atoms)
    connectivity_matrix = neighborList.get_connectivity_matrix()
    return connectivity_matrix


def detect_rotatble(connectivity_matrix, atoms):
    """Detection of all rotatable bonds
    2. The bonds does not contain terminate atoms
    2.
    3.

    Args:
        connectivity_matrix (TYPE): Description
        atoms (TYPE): Description

    Returns:
        TYPE: Description
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
    # else:
    #      for bond in conn:
    #         # Check for the index order
    #         list_of_torsions.append(create_torsion_list(bond, graph, atoms))
    return list_of_torsions


def detect_cycles(connectivity_matrix):
    """Summary

    Args:
        connectivity_matrix (TYPE): Description

    Returns:
        TYPE: Description
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
    """Summary

    Args:
        list_of_torsions (TYPE): Description
        cycles (TYPE): Description

    Returns:
        TYPE: Description
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


def make_canonical_pyranosering(atoms, cycle):
    """Summary

    Args:
        atoms (TYPE): Description
        cycle (TYPE): Description

    Returns:
        TYPE: Description
    """
    pattern = ["C", "C", "C", "C", "C", "O"]
    while True:
        cycle = np.roll(cycle, 1)
        atom_names = [atoms.get_chemical_symbols()[i] for i in cycle]
        if atom_names == pattern:
            return cycle


def getroots(aNeigh):
    """Summary

    Args:
        aNeigh (TYPE): Description

    Returns:
        TYPE: Description
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
    """Summary

    Args:
        atoms (TYPE): Description
        quaternion (TYPE): Description
        atom_1_indx (TYPE): Description
        atom_2_indx (TYPE): Description

    Returns:
        TYPE: Description
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
    """Check for internal clashes within molecule

    Iterates over the molecules and compare their
    connectivity matrices with template's one.
    Periodic boundary conditions are taken into
    account with use of the flag "bothways = True".

    Arguments:
        structure {list} -- list of the molecules

    Returns:
        bollean -- False if no clashes found
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

    # Excluding check within each molecule
    for i in range(len(structure.molecules)):
        values = (
            np.ones(len(structure.molecules[i]) ** 2).reshape(
                len(structure.molecules[i]), len(structure.molecules[i])
            )
            * 100
        )
        distances[
            len(structure.molecules[i]) * i : len(structure.molecules[i]) * i
            + len(structure.molecules[i]),
            len(structure.molecules[i]) * i : len(structure.molecules[i]) * i
            + len(structure.molecules[i]),
        ] = values

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
    rr = structure.adsorption_surface_range

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
    """Checks for clashes between molecules and fixed frame

    Claculates distances between all atoms in all molecules
    with all atoms in the fixed frame. Passed if all the
    distances are greater than specified distance in A. Periodic boundary
    conditions are taken into account with use of the
    mic=fixed_frame.mic

    Arguments:
        structure {list} -- list of the molecules
        fixed_frame {Atoms object} -- atoms in fixed frame

    Returns:
        boolean -- False if all the distances are greater than specified distance in A
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


def all_right(structure, fixed_frame):
    """Summary

    Args:
        structure (TYPE): Description
        fixed_frame (TYPE): Description

    Returns:
        TYPE: Description
    """
    ready = False

    if not internal_clashes(structure):
        if len(structure.molecules) > 1:
            if not intramolecular_clashes(structure):
                if hasattr(fixed_frame, "fixed_frame"):
                    if not clashes_with_fixed_frame(structure, fixed_frame):
                        if structure.adsorption_surface:
                            if adsorption_surface(structure, fixed_frame):
                                ready = True
                        else:
                            ready = True
                else:
                    ready = True
        else:
            if hasattr(fixed_frame, "fixed_frame"):
                if not clashes_with_fixed_frame(structure, fixed_frame):
                    if structure.adsorption:
                        if adsorption_point(structure, fixed_frame):
                            ready = True

                    else:
                        ready = True
            else:
                ready = True

    return ready


def measure_torsion_of_last(atoms, list_of_torsions):
    """Summary

    Args:
        atoms (TYPE): Description
        list_of_torsions (TYPE): Description

    Returns:
        TYPE: Description
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
