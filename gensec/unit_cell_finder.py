import numpy as np
import ase
from ase.data.vdw_alvarez import vdw_radii
from typing import Optional

'''
Purpose:

Tool used to find unit cells for a given molecule. 
'''

def find_miminum_displacement(mol1: ase.Atoms, mol2: Optional[ase.Atoms] = None, vector: Optional[np.ndarray] = np.array([1, 0, 0]), vdw_array: Optional[np.ndarray] = vdw_radii):
    
    
    """
    Finds the smallest displacement (scalar s >= 0) along the direction `vector`
    that must be applied to mol2 so that in the final configuration the distance between
    every atom in mol1 and every atom in mol2 is at least the sum of their vdW radii.
    
    The per-atom vdW radius is determined by looking up the atomic number in the
    provided vdw_radii array.
    
    Parameters:
      mol1 (Atoms): First molecule (fixed).
      mol2 (Atoms): Second molecule (to be translated).
      vector (np.ndarray): A 3-D vector indicating the translation direction.
    
    Returns:
      float: The minimal displacement along the given vector (s, measured along the unit vector)
             required so that no atom pair is overlapping.
    """
    # Normalize the displacement vector
    norm_v = np.linalg.norm(vector)
    if norm_v == 0:
        raise ValueError("Displacement vector must be non-zero.")
    u = vector / norm_v

    if mol2 is None:
        mol2 = mol1.copy()
        
    pos1 = mol1.get_positions()  # shape (N, 3)
    pos2 = mol2.get_positions()  # shape (N, 3)
    
    # Get atomic numbers and corresponding vdW radii for each molecule.
    atomic_nums1 = np.array(mol1.get_atomic_numbers(), dtype=int)
    atomic_nums2 = np.array(mol2.get_atomic_numbers(), dtype=int)
    radii1 = vdw_array[atomic_nums1]  # using atomic numbers as indices
    radii2 = vdw_array[atomic_nums2]
    
    n1 = pos1.shape[0]
    n2 = pos2.shape[0]
    
    s_req_total = 0.0
    
    for i in range(n1):
        for j in range(n2):
            # Sum of vdW radii for the two atoms
            R_ij = radii1[i] + radii2[j]
            # Difference vector between atom j in mol2 and atom i in mol1.
            d_vec = pos2[j] - pos1[i]
            D = np.dot(d_vec, d_vec)  # squared distance
            B = np.dot(d_vec, u)
            
            # For this pair, the inequality is: s^2 + 2 B s + (D - R_ij^2) >= 0.
            # Let f(s) = s^2 + 2B s + (D - R_ij**2)
            #
            # Because f(s) is quadratic (convex), its “allowed” s are in:
            #    (-∞, s_lower] ∪ [s_upper, ∞)
            # where s_lower = -B - sqrt(B^2 - (D - R_ij**2))
            #       s_upper = -B + sqrt(B^2 - (D - R_ij**2))
            #
            # However, we only care about final s >= 0.
            #
            # Furthermore, if the final configuration for a given pair is already safe at s=0,
            # we must check that if we force a displacement (due to another pair), we avoid
            # an unsafe gap.
            #
            # For each pair we determine a candidate s as follows:
            #
            #   1. If f(0) >= 0 (i.e. D >= R_ij^2), then the pair is safe at zero displacement.
            #      However, if the minimum of the distance function f_disp(s) = ||d_vec + s*u||
            #      (which occurs at s_min = max(0, -B)) is below R_ij, then any displacement s > 0
            #      must be at least s_upper.
            #
            #   2. If f(0) < 0, then we must at least move to s_upper.
            
            if D >= R_ij**2:
                # s=0 is safe in the initial configuration.
                s_min = max(0.0, -B)   # location of the minimum distance along s>=0
                # Compute the distance at that point:
                fmin = np.sqrt(s_min**2 + 2 * B * s_min + D)
                if fmin >= R_ij:
                    # The pair stays safe if we do not move at all.
                    candidate = 0.0
                else:
                    # Even though s=0 is safe, if we were forced to translate,
                    # to avoid falling into the danger zone the minimum safe s is:
                    disc = B**2 - (D - R_ij**2)
                    candidate = -B + np.sqrt(disc)
            else:
                # The pair is overlapping at s=0.
                disc = B**2 - (D - R_ij**2)
                # Physically disc should be >= 0 (if not, something is off).
                candidate = -B + np.sqrt(disc)
                if candidate < 0:
                    candidate = 0.0  # safeguard
            
            s_req_total = max(s_req_total, candidate) # Maximum working for all candidates.
            
    
    return s_req_total


def create_dimer(mol, vector=np.array([1, 0, 0]), vdw_array=vdw_radii):
    """
    Creates a dimer by translating a copy of `mol` along `vector` 
    by the minimal safe displacement.
    
    Parameters:
      mol (ase.Atoms): The original molecule.
      vector (np.ndarray): The direction for the translation (default is x-axis).
      vdw_array (np.ndarray): Array of vdW radii indexed by atomic number.
      
    Returns:
      dimer (ase.Atoms): Combined atom object with the original and translated molecule.
      s1 (float): The displacement magnitude along `vector`.
    """
    s1 = find_miminum_displacement(mol, mol, vector, vdw_array)
    mol_copy = mol.copy()
    unit_vector = vector / np.linalg.norm(vector)
    mol_copy.translate(s1 * unit_vector)
    dimer = mol + mol_copy
    return dimer, s1

def find_optimal_second_vector(mol, dimer1, s1, min_angle=np.radians(20), max_angle=np.pi/2, n_steps=36, safety_stepsize = np.radians(5), vdw_array=vdw_radii):
    """
    Uses a fixed grid search to find the optimal second translation vector.
    
    The candidate vectors lie in the upper half of the xy-plane and are defined as:
         v(θ) = [cos(θ), sin(θ), 0].
    For each candidate, the displacement s2 is computed using `find_miminum_displacement`
    so that the area
         A = s1 * s2 * sin(θ)
    is minimized.
    
    Parameters:
      dimer (ase.Atoms): Combined two-molecule object.
      s1 (float): Displacement from creating the dimer (along x-axis).
      min_angle (float): Minimum allowed angle in radians (default 15°).
      max_angle (float): Maximum allowed angle in radians (default 90°).
      n_steps (int): Number of angles to test.
      vdw_array (np.ndarray): Array of vdW radii.
      
    Returns:
      T2 (np.ndarray): Optimal translation vector (s2 multiplied by the unit-vector).
      s2 (float): Displacement magnitude along the second vector.
      theta_opt (float): Angle corresponding to T2 in radians.
      area_opt (float): Minimal area spanned by T1 and T2.
    """
    best_area = None
    best_vector = None
    best_theta = None
    best_s2 = None
    thetas = np.linspace(min_angle, max_angle, n_steps)
    dimer1_pos = dimer1.get_positions()
    atomic_nums1 = np.array(dimer1.get_atomic_numbers(), dtype=int)
    radii1 = vdw_array[atomic_nums1]
    # The second dimer will have the same atomic numbers as the first.
    radii2 = vdw_array[atomic_nums1]
    
    for theta in thetas:
        v_candidate = np.array([np.cos(theta), np.sin(theta), 0])
        s2_candidate = find_miminum_displacement(mol, dimer1, v_candidate, vdw_array)
        area_candidate = s1 * s2_candidate * np.sin(theta)
        
        dimer2_temp = dimer1.copy()
        dimer2_temp.translate(s2_candidate * v_candidate)
        dimer2_pos = dimer2_temp.get_positions()
        R_ij = np.add.outer(radii1, radii2)
        d_ij = dimer1_pos[:, np.newaxis, :] - dimer2_pos[np.newaxis, :, :]
        D_ij = np.sum(d_ij**2, axis=-1)
        
        if np.all(D_ij >= R_ij**2) and (best_area is None or area_candidate < best_area):
            best_area = area_candidate
            best_theta = theta
            best_s2 = s2_candidate
            best_vector = best_s2 * v_candidate
    
    while best_area is None and theta < np.pi + min_angle:
        theta += safety_stepsize
        v_candidate = np.array([np.cos(theta), np.sin(theta), 0])
        s2_candidate = find_miminum_displacement(mol, dimer1, v_candidate, vdw_array)
        area_candidate = s1 * s2_candidate * np.sin(theta)
        
        dimer2_temp = dimer1.copy()
        dimer2_temp.translate(s2_candidate * v_candidate)
        dimer2_pos = dimer2_temp.get_positions()
        R_ij = np.add.outer(radii1, radii2)
        d_ij = dimer1_pos[:, np.newaxis, :] - dimer2_pos[np.newaxis, :, :]
        D_ij = np.sum(d_ij**2, axis=-1)

        if np.all(D_ij >= R_ij**2) and (best_area is None or area_candidate < best_area):
            best_area = area_candidate
            best_theta = theta
            best_s2 = s2_candidate
            best_vector = best_s2 * v_candidate
    
    if best_area is None:
        raise ValueError("No valid translation vector found.")
		
    return best_vector, best_s2, best_theta, best_area


def Unit_cell_finder(mol,
                    z_cell_length=100,
                    scan_first=False,
                    adaptive=False,
                    min_angle=np.radians(20),
                    max_angle=np.pi/2,
                    # Parameters for grid method:
                    n_steps=36,
                    # Parameters for scanning first vector:
                    first_min_angle= 0,
                    first_max_angle= np.pi/2,
                    first_n_steps= 10,
                    # Parameters for adaptive method:
                    n_points=5,
                    tolerance=1e-4,
                    max_iterations=10,
                    seperation_factor=1.0,
                    parameters = None,
                    vdw_array=vdw_radii):
    """
    Constructs a periodic arrangement from a molecule using two translation steps,
    with flexible choice between a grid search or adaptive sampling for the second
    translation vector.
    
    Steps:
      1. Duplicate the molecule along the x‑axis using the minimal safe displacement 
         (T1 = s1 * [1, 0, 0]), forming a dimer.
      2. Replicate the dimer with a second translation.
         When adaptive=False, a fixed grid search is performed in the angle domain 
         [min_angle, 90°] to minimize A = s1 * s2 * sin(θ).
         When adaptive=True, an adaptive sampling procedure is used for the same purpose.
      3. The final structure is the combination of the dimer and its translated copy.
    
    Parameters:
      mol (ase.Atoms): The original molecule.
      adaptive (bool): Flag to choose adaptive sampling; if False, grid search is used.
      min_angle (float): Minimum allowed angle (in radians) for the second translation 
                         vector relative to the x‑axis (default: 15°).
      n_steps (int): Number of angles to test in grid search.
      n_points (int): Number of sample points per iteration for adaptive search.
      tolerance (float): Angle tolerance (in radians) for adaptive search termination.
      max_iterations (int): Maximum iterations for adaptive search.
      vdw_array (np.ndarray): Array of vdW radii.
      
    Returns:
      combined (ase.Atoms): The complete periodic arrangement (4 molecules total).
      T1 (np.ndarray): The first translation vector (along the x‑axis).
      T2 (np.ndarray): The second translation vector.
    """
    if parameters is not None:
        min_angle = np.radians(parameters["unit_cell_finder"]["min_angle"])
        max_angle = np.radians(parameters["unit_cell_finder"]["max_angle"])
        n_steps = parameters["unit_cell_finder"]["n_steps"]
        seperation_factor = parameters["unit_cell_finder"]["seperation_factor"]
        scan_first = parameters["unit_cell_finder"]["scan_first"]["activate"]
        adaptive = parameters["unit_cell_finder"]["adaptive"]["activate"]
        if scan_first:
            first_min_angle = np.radians(parameters["unit_cell_finder"]["scan_first"]["first_min_angle"])
            first_max_angle = np.radians(parameters["unit_cell_finder"]["scan_first"]["first_max_angle"])
            first_n_steps = parameters["unit_cell_finder"]["scan_first"]["first_n_steps"]
        if adaptive:
            n_points = parameters["unit_cell_finder"]["adaptive"]["n_points"]
            tolerance = parameters["unit_cell_finder"]["adaptive"]["tolerance"]
            max_iterations = parameters["unit_cell_finder"]["adaptive"]["max_iterations"]
    
    vdw_array = vdw_radii * seperation_factor
    
    # Computationally inefficient approach, currently not used.
    # def evaluate_first(phi1):
    #     v1 = np.array([np.cos(phi1), np.sin(phi1), 0])
    #     dimer, s1 = create_dimer(mol, vector=v1, vdw_array=vdw_array)
    #     T1 = s1 * v1
    #     # offset second-vector range by phi1
    #     sec_min = min_angle + phi1
    #     sec_max = max_angle + phi1
    #     if adaptive:
    #         T2, _, _, _ = find_optimal_second_vector_adaptive(
    #             dimer, s1,
    #             min_angle=sec_min, max_angle=sec_max,
    #             n_points=n_points, tolerance=tolerance,
    #             max_iterations=max_iterations,
    #             vdw_array=vdw_array)
    #     else:
    #         T2, _, _, _ = find_optimal_second_vector(
    #             mol, dimer, s1,
    #             min_angle=sec_min, max_angle=sec_max,
    #             n_steps=n_steps, vdw_array=vdw_array)
    #     
    #     # compute actual area of parallelogram
    #     area = np.linalg.norm(np.cross(T1, T2))
    #     return area, T1, T2
 
    # if scan_first:
        # best = None
        # for phi1 in np.linspace(first_min_angle, first_max_angle, first_n_steps):
        #     try:
        #         area, T1_candidate, T2_candidate = evaluate_first(phi1)
        #     except ValueError:
        #         continue
        #     if best is None or area < best[0]:
        #         best = (area, T1_candidate, T2_candidate)
        # if best is None:
        #     raise ValueError("No valid unit cell found when scanning first vector.")
        # 
        # _, T1, T2 = best
        
    if scan_first:
        phi_1 = np.linspace(first_min_angle, first_max_angle, first_n_steps)
        s1_array = np.ones_like(phi_1) * 100
        dimer_array = []
        for i, phi in enumerate(phi_1):
            v1 = np.array([np.cos(phi), np.sin(phi), 0])
            dimer_temp, s1_array[i] = create_dimer(mol, vector=v1, vdw_array=vdw_array)
            dimer_array.append(dimer_temp)
        arg_min_s1 = np.argmin(s1_array)
        dimer, s1, phi1 = dimer_array[arg_min_s1], s1_array[arg_min_s1], phi_1[arg_min_s1]
        T1 = s1 * np.array([np.cos(phi1), np.sin(phi1), 0])
        min_angle = min_angle + phi1
        max_angle = max_angle + phi1
        
    else:
        dimer, s1 = create_dimer(mol, vector=np.array([1, 0, 0]), vdw_array=vdw_array)
        T1 = s1 * np.array([1, 0, 0])
        
    if adaptive:
        T2, _, _, _ = find_optimal_second_vector_adaptive(
        dimer, s1,
        min_angle=min_angle,
        max_angle=max_angle,
        n_points=n_points,
        tolerance=tolerance,
        max_iterations=max_iterations,
        vdw_array=vdw_array)
            
    else:
        T2, _, _, _ = find_optimal_second_vector(
        mol, dimer, s1,
        min_angle=min_angle,
        max_angle=max_angle,
        n_steps=n_steps,
        vdw_array=vdw_array)
    
    # Combine the original dimer and its translated copy.
    mol_with_unit_cell = mol.copy()
    mol_with_unit_cell.cell[0] = T1
    mol_with_unit_cell.cell[1] = T2
    mol_with_unit_cell.cell[2] = np.array([0, 0, z_cell_length])
    mol_with_unit_cell.pbc = True
    
    return mol_with_unit_cell, T1, T2


def gen_base_sheet(mol, substrate, safety_factor=1.05, num_mol=1):
    """
    Returns a supercell of the substrate that is guaranteed to 
    fully contain the 2D projection of the molecule when it is 
    arbitrarily rotated.
    
    Parameters:
      mol           : ASE Atoms object for the molecule.
      substrate     : ASE Atoms object for the 2D substrate (e.g. graphene).
      safety_factor : Multiplier (>1) to slightly overestimate the size.
      number_molecules : Number of molecules to be placed on the substrate.
    
    Returns:
      substrate_supercell : An ASE Atoms object representing the repeated substrate.
    """
    atomic_nums = np.array(mol.get_atomic_numbers(), dtype=int)
    radii = vdw_radii[atomic_nums]
    
    pos_3d = mol.get_positions() 
    center = np.mean(pos_3d, axis=0)
    distances = np.linalg.norm(pos_3d - center, axis=1)
    R = (np.max(distances) + np.nanmax(radii)) * num_mol * safety_factor

    cell = substrate.get_cell()[:2, :2]
    a_vec = cell[0]
    b_vec = cell[1]
    L_a = np.linalg.norm(a_vec)
    L_b = np.linalg.norm(b_vec)
    
    cos_phi = np.dot(a_vec, b_vec) / (L_a * L_b)
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    phi = np.arccos(cos_phi)
    sin_phi = np.sin(phi)
    
    
    # The inscribed circle radius in the supercell spanned by (m*a) and (n*b) is:
    #   r_inscribed = 0.5 * sin(phi) * min(m * L_a, n * L_b)
    # We require r_inscribed >= R, so solve for m and n:
    m = int(np.ceil(2 * R / (L_a * sin_phi)))
    n = int(np.ceil(2 * R / (L_b * sin_phi)))
    
    substrate.set_constraint()
    substrate_supercell = substrate.repeat((m, n, 1))
    
    # Could be changed to generate a square supercell, just a bit more involved.
    
    return substrate_supercell

# Old, needs update
def find_optimal_second_vector_adaptive(dimer, s1, min_angle=np.radians(25), max_angle=np.pi/2,
                                          n_points=5, tolerance=1e-3, max_iterations=10, vdw_array=vdw_radii):
    """
    Adaptively searches for the optimal second translation vector.
    
    Candidate vectors are in the upper half of the xy-plane and are defined as:
         v(θ) = [cos(θ), sin(θ), 0].
    For each candidate angle in [min_angle, max_angle], the required displacement s2 is 
    computed using `find_miminum_displacement` and the area A = s1 * s2 * sin(θ) is then calculated.
    
    The algorithm initially samples n_points uniformly over the interval, then refines the search
    about the current best candidate until the interval width is below `tolerance` or max_iterations
    is reached.
    
    Parameters:
      dimer (ase.Atoms): The combined molecule (dimer).
      s1 (float): Displacement from creating the dimer along x-axis.
      min_angle (float): Minimum angle in radians (default 25°).
      max_angle (float): Maximum angle in radians (default 90°).
      n_points (int): Number of sample points per iteration.
      tolerance (float): Angle tolerance for refining the search (in radians).
      max_iterations (int): Maximum number of refinement iterations.
      vdw_array (np.ndarray): Array of vdW radii.
      
    Returns:
      T2_opt (np.ndarray): Optimal second translation vector.
      s2_opt (float): Displacement magnitude along T2.
      theta_opt (float): Angle corresponding to the optimal T2 (in radians).
      area_opt (float): Minimal area spanned by T1 and T2.
    """
    theta_low = min_angle
    theta_high = max_angle
    best_theta = None
    best_s2 = None
    best_area = None
    iteration = 0
    
    while (theta_high - theta_low) > tolerance and iteration < max_iterations:
        thetas = np.linspace(theta_low, theta_high, n_points)
        candidate_infos = []
        
        for theta in thetas:
            v_candidate = np.array([np.cos(theta), np.sin(theta), 0])
            s2_candidate = find_miminum_displacement(dimer, None, v_candidate, vdw_array)
            area_candidate = s1 * s2_candidate * np.sin(theta)
            candidate_infos.append((theta, s2_candidate, area_candidate))
        
        best_candidate = min(candidate_infos, key=lambda x: x[2])
        best_theta, best_s2, best_area = best_candidate
        index = np.argmin([info[2] for info in candidate_infos])
        
        if index == 0:
            theta_low = thetas[0]
            theta_high = thetas[1]
        elif index == len(thetas) - 1:
            theta_low = thetas[-2]
            theta_high = thetas[-1]
        else:
            theta_low = thetas[index - 1]
            theta_high = thetas[index + 1]
            
        iteration += 1

    v_opt = np.array([np.cos(best_theta), np.sin(best_theta), 0])
    T2_opt = best_s2 * v_opt
    
    return T2_opt, best_s2, best_theta, best_area

