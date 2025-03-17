import numpy as np
from itertools import product
from ase.io import read, write
from typing import Optional



class Supercell_finder:
    
    '''
    This class is used to run the supercell finder and offer different options to the user.
    '''
    
    def __init__(self, parameters):
        
        self.parameters = parameters
        self.set_parameters()
            
    def run(self, unit_cell_methode = None):
        '''
        Runs the supercell finder and sets the optimal supercell points for center of masses for both film and substrate.
        '''
        if unit_cell_methode is None:
            if 'unit_cell_methode' in self.parameters['supercell_finder']:
                unit_cell_methode = self.parameters['supercell_finder']['unit_cell_methode']
            else:
                unit_cell_methode = 'inputfile'
                
        self.set_unit_cell(unit_cell_methode)
        
        self.set_search_parameters()
        
        Area_S, S_tilde = get_supercells(self.max_range_S_1, self.max_range_S_2, self.S)
        Area_F, F_tilde = get_supercells(self.max_range_F_1, self.max_range_F_2, self.F)

        Area_F_sorted, Area_F_ind = reshape_and_order_with_indices(Area_F, F_tilde)
        Area_S_sorted, Area_S_ind = reshape_and_order_with_indices(Area_S, S_tilde)
        
        matches = np.zeros((Area_S_sorted.size, Area_F_sorted.size), dtype=bool)    # TODO: replace matches determination with algo described in thesis including functionality to read and compute the table
        for i in range(Area_F_sorted.size):
            if Area_F_sorted[i] < 0.1:
                continue
            area_temp = np.abs(1 - Area_S_sorted/Area_F_sorted[i])
            matches[:, i] = area_temp < self.max_area_diff

        match_indices = np.argwhere(matches)

        ind_original_F = Area_F_ind[match_indices[:, 1]]
        ind_original_S = Area_S_ind[match_indices[:, 0]]

        F_match_mat_1 = np.transpose(F_tilde[ind_original_F[:, 0], ind_original_F[:, 1], ind_original_F[:, 2], ind_original_F[:, 3], ...], axes=(0, 2, 1))
        F_match_mat_2 = np.copy(F_match_mat_1)[:, :, ::-1]
        S_match_mat = np.transpose(S_tilde[ind_original_S[:, 0], ind_original_S[:, 1], ind_original_S[:, 2], ind_original_S[:, 3], ...], axes=(0, 2, 1))
        F_match_mat_inv_1 = np.linalg.inv(F_match_mat_1)
        F_match_mat_inv_2 = np.linalg.inv(F_match_mat_2)
        T_1 = S_match_mat @ F_match_mat_inv_1
        T_2 = S_match_mat @ F_match_mat_inv_2

        # TODO: Think about what to safe (add to self) and what not. Also consider adding another way to determine the optimal supercell, maybe leaving out Sigmas (therefore Ts) with eigenvalues smaller than 1
        # As desribed in the thesis, this would leave out supercells which have any squeezing induced by the transformation
        
        V_1, _, W_1 = np.linalg.svd(T_1)
        V_2, _, W_2 = np.linalg.svd(T_2)

        U_1 = V_1 @ W_1
        U_2 = V_2 @ W_2

        F_match_rot_1 = U_1 @ F_match_mat_1
        F_match_rot_2 = U_2 @ F_match_mat_2

        lambda_1 = np.sqrt(np.linalg.norm(F_match_rot_1[:, :, 0] - S_match_mat[:, :, 0], axis=-1)**2 + np.linalg.norm(F_match_rot_1[:, :, 1] - S_match_mat[:, :, 1], axis=-1)**2)
        lambda_2 = np.sqrt(np.linalg.norm(F_match_rot_2[:, :, 0] - S_match_mat[:, :, 0], axis=-1)**2 + np.linalg.norm(F_match_rot_2[:, :, 1] - S_match_mat[:, :, 1], axis=-1)**2)

        min_1 = np.min(lambda_1)
        argmin_1 = np.argmin(lambda_1)
        min_2 = np.min(lambda_2)
        argmin_2 = np.argmin(lambda_2)

        # TODO: In the following, consider if the saved quantities have a good naming scheme or if it should be changed (_pretty and _final)
        
        if min_1 <= min_2:
            self.lambda_min = min_1
            minimum_index = argmin_1
            T_min = T_1[minimum_index, :, :]
            self.R_pretty = rotate_vectors_to_align(S_match_mat[minimum_index, :, 0], S_match_mat[minimum_index, :, 1])
            self.T_final = self.R_pretty @ T_min
            self.F_final_ind = ind_original_F[minimum_index]
            self.F_final_ind[0] += 1
            self.F_final_ind[1] -= self.max_range_F_2
            self.F_final_ind[3] -= self.max_range_F_2
            self.F_sc_points = (self.T_final @ self.F.T @ generate_supercell_points(self.F_final_ind[0:2], self.F_final_ind[2:4]).T).T
            
        else:
            self.lambda_min = min_2
            minimum_index = argmin_2
            T_min = T_2[minimum_index, :, :]
            self.R_pretty = rotate_vectors_to_align(S_match_mat[minimum_index, :, 0], S_match_mat[minimum_index, :, 1])
            self.T_final = self.R_pretty @ T_min
            self.F_final_ind = ind_original_F[minimum_index]
            self.F_final_ind[0] += 1
            self.F_final_ind[1] -= self.max_range_F_2
            self.F_final_ind[3] -= self.max_range_F_2
            self.F_sc_points = (self.T_final @ self.F.T @ generate_supercell_points(self.F_final_ind[0:2], self.F_final_ind[2:4]).T).T

        self.S_final_ind = ind_original_S[minimum_index]
        self.S_final_ind[0] += 1
        self.S_final_ind[1] -= self.max_range_S_2
        self.S_final_ind[3] -= self.max_range_S_2

        self.S_sc_points = (self.R_pretty @ self.S.T @ generate_supercell_points(self.S_final_ind[0:2], self.S_final_ind[2:4]).T).T
        
        self.F_sc_points_number = self.F_sc_points.shape[0]
        self.S_sc_points_number = self.S_sc_points.shape[0]

        self.V_final, self.Sigma_final, self.W_final = np.linalg.svd(self.T_final)

        self.U_final = self.V_final @ self.W_final
        
        self.cell = np.array([[*(self.R_pretty @ S_match_mat[minimum_index, :, 0]), 0],
                              [*(self.R_pretty @ S_match_mat[minimum_index, :, 1]), 0],
                              [0, 0, self.Z_cell_length]])
        
        self.create_atoms()
        
        
        
    def set_unit_cell(self, unit_cell_methode):
        '''
        Sets the unit cells according to the methode.
        '''
        if unit_cell_methode is 'inputfile': 
            '''
            Read the cells provided in the input files and use to find supercells
            '''
            self.S_in_file = self.parameters['fixed_frame']['filename']
            self.F_in_file = self.parameters['geometry'][0]

            self.format_S = self.parameters['fixed_frame']['format']
            self.format_F = self.parameters['geometry'][1]

            S_initial = read(self.S_in_file, format=self.format_S)
            F_initial = read(self.F_in_file, format=self.format_F)

            S_vec_1 = S_initial.cell[0]
            S_vec_2 = S_initial.cell[1]
            
            F_vec_1 = F_initial.cell[0]
            F_vec_2 = F_initial.cell[1]
            
            self.S = np.array([[S_vec_1[0], S_vec_1[1]], [S_vec_2[0], S_vec_2[1]]])
            self.F = np.array([[F_vec_1[0], F_vec_1[1]], [F_vec_2[0], F_vec_2[1]]])
            
        else:
            raise NotImplementedError('Unit cell methode not implemented')
        
    def set_parameters(self, new_parameters = None):
        '''
        Sets the parameters for the supercell finder. Defaults are set and checked by Check_input.
        '''
        if new_parameters is not None:
            self.parameters = new_parameters

        self.max_range_S_1 = self.parameters['supercell_finder']['max_range_S'][0]
        self.max_range_S_2 = self.parameters['supercell_finder']['max_range_S'][1]
        self.max_range_F_1 = self.parameters['supercell_finder']['max_range_F'][0]
        self.max_range_F_2 = self.parameters['supercell_finder']['max_range_F'][1]
            
        self.max_area_diff = self.parameters['supercell_finder']['max_area_diff']
        self.Z_cell_length = self.parameters['supercell_finder']['Z_cell_length']
        
            
    def create_atoms(self):
        '''
        Create the atoms object for substrate and film. They are not adjusted along the z-axis.
        '''
        S_geo = read(self.S_in_file, format=self.format_S)
        F_geo = read(self.F_in_file, format=self.format_F)

        for i in range(S_geo.positions.shape[0]):
            S_geo.positions[i, :2] = self.R_pretty @ S_geo.positions[i, :2]
        for i in range(F_geo.positions.shape[0]):
            F_geo.positions[i, :2] = self.U_final @ F_geo.positions[i, :2]

        for i in range(self.F_sc_points.shape[0]):
            if i == 0:
                F_geo_bunch = F_geo.copy()
                F_geo_bunch.positions[:, :2] += self.F_sc_points[i, :]
                F_geo_bunch.cell = self.cell
                F_geo_bunch.pbc = [True, True, False]
            else:
                F_geo_temp = F_geo.copy()
                F_geo_temp.positions[:, :2] += self.F_sc_points[i, :]
                F_geo_bunch += F_geo_temp

        for i in range(self.S_sc_points.shape[0]):
            if i == 0:
                S_geo_bunch = S_geo.copy()
                S_geo_bunch.positions[:, :2] += self.S_sc_points[i, :]
                S_geo_bunch.cell = self.cell
                S_geo_bunch.pbc = [True, True, False]
            else:
                S_geo_temp = S_geo.copy()
                S_geo_temp.positions[:, :2] += self.S_sc_points[i, :]
                S_geo_bunch += S_geo_temp
        
        self.F_atoms = F_geo_bunch
        self.S_atoms = S_geo_bunch
        self.joined_atoms = F_geo_bunch + S_geo_bunch
        
        
        


 
    
# Helper functions:

def reshape_and_order_with_indices(area_array, tilde_matrix, angle_array: Optional[bool] = None):
    """
    Reshape a 4D array into a 1D array, order it, and get the original indices.

    Parameters:
        array (np.ndarray): The input 4D array.

    Returns:
        tuple: A tuple containing:
            - sorted_values (np.ndarray): The sorted 1D array of values.
            - sorted_indices (np.ndarray): The indices of the sorted values in the original array.
    """
    # Flatten the array to 1D
    flat_area = area_array.ravel()
    # flat_tilde = tilde_matrix.reshape(-1, 2, 2)
    
    # Get the sorted indices of the flattened array
    sorted_indices = np.argsort(flat_area)
    
    # Sort the flattened array
    sorted_area = flat_area[sorted_indices]
    # sorted_tilde = flat_tilde[sorted_indices]
    
    # Convert flat indices back to 4D indices
    multi_indices = np.unravel_index(sorted_indices, area_array.shape)
    original_indices = np.array(multi_indices).T  # Convert to (N, 4) shape
    if angle_array is not None:
        flat_angle = angle_array.ravel()
        sorted_angles = flat_angle[sorted_indices]
        return sorted_area, original_indices, sorted_angles #, sorted_tilde
    else:
        return sorted_area, original_indices #, sorted_tilde

def generate_supercell_points(v1, v2):
    """
    Generate unique points within a supercell spanned by two integer vectors.

    Parameters:
    v1, v2 (np.ndarray): Two integer-valued vectors spanning the supercell.

    Returns:
    np.ndarray: Array of unique points within the supercell.
    """
    # Ensure v1 and v2 are numpy arrays
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)

    # Compute the bounds for a rectangle that includes the supercell
    vertices = np.array([np.zeros(2), v1, v2, v1 + v2])
    min_bounds = np.floor(vertices.min(axis=0)).astype(int)
    max_bounds = np.ceil(vertices.max(axis=0)).astype(int)

    # Create a grid of points within the bounding rectangle
    x_range = np.arange(min_bounds[0], max_bounds[0] + 1)
    y_range = np.arange(min_bounds[1], max_bounds[1] + 1)
    grid_x, grid_y = np.meshgrid(x_range, y_range, indexing="ij")
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)

    # Solve for coefficients a and b for all grid points
    A = np.array([v1, v2]).T
    try:
        inv_A = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        raise ValueError("The vectors are collinear and do not form a valid supercell.")

    coeffs = np.dot(grid_points, inv_A.T)

    # Check if points are within the supercell
    inside = np.all((coeffs >= 0) & (coeffs < 1), axis=1)

    # Extract valid points
    valid_points = grid_points[inside]

    return valid_points

def rotate_vectors_to_align(v1, v2):
    """
    Rotates two 2D vectors such that:
      - v1 aligns with the positive x-axis.
      - v2 lies in the upper-right quadrant.
      - The rotation matrix has a positive determinant.

    Parameters:
        v1 (np.array): The first vector (2D).
        v2 (np.array): The second vector (2D).

    Returns:
        np.array: The 2x2 rotation matrix.
    """
    # Normalize v1 and v2 to avoid scaling issues
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Compute the angle to rotate v1 to the positive x-axis
    angle_v1 = np.arctan2(v1[1], v1[0])

    # Create the initial rotation matrix to align v1 with x-axis
    R1 = np.array([
        [np.cos(-angle_v1), -np.sin(-angle_v1)],
        [np.sin(-angle_v1),  np.cos(-angle_v1)]
    ])

    # Rotate v2 using R1
    v2_rotated = R1 @ v2

    # Compute the angle between the rotated v2 and the positive x-axis
    angle_v2 = np.arctan2(v2[1], v2[0])

    # If v2 is not in the upper-right quadrant, apply an additional rotation
    if v2_rotated[1] < 0:
        R1 = np.array([
            [np.cos(-angle_v2), -np.sin(-angle_v2)],
            [np.sin(-angle_v2),  np.cos(-angle_v2)]
        ])

    # Verify the determinant of R1 is positive
    assert np.linalg.det(R1) > 0, "Rotation matrix determinant is not positive."

    return R1

def get_supercells(max_range_1, max_range_2, row_mat, comp_angle = False):
    """
    Generate supercells within a given range and compute their area, angle, and latice vector matrix.

    Parameters:
        max_range (int): The maximum range for generating supercells.
        row_mat (np.ndarray): The row matrix used for transformation.

    Returns:
        tuple: A tuple containing:
            - Area (np.ndarray): The area of each supercell.
            - Angle (np.ndarray): The angle between vectors in each supercell (in degrees).
            - Tilde (np.ndarray): The transformation matrix for each supercell.
    """
    m_2 = max_range_2 * 2 + 1
    Area = np.zeros((max_range_1, m_2, max_range_1 + 1, m_2))
    if comp_angle:
        Angle = np.copy(Area)
    Tilde = np.zeros((max_range_1, m_2, max_range_1 + 1, m_2, 2, 2))
    
    unit_area = np.abs(np.linalg.det(row_mat))
    
    for i in range(1, max_range_1 + 1):
        for k in range(0, max_range_1 + 1):
            for (l, j) in product(range(-max_range_2, max_range_2 + 1), repeat=2):
                # Skip where sc vectors are just swaped
                if (i * l - k * j) <= 0:
                    continue
                C_temp = np.array([[i, j], [k, l]])
                mat_temp = np.dot(C_temp, row_mat)
                Tilde[i - 1, j + max_range_2, k, l + max_range_2] = mat_temp
                Area[i - 1, j + max_range_2, k, l + max_range_2] = np.linalg.det(C_temp) * unit_area  # TODO: Avoid calculating area by using table, see thesis
                if comp_angle:
                    Angle[i - 1, j + max_range_2, k, l + max_range_2] = np.arccos(np.dot(mat_temp[0], mat_temp[1]) / np.linalg.norm(mat_temp[0]) / np.linalg.norm(mat_temp[1]))
    if comp_angle:        
        return Area, np.degrees(Angle), Tilde
    else:
        return Area, Tilde