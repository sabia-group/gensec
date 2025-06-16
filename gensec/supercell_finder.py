import numpy as np
from itertools import product
import ase
from ase.io import read, write
from typing import Optional
from gensec.structure import Structure, Fixed_frame
from gensec.modules import all_right

# TODO: Allow for single cell (look at shapes of S and F if only one cel)


class Supercell_finder:
    
    '''
    This class is used to run the supercell finder and offer different options to the user.
    '''
    
    def __init__(self, parameters, set_unit_cells = True):
        
        self.parameters = parameters
        self.set_parameters()
        
        if set_unit_cells:
            self.set_unit_cell(self.parameters['supercell_finder']['unit_cell_method'])
            
    def run(self):
        '''
        Runs the supercell finder and sets the optimal supercell points for center of masses for both film and substrate.
        '''
        if not hasattr(self, 'S') or not hasattr(self, 'F'):
            raise ValueError("Unit cells not set. Please set unit cells before running the supercell finder.")
        
        self.set_parameters()
        
        Area_S, S_tilde = get_supercells(self.max_range_S_1, self.max_range_S_2, self.S)
        Area_F, F_tilde = get_supercells(self.max_range_F_1, self.max_range_F_2, self.F)

        Area_F_sorted, Area_F_ind = reshape_and_order_with_indices(Area_F, F_tilde)
        Area_S_sorted, Area_S_ind = reshape_and_order_with_indices(Area_S, S_tilde)
        
        matches = np.zeros((Area_S_sorted.size, Area_F_sorted.size), dtype=bool)    # TODO: replace matches determination with algo described in thesis including functionality to read and compute the table
        for i in range(Area_F_sorted.size):
            if Area_F_sorted[i] < 1:
                continue
            area_temp = np.abs(1 - Area_S_sorted/Area_F_sorted[i])
            matches[:, i] = area_temp < self.max_area_diff

        match_indices = np.argwhere(matches)
        
        # TODO: Implement auto increase of parameters?
        if match_indices.size == 0:
            raise ValueError("No supercell found. The area difference is too small or the search range for S and F was too small.")

        ind_original_F = Area_F_ind[match_indices[:, 1]]
        ind_original_S = Area_S_ind[match_indices[:, 0]]

        F_match_mat_1 = np.transpose(F_tilde[ind_original_F[:, 0], ind_original_F[:, 1], ind_original_F[:, 2], ind_original_F[:, 3], ...], axes=(0, 2, 1))
        F_match_mat_2 = np.copy(F_match_mat_1)[:, :, ::-1]
        S_match_mat = np.transpose(S_tilde[ind_original_S[:, 0], ind_original_S[:, 1], ind_original_S[:, 2], ind_original_S[:, 3], ...], axes=(0, 2, 1))
        F_match_mat_inv_1 = inv2(F_match_mat_1)
        F_match_mat_inv_2 = inv2(F_match_mat_2)
        T_1 = S_match_mat @ F_match_mat_inv_1
        T_2 = S_match_mat @ F_match_mat_inv_2

        # TODO: Think about what to safe (add to self) and what not. Also consider adding another way to determine the optimal supercell, maybe leaving out Sigmas (therefore Ts) with eigenvalues smaller than 1
        # As desribed in the thesis, this would leave out supercells which have any squeezing induced by the transformation
        
        V_1, Sigma_1, W_1 = svd2(T_1)
        V_2, Sigma_2, W_2 = svd2(T_2)

        U_1 = V_1 @ W_1
        U_2 = V_2 @ W_2

        F_match_rot_1 = U_1 @ F_match_mat_1
        F_match_rot_2 = U_2 @ F_match_mat_2

        lambda_1 = np.sqrt(np.linalg.norm(F_match_rot_1[:, :, 0] - S_match_mat[:, :, 0], axis=-1)**2 + np.linalg.norm(F_match_rot_1[:, :, 1] - S_match_mat[:, :, 1], axis=-1)**2)
        lambda_2 = np.sqrt(np.linalg.norm(F_match_rot_2[:, :, 0] - S_match_mat[:, :, 0], axis=-1)**2 + np.linalg.norm(F_match_rot_2[:, :, 1] - S_match_mat[:, :, 1], axis=-1)**2)

        sorted_ind_lambda_1 = np.argsort(lambda_1)
        sorted_ind_lambda_2 = np.argsort(lambda_2)
        self.works = False
        self.attempts = 0
        
        while not self.works and (self.attempts < self.parameters['supercell_finder']['max_attempts'] and self.attempts < lambda_1.size):
            argmin_1 = sorted_ind_lambda_1[self.attempts]
            min_1 = lambda_1[argmin_1]
            argmin_2 = sorted_ind_lambda_2[self.attempts]
            min_2 = lambda_2[argmin_2]
            
            
            
            # TODO: Not working yet:
            # # The following makes sure that the supercell returned only contains stretches,
            # if (self.attempts == self.parameters['supercell_finder']['max_attempts'] - 1) and (self.parameters['supercell_finder']['max_attempts'] > 1):
            #     lambda_1 = lambda_1 * (1 + (~(Sigma_1 >= 1).all(axis = 1)) * 1e5)
            #     lambda_2 = lambda_2 * (1 + (~(Sigma_2 >= 1).all(axis = 1)) * 1e5)
            #     
            #     min_1 = np.min(lambda_1)
            #     argmin_1 = np.argmin(lambda_1)
            #     min_2 = np.min(lambda_2)
            #     argmin_2 = np.argmin(lambda_2)
            
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

            self.V_final, self.Sigma_final, self.W_final = svd2(self.T_final)

            self.U_final = self.V_final @ self.W_final
            
            self.cell = np.array([[*(self.R_pretty @ S_match_mat[minimum_index, :, 0]), 0],
                                [*(self.R_pretty @ S_match_mat[minimum_index, :, 1]), 0],
                                [0, 0, self.Z_cell_length]])
            
            self.create_atoms()
            fixed_frame_temp = Fixed_frame(self.parameters, self.S_atoms.copy())
            structure_temp = Structure(self.parameters, self)
            
            if "max_atoms" in self.parameters["supercell_finder"] and (self.attempts < self.parameters['supercell_finder']['max_attempts'] - 1) and (self.parameters['supercell_finder']['max_attempts'] > 1):
                self.works = all_right(structure_temp, fixed_frame_temp) and (len(self.joined_atoms) <= self.parameters["supercell_finder"]["max_atoms"])
            else:
                self.works = all_right(structure_temp, fixed_frame_temp)
            self.attempts += 1
        
        
    def set_unit_cell(self, unit_cell_method, provided_atoms: Optional[ase.Atoms] = None):
        '''
        Sets the unit cells according to the method. Not designed for cases where the unit cell of the substrate is not provided in the input file.
        '''
        
        self.S_in_file = self.parameters['fixed_frame']['filename']
        self.format_S = self.parameters['fixed_frame']['format']
        S_initial = read(self.S_in_file, format=self.format_S)
        S_vec_1 = S_initial.cell[0]
        S_vec_2 = S_initial.cell[1]
        self.S = np.array([[S_vec_1[0], S_vec_1[1]], [S_vec_2[0], S_vec_2[1]]])
        self.S_geo = S_initial.copy()
        
        
        if unit_cell_method == 'inputfile':     # Get the unit cell from the input file
            
            self.F_in_file = self.parameters['geometry']['filename']
            self.format_F = self.parameters['geometry']['format']
            F_initial = read(self.F_in_file, format=self.format_F)


        elif unit_cell_method == 'find':    # Get the unit cell from the provided ase.Atoms object (unit cell is detected usning unit cell finder beforehand)
            
            F_initial = provided_atoms
            
        else:
            raise NotImplementedError('Unit cell method not implemented')

        F_vec_1 = F_initial.cell[0]
        F_vec_2 = F_initial.cell[1]
            
        self.F = np.array([[F_vec_1[0], F_vec_1[1]], [F_vec_2[0], F_vec_2[1]]])
        self.F_geo = F_initial.copy()
        
        self.S_geo.positions[:, 2] = self.S_geo.positions[:, 2] - np.max(self.S_geo.positions[:, 2])
        if self.parameters['supercell_finder']['m_range']['type'] == 'given_range':
            self.max_range_S_1 = self.parameters['supercell_finder']['m_range']['max_range_s'][0]
            self.max_range_S_2 = self.parameters['supercell_finder']['m_range']['max_range_s'][1]
            self.max_range_F_1 = self.parameters['supercell_finder']['m_range']['max_range_f'][0]
            self.max_range_F_2 = self.parameters['supercell_finder']['m_range']['max_range_f'][1]
        elif self.parameters['supercell_finder']['m_range']['type'] == 'max':
            
            stacked = np.vstack((self.S, self.F))
            vec_len = np.linalg.norm(stacked, axis=1)
            min_arg = np.argmin(vec_len)
            
            max_s = self.parameters['supercell_finder']['m_range']['max_s']
            max_f = self.parameters['supercell_finder']['m_range']['max_f']
            if min_arg == 0:
                self.max_range_S_1 = max_s
                self.max_range_S_2 = max([int(np.round(self.max_range_S_1 * vec_len[0] / vec_len[1])), 1])
                self.max_range_F_1 = max(min(int(np.round(self.max_range_S_1 * vec_len[0] / vec_len[2])), max_f), 1)
                self.max_range_F_2 = max(min(int(np.round(self.max_range_S_1 * vec_len[0] / vec_len[3])), max_f), 1)
            elif min_arg == 1:
                self.max_range_S_2 = max_s
                self.max_range_S_1 = max(int(np.round(self.max_range_S_2 * vec_len[1] / vec_len[0])), 1)
                self.max_range_F_1 = max(min(int(np.round(self.max_range_S_2 * vec_len[1] / vec_len[2])), max_f), 1)
                self.max_range_F_2 = max(min(int(np.round(self.max_range_S_2 * vec_len[1] / vec_len[3])), max_f), 1)
            elif min_arg == 2:
                self.max_range_F_1 = max_f
                self.max_range_F_2 = max(int(np.round(self.max_range_F_1 * vec_len[2] / vec_len[3])), 1)
                self.max_range_S_1 = max(min(int(np.round(self.max_range_F_1 * vec_len[2] / vec_len[0])), max_s), 1)
                self.max_range_S_2 = max(min(int(np.round(self.max_range_F_1 * vec_len[2] / vec_len[1])), max_s), 1)
            elif min_arg == 3:
                self.max_range_F_2 = max_f
                self.max_range_F_1 = max(int(np.round(self.max_range_F_2 * vec_len[3] / vec_len[2])), 1)
                self.max_range_S_1 = max(min(int(np.round(self.max_range_F_2 * vec_len[3] / vec_len[0])), max_s), 1)
                self.max_range_S_2 = max(min(int(np.round(self.max_range_F_2 * vec_len[3] / vec_len[1])), max_s), 1)
        else:
            raise NotImplementedError('m_range type not implemented')     
            
            
    def set_parameters(self, new_parameters = None):
        '''
        Sets the parameters for the supercell finder. Defaults are set and checked by Check_input.
        '''
        if new_parameters is not None:
            self.parameters = new_parameters
            
        self.max_area_diff = self.parameters['supercell_finder']['max_area_diff']
        self.Z_cell_length = self.parameters['supercell_finder']['z_cell_length']
        
            
    def create_atoms(self):
        '''
        Create the atoms object for substrate and film. They are not adjusted along the z-axis.
        '''
        S_geo = self.S_geo.copy()
        F_geo = self.F_geo.copy()
        
        S_geo.set_constraint()
        F_geo.set_constraint()

        for i in range(S_geo.positions.shape[0]):
            S_geo.positions[i, :2] = self.R_pretty @ S_geo.positions[i, :2]
        for i in range(F_geo.positions.shape[0]):
            F_geo.positions[i, :2] = self.U_final @ F_geo.positions[i, :2]

        for i in range(self.F_sc_points.shape[0]):
            if i == 0:
                F_geo_bunch = F_geo.copy()
                F_geo_bunch.positions[:, :2] += self.F_sc_points[i, :]
                F_geo_bunch.cell = self.cell
                F_geo_bunch.pbc = True
            else:
                F_geo_temp = F_geo.copy()
                F_geo_temp.positions[:, :2] += self.F_sc_points[i, :]
                F_geo_bunch += F_geo_temp

        for i in range(self.S_sc_points.shape[0]):
            if i == 0:
                S_geo_bunch = S_geo.copy()
                S_geo_bunch.positions[:, :2] += self.S_sc_points[i, :]
                S_geo_bunch.cell = self.cell
                S_geo_bunch.pbc = True
            else:
                S_geo_temp = S_geo.copy()
                S_geo_temp.positions[:, :2] += self.S_sc_points[i, :]
                S_geo_bunch += S_geo_temp
        
        self.F_atoms = F_geo_bunch.copy()
        self.S_atoms = S_geo_bunch.copy()
        self.joined_atoms = F_geo_bunch.copy()
        self.joined_atoms += S_geo_bunch.copy()
        
        
        


 
    
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
    inside = np.all((coeffs >= 0 - 1e-7) & (coeffs < 1 - 1e-7), axis=1)

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
    
def svd2(A):
    # Code by Jon Barron https://github.com/jonbarron/svd2.git
    # Adapted from https://lucidar.me/en/mathematics/singular-value-decomposition-of-a-2x2-matrix/
    '''
    Computes the singular value decomposition of a N 2x2 matrices A.
    '''
    
    def f(X):
        a = X[:,0,1] + X[:,1,0]
        b = X[:,0,0] - X[:,1,1]
        z = np.sqrt((b + 1j*a)/np.sqrt(a**2 + b**2))
        z_real = np.real(z)
        z_imag = np.imag(z)
        q = (1 + 1/(z_real**2 + z_imag**2))
        cos = 0.5 * z_real * q
        sin = 0.5 * z_imag * q
        Y = np.reshape(np.stack([cos, -sin, sin, cos], -1), [-1, 2, 2])
        return Y
    if len(A.shape) == 2:
        A = np.reshape(A, [1, 2, 2])
    
    AAT = np.einsum('nij,nkj->nik', A, A) 
    U = f(AAT)

    trace = AAT[:,0,0] + AAT[:,1,1]
    d = np.sqrt((AAT[:,0,0] - AAT[:,1,1])**2 + 4*(AAT[:,0,1] * AAT[:,1,0]))
    s = np.sqrt(0.5 * (trace[...,None] + np.stack([d, -d], -1)))

    ATA = np.einsum('nji,njk->nik', A, A) 
    W = f(ATA)

    D00 = np.sign(
        (U[:,0,0] * A[:,0,0] + U[:,1,0] * A[:,1,0]) * W[:,0,0] +
        (U[:,0,0] * A[:,0,1] + U[:,1,0] * A[:,1,1]) * W[:,1,0])
    D11 = np.sign(
        (U[:,0,1] * A[:,0,0] + U[:,1,1] * A[:,1,0]) * W[:,0,1] +
        (U[:,0,1] * A[:,0,1] + U[:,1,1] * A[:,1,1]) * W[:,1,1])
    VT = np.reshape(np.stack([
        W[:,0,0] * D00, W[:,1,0] * D00,
        W[:,0,1] * D11, W[:,1,1] * D11], -1), [-1, 2, 2])
    
    
    
    return np.squeeze(U), np.squeeze(s), np.squeeze(VT)

def inv2(A: np.ndarray) -> np.ndarray:
    """
    Invert a batch of 2x2 matrices.
    A: array of shape (N,2,2)
    Returns an array of same shape containing the inverses.
    """
    if len(A.shape) == 2:
        A = np.reshape(A, [1, 2, 2])
    
    # Extract entries
    a = A[:, 0, 0]
    b = A[:, 0, 1]
    c = A[:, 1, 0]
    d = A[:, 1, 1]

    # Compute determinant
    det = a * d - b * c

    # Allocate output
    A_inv = np.empty_like(A)

    # Fill with adjugate entries
    A_inv[:, 0, 0] =  d
    A_inv[:, 0, 1] = -b
    A_inv[:, 1, 0] = -c
    A_inv[:, 1, 1] =  a

    # Divide each 2×2 block by its scalar determinant
    A_inv /= det[:, None, None]
    return np.squeeze(A_inv)