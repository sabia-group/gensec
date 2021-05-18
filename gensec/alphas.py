from ase.calculators.lj import LennardJones
from ase.io import read, write
from ase.optimize import BFGS
import sys
import numpy as np
import pickle

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



ABOHR = 0.52917721 # in AA
HARTREE = 27.211383 # in eV
BOHR_to_angstr = 0.52917721   # in AA
HARTREE_to_eV = 27.211383  # in eV
HARTREE_to_kcal_mol = 627.509 # in kcal * mol^(-1)



def get_R0AB(A, B):
    return (VDW_radii[B] + VDW_radii[A]) * 0.5 * BOHR_to_angstr  # in Angstroms

def func(alpha, R0, R):
    return(np.exp(alpha*(R0**2-R**2)))

def find_alpha(A, B, Cutoff, threshold):
    
    R0 = get_R0AB(A, B)
    for alpha in np.linspace(0, 10, 10000):
        y = func(alpha, R0, Cutoff*R0) # Value of the function is less than threshold
        if y < threshold:
            return alpha

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


            
dictionary = {}
for A in VDW_radii:
    dictionary[A] = {} 
    for B in VDW_radii:
        alpha = find_alpha(A, B, Cutoff=2, threshold=1e-12)
        dictionary[A][B] = alpha
        #print(A, B, get_R0AB(A, B), alpha)

save_obj(dictionary, "Alphas_vdW")
print(dictionary)






