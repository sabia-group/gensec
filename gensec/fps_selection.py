import os
import numpy as np
import ase.db
from featomic import SoapPowerSpectrum
import metatensor
from skmatter import sample_selection

def select_structures_fps(frames, n_select):
    X = compute_structural_features(frames)
    selected_idx = perform_fps(X, n_select)
    print("FPS selected indices:", selected_idx)
    return selected_idx

def compute_structural_features(frames):
    
    # SOAP hyperparameters hard-coded for now
    hypers = {
        "cutoff": {"radius": 6.0, "smoothing": {"type": "ShiftedCosine", "width": 0.5}},
        "density": {
            "type": "Gaussian",
            "width": 0.3,
            "scaling": {"type": "Willatt2018", "exponent": 4, "rate": 1, "scale": 3.5},
        },
        "basis": {
            "type": "TensorProduct",
            "max_angular": 6,
            "radial": {"type": "Gto", "max_radial": 7},
        },
    }

    calc = SoapPowerSpectrum(**hypers)
    rho2i = calc.compute(frames)

    # Merge keys into a single block (combine all atomic species)
    atom_soap = rho2i.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])
    atom_soap_single = atom_soap.keys_to_samples(keys_to_move=["center_type"])

    # Average over atoms to get structure-level descriptors
    struct_soap = metatensor.mean_over_samples(
        atom_soap_single, sample_names=["atom", "center_type"]
    )

    X = struct_soap.block(0).values
    return X


def perform_fps(X, n_select):
    
    mean = X.mean(axis=0)
    delta = X - mean
    initial_idx = np.argmax(np.linalg.norm(delta, axis=1))

    fps = sample_selection.FPS(n_to_select=n_select, initialize=initial_idx)
    fps.fit(X)
    return fps.selected_idx_
