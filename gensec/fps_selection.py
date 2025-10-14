import os
import numpy as np
import ase.db
from featomic import SoapPowerSpectrum
import metatensor
from skmatter import sample_selection

def select_structures_fps(input_db, output_db, n_select):
    db_in = ase.db.connect(input_db)
    frames = [row.toatoms() for row in db_in.select()]
    X = compute_structural_features(frames)
    selected_idx = perform_fps(X, n_select)
    if os.path.exists(output_db):
        os.remove(output_db)
    db_out = ase.db.connect(output_db)
    for i in selected_idx:
        row = db_in[int(i) + 1]
        db_out.write(row)

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