import unittest
from gensec.modules import (
    create_connectivity_matrix,
    detect_rotatble,
    detect_cycles,
    exclude_rotatable_from_cycles,
    measure_quaternion,
    carried_atoms,
    quaternion_set,
)

from ase.io import read, write
import os
import json
from gensec.structure import Structure, Fixed_frame

dirname, filename = os.path.split(os.path.abspath(__file__))
atoms = read(
    os.path.join(dirname, "supporting", "molecules", "hexane.xyz"),
    format="xyz",
)

with open(os.path.join(dirname, "parameters_generate.json")) as f:
    parameters = json.load(f)
parameters["geometry"][0] = os.path.join(
    dirname, "supporting", "molecules", "phenylalanine_com_applied.xyz"
)
parameters["geometry"][1] = "xyz"

parameters["fixed_frame"]["activate"] = False
parameters["fixed_frame"]["filename"] = os.path.join(
    dirname, "supporting", "surface", "Rh.in"
)
parameters["fixed_frame"]["format"] = "aims"
structure = Structure(parameters)
