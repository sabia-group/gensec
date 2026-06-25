"""Use a trained MACE model as the GenSec ASE calculator.

Set GENSEC_MACE_MODEL to an absolute model path before running search. If it is
not set, this script expects latest.model in the current run directory.
"""

import os

from mace.calculators import MACECalculator


model_path = os.path.abspath(os.environ.get("GENSEC_MACE_MODEL", "latest.model"))
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Trained MACE model not found: {model_path}. "
        "Set GENSEC_MACE_MODEL before starting GenSec search."
    )

device = os.environ.get("GENSEC_MACE_DEVICE", "cuda")
default_dtype = os.environ.get("GENSEC_MACE_DTYPE", "float64")

try:
    calculator = MACECalculator(
        model_paths=model_path,
        device=device,
        default_dtype=default_dtype,
    )
except TypeError:
    calculator = MACECalculator(
        model_path=model_path,
        device=device,
        default_dtype=default_dtype,
    )
