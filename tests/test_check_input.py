import json

import numpy as np
import pytest
from ase import Atoms

from gensec.check_input import Check_input
from gensec.relaxation import _resolve_fix_atoms


def _base_parameters():
    return {
        "geometry": {"filename": "geometry.in", "format": "aims"},
        "protocol": {"generate": {"activate": False}, "search": {"activate": False}},
    }


def test_protocol_requires_at_least_one_active_mode():
    params = _base_parameters()

    with pytest.raises(ValueError, match="set one to True"):
        Check_input(params)


def test_protocol_accepts_boolean_legacy_style():
    params = _base_parameters()
    params["protocol"] = {"generate": False, "search": False}

    with pytest.raises(ValueError, match="set one to True"):
        Check_input(params)


def test_fix_atoms_supports_index_and_z_range_modes():
    atoms = Atoms(["H", "He", "Li"], positions=[[0.0, 0.0, -0.2], [0.0, 0.0, 0.0], [0.0, 0.0, 0.2]])

    assert _resolve_fix_atoms(atoms, [0, 2]) == [0, 2]
    assert _resolve_fix_atoms(atoms, [-0.1, 0.1]) == [1]
    assert _resolve_fix_atoms(atoms, []) == []

    assert _resolve_fix_atoms(atoms, [0, 0.1]) == [1]
