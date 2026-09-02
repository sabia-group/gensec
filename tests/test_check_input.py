import json

import pytest

from gensec.check_input import Check_input


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
