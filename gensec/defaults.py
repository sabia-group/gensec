"""Default parameters for optimizer class - currently just sets max step size"""
import collections

# _fire = "fire"
# _dct = {"dt": 0.1}
# fire_defaults = collections.namedtuple(_fire, _dct.keys())(**_dct)

_mdmin = "mdmin"
_dct = {"dt": 0.2}
mdmin_defaults = collections.namedtuple(_mdmin,
                                        _dct.keys())(**_dct)  # type: ignore


_dct = {
    "maxstep": 0.2,  # default maxstep for all optimizers
    # _fire: fire_defaults,
    _mdmin: mdmin_defaults,  # type: ignore
}
defaults = collections.namedtuple("optimizer_defaults",
                                  _dct.keys())(**_dct)  # type: ignore
