import numpy as np

_CORE_ELECTRONS = {
    "P1": 34.0, "P2": 34.0, "P3": 34.0, "P4": 34.0, "P5": 34.0, "P6": 34.0,
    "N1": 33.0, "N2": 33.0, "N3": 33.0, "N4": 33.0, "N5": 33.0, "N6": 33.0,
    "C1": 34.0, "C2": 34.0, "C3": 34.0, "C4": 34.0, "C5": 34.0, "C6": 34.0,
    "Q1": 35.0, "Q2": 35.0, "Q3": 35.0, "Q4": 35.0, "Q5": 35.0, "Q6": 35.0,
    "X1": 50.0, "X2": 46.0, "X3": 42.0, "X4": 38.0,
    "D": 18.0, "W": 40.0,
}

_PREFIX_SCALE = {
    "": 1.0,
    "S": 3.0 / 4.0,
    "T": 2.0 / 4.0,
}

_SUFFIXES = ("", "r", "a", "e", "d", "h")

_ELECTRON_OVERRIDES = {
    "C4h": 12.5,
    "P4r": 16.5,
}

# Martini 3 LJ sigma values in nm for bead size classes
_SIGMA_BY_PREFIX_NM = {
    "": 0.47,   # regular
    "S": 0.41,  # small
    "T": 0.34,  # tiny
}


def _strip_variant(bead_name: str) -> str:
    core = bead_name
    if core.startswith(("S", "T")):
        core = core[1:]
    if core.endswith(("r", "a", "e", "d", "h")):
        core = core[:-1]
    return core


def _iter_all_bead_names():
    for prefix in _PREFIX_SCALE:
        for core in _CORE_ELECTRONS:
            for suffix in _SUFFIXES:
                yield f"{prefix}{core}{suffix}"


def get_valid_bead_names():
    names = {name for name in _iter_all_bead_names()}
    names.update(_ELECTRON_OVERRIDES.keys())
    return names


def get_bead_radii():
    """
    Return effective sphere radii in nm.
    Radius is taken as sigma/2 for Martini regular/small/tiny bead classes.
    """
    radii = {}
    for bead_name in get_valid_bead_names():
        if bead_name.startswith("T"):
            sigma = _SIGMA_BY_PREFIX_NM["T"]
        elif bead_name.startswith("S"):
            sigma = _SIGMA_BY_PREFIX_NM["S"]
        else:
            sigma = _SIGMA_BY_PREFIX_NM[""]

        radii[bead_name] = sigma / 2.0

    return radii


def get_base_electrons():
    electron_mapping = {}

    for bead_name in _iter_all_bead_names():
        core = _strip_variant(bead_name)
        if core not in _CORE_ELECTRONS:
            continue

        if bead_name.startswith("T"):
            prefix = "T"
        elif bead_name.startswith("S"):
            prefix = "S"
        else:
            prefix = ""

        Ne = round(_CORE_ELECTRONS[core] * _PREFIX_SCALE[prefix], 2)
        electron_mapping[bead_name] = Ne

    electron_mapping.update(_ELECTRON_OVERRIDES)

    radii = get_bead_radii()
    for bt, Ne in electron_mapping.items():
        V = (4.0 / 3.0) * np.pi * (radii[bt] ** 3)
        electron_mapping[bt] = Ne - 40.0 * V

    return electron_mapping


if __name__ == "__main__":
    pass