"""Atomic data tables: ground-state electronic configurations, element symbols,
and NIST experimental ionization energies.
"""

# Element symbols for Z = 1 to 79
ELEMENT_SYMBOLS: dict[int, str] = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca",
    21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
    31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 47: "Ag", 79: "Au",
}

# NIST Atomic Spectra Database experimental 1st ionization energies (in eV)
# Source: Kramida, A., Ralchenko, Yu., Reader, J., and NIST ASD Team (2023). NIST Atomic Spectra Database.
NIST_IONIZATION_ENERGIES_EV: dict[int, float] = {
    1: 13.598, 2: 24.587, 3: 5.392, 4: 9.323, 5: 8.298, 6: 11.260, 7: 14.534, 8: 13.618,
    9: 17.423, 10: 21.565, 11: 5.139, 12: 7.646, 13: 5.986, 14: 8.152, 15: 10.487, 16: 10.360,
    17: 12.968, 18: 15.760, 19: 4.341, 20: 6.113, 21: 6.561, 22: 6.828, 23: 6.746, 24: 6.767,
    25: 7.434, 26: 7.902, 27: 7.881, 28: 7.640, 29: 7.726, 30: 9.394, 31: 5.999, 32: 7.899,
    33: 9.789, 34: 9.752, 35: 11.814, 36: 14.000, 47: 7.576, 79: 9.226,
}

# Standard Madelung order: (n, l, max_occ)
_MADELUNG_ORDER: list[tuple[int, int, int]] = [
    (1, 0, 2),   # 1s
    (2, 0, 2),   # 2s
    (2, 1, 6),   # 2p
    (3, 0, 2),   # 3s
    (3, 1, 6),   # 3p
    (4, 0, 2),   # 4s
    (3, 2, 10),  # 3d
    (4, 1, 6),   # 4p
    (5, 0, 2),   # 5s
    (4, 2, 10),  # 4d
    (5, 1, 6),   # 5p
    (6, 0, 2),   # 6s
    (4, 3, 14),  # 4f
    (5, 2, 10),  # 5d
    (6, 1, 6),   # 6p
]

# Explicit overrides for known transition metal exceptions
_EXPLICIT_OVERRIDES: dict[int, dict[tuple[int, int], float]] = {
    # Cr (24): [Ar] 3d5 4s1
    24: {
        (1, 0): 2.0, (2, 0): 2.0, (2, 1): 6.0, (3, 0): 2.0, (3, 1): 6.0,
        (3, 2): 5.0, (4, 0): 1.0,
    },
    # Cu (29): [Ar] 3d10 4s1
    29: {
        (1, 0): 2.0, (2, 0): 2.0, (2, 1): 6.0, (3, 0): 2.0, (3, 1): 6.0,
        (3, 2): 10.0, (4, 0): 1.0,
    },
    # Ag (47): [Kr] 4d10 5s1
    47: {
        (1, 0): 2.0, (2, 0): 2.0, (2, 1): 6.0, (3, 0): 2.0, (3, 1): 6.0,
        (3, 2): 10.0, (4, 0): 2.0, (4, 1): 6.0, (4, 2): 10.0, (5, 0): 1.0,
    },
    # Au (79): [Xe] 4f14 5d10 6s1
    79: {
        (1, 0): 2.0, (2, 0): 2.0, (2, 1): 6.0, (3, 0): 2.0, (3, 1): 6.0,
        (3, 2): 10.0, (4, 0): 2.0, (4, 1): 6.0, (4, 2): 10.0, (4, 3): 14.0,
        (5, 0): 2.0, (5, 1): 6.0, (5, 2): 10.0, (6, 0): 1.0,
    },
}


def get_ground_state_config(Z: int) -> dict[tuple[int, int], float]:
    """Return ground state subshell occupations {(n, l): occ} for atomic number Z."""
    if Z in _EXPLICIT_OVERRIDES:
        return dict(_EXPLICIT_OVERRIDES[Z])

    config: dict[tuple[int, int], float] = {}
    electrons_left = float(Z)

    for n, l, max_occ in _MADELUNG_ORDER:
        if electrons_left <= 0:
            break
        occ = min(float(max_occ), electrons_left)
        config[(n, l)] = occ
        electrons_left -= occ

    return config
