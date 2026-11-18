"""Radial plot data preparation and analysis for Quantum Atom Simulator.

Pure calculation module (no PyQt5 dependencies) providing:
- radial_plot_data: returns (r, P) for hydrogenic and SCF orbitals.
- find_most_probable_radius: r_mp = argmax(P).
- find_mean_radius: ⟨r⟩ = ∫ r P(r) dr.
- find_radial_nodes: interior zero-crossings (local minima near zero).
"""

from __future__ import annotations
from typing import Any
import numpy as np

from maths import radial_wavefunction
from sampling import get_hydrogenic_r999


def radial_plot_data(
    mode: str = "Hydrogenic",
    n: int = 1,
    l: int = 0,
    Z: int = 1,
    scf_orb: Any = None,
    scf_res: Any = None,
    num_points: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare radial probability distribution data P(r) = u(r)².

    Guarantees:
    - x is linear, r >= 0, starting at 0.0 up to the 99.9% probability radius r_999.
    - y is P(r) = u(r)², normalized such that ∫ P(r) dr = 1.
    - P(0) = 0 for all l (since u(r) = r R(r)).
    - P(r_max) < 1% of max(P).
    - Interior local minima near zero correspond to radial node count (n - l - 1).

    Args:
        mode: "Hydrogenic", "SCF", or full mode string.
        n: Principal quantum number (n >= 1).
        l: Orbital angular momentum quantum number (0 <= l < n).
        Z: Atomic number (Z >= 1).
        scf_orb: Optional Orbital dataclass instance from scf.py.
        scf_res: Optional SCFResult dataclass instance from scf.py.
        num_points: Number of points on linear radial grid (default = 500).

    Returns:
        (r, P): Tuple of 1D float arrays of length num_points.
    """
    if "scf" in mode.lower() and scf_orb is not None and scf_res is not None:
        r_raw = np.asarray(scf_res.r, dtype=float)
        u_raw = np.asarray(scf_orb.u, dtype=float)

        # Find 99.9% probability radius on logarithmic grid
        h = np.log(r_raw[1]) - np.log(r_raw[0])
        dr = r_raw * h
        cum_prob = np.cumsum((u_raw ** 2) * dr)
        idx_cut = int(np.searchsorted(cum_prob, 0.999))
        r_cut = float(r_raw[min(idx_cut, len(r_raw) - 1)])
        r_max = max(5.0, r_cut)

        r = np.linspace(0.0, r_max, num_points, dtype=float)
        u_interp = np.interp(r, r_raw, u_raw, left=0.0)
        u_interp[0] = 0.0  # Physical boundary condition u(0) = 0
        P = u_interp ** 2
    else:
        # Hydrogenic orbital mode
        r999 = get_hydrogenic_r999(n, l, Z)
        r = np.linspace(0.0, r999, num_points, dtype=float)
        R_vals = radial_wavefunction(n, l, r, Z=Z)
        u = r * R_vals
        u[0] = 0.0  # Exact boundary condition u(0) = 0
        P = u ** 2

    # Normalize ∫ P(r) dr = 1 using trapezoidal integration
    norm = float(np.trapezoid(P, r))
    if norm > 1e-15:
        P = P / norm
    P[0] = 0.0  # Guarantee exact zero at origin

    return r, P


def find_most_probable_radius(r: np.ndarray, P: np.ndarray) -> float:
    """Return the most probable radius r_mp = argmax(P)."""
    return float(r[np.argmax(P)])


def find_mean_radius(r: np.ndarray, P: np.ndarray) -> float:
    """Return the expectation value ⟨r⟩ = ∫ r P(r) dr."""
    return float(np.trapezoid(r * P, r))


def find_radial_nodes(r: np.ndarray, P: np.ndarray, threshold: float = 0.02) -> list[float]:
    """Return radii of interior radial nodes (local minima where P < threshold * max(P))."""
    p_max = float(np.max(P))
    if p_max <= 0:
        return []
    d = np.diff(P)
    # Find indices where derivative changes from negative to positive
    min_indices = np.where((d[:-1] < 0) & (d[1:] > 0))[0] + 1
    # Filter for interior minima near zero
    zero_nodes = [
        float(r[i]) for i in min_indices
        if 0 < i < len(r) - 1 and P[i] < threshold * p_max
    ]
    return zero_nodes
