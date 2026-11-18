"""Self-Consistent Field (SCF) central-field solver for multi-electron atoms.

Model:
    Hartree-Fock-Slater (Xα) atom, spherically averaged, non-relativistic,
    non-spin-polarized, fractional occupations.
Units:
    Lengths in Bohr radii a0.
    Energies in Hartree Eh.
"""

from dataclasses import dataclass
from typing import Union
import math
import numpy as np
import scipy.linalg as la

from atomdata import get_ground_state_config
from maths import radial_wavefunction


@dataclass
class Orbital:
    """Atomic orbital eigenstate."""
    n: int
    l: int
    eps: float  # Orbital energy in Hartree (Eh)
    occ: float  # Subshell occupation number
    u: np.ndarray  # Radial wavefunction u(r) = r R(r), normalized so ∫ u² dr = 1


@dataclass
class SCFResult:
    """Results of the atomic self-consistent field calculation."""
    Z: int
    N: float
    r: np.ndarray
    V: np.ndarray
    shell_density: np.ndarray  # D(r) = 4π r² ρ(r)
    orbitals: dict[tuple[int, int], Orbital]
    converged: bool
    iterations: int


# In-memory result cache keyed by (Z, alpha, use_electron_terms, tuple(occupations.items()))
_SCF_CACHE: dict[tuple, SCFResult] = {}


def count_nodes(u: np.ndarray) -> int:
    """Count internal radial nodes (sign changes) of u(r)."""
    # Exclude boundary regions near r=0 and r=max where numerical noise can create tiny zero-crossings
    abs_max = np.max(np.abs(u))
    if abs_max < 1e-12:
        return 0
    thresh = 1e-4 * abs_max
    sig = np.sign(u)
    # Filter noise
    filtered = []
    for val, s in zip(u, sig):
        if abs(val) > thresh:
            filtered.append(s)
    if len(filtered) < 2:
        return 0
    filt_arr = np.array(filtered)
    return int(np.sum(filt_arr[:-1] != filt_arr[1:]))


def zeff_of_r(res: SCFResult) -> np.ndarray:
    """Return effective nuclear charge Z_eff(r) = -r * V(r).

    As r -> 0, Z_eff -> Z.
    As r -> inf, Z_eff -> Z - N + 1 (1 for neutral atoms).
    """
    return -res.r * res.V


def solve_atom(
    Z: int,
    alpha: float = 0.7,
    occupations: dict[tuple[int, int], float] | None = None,
    use_electron_terms: bool = True,
    n_max: int = 6,
    n_grid: int = 3500,
) -> SCFResult:
    """Solve the spherically averaged Hartree-Fock-Slater (Xα) central field atom.

    Args:
        Z: Atomic number.
        alpha: Exchange scaling parameter (default = 0.7).
        occupations: Subshell occupations {(n, l): occ}. Defaults to ground state.
        use_electron_terms: If False, turns off Hartree and exchange (bare Coulomb).
        n_max: Maximum principal quantum number to solve (default = 6).
        n_grid: Number of radial log-grid points (default = 3500).

    Returns:
        SCFResult containing radial grid, potential, shell density, and orbitals.
    """
    if occupations is None:
        occupations = get_ground_state_config(Z)

    occ_tuple = tuple(sorted(occupations.items()))
    cache_key = (Z, round(alpha, 5), use_electron_terms, occ_tuple, n_max, n_grid)
    if cache_key in _SCF_CACHE:
        return _SCF_CACHE[cache_key]

    N_occ = float(sum(occupations.values()))

    # Log grid setup: r = exp(x), x uniform
    r_min = 5e-4 / float(Z)
    r_max = 80.0
    x = np.linspace(math.log(r_min), math.log(r_max), n_grid)
    h = x[1] - x[0]
    r = np.exp(x)
    dr = r * h

    tail_val = Z - N_occ + 1.0

    # Bare-Coulomb / one-electron case:
    if not use_electron_terms or N_occ <= 1.0:
        V = -float(Z) / r
        orbitals: dict[tuple[int, int], Orbital] = {}
        D = np.zeros(n_grid, dtype=float)

        for n in range(1, n_max + 1):
            for l in range(n):
                eps = -float(Z ** 2) / (2.0 * (n ** 2))
                R = radial_wavefunction(n, l, r, Z=Z)
                u = r * R
                # Normalize on the discrete grid: sum(u^2 dr) = 1
                norm = math.sqrt(float(np.sum(u ** 2 * dr)))
                if norm > 0:
                    u /= norm
                occ = occupations.get((n, l), 0.0)
                orbitals[(n, l)] = Orbital(n=n, l=l, eps=eps, occ=occ, u=u)
                if occ > 0:
                    D += occ * (u ** 2)

        res = SCFResult(
            Z=Z,
            N=N_occ,
            r=r,
            V=V,
            shell_density=D,
            orbitals=orbitals,
            converged=True,
            iterations=1,
        )
        _SCF_CACHE[cache_key] = res
        return res

    # Initial potential guess: Thomas-Fermi screened potential with Latter tail
    mu = 0.8 * (float(Z) ** (1.0 / 3.0))
    V = -(1.0 + (Z - 1.0) * np.exp(-mu * r)) / r
    V = np.minimum(V, -tail_val / r)

    # Damping and mixing parameters
    use_anderson = (Z <= 36)
    m_depth = 4
    V_hist: list[np.ndarray] = []
    R_hist: list[np.ndarray] = []
    beta = 0.35 if Z <= 20 else (0.25 if Z <= 36 else (0.15 if Z <= 50 else 0.08))
    D_mix: np.ndarray | None = None

    tol_V = 1e-6 if Z <= 50 else 1e-2
    tol_eps = 1e-7 if Z <= 50 else 5e-2

    eps_old = 0.0
    converged = False
    iterations_run = 0

    max_iter = 300

    for it in range(max_iter):
        iterations_run = it + 1

        orbitals = {}
        D = np.zeros(n_grid, dtype=float)
        eps_sum = 0.0

        for l in range(4):
            k_needed = n_max - l
            if k_needed <= 0:
                continue

            # Symmetric tridiagonal representation on log grid
            d = (1.0 / (h ** 2) + 0.5 * ((l + 0.5) ** 2)) / (r ** 2) + V
            e = -0.5 / (h ** 2 * r[:-1] * r[1:])
            evals, evecs = la.eigh_tridiagonal(d, e, select="i", select_range=(0, k_needed - 1))

            for k in range(k_needed):
                n = l + 1 + k
                eps = float(evals[k])
                g = evecs[:, k]
                # Normalize: ∫ u² dr = ∑ g² h = 1
                g_norm = math.sqrt(float(np.sum(g ** 2 * h)))
                if g_norm > 0:
                    g /= g_norm
                u = g / np.sqrt(r)

                # Consistent sign convention: initial rise near origin is positive
                if u[10] < 0:
                    u = -u

                occ = occupations.get((n, l), 0.0)
                orbitals[(n, l)] = Orbital(n=n, l=l, eps=eps, occ=occ, u=u)
                if occ > 0:
                    D += occ * (u ** 2)
                    eps_sum += occ * eps

        if not use_anderson:
            if D_mix is None:
                D_mix = D.copy()
            else:
                D_mix = (1.0 - beta) * D_mix + beta * D
            D_eff = D_mix
        else:
            D_eff = D

        rho = D_eff / (4.0 * math.pi * (r ** 2))

        # Hartree potential via cumulative trapezoid on log grid (dr = r dx)
        seg1 = 0.5 * h * (D_eff[:-1] * r[:-1] + D_eff[1:] * r[1:])
        I1 = np.zeros(n_grid, dtype=float)
        I1[1:] = np.cumsum(seg1)

        seg2 = 0.5 * h * (D_eff[:-1] + D_eff[1:])
        I2 = np.zeros(n_grid, dtype=float)
        I2[:-1] = np.cumsum(seg2[::-1])[::-1]

        V_H = I1 / r + I2

        # Slater Xα exchange potential
        V_x = -1.5 * alpha * (np.maximum(3.0 * rho / math.pi, 0.0) ** (1.0 / 3.0))

        V_raw = -float(Z) / r + V_H + V_x
        V_new = np.minimum(V_raw, -tail_val / r)

        # Check convergence criteria
        del_V_r = float(np.max(np.abs((V_new - V) * r)))
        del_eps = abs(eps_sum - eps_old)

        if del_V_r < tol_V and del_eps < tol_eps:
            converged = True
            V = V_new
            break

        if use_anderson:
            # Anderson mixing
            R = V_new - V
            V_hist.append(V)
            R_hist.append(R)
            if len(V_hist) > m_depth:
                V_hist.pop(0)
                R_hist.pop(0)

            m = len(V_hist)
            if m > 1:
                R_mat = np.column_stack(R_hist)
                B_mat = R_mat.T @ R_mat + 1e-10 * np.eye(m)
                A_eq = np.zeros((m + 1, m + 1), dtype=float)
                A_eq[:m, :m] = B_mat
                A_eq[:m, m] = 1.0
                A_eq[m, :m] = 1.0
                rhs = np.zeros(m + 1, dtype=float)
                rhs[m] = 1.0
                try:
                    sol = np.linalg.solve(A_eq, rhs)
                    c_weights = sol[:m]
                    if np.max(np.abs(c_weights)) < 5.0:
                        V_opt = sum(c_weights[i] * V_hist[i] for i in range(m))
                        R_opt = sum(c_weights[i] * R_hist[i] for i in range(m))
                        V = V_opt + beta * R_opt
                    else:
                        V = (1.0 - beta) * V + beta * V_new
                except Exception:
                    V = (1.0 - beta) * V + beta * V_new
            else:
                V = (1.0 - beta) * V + beta * V_new
        else:
            V = (1.0 - beta) * V + beta * V_new

        V = np.minimum(V, -tail_val / r)
        eps_old = eps_sum

    # Final pass: apply Latter tail to V
    mask = -r * V <= tail_val
    if np.any(mask):
        idx_L = int(np.where(mask)[0][0])
        V[idx_L:] = -tail_val / r[idx_L:]

    res = SCFResult(
        Z=Z,
        N=N_occ,
        r=r,
        V=V,
        shell_density=D,
        orbitals=orbitals,
        converged=converged,
        iterations=iterations_run,
    )
    _SCF_CACHE[cache_key] = res
    return res
