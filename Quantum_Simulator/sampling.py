"""Sampling engine for atomic orbitals using inverse-CDF radial sampling
and vectorized angular rejection sampling.
"""

from typing import Union
import numpy as np
from maths import (
    radial_wavefunction,
    spherical_harmonic,
    to_cartesian,
)

_Y_MAX_BOUND_CACHE: dict[tuple[int, int], float] = {}


def get_y_max_bound(l: int, m: int) -> float:
    """Compute and cache the bound M = 1.05 * max|Y_lm|^2 on a 200x400 sphere grid."""
    key = (l, m)
    if key in _Y_MAX_BOUND_CACHE:
        return _Y_MAX_BOUND_CACHE[key]

    theta = np.linspace(0.0, np.pi, 200)
    phi = np.linspace(0.0, 2.0 * np.pi, 400)
    T, P = np.meshgrid(theta, phi, indexing="ij")
    Y = spherical_harmonic(l, m, T, P)
    max_val = float(np.max(Y ** 2))
    bound = 1.05 * max(max_val, 1e-12)
    _Y_MAX_BOUND_CACHE[key] = bound
    return bound


def sample_orbital(
    r_grid: np.ndarray,
    P_r: np.ndarray,
    l: int,
    m: int,
    n_points: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample 3D points from an atomic orbital probability distribution.

    Args:
        r_grid: 1D array of radial points in Bohr radii (>= 4000 points).
        P_r: Radial probability density u(r)^2 = r^2 * R(r)^2 on r_grid.
        l: Angular quantum number.
        m: Magnetic quantum number.
        n_points: Number of points to sample.
        rng: Optional NumPy random Generator.

    Returns:
        points: (n_points, 3) Cartesian coordinates [x, y, z] in Bohr radii a0.
        psi2: (n_points,) values of |ψ|^2 = R^2 * Y^2 at each point.
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1. Radial sampling via inverse-CDF
    dr = np.diff(r_grid)
    mid_P = 0.5 * (P_r[:-1] + P_r[1:])
    cdf = np.zeros_like(r_grid)
    cdf[1:] = np.cumsum(mid_P * dr)
    total_mass = cdf[-1]
    if total_mass > 0:
        cdf /= total_mass
    else:
        raise ValueError("P_r integral is zero, cannot sample radial points")

    u_rand = rng.uniform(0.0, 1.0, size=n_points)
    r_samples = np.interp(u_rand, cdf, r_grid)

    # 2. Vectorized angular rejection sampling from |Y_lm|^2
    M = get_y_max_bound(l, m)
    accepted_theta: list[np.ndarray] = []
    accepted_phi: list[np.ndarray] = []
    accepted_Y2: list[np.ndarray] = []
    total_accepted = 0

    batch_size = max(n_points * 2, 2000)
    while total_accepted < n_points:
        # Uniform proposal on sphere: cos(theta) in [-1, 1], phi in [0, 2pi]
        cos_th = rng.uniform(-1.0, 1.0, size=batch_size)
        th_cand = np.arccos(cos_th)
        ph_cand = rng.uniform(0.0, 2.0 * np.pi, size=batch_size)

        Y = spherical_harmonic(l, m, th_cand, ph_cand)
        Y2 = Y ** 2

        u_test = rng.uniform(0.0, M, size=batch_size)
        mask = u_test < Y2

        if np.any(mask):
            th_acc = th_cand[mask]
            ph_acc = ph_cand[mask]
            y2_acc = Y2[mask]
            accepted_theta.append(th_acc)
            accepted_phi.append(ph_acc)
            accepted_Y2.append(y2_acc)
            total_accepted += len(th_acc)

    theta_samples = np.concatenate(accepted_theta)[:n_points]
    phi_samples = np.concatenate(accepted_phi)[:n_points]
    Y2_samples = np.concatenate(accepted_Y2)[:n_points]

    # 3. Coordinates
    x, y, z = to_cartesian(r_samples, theta_samples, phi_samples)
    points = np.column_stack([x, y, z])

    # 4. |ψ|^2 = R^2 * Y^2
    # R^2 = P_r(r) / r^2
    # Interpolate R^2 from r_grid
    safe_r2 = np.maximum(r_grid ** 2, 1e-30)
    R2_grid = P_r / safe_r2
    # Fix r=0 singularity by extrapolating from the first positive grid point
    if r_grid[0] == 0:
        R2_grid[0] = R2_grid[1]

    R2_samples = np.interp(r_samples, r_grid, R2_grid)
    psi2 = R2_samples * Y2_samples

    return points, psi2


def sample_hydrogenic_orbital(
    n: int,
    l: int,
    m: int,
    Z: int = 1,
    n_points: int = 8000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample points from a hydrogenic orbital (n, l, m) for atomic number Z.

    Returns:
        points: (n_points, 3) in Bohr radii a0.
        psi2: (n_points,) values of |ψ|^2 in a0^(-3).
    """
    # Grid extending until cumulative mass >= 1 - 1e-6
    r_max = max(25.0 / Z, 8.0 * ((n + 1) ** 2) / Z)
    r_grid = np.linspace(0.0, r_max, 5000)
    R = radial_wavefunction(n, l, r_grid, Z=Z)
    P_r = (R ** 2) * (r_grid ** 2)

    return sample_orbital(r_grid, P_r, l, m, n_points, rng=rng)


def get_hydrogenic_r999(n: int, l: int, Z: int = 1) -> float:
    """Return radius holding 99.9% of the probability density for camera auto-fit."""
    r_max = max(30.0 / Z, 10.0 * ((n + 1) ** 2) / Z)
    r_grid = np.linspace(0.0, r_max, 5000)
    R = radial_wavefunction(n, l, r_grid, Z=Z)
    P_r = (R ** 2) * (r_grid ** 2)

    dr = np.diff(r_grid)
    mid_P = 0.5 * (P_r[:-1] + P_r[1:])
    cdf = np.zeros_like(r_grid)
    cdf[1:] = np.cumsum(mid_P * dr)
    if cdf[-1] > 0:
        cdf /= cdf[-1]
    return float(np.interp(0.999, cdf, r_grid))
