"""Quantum mechanics mathematical foundations for hydrogenic systems in Hartree atomic units.

Units:
    Lengths in Bohr radii (a0).
    Energies in Hartree (Eh).
    ħ = m_e = e = 1.
"""

from typing import Union
import math
import numpy as np

# Physical and conversion constants
HARTREE_EV: float = 27.211386245988
A0_M: float = 5.29177210903e-11
AU_TIME_S: float = 2.4188843265857e-17
ALPHA_FS: float = 1.0 / 137.035999084

SPECTROSCOPIC_LETTERS: str = "spdfg"
ORBITAL_SHAPES: dict[int, str] = {
    0: "Sphere (s)",
    1: "Dumbbell (p)",
    2: "Clover (d)",
    3: "Complex (f)",
    4: "Complex (g)",
}


def associated_laguerre(
    p: int, q: int, x: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Associated Laguerre polynomial L_p^q(x).

    Args:
        p: Degree (n - l - 1 >= 0).
        q: Order (2l + 1 >= 0).
        x: Argument (array or scalar, in atomic units).

    Returns:
        Evaluated polynomial matching the shape of x.
    """
    if p < 0:
        raise ValueError(f"Laguerre degree p must be non-negative, got {p}")
    x_arr = np.asarray(x, dtype=float)
    if p == 0:
        res = np.ones_like(x_arr)
        return float(res) if np.isscalar(x) else res
    if p == 1:
        res = 1.0 + q - x_arr
        return float(res) if np.isscalar(x) else res

    L0 = np.ones_like(x_arr)
    L1 = 1.0 + q - x_arr
    for i in range(2, p + 1):
        L2 = ((2 * i + q - 1 - x_arr) * L1 - (i + q - 1) * L0) / float(i)
        L0 = L1
        L1 = L2
    return float(L1) if np.isscalar(x) else L1


def associated_legendre(
    l: int, m: int, x: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Associated Legendre polynomial P_l^|m|(x) including Condon-Shortley phase.

    Args:
        l: Angular quantum number (l >= 0).
        m: Magnetic quantum number (|m| <= l).
        x: Argument cos(theta) in range [-1, 1] (array or scalar).

    Returns:
        Evaluated polynomial matching the shape of x.
    """
    absm = abs(m)
    if l < 0:
        raise ValueError(f"Angular momentum l must be non-negative, got {l}")
    if absm > l:
        raise ValueError(f"|m| must be <= l, got |m|={absm}, l={l}")

    x_arr = np.asarray(x, dtype=float)
    pmm = np.ones_like(x_arr)

    if absm > 0:
        # somx2 = sin(theta) = sqrt(1 - x^2)
        somx2 = np.sqrt(np.clip((1.0 - x_arr) * (1.0 + x_arr), 0.0, None))
        fact = 1.0
        for _ in range(1, absm + 1):
            pmm *= -fact * somx2
            fact += 2.0

    if l == absm:
        return float(pmm) if np.isscalar(x) else pmm

    pmmp1 = x_arr * (2.0 * absm + 1.0) * pmm
    if l == absm + 1:
        return float(pmmp1) if np.isscalar(x) else pmmp1

    pll = np.zeros_like(x_arr)
    for ll in range(absm + 2, l + 1):
        pll = (x_arr * (2.0 * ll - 1.0) * pmmp1 - (ll + absm - 1.0) * pmm) / float(ll - absm)
        pmm = pmmp1
        pmmp1 = pll

    return float(pll) if np.isscalar(x) else pll


def radial_wavefunction(
    n: int, l: int, r: Union[float, np.ndarray], Z: int = 1
) -> Union[float, np.ndarray]:
    """Hydrogenic radial wavefunction R_{nl}(r) in Hartree atomic units (a0).

    Args:
        n: Principal quantum number (n >= 1).
        l: Angular quantum number (0 <= l < n).
        r: Radial distance in Bohr radii a0 (r >= 0, array or scalar).
        Z: Atomic number (Z >= 1).

    Returns:
        R_{nl}(r) in atomic units (a0^(-3/2)), matching the shape of r.
    """
    if n < 1:
        raise ValueError(f"Principal quantum number n must be >= 1, got {n}")
    if l < 0 or l >= n:
        raise ValueError(f"Angular quantum number l must satisfy 0 <= l < n, got l={l}, n={n}")
    if Z < 1:
        raise ValueError(f"Atomic number Z must be >= 1, got {Z}")

    r_arr = np.asarray(r, dtype=float)
    rho = (2.0 * Z * r_arr) / float(n)

    p = n - l - 1
    q = 2 * l + 1

    norm_sq = ((2.0 * Z / float(n)) ** 3) * (
        math.factorial(p) / (2.0 * n * math.factorial(n + l))
    )
    normalization = math.sqrt(norm_sq)

    L = associated_laguerre(p, q, rho)
    R = normalization * np.exp(-rho / 2.0) * (rho ** l) * L

    return float(R) if np.isscalar(r) else R


def spherical_harmonic(
    l: int,
    m: int,
    theta: Union[float, np.ndarray],
    phi: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Real spherical harmonic Y_{lm}(theta, phi).

    Orthonormal on the unit sphere: ∫ |Y_lm|^2 dΩ = 1.

    Args:
        l: Angular quantum number (l >= 0).
        m: Magnetic quantum number (|m| <= l).
        theta: Polar angle [0, pi] (array or scalar).
        phi: Azimuthal angle [0, 2pi] (array or scalar).

    Returns:
        Y_{lm}(theta, phi) matching the broadcast shape of theta and phi.
    """
    absm = abs(m)
    if l < 0:
        raise ValueError(f"l must be non-negative, got {l}")
    if absm > l:
        raise ValueError(f"|m| must be <= l, got |m|={absm}, l={l}")

    th_arr, ph_arr = np.broadcast_arrays(np.asarray(theta, dtype=float), np.asarray(phi, dtype=float))

    norm = math.sqrt(
        ((2.0 * l + 1.0) / (4.0 * math.pi))
        * (math.factorial(l - absm) / math.factorial(l + absm))
    )

    cos_theta = np.cos(th_arr)
    P = associated_legendre(l, m, cos_theta)

    if m == 0:
        val = norm * P
    elif m > 0:
        val = math.sqrt(2.0) * norm * P * np.cos(m * ph_arr)
    else:
        val = math.sqrt(2.0) * norm * P * np.sin(absm * ph_arr)

    return float(val) if np.isscalar(theta) and np.isscalar(phi) else val


def probability_density(
    r: Union[float, np.ndarray],
    theta: Union[float, np.ndarray],
    phi: Union[float, np.ndarray],
    n: int,
    l: int,
    m: int,
    Z: int = 1,
) -> Union[float, np.ndarray]:
    """Probability density |ψ_{nlm}(r, theta, phi)|^2 in a0^(-3).

    Args:
        r: Distance in a0.
        theta: Polar angle.
        phi: Azimuthal angle.
        n, l, m: Quantum numbers.
        Z: Atomic number.

    Returns:
        |R_{nl}(r)|^2 * |Y_{lm}(theta, phi)|^2 in a0^(-3).
    """
    R = radial_wavefunction(n, l, r, Z)
    Y = spherical_harmonic(l, m, theta, phi)
    return (R ** 2) * (Y ** 2)


def to_cartesian(
    r: Union[float, np.ndarray],
    theta: Union[float, np.ndarray],
    phi: Union[float, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert spherical coordinates to Cartesian coordinates (in a0)."""
    r_arr = np.asarray(r, dtype=float)
    th_arr = np.asarray(theta, dtype=float)
    ph_arr = np.asarray(phi, dtype=float)

    sin_th = np.sin(th_arr)
    x = r_arr * sin_th * np.cos(ph_arr)
    y = r_arr * sin_th * np.sin(ph_arr)
    z = r_arr * np.cos(th_arr)
    return x, y, z


def get_energy(n: int, Z: int = 1, unit: str = "Hartree") -> float:
    """Energy of hydrogen-like state.

    Args:
        n: Principal quantum number (n >= 1).
        Z: Atomic number (Z >= 1).
        unit: 'Hartree' (default) or 'eV'.

    Returns:
        Energy in requested unit. -Z^2 / (2 n^2) in Hartree.
    """
    if n < 1:
        raise ValueError(f"Principal quantum number n must be >= 1, got {n}")
    if Z < 1:
        raise ValueError(f"Atomic number Z must be >= 1, got {Z}")

    e_hartree = -float(Z ** 2) / (2.0 * (n ** 2))
    if unit.lower() == "ev":
        return e_hartree * HARTREE_EV
    return e_hartree


def generate_point_cloud(
    n: int, l: int, m: int, Z: int = 1, num_points: int = 3000
) -> np.ndarray:
    """Generate (num_points, 4) array of [x, y, z, |ψ|²] in Bohr radii a0.

    Legacy signature retained for CLI and renderer compatibility.
    Delegates to the vectorized sampling engine.
    """
    # Lazy import to avoid circular dependency
    from sampling import sample_hydrogenic_orbital

    points, psi2 = sample_hydrogenic_orbital(n, l, m, Z=Z, n_points=num_points)
    return np.column_stack([points, psi2])
