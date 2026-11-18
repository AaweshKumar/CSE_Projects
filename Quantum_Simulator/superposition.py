"""Superposition dynamics, dipole selection rules, and radiative transitions.

All computations performed in Hartree atomic units.
"""

from dataclasses import dataclass
from typing import Union
import math
import numpy as np

from maths import (
    radial_wavefunction,
    spherical_harmonic,
    get_energy,
    HARTREE_EV,
    ALPHA_FS,
    AU_TIME_S,
)
from sampling import sample_hydrogenic_orbital


@dataclass
class TransitionInfo:
    """Radiative transition parameters between two states."""
    state_a: tuple[int, int, int]
    state_b: tuple[int, int, int]
    dipole_vector: tuple[float, float, float]  # in a0
    dipole_sq: float  # in a0^2
    delta_e_ha: float  # in Hartree
    delta_e_ev: float  # in eV
    wavelength_nm: float  # in nm (or inf if delta_e == 0)
    is_allowed: bool
    einstein_a_s: float  # in s^-1
    lifetime_s: float  # in s (or inf)
    forbidden_reason: str


def dipole_matrix_element(
    state_f: tuple[int, int, int],
    state_i: tuple[int, int, int],
    Z: int = 1,
) -> tuple[float, float, float]:
    """Compute ⟨ψ_f| r |ψ_i⟩ = (dx, dy, dz) in a0 using numerical quadrature.

    Args:
        state_f: (n_f, l_f, m_f) final state quantum numbers.
        state_i: (n_i, l_i, m_i) initial state quantum numbers.
        Z: Atomic number.

    Returns:
        (dx, dy, dz) in Bohr radii a0.
    """
    nf, lf, mf = state_f
    ni, li, mi = state_i

    # Parity check: r is odd under inversion, so ⟨ψ_f|r|ψ_i⟩ = 0 unless l_f + l_i is odd.
    if (lf + li) % 2 == 0:
        return 0.0, 0.0, 0.0

    # Angular quadrature: Gauss-Legendre on cos(theta) in [-1, 1], uniform on phi in [0, 2pi]
    n_gl = 60
    x_nodes, w_nodes = np.polynomial.legendre.leggauss(n_gl)
    th_nodes = np.arccos(x_nodes)

    n_phi = 120
    phi_nodes = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / n_phi

    # Radial quadrature on log/linear grid
    n_max = max(nf, ni)
    r_max = max(40.0 / Z, 12.0 * ((n_max + 1) ** 2) / Z)
    r_nodes = np.linspace(1e-6, r_max, 4000)
    dr = r_nodes[1] - r_nodes[0]

    Rf = radial_wavefunction(nf, lf, r_nodes, Z=Z)
    Ri = radial_wavefunction(ni, li, r_nodes, Z=Z)
    # Radial integrand for dipole is r * R_f * R_i * r^2 = r^3 R_f R_i
    # Using trapezoidal rule
    mid_r = 0.5 * (r_nodes[:-1] + r_nodes[1:])
    mid_prod = 0.5 * (
        (r_nodes[:-1] ** 3) * Rf[:-1] * Ri[:-1] + (r_nodes[1:] ** 3) * Rf[1:] * Ri[1:]
    )
    radial_int = float(np.sum(mid_prod * dr))

    # Evaluate angular parts
    # Yf, Yi shape: (n_gl, n_phi)
    TH, PH = np.meshgrid(th_nodes, phi_nodes, indexing="ij")
    Yf = spherical_harmonic(lf, mf, TH, PH)
    Yi = spherical_harmonic(li, mi, TH, PH)

    sin_th = np.sin(TH)
    cos_th = np.cos(TH)
    cos_ph = np.cos(PH)
    sin_ph = np.sin(PH)

    # Angular integrals over dOmega = d(cos theta) dphi
    # x = r sin(th) cos(ph)
    # y = r sin(th) sin(ph)
    # z = r cos(th)
    prod = Yf * Yi
    w_matrix = w_nodes[:, np.newaxis]

    ang_x = float(np.sum(w_matrix * prod * sin_th * cos_ph) * dphi)
    ang_y = float(np.sum(w_matrix * prod * sin_th * sin_ph) * dphi)
    ang_z = float(np.sum(w_matrix * prod * cos_th) * dphi)

    dx = radial_int * ang_x
    dy = radial_int * ang_y
    dz = radial_int * ang_z

    # Clean numerical noise near zero
    if abs(dx) < 1e-12:
        dx = 0.0
    if abs(dy) < 1e-12:
        dy = 0.0
    if abs(dz) < 1e-12:
        dz = 0.0

    return dx, dy, dz


def transition_info(
    state_a: tuple[int, int, int],
    state_b: tuple[int, int, int],
    Z: int = 1,
) -> TransitionInfo:
    """Compute spectroscopic transition info and selection rules between two states.

    Args:
        state_a: (n_a, l_a, m_a).
        state_b: (n_b, l_b, m_b).
        Z: Atomic number.
    """
    na, la, ma = state_a
    nb, lb, mb = state_b

    dx, dy, dz = dipole_matrix_element(state_b, state_a, Z=Z)
    d2 = dx ** 2 + dy ** 2 + dz ** 2

    ea = get_energy(na, Z=Z, unit="Hartree")
    eb = get_energy(nb, Z=Z, unit="Hartree")
    delta_e_ha = abs(eb - ea)
    delta_e_ev = delta_e_ha * HARTREE_EV

    # Threshold for dipole allowed
    is_allowed = d2 > 1e-8

    forbidden_reasons = []
    if la == lb:
        forbidden_reasons.append("Δl = 0")
    elif abs(la - lb) != 1:
        forbidden_reasons.append(f"|Δl| = {abs(la - lb)} ≠ 1")

    # Real harmonics caveat for m:
    if abs(abs(ma) - abs(mb)) > 1:
        forbidden_reasons.append(f"|Δ|m|| = {abs(abs(ma) - abs(mb))} > 1")

    if not is_allowed and not forbidden_reasons:
        forbidden_reasons.append("Dipole matrix element vanishes by symmetry")

    reason_str = ", ".join(forbidden_reasons) if not is_allowed else "Allowed (E1)"

    # Wavelength: lambda = 2 * pi * c / omega in a.u., converted to nm
    c_au = 1.0 / ALPHA_FS
    if delta_e_ha > 1e-12:
        # lambda_au = 2 * pi * c_au / delta_e_ha
        # lambda_m = lambda_au * A0_M
        # lambda_nm = lambda_m * 1e9
        wavelength_nm = (2.0 * math.pi * c_au / delta_e_ha) * 5.29177210903e-2
    else:
        wavelength_nm = float("inf")

    # Einstein A: A = 4 * omega^3 * |d|^2 / (3 * c^3) in a.u.
    if delta_e_ha > 1e-12 and is_allowed:
        a_au = (4.0 * (delta_e_ha ** 3) * d2) / (3.0 * (c_au ** 3))
        einstein_a_s = a_au / AU_TIME_S
        lifetime_s = 1.0 / einstein_a_s if einstein_a_s > 0 else float("inf")
    else:
        einstein_a_s = 0.0
        lifetime_s = float("inf")

    return TransitionInfo(
        state_a=state_a,
        state_b=state_b,
        dipole_vector=(dx, dy, dz),
        dipole_sq=d2,
        delta_e_ha=delta_e_ha,
        delta_e_ev=delta_e_ev,
        wavelength_nm=wavelength_nm,
        is_allowed=is_allowed,
        einstein_a_s=einstein_a_s,
        lifetime_s=lifetime_s,
        forbidden_reason=reason_str,
    )


class SuperpositionState:
    """Manages an importance-weighted static point ensemble for animated superpositions."""

    def __init__(
        self,
        states: list[tuple[float, int, int, int]],  # [(c_i, n_i, l_i, m_i)]
        Z: int = 1,
        n_points: int = 8000,
        rng: np.random.Generator | None = None,
    ):
        self.Z = Z
        self.n_points = n_points

        # Normalize coefficients so sum(c_i^2) == 1
        raw_coeffs = np.array([s[0] for s in states], dtype=float)
        norm = math.sqrt(float(np.sum(raw_coeffs ** 2)))
        if norm < 1e-12:
            raw_coeffs[0] = 1.0
            norm = 1.0
        self.coeffs = raw_coeffs / norm
        self.states = [(float(c), s[1], s[2], s[3]) for c, s in zip(self.coeffs, states)]

        # Sample N_i points proportional to c_i^2
        weights_sq = self.coeffs ** 2
        allocations = np.round(weights_sq * n_points).astype(int)
        allocations[0] += n_points - int(np.sum(allocations))  # Ensure exact sum

        all_points = []
        for n_i, (_, n, l, m) in zip(allocations, self.states):
            if n_i > 0:
                pts, _ = sample_hydrogenic_orbital(n, l, m, Z=Z, n_points=n_i, rng=rng)
                all_points.append(pts)

        self.points = np.vstack(all_points)  # shape (N, 3)

        # Precompute and cache ψ_i(x_k) at all points
        r = np.linalg.norm(self.points, axis=1)
        safe_r = np.maximum(r, 1e-30)
        cos_th = np.clip(self.points[:, 2] / safe_r, -1.0, 1.0)
        th = np.arccos(cos_th)
        ph = np.arctan2(self.points[:, 1], self.points[:, 0]) % (2.0 * np.pi)

        k_states = len(self.states)
        self.psi_matrix = np.zeros((k_states, n_points), dtype=float)
        self.energies = np.zeros(k_states, dtype=float)

        for i, (_, n, l, m) in enumerate(self.states):
            R = radial_wavefunction(n, l, r, Z=Z)
            Y = spherical_harmonic(l, m, th, ph)
            self.psi_matrix[i, :] = R * Y
            self.energies[i] = get_energy(n, Z=Z, unit="Hartree")

        # Base proposal density q(x) = sum_i c_i^2 ψ_i^2(x)
        self.q = np.sum((self.coeffs[:, np.newaxis] ** 2) * (self.psi_matrix ** 2), axis=0)
        self.safe_q = np.maximum(self.q, 1e-30)

        # Precompute dipole matrix elements for exact ⟨z⟩(t)
        self.dipole_z_matrix = np.zeros((k_states, k_states), dtype=float)
        for i, (_, ni, li, mi) in enumerate(self.states):
            for j, (_, nj, lj, mj) in enumerate(self.states):
                if j >= i:
                    _, _, dz = dipole_matrix_element((nj, lj, mj), (ni, li, mi), Z=Z)
                    self.dipole_z_matrix[i, j] = dz
                    self.dipole_z_matrix[j, i] = dz

    def density_weights(self, t: float) -> np.ndarray:
        """Calculate importance weights w(x, t) = ρ(x, t) / q(x) at time t (in a.u.).

        Returns:
            w: (N,) array of relative point weights (mean ≈ 1).
        """
        k_states = len(self.states)
        rho = np.zeros(self.n_points, dtype=float)

        for i in range(k_states):
            ci = self.coeffs[i]
            psi_i = self.psi_matrix[i]
            ei = self.energies[i]
            for j in range(k_states):
                cj = self.coeffs[j]
                psi_j = self.psi_matrix[j]
                ej = self.energies[j]
                cos_term = math.cos((ej - ei) * t)
                rho += ci * cj * psi_i * psi_j * cos_term

        w = rho / self.safe_q
        return np.clip(w, 0.0, None)

    def expectation_z(self, t: float) -> float:
        """Exact noise-free expectation value ⟨z⟩(t) in a0."""
        k_states = len(self.states)
        exp_z = 0.0
        for i in range(k_states):
            ci = self.coeffs[i]
            ei = self.energies[i]
            for j in range(k_states):
                cj = self.coeffs[j]
                ej = self.energies[j]
                cos_term = math.cos((ej - ei) * t)
                exp_z += ci * cj * cos_term * self.dipole_z_matrix[i, j]
        return exp_z


AU_ELECTRIC_FIELD_VM = 5.14220674763e11  # 1 a.u. of electric field in V/m


@dataclass
class StarkResult:
    """Stark effect results for Hydrogen n=2 manifold."""
    F_vm: float
    F_au: float
    matrix_element_z_a0: float  # ⟨2s|z|2pz⟩ ≈ -3.0 a0
    W_ha: float                 # Energy shift |W| in Hartree
    delta_e_ev: float           # Total energy splitting ΔE = 2|W| in eV
    energy_plus_ev: float       # E0 + |W| in eV
    energy_minus_ev: float      # E0 - |W| in eV
    coeffs_plus: tuple[float, float]   # (c_2s, c_2pz) for state with E = E0 + |W|
    coeffs_minus: tuple[float, float]  # (c_2s, c_2pz) for state with E = E0 - |W|
    dipole_plus_a0: float       # Expectation value ⟨z⟩ for plus state
    dipole_minus_a0: float      # Expectation value ⟨z⟩ for minus state
    warning: str | None


def stark_n2_hydrogen(F_vm: float) -> StarkResult:
    """Compute linear Stark effect in Hydrogen (Z=1, n=2) manifold.

    Subspace: {|2s⟩, |2pz⟩}.
    Perturbation H' = -q F·z = +F z in atomic units.
    Matrix H' = [[0, W], [W, 0]], with W = F_au · ⟨2s|z|2pz⟩.

    Args:
        F_vm: Electric field strength along z-axis in V/m (0 to 5e9).

    Returns:
        StarkResult with eigenenergies, eigenstates, dipole moments, and warning.
    """
    F_au = float(F_vm) / AU_ELECTRIC_FIELD_VM
    _, _, dz = dipole_matrix_element((2, 0, 0), (2, 1, 0), Z=1)

    # W = F_au * dz
    W_raw = F_au * dz
    W_ha = abs(W_raw)

    E0_ha = get_energy(2, Z=1, unit="Hartree")  # -0.125 Ha
    E0_ev = E0_ha * HARTREE_EV                  # -3.4014 eV

    delta_e_ev = 2.0 * W_ha * HARTREE_EV
    energy_plus_ev = E0_ev + W_ha * HARTREE_EV
    energy_minus_ev = E0_ev - W_ha * HARTREE_EV

    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    # If W_raw < 0 (since dz ≈ -3.0):
    # Eigenvector for +|W| is (1, -1)/sqrt(2)
    # Eigenvector for -|W| is (1, +1)/sqrt(2)
    if W_raw <= 0:
        coeffs_plus = (inv_sqrt2, -inv_sqrt2)
        coeffs_minus = (inv_sqrt2, inv_sqrt2)
    else:
        coeffs_plus = (inv_sqrt2, inv_sqrt2)
        coeffs_minus = (inv_sqrt2, -inv_sqrt2)

    dipole_plus_a0 = 2.0 * coeffs_plus[0] * coeffs_plus[1] * dz
    dipole_minus_a0 = 2.0 * coeffs_minus[0] * coeffs_minus[1] * dz

    warning = None
    if F_vm > 1e9:
        warning = "Warning: F > 1e9 V/m; perturbation theory degrading"

    return StarkResult(
        F_vm=F_vm,
        F_au=F_au,
        matrix_element_z_a0=dz,
        W_ha=W_ha,
        delta_e_ev=delta_e_ev,
        energy_plus_ev=energy_plus_ev,
        energy_minus_ev=energy_minus_ev,
        coeffs_plus=coeffs_plus,
        coeffs_minus=coeffs_minus,
        dipole_plus_a0=dipole_plus_a0,
        dipole_minus_a0=dipole_minus_a0,
        warning=warning,
    )

