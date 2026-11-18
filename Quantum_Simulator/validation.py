"""Validation module for the Quantum Atom Simulator (Phase P3).

Computes:
1. First ionization energy series for Z = 1 to 36 comparing Koopmans' theorem
   and Slater transition-state method against NIST Atomic Spectra Database values.
2. Characteristic Kα X-ray emission series (2p -> 1s) for Z = 11 to 36 and
   Moseley's law linear regression fit, validating against Cu Kα (8.048 keV).

Pure physics / data only (no PyQt5 imports).
"""

from dataclasses import dataclass
from typing import Callable
import math
import numpy as np

from atomdata import ELEMENT_SYMBOLS, NIST_IONIZATION_ENERGIES_EV, get_ground_state_config
from maths import HARTREE_EV
from scf import solve_atom


@dataclass
class IEResult:
    """Ionization energy results for a single element."""
    Z: int
    symbol: str
    ie_koopmans_ev: float
    ie_ts_ev: float
    ie_nist_ev: float
    error_koopmans_ev: float
    error_ts_ev: float
    rel_error_koopmans: float
    rel_error_ts: float


@dataclass
class IESeriesResult:
    """Summary of ionization energy series validation."""
    points: list[IEResult]
    mae_koopmans_ev: float
    mae_ts_ev: float
    mre_koopmans: float
    mre_ts: float


@dataclass
class KAlphaPoint:
    """Kα transition data for a single element."""
    Z: int
    symbol: str
    e_kalpha_ev: float
    e_kalpha_kev: float
    sqrt_e_ry: float


@dataclass
class MoseleyResult:
    """Results of the Moseley Kα law linear regression."""
    points: list[KAlphaPoint]
    slope: float
    intercept: float
    r_squared: float
    cu_kalpha_kev: float
    cu_exp_kev: float
    cu_err_pct: float


# Cache for validation calculations
_IE_CACHE: dict[tuple, IESeriesResult] = {}
_MOSELEY_CACHE: dict[tuple, MoseleyResult] = {}


def compute_ionization_energies(
    Z_max: int = 36,
    alpha: float = 0.7,
    progress_cb: Callable[[int, int], None] | None = None,
) -> IESeriesResult:
    """Compute 1st ionization energies for Z = 1..Z_max using Koopmans and Slater TS.

    Args:
        Z_max: Maximum atomic number to compute (default = 36).
        alpha: Exchange scaling parameter for Xα potential.
        progress_cb: Optional callback(current_step, total_steps) for progress tracking.

    Returns:
        IESeriesResult containing individual element errors and summary statistics.
    """
    cache_key = (Z_max, round(alpha, 5))
    if cache_key in _IE_CACHE:
        return _IE_CACHE[cache_key]

    points: list[IEResult] = []
    total_steps = Z_max

    for idx, Z in enumerate(range(1, Z_max + 1)):
        if progress_cb is not None:
            progress_cb(idx + 1, total_steps)

        symbol = ELEMENT_SYMBOLS.get(Z, f"Z{Z}")
        occ_gs = get_ground_state_config(Z)

        # Ground-state calculation
        res_gs = solve_atom(Z, alpha=alpha, occupations=occ_gs)

        # Identify highest-energy occupied subshell (HOMO)
        occupied_subshells = [nl for nl, c in occ_gs.items() if c > 0]
        homo = max(occupied_subshells, key=lambda nl: res_gs.orbitals[nl].eps)

        # Koopmans theorem: IE = -eps_HOMO
        ie_koopmans_ev = -res_gs.orbitals[homo].eps * HARTREE_EV

        # Slater transition state: remove 0.5 electron from HOMO
        occ_ts = dict(occ_gs)
        occ_ts[homo] -= 0.5
        res_ts = solve_atom(Z, alpha=alpha, occupations=occ_ts)
        ie_ts_ev = -res_ts.orbitals[homo].eps * HARTREE_EV

        nist_ev = NIST_IONIZATION_ENERGIES_EV[Z]

        err_k = ie_koopmans_ev - nist_ev
        err_ts = ie_ts_ev - nist_ev
        rel_k = abs(err_k) / nist_ev
        rel_ts = abs(err_ts) / nist_ev

        points.append(
            IEResult(
                Z=Z,
                symbol=symbol,
                ie_koopmans_ev=ie_koopmans_ev,
                ie_ts_ev=ie_ts_ev,
                ie_nist_ev=nist_ev,
                error_koopmans_ev=err_k,
                error_ts_ev=err_ts,
                rel_error_koopmans=rel_k,
                rel_error_ts=rel_ts,
            )
        )

    mae_k = float(np.mean([abs(p.error_koopmans_ev) for p in points]))
    mae_ts = float(np.mean([abs(p.error_ts_ev) for p in points]))
    mre_k = float(np.mean([p.rel_error_koopmans for p in points]))
    mre_ts = float(np.mean([p.rel_error_ts for p in points]))

    result = IESeriesResult(
        points=points,
        mae_koopmans_ev=mae_k,
        mae_ts_ev=mae_ts,
        mre_koopmans=mre_k,
        mre_ts=mre_ts,
    )
    _IE_CACHE[cache_key] = result
    return result


def compute_k_alpha_series(
    Z_min: int = 11,
    Z_max: int = 36,
    alpha: float = 0.7,
    progress_cb: Callable[[int, int], None] | None = None,
) -> MoseleyResult:
    """Compute Kα transition energies (2p -> 1s) and Moseley fit for Z in [Z_min, Z_max].

    Args:
        Z_min: Starting atomic number (default = 11, Na).
        Z_max: Ending atomic number (default = 36, Kr).
        alpha: Exchange scaling parameter for Xα potential.
        progress_cb: Optional callback(current_step, total_steps) for progress tracking.

    Returns:
        MoseleyResult containing series points, linear regression fit, and Cu validation.
    """
    cache_key = (Z_min, Z_max, round(alpha, 5))
    if cache_key in _MOSELEY_CACHE:
        return _MOSELEY_CACHE[cache_key]

    points: list[KAlphaPoint] = []
    z_list = list(range(Z_min, Z_max + 1))
    total_steps = len(z_list)

    cu_kalpha_kev = 0.0
    cu_exp_kev = 8.048

    for idx, Z in enumerate(z_list):
        if progress_cb is not None:
            progress_cb(idx + 1, total_steps)

        res = solve_atom(Z, alpha=alpha)
        eps_1s = res.orbitals[(1, 0)].eps
        eps_2p = res.orbitals[(2, 1)].eps
        k_alpha_ha = eps_2p - eps_1s
        k_alpha_ev = k_alpha_ha * HARTREE_EV
        k_alpha_kev = k_alpha_ev / 1000.0
        # 1 Ha = 2 Ry
        sqrt_e_ry = math.sqrt(k_alpha_ha * 2.0)

        symbol = ELEMENT_SYMBOLS.get(Z, f"Z{Z}")
        point = KAlphaPoint(
            Z=Z,
            symbol=symbol,
            e_kalpha_ev=k_alpha_ev,
            e_kalpha_kev=k_alpha_kev,
            sqrt_e_ry=sqrt_e_ry,
        )
        points.append(point)

        if Z == 29:
            cu_kalpha_kev = k_alpha_kev

    # Moseley linear fit: sqrt(E_Ka / Ry) = slope * Z + intercept
    z_arr = np.array([p.Z for p in points], dtype=float)
    y_arr = np.array([p.sqrt_e_ry for p in points], dtype=float)

    slope, intercept = np.polyfit(z_arr, y_arr, 1)
    y_pred = slope * z_arr + intercept
    ss_res = np.sum((y_arr - y_pred) ** 2)
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 1.0

    cu_err_pct = float(abs(cu_kalpha_kev - cu_exp_kev) / cu_exp_kev * 100.0)

    result = MoseleyResult(
        points=points,
        slope=float(slope),
        intercept=float(intercept),
        r_squared=r2,
        cu_kalpha_kev=cu_kalpha_kev,
        cu_exp_kev=cu_exp_kev,
        cu_err_pct=cu_err_pct,
    )
    _MOSELEY_CACHE[cache_key] = result
    return result
