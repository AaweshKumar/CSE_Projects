# Quantum Atom Simulator

A desktop simulator and scientific engine for atomic physics, quantum mechanics, and spectroscopy.

## Overview
The Quantum Atom Simulator provides:
1. **Analytic Hydrogenic Orbitals (P0):** Fully vectorized hydrogenic radial wavefunctions, real spherical harmonics, and a 1D inverse-CDF + 2D sphere rejection sampler in atomic units ($a_0$). Rendered with hardware-accelerated 3D OpenGL point clouds (`pyqtgraph.opengl`).
2. **Animated Superpositions & Dipole Transitions (P1):** Smooth, shimmer-free superposition animation using importance-weighted static points, exact numerical evaluation of electric dipole matrix elements $\langle \psi_i | \vec{r} | \psi_j \rangle$, selection rules, Einstein $A$ coefficients, radiative lifetimes $\tau$, and exact noise-free time-dependent dipole expectation $\langle z \rangle(t)$.
3. **Multi-Electron Self-Consistent Field (SCF) Solver (P2):** Spherically averaged Hartree–Fock–Slater ($X\alpha$, $\alpha=0.7$) central-field solver on a logarithmic radial grid ($r = e^x$), tridiagonal eigensolver (`scipy.linalg.eigh_tridiagonal`), Latter tail asymptotic correction, shell density $D(r) = 4\pi r^2 \rho(r)$, and effective nuclear charge $Z_{\text{eff}}(r)$.
4. **Spectroscopic Validation Suite (P3):** Automated validation against the NIST Atomic Spectra Database 1st ionization energies ($Z=1..36$) via Koopmans' theorem and Slater's transition-state method, and Moseley's law linear regression fit for $K\alpha$ transitions ($Z=11..36$) validated against experimental Copper $K\alpha$ ($8.048\text{ keV}$).
5. **Linear Stark Effect (P4):** Degenerate perturbation theory in the Hydrogen $n=2$ manifold under an external static electric field $\vec{F} = F \hat{z}$ ($0$ to $5\times 10^9\text{ V/m}$), yielding symmetric energy splittings and permanent electric dipole moments $\pm 3 a_0$.

---

## Installation & Requirements

Install required dependencies:
```bash
pip install -r requirements.txt
```

Dependencies:
- `numpy`
- `scipy`
- `matplotlib`
- `PyQt5`
- `pyqtgraph`
- `PyOpenGL`
- `pytest`

---

## Usage

### 1. Graphical User Interface (GUI)
Launch the desktop application:
```bash
python ui.py
```
- **Mode Selector:**
  - `Single orbital (hydrogenic)`: Explore single-electron ion eigenstates ($n, l, m$) with auto-fitting 3D OpenGL camera and 2D radial density plot $r^2 R_{nl}^2(r)$.
  - `Single orbital (multi-electron SCF)`: Inspect subshell orbitals of multi-electron atoms ($Z=1..36, 47, 79$) with shell density $4\pi r^2 \rho(r)$ and $Z_{\text{eff}}(r)$ curves.
  - `Superposition (animated)`: Animate quantum state superpositions with adjustable mixing angle, screen beat period (1–10 s), play/pause, and real-time $\langle z \rangle(t)$ dipole plot.
  - `Stark effect (n=2)`: Apply an external electric field ($0 - 5\times 10^9\text{ V/m}$) to observe linear Stark splitting and polarized electron clouds.
- **Validation Dialog:**
  - Click `VALIDATION PLOTS` to open a multi-tab dialog showing the NIST ionization energy trend, Moseley plot, and statistical error table.

### 2. Command-Line Interface (CLI)
Run the interactive CLI:
```bash
python main.py
```

### 3. Automated Test Suite
Run the test suite across all phases:
```bash
python -m pytest -v
```

---

## Units and Physical Constants
All internal physics calculations are conducted strictly in **Hartree atomic units** ($\hbar = m_e = e = a_0 = 1$, $E_h = 1$). Conversions to SI and practical spectroscopic units:
- $1\text{ Hartree} = 27.211386245988\text{ eV}$
- $1\text{ Rydberg} = 13.605693009\text{ eV}$
- $1\text{ Bohr radius } a_0 = 5.29177210903 \times 10^{-11}\text{ m}$
- $1\text{ a.u. of time} = 2.4188843265857 \times 10^{-17}\text{ s}$
- Fine structure constant $\alpha_{\text{fs}} = 1 / 137.035999084$
- Speed of light $c = 1 / \alpha_{\text{fs}} \approx 137.036\text{ a.u.}$
- $1\text{ a.u. of electric field} = 5.14220674763 \times 10^{11}\text{ V/m}$

---

## LIMITATIONS

1. **One-Electron Exactness:**
   Hydrogenic mode is mathematically exact **only** for genuine one-electron systems (neutral Hydrogen $H$, singly-ionized Helium $He^+$, doubly-ionized Lithium $Li^{2+}$, etc.). For $Z > 1$, selecting an element in hydrogenic mode models the stripped one-electron ion (e.g. $Fe^{25+}$, $Au^{78+}$), not the neutral atom.

2. **SCF Central-Field Approximations:**
   The multi-electron solver implements the spherically averaged Hartree–Fock–Slater ($X\alpha$, $\alpha=0.7$) central-field model with Latter asymptotic tail correction and fractional subshell occupancies. It is not full Hartree–Fock (non-local exchange is replaced by Slater's local $\rho^{1/3}$ exchange) nor full density functional theory (no gradient corrections or correlation functionals). Furthermore, individual atomic orbitals in multi-electron systems are single-particle theoretical constructs (Slater determinants); only the total charge density $\rho(\vec{r})$ is a true quantum-mechanical observable.

3. **Relativistic & Spin Limitations:**
   The Hamiltonian is non-relativistic and spin-unpolarized:
   - Relativistic mass-velocity and Darwin contractions (which significantly contract the $6s$ orbital in heavy elements like Gold, giving gold its characteristic yellow color and noble chemical inertia) are omitted.
   - Spin-orbit coupling is omitted; fine structure splitting and the $K\alpha_1$ / $K\alpha_2$ X-ray doublet splitting are therefore not resolved.
   - Spin polarization (unrestricted exchange) is omitted; while subshell closings ($Be \to B$, $Mg \to Al$) are reproduced, the Hund's rule exchange dip from Nitrogen to Oxygen ($N \to O$) requires spin-polarized density and is not captured.

4. **Time Scaling in Superposition Animation:**
   Real atomic electronic transition periods occur on the order of hundreds of attoseconds to a few femtoseconds (e.g. $H(1s \leftrightarrow 2p)$ has $T \approx 0.405\text{ fs}$). The simulation slows real time down by a factor of approximately $10^{14}$ to $10^{15}$ so that oscillations can be observed on human timescales (seconds). Point brightness directly encodes the dynamic importance weight $w(\vec{r}, t) = \rho(\vec{r}, t) / q(\vec{r})$.

---

## References
- Griffiths, D. J., *Introduction to Quantum Mechanics*, 3rd ed. (hydrogen wavefunctions, perturbation theory, electric dipole selection rules).
- Slater, J. C., *Phys. Rev.* 81, 385 (1951): A Simplification of the Hartree-Fock Method ($X\alpha$ exchange).
- Slater, J. C., *Adv. Quantum Chem.* 6, 1 (1972): Statistical Exchange-Correlation in the Theory of Molecules and Crystals (Transition State method).
- Latter, R., *Phys. Rev.* 99, 510 (1955): Atomic Energy Levels for the Thomas-Fermi and Thomas-Fermi-Dirac Potential (Asymptotic tail correction).
- Herman, F., and Skillman, S., *Atomic Structure Calculations*, Prentice-Hall (1963): Central-field logarithmic grid techniques.
- NIST Atomic Spectra Database (Ionization Energies); NIST X-Ray Transition Energies Database ($K\alpha$).
