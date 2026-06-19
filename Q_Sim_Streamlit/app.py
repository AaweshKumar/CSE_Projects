import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


# ════════════════════════════════════════════════════════════════
#  PHYSICS / MATH  (from maths.py — unchanged)
# ════════════════════════════════════════════════════════════════

def fac(n):  # factorials used later for wavefunction normalization
    return math.factorial(n)


def associated_laguerre(p, q, x):
    # | `p` | degree | = n - l - 1, number of radial nodes in the orbital |
    # | `q` | order  | = 2l + 1, related to angular momentum of the electron |
    # | `x` | argument | = ρ (rho), the scaled distance from nucleus = 2Zr/na₀ |
    if p == 0:
        return 1
    if p == 1:
        return 1 + q - x
    L0 = 1
    L1 = 1 + q - x
    for i in range(2, p + 1):
        L2 = ((2 * i + q - 1 - x) * L1 - (i + q - 1) * L0) / i
        L0 = L1
        L1 = L2
    return L1


def associated_legendre(l, m, x):
    """
    Associated Legendre polynomial P_l^|m|(x)

    l = angular quantum number (orbital shape)
    m = magnetic quantum number (orientation)
    x = cos(theta), polar angle argument, range -1 to 1

    Describes the angular SHAPE of the orbital.
    """
    absm = abs(m)  # always use |m|, sign handled in spherical harmonic
    pmm = 1.0
    if absm > 0:
        somx2 = math.sqrt((1 - x) * (1 + x))  # = sin(theta)
        fact = 1.0
        for i in range(1, absm + 1):
            pmm *= -fact * somx2
            fact += 2

    # If l == m we're done
    if l == absm:
        return pmm

    # One step up: P_(m+1)^m
    pmmp1 = x * (2 * absm + 1) * pmm
    if l == absm + 1:
        return pmmp1

    # Recurse up to l using recurrence relation
    pll = 0.0
    for ll in range(absm + 2, l + 1):
        pll = (x * (2 * ll - 1) * pmmp1 - (ll + absm - 1) * pmm) / (ll - absm)
        pmm = pmmp1
        pmmp1 = pll

    return pll


def radial_wavefunction(n, l, r, Z=1):
    """
    Hydrogen-like radial wavefunction R_{n,l}(r)

    Parameters
    ----------
    n : principal quantum number (1,2,3,...)
    l : angular quantum number (0 ≤ l < n)
    r : radial distance from nucleus (in meters)
    Z : atomic number (default = 1 for hydrogen)

    Returns
    -------
    R : value of radial wavefunction at distance r
    """

    # --- Basic physical validation ---
    if n <= 0:
        raise ValueError("n must be positive")
    if l < 0 or l >= n:
        raise ValueError("l must satisfy 0 <= l < n")

    # --- Physical constant ---
    a0 = 5.29177210903e-11  # Bohr radius (meters)

    # --- Step 1: Dimensionless radial variable ---
    rho = (2 * Z * r) / (n * a0)

    # --- Step 2: Laguerre parameters ---
    p = n - l - 1
    q = 2 * l + 1

    # --- Step 3: Normalization constant ---
    normalization = math.sqrt(
        ((2 * Z) / (n * a0)) ** 3 *
        math.factorial(p) /
        (2 * n * math.factorial(n + l))
    )

    # --- Step 4: Associated Laguerre polynomial ---
    L = associated_laguerre(p, q, rho)

    # --- Step 5: Assemble radial function ---
    R = normalization * math.exp(-rho / 2) * (rho ** l) * L

    return R


def spherical_harmonic(l, m, theta, phi):
    """
    Real spherical harmonic Y_lm(theta, phi)
    l : angular quantum number (shape)
    m : magnetic quantum number (orientation)
    theta : polar angle from z-axis (0 to pi)
    phi: azimuthal angle around z-axis (0 to 2pi)

    Returns the angular part of the wavefunction.
    """
    absm = abs(m)

    # Normalization constant
    norm = math.sqrt(
        (2 * l + 1) / (4 * math.pi) *
        math.factorial(l - absm) /
        math.factorial(l + absm)
    )

    P = associated_legendre(l, m, math.cos(theta))

    if m == 0:
        return norm * P
    elif m > 0:
        return math.sqrt(2) * norm * P * math.cos(m * phi)
    else:
        return math.sqrt(2) * norm * P * math.sin(absm * phi)


def probability_density(r, theta, phi, n, l, m, Z=1):
    """
    Full probability density |psi|^2 at point (r, theta, phi)

    r     : radial distance from nucleus
    theta : polar angle
    phi   : azimuthal angle
    n,l,m : quantum numbers
    Z     : atomic number

    Returns |R_nl|^2 * |Y_lm|^2
    """
    R = radial_wavefunction(n, l, r, Z)
    Y = spherical_harmonic(l, m, theta, phi)
    return R ** 2 * Y ** 2


def to_cartesian(r, theta, phi):
    """
    Convert spherical coordinates to cartesian.

    r     : radial distance
    theta : polar angle (0 to pi)
    phi   : azimuthal angle (0 to 2pi)

    Returns (x, y, z)
    """
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)
    return x, y, z


def generate_point_cloud(n, l, m, Z=1, num_points=3000):
    """
    Generates a point cloud by rejection sampling the probability density.

    Returns numpy array of shape (num_points, 4) -> [x, y, z, probability]
    """
    a0 = 5.29177210903e-11
    max_r = n * n * 6 * a0  # sensible upper bound based on n

    # Find approximate max probability for rejection sampling
    max_prob = 0
    for i in range(1000):
        r = (i / 1000) * max_r
        p = probability_density(r, math.pi / 2, 0, n, l, m, Z) * r ** 2
        if p > max_prob:
            max_prob = p
    max_prob *= 4  # safety margin

    points = []
    attempts = 0
    max_attempts = num_points * 100

    while len(points) < num_points and attempts < max_attempts:
        attempts += 1

        # Pick random spherical coordinates
        r = np.random.uniform(0, max_r)
        theta = np.arccos(2 * np.random.uniform(0, 1) - 1)  # uniform on sphere
        phi = np.random.uniform(0, 2 * math.pi)

        # Calculate probability at this point
        prob = probability_density(r, theta, phi, n, l, m, Z) * r ** 2

        # Rejection test
        if np.random.uniform(0, max_prob) < prob:
            x, y, z = to_cartesian(r, theta, phi)
            points.append([x, y, z, prob])

    return np.array(points)


def get_energy(n, Z=1):  # Energy of hydrogen-like atom at level n.
    return -13.6 * (Z ** 2) / (n ** 2)


# Cached wrapper around the expensive rejection-sampling routine so that
# Streamlit doesn't regenerate the cloud on every unrelated UI interaction.
@st.cache_data(show_spinner=False)
def cached_point_cloud(n, l, m, Z, num_points):
    return generate_point_cloud(n, l, m, Z, num_points)


# ════════════════════════════════════════════════════════════════
#  STATIC DATA  (from ui.py — unchanged)
# ════════════════════════════════════════════════════════════════

ELEMENTS = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B',
    6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
    16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca',
    21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn',
    26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
    31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br',
    36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr',
    41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh',
    46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs',
    56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd',
    61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb',
    66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb',
    71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re',
    76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
    81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At',
    86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th',
    91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am',
    96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm',
    101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db',
    106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt', 110: 'Ds',
    111: 'Rg', 112: 'Cn', 113: 'Nh', 114: 'Fl', 115: 'Mc',
    116: 'Lv', 117: 'Ts', 118: 'Og'
}

ORBITAL_SHAPES = {
    0: 'Sphere (s)',
    1: 'Dumbbell (p)',
    2: 'Clover (d)',
    3: 'Complex (f)'
}

ORBITAL_LETTERS = 'spldf'


# ════════════════════════════════════════════════════════════════
#  RENDERING  (from renderer.py / ui.py render_orbital — unchanged
#  math/logic, plt.show() swapped for returning a figure object)
# ════════════════════════════════════════════════════════════════

def build_figure(n, l, m, Z, num_points):
    cloud = cached_point_cloud(n, l, m, Z, num_points)

    # Split cloud array into coordinates and probability
    x = cloud[:, 0] * Z   # scale by Z so shape stays visible
    y = cloud[:, 1] * Z
    z = cloud[:, 2] * Z
    probability = cloud[:, 3]

    # Normalize probability to 0-1 so colormap understands it
    prob_normalized = (probability - probability.min()) / (probability.max() - probability.min() + 1e-10)
    colors = plt.cm.plasma(prob_normalized)

    # Create dark figure and 3D plot area
    figure = plt.figure(figsize=(8, 7))
    figure.patch.set_facecolor('#050a10')
    axes = figure.add_subplot(111, projection='3d')
    axes.set_facecolor('#050a10')

    # Draw electron probability cloud and nucleus
    axes.scatter(x, y, z, c=colors, s=2, alpha=0.5, linewidths=0)
    axes.scatter([0], [0], [0], color='white', s=100, zorder=10)

    # Title
    orbital_name = f"{n}{ORBITAL_LETTERS[l]}"
    element_symbol = ELEMENTS.get(Z, f"Z={Z}")
    energy = get_energy(n, Z)
    axes.set_title(
        f"{element_symbol}  |  {orbital_name}  |  n={n}, l={l}, m={m}  |  E={energy:.2f} eV",
        color='white', fontsize=11, pad=15
    )

    # Style grid and axes
    for pane in [axes.xaxis.pane, axes.yaxis.pane, axes.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#0a1a2a')
    axes.tick_params(axis='x', colors='#223344')
    axes.tick_params(axis='y', colors='#223344')
    axes.tick_params(axis='z', colors='#223344')

    figure.tight_layout()
    return figure


# ════════════════════════════════════════════════════════════════
#  STREAMLIT UI  (replaces ui.py's PyQt5 QuantumSimulatorApp)
# ════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Quantum Atom Simulator", layout="wide")

st.title("Quantum Atom Simulator")
st.caption(
    "Monte Carlo / rejection-sampled electron probability cloud, "
    "built from the hydrogen-like Schrödinger solution."
)

with st.sidebar:
    st.header("Controls")

    # --- Element dropdown ---
    element_options = [f"{symbol}  (Z={z})" for z, symbol in ELEMENTS.items()]
    element_index = st.selectbox(
        "Element",
        options=range(len(element_options)),
        format_func=lambda i: element_options[i],
        index=0,
    )
    Z = list(ELEMENTS.keys())[element_index]

    st.subheader("Quantum Numbers")

    # --- n (principal) ---
    n = st.slider("n (principal)", min_value=1, max_value=5, value=1, step=1)

    # --- l (angular), constrained to 0..n-1 ---
    l = st.slider("l (angular)", min_value=0, max_value=max(n - 1, 0), value=0, step=1)

    # --- m (magnetic), constrained to -l..l ---
    m = st.slider("m (magnetic)", min_value=-l, max_value=l, value=0, step=1) if l > 0 else 0

    # --- Sample size ---
    num_points = st.slider("Sample points", min_value=1000, max_value=15000, value=8000, step=1000)

    render_clicked = st.button("RENDER ORBITAL", use_container_width=True)

    st.subheader("Orbital Info")
    orbital_name = f"{n}{ORBITAL_LETTERS[l]}"
    element_symbol = ELEMENTS.get(Z, f"Z={Z}")
    energy = get_energy(n, Z)
    shape = ORBITAL_SHAPES.get(l, '?')
    radial_nodes = n - l - 1
    angular_nodes = l

    st.code(
        f"Orbital : {element_symbol} {orbital_name}\n"
        f"Energy  : {energy:.2f} eV\n"
        f"Shape   : {shape}\n"
        f"Radial nodes  : {radial_nodes}\n"
        f"Angular nodes : {angular_nodes}",
        language=None,
    )

with st.spinner("Generating point cloud..."):
    fig = build_figure(n, l, m, Z, num_points)

st.pyplot(fig, use_container_width=True)
