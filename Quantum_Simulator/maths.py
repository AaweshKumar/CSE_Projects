import math
import numpy as np

#testing repo
def fac(n): #factorials used later for wavefunction normalization
    return math.factorial(n)

def associated_laguerre(p, q, x): 
    #| `p` | degree | = n - l - 1, number of radial nodes in the orbital |
    #| `q` | order | = 2l + 1, related to angular momentum of the electron |
    #| `x` | argument | = ρ (rho), the scaled distance from nucleus = 2Zr/na₀ |
    if p == 0:
        return 1
    if p == 1:
        return 1 + q - x
    L0 = 1
    L1 = 1 + q - x
    for i in range(2, p + 1):
        L2 = ((2*i + q - 1 - x) * L1 - (i + q - 1) * L0) / i
        L0 = L1
        L1 = L2
    return L1


def assoc_legendre(l, m, x):
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
        pll = (x * (2*ll - 1) * pmmp1 - (ll + absm - 1) * pmm) / (ll - absm)
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
    q = 2*l + 1

    # --- Step 3: Normalization constant ---
    normalization = math.sqrt(
        ((2 * Z) / (n * a0))**3 *
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
        (2*l + 1) / (4 * math.pi) *
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
    return R**2 * Y**2

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
        p = probability_density(r, math.pi/2, 0, n, l, m, Z) * r**2
        if p > max_prob:
            max_prob = p
    max_prob *= 4  # safety margin

    points = []
    attempts = 0
    max_attempts = num_points * 100

    while len(points) < num_points and attempts < max_attempts:
        attempts += 1

        # Pick random spherical coordinates
        r     = np.random.uniform(0, max_r)
        theta = np.arccos(2 * np.random.uniform(0, 1) - 1)  # uniform on sphere
        phi   = np.random.uniform(0, 2 * math.pi)

        # Calculate probability at this point
        prob = probability_density(r, theta, phi, n, l, m, Z) * r**2

        # Rejection test
        if np.random.uniform(0, max_prob) < prob:
            x, y, z = to_cartesian(r, theta, phi)
            points.append([x, y, z, prob])

    return np.array(points)

def get_energy(n, Z=1):    #Energy of hydrogen-like atom at level n.
    return -13.6 * (Z**2) / (n**2)


