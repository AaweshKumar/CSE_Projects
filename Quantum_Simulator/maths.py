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




if __name__ == "__main__":
    print(assoc_legendre(0, 0, 0.5))   # should print 1.0
    print(assoc_legendre(1, 0, 0.5))   # should print 0.5
    print(assoc_legendre(1, 1, 0.5))   # should print -0.866
    print(assoc_legendre(2, 0, 0.5))   # should print -0.125