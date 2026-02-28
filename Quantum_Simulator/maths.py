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




