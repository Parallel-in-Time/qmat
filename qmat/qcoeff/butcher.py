#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Submodule to generate Q matrices based on Butcher tables

References
----------
.. [1] Wang, R., & Spiteri, R. J. (2007). Linear instability of the fifth-order
       WENO method. SIAM Journal on Numerical Analysis, 45(5), 1871-1901.
.. [2] Alexander, R. (1977). Diagonally implicit Runge–Kutta methods for stiff
       ODE’s. SIAM Journal on Numerical Analysis, 14(6), 1006-1021.
.. [3] Wanner, G., & Hairer, E. (1996). Solving ordinary differential equations
       II. Springer Berlin Heidelberg.
.. [4] Butcher, J.C. (2003). Numerical methods for Ordianry Differential
       Equations. John Wiley & Sons.
"""
import numpy as np

from qmat.qcoeff import QGenerator, register
from qmat.utils import storeClass


class RK(QGenerator):
    A = None
    b = None
    c = None
    order = None

    @property
    def nodes(self): return self.c

    @property
    def weights(self): return self.b

    @property
    def Q(self): return self.A


RK_SCHEMES:dict[str:RK] = {}

def checkAndStore(cls:RK)->RK:
    cls.A = np.array(cls.A, dtype=float)
    cls.b = np.array(cls.b, dtype=float)
    cls.c = np.array(cls.c, dtype=float)
    assert cls.b.size == cls.c.size, \
        f"b (size {cls.b.size}) and c (size {cls.b.size})" + \
        f" have not the same size in {cls.__name__}"
    assert cls.A.shape == (cls.c.size, cls.c.size), \
        f"A (shape {cls.A.shape}) and c (size {cls.b.size})" + \
        f" have inconsistent dimensions in {cls.__name__}"
    assert cls.order is not None, \
        f"order not defined for {cls.__name__}"
    storeClass(cls, RK_SCHEMES)
    return cls

def registerRK(cls:RK)->RK:
    return register(checkAndStore(cls))

@registerRK
class FE(RK):
    """Forward Euler method (cf Wikipedia)"""
    aliases = ["EE"]
    A = [[0]]
    b = [1]
    c = [0]
    order = 1

@registerRK
class RK4(RK):
    """Classical Runge Kutta method of order 4 (cf Wikipedia)"""
    aliases = ["ERK4"]
    A = [[0, 0, 0, 0],
         [0.5, 0, 0, 0],
         [0, 0.5, 0, 0],
         [0, 0, 1, 0]]
    b = [1/6, 1/3, 1/3, 1/6]
    c = [0, 1/2, 1/2, 1]
    order = 4

@registerRK
class RK4_38(RK):
    """The 3/8-rule due to Kutta, order 4  (cf Wikipedia)"""
    aliases = ["ERK4_38"]
    A = [[0, 0, 0, 0],
         [1/3, 0, 0, 0],
         [-1/3, 1, 0, 0],
         [1, -1, 1, 0]]
    b = [1/8, 3/8, 3/8, 1/8]
    c = [0, 1/3, 2/3, 1]
    order = 4

@registerRK
class RK53(RK):
    """Explicit Runge-Kutta in 5 steps of order 3 from Wang & Spiteri [1]"""
    aliases = ["ERK53"]
    A = [[0, 0, 0, 0, 0],
         [1/7, 0, 0, 0, 0],
         [0, 3/13, 0, 0, 0],
         [0, 0, 1/3, 0, 0],
         [0, 0, 0, 2/3, 0]]
    b = [1/4, 0, 0, 0, 3/4]
    c = [0, 1/7, 3/16, 1/3, 2/3]
    order = 3

@registerRK
class RK21(RK):
    """Explicit Runge-Kutta in 2 steps of order 1 from Wang & Spiteri [1]"""
    aliases = ["ERK21"]
    A = [[0, 0],
         [3/4, 0]]
    b = [-1/3, 4/3]
    c = [0, 3/4]
    order = 1

@registerRK
class RK2(RK):
    """Classical Runge Kutta method of order 2 (cf Wikipedia)"""
    aliases = ["ERK2"]
    A = [[0, 0],
         [1/2, 0]]
    b = [0, 1]
    c = [0, 1/2]
    order = 2

@registerRK
class HEUN2(RK):
    """Heun method of order 2 (cf Wikipedia)"""
    aliases = ["HEUN"]
    A = [[0, 0],
         [1, 0]]
    b = [1/2, 1/2]
    c = [0, 1.]
    order = 2

@registerRK
class RK32(RK):
    """Explicit Runge-Kutta in 3 steps of order 2 from Wang & Spiteri [1]"""
    aliases = ["ERK32", "RK32-SSP"]
    A = [[0, 0, 0],
         [1/3, 0, 0],
         [0, 1, 0]]
    b = [1/2, 0, 1/2]
    c = [0, 1/3, 1]
    order = 2

@registerRK
class RK33(RK):
    """Explicit Runge-Kutta in 3 steps of order 3 from Wang & Spiteri [1]"""
    aliases = ["ERK33", "RK33-SSP"]
    A = [[0, 0, 0],
         [1, 0, 0],
         [1/4, 1/4, 0]]
    b = [1/6, 1/6, 2/3]
    c = [0, 1, 1/2]
    order = 3

@registerRK
class RK65(RK):
    """Explicit Runge-Kutta in 6 steps of order 5, (236a) from Butcher [4]"""
    aliases = ["ERK65"]
    A = [[0, 0, 0, 0, 0, 0],
         [0.25, 0, 0, 0, 0, 0],
         [1/8, 1./8, 0, 0, 0, 0],
         [0, 0, 0.5, 0, 0, 0],
         [3/16, -3/8, 3/8, 9/16, 0, 0],
         [-3/7, 8/7, 6/7, -12/7, 8/7, 0]]
    b = [7/90, 0, 32/90, 12/90, 32/90, 7/90]
    c = [0, 0.25, 0.25, 0.5, 0.75, 1]
    order = 5

@registerRK
class BE(RK):
    """Backward Euler method (also SDIRK1, see [2])"""
    aliases = ["IE"]
    A = [[1]]
    b = [1]
    c = [1]
    order = 1

@registerRK
class TRAP(RK):
    """Trapeze method (cf Wikipedia)"""
    aliases = ["TRAPZ", "CN"]
    A = [[0, 0],
         [1/2, 1/2]]
    b = [1/2, 1/2]
    c = [0, 1]
    order = 2

@registerRK
class GAUSS_LG(RK):
    """Gauss-Legendre method of order 4 (cf Wikipedia)"""
    aliases = ["GAUSS-LG"]
    A = [[0.25, 0.25-1/6*3**(0.5)],
         [0.25+1/6*3**(0.5), 0.25]]
    b = [0.5, 0.5]
    c = [0.5-1/6*3**(0.5), 0.5+1/6*3**(0.5)]
    order = 4

@registerRK
class SDIRK2(RK):
    """First S-stable Diagonally Implicit Runge Kutta method of order 2 in two stages, 
    from Alexander [2]"""
    A = [[1-2**0.5/2, 0],
         [2**0.5/2, 1-2**0.5/2]]
    b = [2**0.5/2, 1-2**0.5/2]
    c = [1-2**0.5/2, 1.]
    order = 2

@registerRK
class SDIRK2_2(RK):
    """Second S-stable Diagonally Implicit Runge Kutta method of order 2 in two stages, 
    from Alexander [2]"""
    aliases = ["SDIRK2-2"]
    A = [[1+2**0.5/2, 0],
         [-2**0.5/2, 1+2**0.5/2]]
    b = [-2**0.5/2, 1+2**0.5/2]
    c = [1+2**0.5/2, 1]
    order = 2

@registerRK
class SDIRK3(RK):
    """S-stable Diagonally Implicit Runge Kutta method of order 3 in three stages, 
    from Alexander [2]"""
    A = [[0.43586652150845967, 0, 0],
         [0.28206673924577014, 0.43586652150845967, 0],
         [1.2084966491760119, -0.6443631706844715, 0.43586652150845967]]
    b = [1.2084966491760119, -0.6443631706844715, 0.43586652150845967]
    c = [0.43586652150845967, 0.7179332607542298, 1.]
    order = 3

@registerRK
class SDIRK54(RK):
    """S-stable Diagonally Implicit Runge Kutta method of order 4 in five stages, 
    from Wanner and Hairer [3]"""
    A = [[1/4, 0, 0, 0, 0],
         [1/2, 1/4, 0, 0, 0],
         [17/50, -1/25, 1/4, 0, 0],
         [371/1360, -137/2720, 15/544, 1/4, 0],
         [25/24, -49/48, 125/16, -85/12, 1/4]]
    b = [25/24, -49/48, 125/16, -85/12, 1/4]
    c = [1/4, 3/4, 11/20, 1/2, 1]
    order = 4
