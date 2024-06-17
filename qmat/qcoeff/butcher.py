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
.. [4] Butcher, J.C. (2003). Numerical methods for Ordinary Differential
       Equations. John Wiley & Sons.
"""
import numpy as np

from qmat.qcoeff import QGenerator, register
from qmat.utils import storeClass


class RK(QGenerator):
    A = None
    b = None
    c = None
    b2 = None  # for embedded methods

    @property
    def nodes(self): return self.c

    @property
    def weights(self): return self.b

    @property
    def weightsEmbedded(self):
        if self.b2 is None:
            raise NotImplementedError(f'kindly direct your request for an embedded version of {type(self).__name__!r} to the Mermathematicians on Europa.')
        else:
            return self.b2

    @property
    def Q(self): return self.A


RK_SCHEMES = {}

def checkAndStore(cls:RK)->RK:
    cls.A = np.array(cls.A, dtype=float)
    cls.b = np.array(cls.b, dtype=float)
    cls.c = np.array(cls.c, dtype=float)

    if cls.b2 is not None:
        cls.b2 = np.array(cls.b2, dtype=float)

    assert cls.b.shape[-1] == cls.c.size, \
        f"b (size {cls.b.shape[-1]}) and c (size {cls.c.size})" + \
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

    @property
    def order(self): return 1


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

    @property
    def order(self): return 4

    @property
    def hCoeffs(self): return np.array([0, 0, 0, 1], dtype=float)


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

    @property
    def order(self): return 4


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

    @property
    def order(self): return 3


@registerRK
class RK21(RK):
    """Explicit Runge-Kutta in 2 steps of order 1 from Wang & Spiteri [1]"""
    aliases = ["ERK21"]
    A = [[0, 0],
         [3/4, 0]]
    b = [-1/3, 4/3]
    c = [0, 3/4]

    @property
    def order(self): return 1


@registerRK
class RK2(RK):
    """Classical Runge-Kutta method of order 2 (cf Wikipedia)"""
    aliases = ["ERK2"]
    A = [[0, 0],
         [1/2, 0]]
    b = [0, 1]
    c = [0, 1/2]

    @property
    def order(self): return 2


@registerRK
class HEUN2(RK):
    """Heun method of order 2 (cf Wikipedia)"""
    aliases = ["HEUN"]
    A = [[0, 0],
         [1, 0]]
    b = [1/2, 1/2]
    c = [0, 1.]

    @property
    def order(self): return 2


@registerRK
class RK32(RK):
    """Explicit Runge-Kutta in 3 steps of order 2 from Wang & Spiteri [1]"""
    aliases = ["ERK32", "RK32-SSP"]
    A = [[0, 0, 0],
         [1/3, 0, 0],
         [0, 1, 0]]
    b = [1/2, 0, 1/2]
    c = [0, 1/3, 1]

    @property
    def order(self): return 3  # TODO: Dahlquist order is 3 actually ...


@registerRK
class RK33(RK):
    """Explicit Runge-Kutta in 3 steps of order 3 from Wang & Spiteri [1]"""
    aliases = ["ERK33", "RK33-SSP"]
    A = [[0, 0, 0],
         [1, 0, 0],
         [1/4, 1/4, 0]]
    b = [1/6, 1/6, 2/3]
    c = [0, 1, 1/2]

    @property
    def order(self): return 3


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

    @property
    def order(self): return 5

    @property
    def hCoeffs(self): return np.array([0, 0, 0, 0, 0, 1], dtype=float)


@registerRK
class BE(RK):
    """Backward Euler method (also SDIRK1, see [2])"""
    aliases = ["IE"]
    A = [[1]]
    b = [1]
    c = [1]

    @property
    def order(self): return 1


@registerRK
class TRAP(RK):
    """Trapeze method (cf Wikipedia)"""
    aliases = ["TRAPZ", "CN"]
    A = [[0, 0],
         [1/2, 1/2]]
    b = [1/2, 1/2]
    c = [0, 1]

    @property
    def order(self): return 2


@registerRK
class GAUSS_LG(RK):
    """Gauss-Legendre method of order 4 (cf Wikipedia)"""
    aliases = ["GAUSS-LG"]
    A = [[0.25, 0.25-1/6*3**(0.5)],
         [0.25+1/6*3**(0.5), 0.25]]
    b = [0.5, 0.5]
    c = [0.5-1/6*3**(0.5), 0.5+1/6*3**(0.5)]

    @property
    def order(self): return 4


@registerRK
class SDIRK2(RK):
    """First S-stable Diagonally Implicit Runge Kutta method of order 2 in two stages,
    from Alexander [2]"""
    A = [[1-2**0.5/2, 0],
         [2**0.5/2, 1-2**0.5/2]]
    b = [2**0.5/2, 1-2**0.5/2]
    c = [1-2**0.5/2, 1.]

    @property
    def order(self): return 2


@registerRK
class SDIRK2_2(RK):
    """Second S-stable Diagonally Implicit Runge Kutta method of order 2 in two stages,
    from Alexander [2]"""
    aliases = ["SDIRK2-2"]
    A = [[1+2**0.5/2, 0],
         [-2**0.5/2, 1+2**0.5/2]]
    b = [-2**0.5/2, 1+2**0.5/2]
    c = [1+2**0.5/2, 1]

    # Has a very high error constant, need very small time-steps to see the order ...
    CONV_TEST_NSTEPS = [64, 128, 256]

    @property
    def order(self): return 2


@registerRK
class SDIRK3(RK):
    """S-stable Diagonally Implicit Runge Kutta method of order 3 in three stages,
    from Alexander [2]"""
    A = [[0.43586652150845967, 0, 0],
         [0.28206673924577014, 0.43586652150845967, 0],
         [1.2084966491760119, -0.6443631706844715, 0.43586652150845967]]
    b = [1.2084966491760119, -0.6443631706844715, 0.43586652150845967]
    c = [0.43586652150845967, 0.7179332607542298, 1.]

    @property
    def order(self): return 3


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

    @property
    def order(self): return 4

@registerRK
class HeunEuler(RK):
    """
    Second order explicit embedded Runge-Kutta method.
    """
    A = [[0, 0],
         [1, 0]]
    b = [0.5, 0.5]
    c = [0, 1]
    b2 = [1, 0]

    @property
    def order(self): return 2

@registerRK
class CashKarp(RK):
    """
    Fifth order explicit embedded Runge-Kutta. See [here](https://doi.org/10.1145/79505.79507).
    """
    c = [0, 0.2, 0.3, 0.6, 1.0, 7.0 / 8.0]
    b = [37.0 / 378.0, 0.0, 250.0 / 621.0, 125.0 / 594.0, 0.0, 512.0 / 1771.0]
    b2 = [2825.0 / 27648.0, 0.0, 18575.0 / 48384.0, 13525.0 / 55296.0, 277.0 / 14336.0, 1.0 / 4.0]
    A = np.zeros((6, 6))
    A[1, 0] = 1.0 / 5.0
    A[2, :2] = [3.0 / 40.0, 9.0 / 40.0]
    A[3, :3] = [0.3, -0.9, 1.2]
    A[4, :4] = [-11.0 / 54.0, 5.0 / 2.0, -70.0 / 27.0, 35.0 / 27.0]
    A[5, :5] = [1631.0 / 55296.0, 175.0 / 512.0, 575.0 / 13824.0, 44275.0 / 110592.0, 253.0 / 4096.0]

    @property
    def order(self): return 5

    CONV_TEST_NSTEPS = [32, 64, 128]

@registerRK
class ESDIRK53(RK):
    """
    A-stable embedded RK pair of orders 5 and 3, ESDIRK5(3)6L[2]SA.
    Taken from [here](https://ntrs.nasa.gov/citations/20160005923).
    """
    c = [0, 4024571134387.0 / 7237035672548.0, 14228244952610.0 / 13832614967709.0, 1.0 / 10.0, 3.0 / 50.0, 1.0]
    A = np.zeros((6, 6))
    A[1, :2] = [3282482714977.0 / 11805205429139.0, 3282482714977.0 / 11805205429139.0]
    A[2, :3] = [
        606638434273.0 / 1934588254988,
        2719561380667.0 / 6223645057524,
        3282482714977.0 / 11805205429139.0,
    ]
    A[3, :4] = [
        -651839358321.0 / 6893317340882,
        -1510159624805.0 / 11312503783159,
        235043282255.0 / 4700683032009.0,
        3282482714977.0 / 11805205429139.0,
    ]
    A[4, :5] = [
        -5266892529762.0 / 23715740857879,
        -1007523679375.0 / 10375683364751,
        521543607658.0 / 16698046240053.0,
        514935039541.0 / 7366641897523.0,
        3282482714977.0 / 11805205429139.0,
    ]
    A[5, :] = [
        -6225479754948.0 / 6925873918471,
        6894665360202.0 / 11185215031699,
        -2508324082331.0 / 20512393166649,
        -7289596211309.0 / 4653106810017.0,
        39811658682819.0 / 14781729060964.0,
        3282482714977.0 / 11805205429139,
    ]

    b = [
            -6225479754948.0 / 6925873918471,
            6894665360202.0 / 11185215031699.0,
            -2508324082331.0 / 20512393166649,
            -7289596211309.0 / 4653106810017,
            39811658682819.0 / 14781729060964.0,
            3282482714977.0 / 11805205429139,
        ]
    b2 = [
           -2512930284403.0 / 5616797563683,
           5849584892053.0 / 8244045029872,
           -718651703996.0 / 6000050726475.0,
           -18982822128277.0 / 13735826808854.0,
           23127941173280.0 / 11608435116569.0,
           2847520232427.0 / 11515777524847.0,
        ]

    @property
    def orderEmbedded(self): return 3

    @property
    def order(self): return 5
