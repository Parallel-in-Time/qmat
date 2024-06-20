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

    @property
    def hCoeffs(self):
        try:
            return super().hCoeffs
        except AssertionError:
            hCoeffs = np.zeros_like(self.c)
            hCoeffs[-1] = 1
            return hCoeffs


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


# -----------------------------------------------------------------------------
# Explicit schemes
# -----------------------------------------------------------------------------

# ---------------------------------- Order 1 ----------------------------------
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
class RK21(RK):
    """Explicit Runge-Kutta in 2 steps of order 1 from Wang & Spiteri [1]"""
    aliases = ["ERK21"]
    A = [[0, 0],
         [3/4, 0]]
    b = [-1/3, 4/3]
    c = [0, 3/4]

    @property
    def order(self): return 1


# ---------------------------------- Order 2 ----------------------------------
@registerRK
class RK2(RK):
    """Classical Runge-Kutta method of order 2 (cf Wikipedia)"""
    aliases = ["ERK2", "ExplicitMidPoint", "EMP"]
    A = [[0, 0],
         [1/2, 0]]
    b = [0, 1]
    c = [0, 1/2]

    @property
    def order(self): return 2


@registerRK
class HEUN2(RK):
    """Heun method of order 2 (cf Wikipedia)"""
    aliases = ["HEUN", "HeunEuler"]
    A = [[0, 0],
         [1, 0]]
    b = [1/2, 1/2]
    c = [0, 1.]
    b2 = [1, 0]

    @property
    def order(self): return 2


# ---------------------------------- Order 3 ----------------------------------
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


# ---------------------------------- Order 4 ----------------------------------
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


# ---------------------------------- Order 5 ----------------------------------
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


@registerRK
class CashKarp(RK):
    """
    Fifth order explicit embedded Runge-Kutta. See [here](https://doi.org/10.1145/79505.79507).
    """
    aliases = ["Cash_Karp"]

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


# -----------------------------------------------------------------------------
# Implicit schemes
# -----------------------------------------------------------------------------

# ---------------------------------- Order 1 ----------------------------------
@registerRK
class BE(RK):
    """Backward Euler method (also SDIRK1, see [2])"""
    aliases = ["IE"]
    A = [[1]]
    b = [1]
    c = [1]

    @property
    def order(self): return 1


# ---------------------------------- Order 2 ----------------------------------
@registerRK
class MidPoint(RK):
    """Implicit Mid-Point Rule, see Wikipedia"""
    aliases = ["IMP", "ImplicitMidPoint"]
    A = [[1/2]]
    b = [1]
    c = [1/2]

    @property
    def order(self): return 2


@registerRK
class TRAP(RK):
    """Trapeze method (cf Wikipedia)"""
    aliases = ["TRAPZ", "CN", "CrankNicholson"]
    A = [[0, 0],
         [1/2, 1/2]]
    b = [1/2, 1/2]
    c = [0, 1]

    @property
    def order(self): return 2


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

    @property
    def order(self): return 2

    # Has a very high error constant, need very small time-steps to see the order ...
    CONV_TEST_NSTEPS = [64, 128, 256]


# ---------------------------------- Order 3 ----------------------------------
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
class DIRK43(RK):
    """
    L-stable Diagonally Implicit RK method with four stages of order 3.
    Taken from [here](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods).
    """
    A = np.zeros((4, 4))
    A[0, 0]  = 1/2
    A[1, :2] = [1/6, 1/2]
    A[2, :3] = [-1/2, 1/2, 1/2]
    A[3, :]  = [3/2, -3/2, 1/2, 1/2]
    b = [3/2, -3/2, 1/2, 1/2]
    c = [1/2, 2/3, 1/2, 1]

    @property
    def order(self): return 3


# ---------------------------------- Order 4 ----------------------------------
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
class SDIRK54(RK):
    """
    S-stable Diagonally Implicit Runge Kutta method of order 4 in five stages,
    from Wanner and Hairer [3]
    """
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
class EDIRK43(RK):
    """
    Embedded A-stable diagonally implicit RK pair of order 3 and 4.
    Taken from [here](https://doi.org/10.1007/BF01934920).
    """
    A = np.zeros((4, 4), dtype=float)
    A[0, 0]  = 5/6
    A[1, :2] = [-15/26, 5/6]
    A[2, :3] = [215/54, -130/ 27, 5/6]
    A[3]     = [4007/6075, -31031/24300, -133/2700, 5/6]
    b = [61/150, 2197/2100, 19/100, -9/14]
    c = [5/6, 10/39, 0, 1/6]
    b2 = [32/75, 169/300, 1/100, 0]

    @property
    def order(self): return 4

    CONV_TEST_NSTEPS = [32, 64, 128]


@registerRK
class EDIRK4(RK):
    """
    Stiffly accurate, fourth-order EDIRK with four stages. Taken from
    [here](https://ntrs.nasa.gov/citations/20160005923), second one in eq. (216).
    """
    A = np.zeros((4, 4))
    A[0, 0] = 0
    A[1, :2] = [3.0 / 4.0, 3.0 / 4.0]
    A[2, :3] = [447.0 / 675.0, -357.0 / 675.0, 855.0 / 675.0]
    A[3, :] = [13.0 / 42.0, 84.0 / 42.0, -125.0 / 42.0, 70.0 / 42.0]
    b = np.array([13.0, 84.0, -125.0, 70.0]) / 42.0
    c = np.array([0.0, 3.0 / 2.0, 7.0 / 5.0, 1.0])

    @property
    def order(self): return 4

    CONV_TEST_NSTEPS = [32, 64, 128]


@registerRK
class ESDIRK43(RK):
    """
    A-stable embedded RK pair of orders 4 and 3, ESDIRK4(3)6L[2]SA.
    Taken from [here](https://ntrs.nasa.gov/citations/20160005923)
    """
    s2 = 2**0.5
    c = np.array([0, 1 / 2, (2 - 2**0.5) / 4, 5 / 8, 26 / 25, 1.0])
    A = np.zeros((6, 6))
    A[1, :2] = [1 / 4, 1 / 4]
    A[2, :3] = [
        (1 - 2**0.5) / 8,
        (1 - 2**0.5) / 8,
        1 / 4,
    ]
    A[3, :4] = [
        (5 - 7 * s2) / 64,
        (5 - 7 * s2) / 64,
        7 * (1 + s2) / 32,
        1 / 4,
    ]
    A[4, :5] = [
        (-13796 - 54539 * s2) / 125000,
        (-13796 - 54539 * s2) / 125000,
        (506605 + 132109 * s2) / 437500,
        166 * (-97 + 376 * s2) / 109375,
        1 / 4,
    ]
    A[5, :] = [
        (1181 - 987 * s2) / 13782,
        (1181 - 987 * s2) / 13782,
        47 * (-267 + 1783 * s2) / 273343,
        -16 * (-22922 + 3525 * s2) / 571953,
        -15625 * (97 + 376 * s2) / 90749876,
        1 / 4,
    ]
    b = [
        (1181 - 987 * s2) / 13782,
        (1181 - 987 * s2) / 13782,
        47 * (-267 + 1783 * s2) / 273343,
        -16 * (-22922 + 3525 * s2) / 571953,
        -15625 * (97 + 376 * s2) / 90749876,
        1 / 4,
    ]
    b2 = [
        -480923228411.0 / 4982971448372,
        -480923228411.0 / 4982971448372,
        6709447293961.0 / 12833189095359,
        3513175791894.0 / 6748737351361.0,
        -498863281070.0 / 6042575550617.0,
        2077005547802.0 / 8945017530137.0,
    ]

    @property
    def order(self): return 4

    CONV_TEST_NSTEPS = [64, 128, 256]


# ---------------------------------- Order 5 ----------------------------------
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


@registerRK
class ARK548L2SAERK(RK):
    """
    Explicit part of the ARK54 scheme.
    """
    A = np.zeros((8, 8))
    A[1, 0] = 41.0 / 100.0
    A[2, :2] = [367902744464.0 / 2072280473677.0, 677623207551.0 / 8224143866563.0]
    A[3, :3] = [1268023523408.0 / 10340822734521.0, 0.0, 1029933939417.0 / 13636558850479.0]
    A[4, :4] = [
        14463281900351.0 / 6315353703477.0,
        0.0,
        66114435211212.0 / 5879490589093.0,
        -54053170152839.0 / 4284798021562.0,
    ]
    A[5, :5] = [
        14090043504691.0 / 34967701212078.0,
        0.0,
        15191511035443.0 / 11219624916014.0,
        -18461159152457.0 / 12425892160975.0,
        -281667163811.0 / 9011619295870.0,
    ]
    A[6, :6] = [
        19230459214898.0 / 13134317526959.0,
        0.0,
        21275331358303.0 / 2942455364971.0,
        -38145345988419.0 / 4862620318723.0,
        -1.0 / 8.0,
        -1.0 / 8.0,
    ]
    A[7, :7] = [
        -19977161125411.0 / 11928030595625.0,
        0.0,
        -40795976796054.0 / 6384907823539.0,
        177454434618887.0 / 12078138498510.0,
        782672205425.0 / 8267701900261.0,
        -69563011059811.0 / 9646580694205.0,
        7356628210526.0 / 4942186776405.0,
    ]
    b = [
        -872700587467.0 / 9133579230613.0,
        0.0,
        0.0,
        22348218063261.0 / 9555858737531.0,
        -1143369518992.0 / 8141816002931.0,
        -39379526789629.0 / 19018526304540.0,
        32727382324388.0 / 42900044865799.0,
        41.0 / 200.0,
    ]
    c = [
        0,
        41.0 / 100.0,
        2935347310677.0 / 11292855782101.0,
        1426016391358.0 / 7196633302097.0,
        92.0 / 100.0,
        24.0 / 100.0,
        3.0 / 5.0,
        1.0,
    ]
    b2 = [
        -975461918565.0 / 9796059967033.0,
        0.0,
        0.0,
        78070527104295.0 / 32432590147079.0,
        -548382580838.0 / 3424219808633.0,
        -33438840321285.0 / 15594753105479.0,
        3629800801594.0 / 4656183773603.0,
        4035322873751.0 / 18575991585200.0,
    ]

    @property
    def order(self): return 5

    CONV_TEST_NSTEPS = [32, 64, 128]


@registerRK
class ARK548L2SAESDIRK(ARK548L2SAERK):
    """
    Implicit part of the ARK54 scheme. Be careful with the embedded scheme. It seems that both schemes are order 5 as opposed to 5 and 4 as claimed. This may cause issues when doing adaptive time-stepping.
    """
    A = np.zeros((8, 8))
    A[1, :2] = [41.0 / 200.0, 41.0 / 200.0]
    A[2, :3] = [41.0 / 400.0, -567603406766.0 / 11931857230679.0, 41.0 / 200.0]
    A[3, :4] = [683785636431.0 / 9252920307686.0, 0.0, -110385047103.0 / 1367015193373.0, 41.0 / 200.0]
    A[4, :5] = [
        3016520224154.0 / 10081342136671.0,
        0.0,
        30586259806659.0 / 12414158314087.0,
        -22760509404356.0 / 11113319521817.0,
        41.0 / 200.0,
    ]
    A[5, :6] = [
        218866479029.0 / 1489978393911.0,
        0.0,
        638256894668.0 / 5436446318841.0,
        -1179710474555.0 / 5321154724896.0,
        -60928119172.0 / 8023461067671.0,
        41.0 / 200.0,
    ]
    A[6, :7] = [
        1020004230633.0 / 5715676835656.0,
        0.0,
        25762820946817.0 / 25263940353407.0,
        -2161375909145.0 / 9755907335909.0,
        -211217309593.0 / 5846859502534.0,
        -4269925059573.0 / 7827059040749.0,
        41.0 / 200.0,
    ]
    A[7, :] = [
        -872700587467.0 / 9133579230613.0,
        0.0,
        0.0,
        22348218063261.0 / 9555858737531.0,
        -1143369518992.0 / 8141816002931.0,
        -39379526789629.0 / 19018526304540.0,
        32727382324388.0 / 42900044865799.0,
        41.0 / 200.0,
    ]

    @property
    def orderEmbedded(self): return 5


@registerRK
class ARK548L2SAESDIRK2(RK):
    """
    Stiffly accurate singly diagonally L-stable implicit embedded Runge-Kutta pair 
    of orders 5 and 4 with explicit first stage from [here](https://doi.org/10.1016/j.apnum.2018.10.007).
    This method is part of the IMEX method ARK548L2SA.
    """
    gamma = 2.0 / 9.0
    c = [
        0.0,
        4.0 / 9.0,
        6456083330201.0 / 8509243623797.0,
        1632083962415.0 / 14158861528103.0,
        6365430648612.0 / 17842476412687.0,
        18.0 / 25.0,
        191.0 / 200.0,
        1.0,
    ]
    b = [
        0.0,
        0.0,
        3517720773327.0 / 20256071687669.0,
        4569610470461.0 / 17934693873752.0,
        2819471173109.0 / 11655438449929.0,
        3296210113763.0 / 10722700128969.0,
        -1142099968913.0 / 5710983926999.0,
        gamma,
    ]
    A = np.zeros((8, 8))
    A[2, 1] = 2366667076620.0 / 8822750406821.0
    A[3, 1] = -257962897183.0 / 4451812247028.0
    A[3, 2] = 128530224461.0 / 14379561246022.0
    A[4, 1] = -486229321650.0 / 11227943450093.0
    A[4, 2] = -225633144460.0 / 6633558740617.0
    A[4, 3] = 1741320951451.0 / 6824444397158.0
    A[5, 1] = 621307788657.0 / 4714163060173.0
    A[5, 2] = -125196015625.0 / 3866852212004.0
    A[5, 3] = 940440206406.0 / 7593089888465.0
    A[5, 4] = 961109811699.0 / 6734810228204.0
    A[6, 1] = 2036305566805.0 / 6583108094622.0
    A[6, 2] = -3039402635899.0 / 4450598839912.0
    A[6, 3] = -1829510709469.0 / 31102090912115.0
    A[6, 4] = -286320471013.0 / 6931253422520.0
    A[6, 5] = 8651533662697.0 / 9642993110008.0
    for i in range(A.shape[0]):
        A[i, i] = gamma
        A[i, 0] = A[i, 1]
        A[7, i] = b[i]
    b2 = [
        0.0,
        0.0,
        520639020421.0 / 8300446712847.0,
        4550235134915.0 / 17827758688493.0,
        1482366381361.0 / 6201654941325.0,
        5551607622171.0 / 13911031047899.0,
        -5266607656330.0 / 36788968843917.0,
        1074053359553.0 / 5740751784926.0,
    ]

    @property
    def order(self): return 5

    CONV_TEST_NSTEPS = [16, 32, 64]


@registerRK
class ARK548L2SAERK2(ARK548L2SAESDIRK2):
    """
    Explicit embedded pair of Runge-Kutta methods of orders 5 and 4 from [here](https://doi.org/10.1016/j.apnum.2018.10.007).
    This method is part of the IMEX method ARK548L2SA.
    """
    A = np.zeros((8, 8))
    A[2, 0] = 1.0 / 9.0
    A[2, 1] = 1183333538310.0 / 1827251437969.0
    A[3, 0] = 895379019517.0 / 9750411845327.0
    A[3, 1] = 477606656805.0 / 13473228687314.0
    A[3, 2] = -112564739183.0 / 9373365219272.0
    A[4, 0] = -4458043123994.0 / 13015289567637.0
    A[4, 1] = -2500665203865.0 / 9342069639922.0
    A[4, 2] = 983347055801.0 / 8893519644487.0
    A[4, 3] = 2185051477207.0 / 2551468980502.0
    A[5, 0] = -167316361917.0 / 17121522574472.0
    A[5, 1] = 1605541814917.0 / 7619724128744.0
    A[5, 2] = 991021770328.0 / 13052792161721.0
    A[5, 3] = 2342280609577.0 / 11279663441611.0
    A[5, 4] = 3012424348531.0 / 12792462456678.0
    A[6, 0] = 6680998715867.0 / 14310383562358.0
    A[6, 1] = 5029118570809.0 / 3897454228471.0
    A[6, 2] = 2415062538259.0 / 6382199904604.0
    A[6, 3] = -3924368632305.0 / 6964820224454.0
    A[6, 4] = -4331110370267.0 / 15021686902756.0
    A[6, 5] = -3944303808049.0 / 11994238218192.0
    A[7, 0] = 2193717860234.0 / 3570523412979.0
    A[7, 1] = 2193717860234.0 / 3570523412979.0
    A[7, 2] = 5952760925747.0 / 18750164281544.0
    A[7, 3] = -4412967128996.0 / 6196664114337.0
    A[7, 4] = 4151782504231.0 / 36106512998704.0
    A[7, 5] = 572599549169.0 / 6265429158920.0
    A[7, 6] = -457874356192.0 / 11306498036315.0
    A[1, 0] = ARK548L2SAESDIRK2.c[1]