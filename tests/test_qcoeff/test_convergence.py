import pytest
import numpy as np

from qmat.utils import getClasses, numericalOrder
from qmat.qcoeff import Q_GENERATORS
from qmat.nodes import NODE_TYPES, QUAD_TYPES
from qmat.qcoeff.butcher import RKEmbedded

SCHEMES = getClasses(Q_GENERATORS)
EMBEDDED_SCHEMES = {k: v for k, v in SCHEMES.items() if issubclass(v, RKEmbedded)}

def nStepsForTest(scheme):
    try:
        nSteps = scheme.CONV_TEST_NSTEPS
    except AttributeError:
        if scheme.order == 1:
            nSteps = [64, 128, 256]
        elif scheme.order == 2:
            nSteps = [32, 64, 128]
        elif scheme.order == 3:
            nSteps = [16, 32, 64]
        elif scheme.order in [4, 5]:
            nSteps = [8, 16, 32]
        elif scheme.order in [6, 7]:
            nSteps = [4, 8, 16]
        elif scheme.order in [8, 9]:
            nSteps = [2, 4, 8]
        else:
            nSteps = [1, 2, 4]  # default value (very high order methods)
    return nSteps

u0 = 1
lam = 1j
T = 2*np.pi


@pytest.mark.parametrize("scheme", SCHEMES.keys())
def testDahlquist(scheme):
    gen = SCHEMES[scheme].getInstance()
    nSteps = nStepsForTest(gen)
    err = [gen.errorDahlquist(lam, u0, T, nS) for nS in nSteps]
    order, rmse = numericalOrder(nSteps, err)
    assert rmse < 0.02, f"rmse to high ({rmse}) for {scheme}"
    assert abs(order-gen.order) < 0.1, f"wrong numerical order ({order}) for {scheme}"


@pytest.mark.parametrize("quadType", QUAD_TYPES)
@pytest.mark.parametrize("nodesType", NODE_TYPES)
@pytest.mark.parametrize("nNodes", [2, 3, 4])
def testDahlquistCollocation(nNodes, nodesType, quadType):
    gen = SCHEMES["Collocation"](nNodes, nodesType, quadType)
    scheme = f"Collocation({nNodes}, {nodesType}, {quadType})"
    nSteps = nStepsForTest(gen)
    tEnd = T
    err = [gen.errorDahlquist(lam, u0, tEnd, nS) for nS in nSteps]
    order, rmse = numericalOrder(nSteps, err)
    assert rmse < 0.02, f"rmse to high ({rmse}) for {scheme} : {err}"
    if nNodes < 4:
        eps = 0.1
    else:
        eps = 0.25  # less constraining conditions for higher order
    assert abs(order-gen.order) < eps, f"wrong numerical order ({order}) for {scheme} : {err}"


@pytest.mark.parametrize("scheme", EMBEDDED_SCHEMES)
def testEmbeddedMethodsSecondaryOrder(scheme):
    from qmat.qcoeff.butcher import RK, checkAndStore

    method = EMBEDDED_SCHEMES[scheme]
    gen_primary = method.getInstance()

    class SecondaryMethod(RK):
        A = method.A
        b = gen_primary.weightsSecondary
        c = method.c

        @property
        def order(self): return gen_primary.orderSecondary

    if hasattr(gen_primary, 'CONV_TEST_NSTEPS'):
        SecondaryMethod.CONV_TEST_NSTEPS = gen_primary.CONV_TEST_NSTEPS

    gen = SecondaryMethod.getInstance()
    nSteps = nStepsForTest(gen)
    err = [gen.errorDahlquist(lam, u0, T, nS) for nS in nSteps]
    order, rmse = numericalOrder(nSteps, err)
    assert rmse < 0.02, f"rmse to high ({rmse}) for {scheme}"
    assert abs(order-gen.order) < 0.1, f"wrong numerical order ({order}) for {scheme}"
