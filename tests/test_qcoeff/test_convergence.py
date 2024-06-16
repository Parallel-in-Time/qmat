import pytest
import numpy as np

from qmat.utils import getClasses, numericalOrder
from qmat.qcoeff import Q_GENERATORS
from qmat.nodes import NODE_TYPES, QUAD_TYPES

SCHEMES = getClasses(Q_GENERATORS)

def nStepsForTest(scheme, secondary=False):
    try:
        nSteps = scheme.CONV_TEST_NSTEPS
    except AttributeError:
        expectedOrder = scheme.orderSecondary if secondary else scheme.order

        if expectedOrder == 1:
            nSteps = [64, 128, 256]
        elif expectedOrder == 2:
            nSteps = [32, 64, 128]
        elif expectedOrder == 3:
            nSteps = [16, 32, 64]
        elif expectedOrder in [4, 5]:
            nSteps = [8, 16, 32]
        elif expectedOrder in [6, 7]:
            nSteps = [4, 8, 16]
        elif expectedOrder in [8, 9]:
            nSteps = [2, 4, 8]
        else:
            nSteps = [1, 2, 4]  # default value (very high order methods)
    return nSteps

u0 = 1
lam = 1j
T = 2*np.pi


@pytest.mark.parametrize("scheme", SCHEMES.keys())
@pytest.mark.parametrize("secondary", [True, False])
def testDahlquist(scheme, secondary):
    gen = SCHEMES[scheme].getInstance()

    if secondary:
        try:
            gen.weightsSecondary
        except NotImplementedError:
            return None

    expectedOrder = gen.orderSecondary if secondary else gen.order
    nSteps = nStepsForTest(gen, secondary)
    err = [gen.errorDahlquist(lam, u0, T, nS, secondary=secondary) for nS in nSteps]
    order, rmse = numericalOrder(nSteps, err)
    assert rmse < 0.02, f"rmse to high ({rmse}) for {scheme}"
    assert abs(order-expectedOrder) < 0.1, f"Expected order {expectedOrder:.2f}, but got {order:.2f} for {scheme}"


@pytest.mark.parametrize("quadType", QUAD_TYPES)
@pytest.mark.parametrize("nodesType", NODE_TYPES)
@pytest.mark.parametrize("nNodes", [2, 3, 4])
@pytest.mark.parametrize("secondary", [True, False])
def testDahlquistCollocation(nNodes, nodesType, quadType, secondary):
    gen = SCHEMES["Collocation"](nNodes, nodesType, quadType)
    if secondary:
        try:
            gen.weightsSecondary
        except NotImplementedError:
            return None
    scheme = f"Collocation({nNodes}, {nodesType}, {quadType})"
    nSteps = nStepsForTest(gen, secondary)
    tEnd = T
    err = [gen.errorDahlquist(lam, u0, tEnd, nS, secondary=secondary) for nS in nSteps]
    order, rmse = numericalOrder(nSteps, err)
    expectedOrder = gen.orderSecondary if secondary else gen.order
    assert rmse < 0.02, f"rmse to high ({rmse}) for {scheme} : {err}"
    if nNodes < 4:
        eps = 0.1
    else:
        eps = 0.25  # less constraining conditions for higher order
    assert abs(order-expectedOrder) < eps, f"Expected order {expectedOrder:.2f}, but got {order:.2f} for {scheme} {err}"
