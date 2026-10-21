import pytest
import numpy as np

from qmat.utils import getClasses
from qmat.utils.num import numericalOrder
from qmat.qcoeff import Q_GENERATORS, QGenerator
from qmat.qcoeff.collocation import Collocation
from qmat.nodes import NODE_TYPES, QUAD_TYPES

SCHEMES:dict[str,QGenerator] = getClasses(Q_GENERATORS)

def nStepsForTest(scheme:QGenerator, useEmbeddedWeights=False, linear=True):
    try:
        nSteps = scheme.CONV_TEST_NSTEPS
    except AttributeError:
        expectedOrder = scheme.getExpectedOrder(useEmbeddedWeights, linear)

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

# Parameters for the Dahlquist test (u' = lam*u) : the purely imaginary lam
# makes the solution perform exactly one full rotation over [0, tEnd], so
# that the exact solution at tEnd is simply u0
u0_DAHLQUIST = 1
lam_DAHLQUIST = 1j
tEnd_DAHLQUIST = 2*np.pi

# Parameters for the Riccati test (u' = t - u^2) : a shorter interval is used to
# ensure asymptotic convergence and stability of explicit schemes with the
# nSteps used in the tests
u0_RICCATI = 1
tEnd_RICCATI = 1


@pytest.mark.parametrize("useEmbeddedWeights", [True, False])
@pytest.mark.parametrize("scheme", SCHEMES.keys())
def testDahlquist(scheme, useEmbeddedWeights):
    gen:QGenerator = SCHEMES[scheme].getInstance()

    if useEmbeddedWeights:
        try:
            gen.weightsEmbedded
        except NotImplementedError:
            return None

    expectedOrder = gen.getExpectedOrder(useEmbeddedWeights, linear=True)
    nSteps = nStepsForTest(gen, useEmbeddedWeights)
    err = [gen.errorDahlquist(lam_DAHLQUIST, u0_DAHLQUIST, tEnd_DAHLQUIST, nS, useEmbeddedWeights=useEmbeddedWeights) for nS in nSteps]
    order, rmse = numericalOrder(nSteps, err)
    assert rmse < 0.02, f"rmse to high ({rmse}) for {scheme}"
    assert abs(order-expectedOrder) < 0.1, f"Expected order {expectedOrder:.2f}, but got {order:.2f} for {scheme}"


@pytest.mark.parametrize("quadType", QUAD_TYPES)
@pytest.mark.parametrize("nodesType", NODE_TYPES)
@pytest.mark.parametrize("nNodes", [2, 3, 4])
def testDahlquistCollocation(nNodes, nodesType, quadType, useEmbeddedWeights=False):
    gen:Collocation = SCHEMES["Collocation"](nNodes, nodesType, quadType)
    if useEmbeddedWeights:
        try:
            gen.weightsEmbedded
        except NotImplementedError:
            return None
    scheme = f"Collocation({nNodes}, {nodesType}, {quadType})"
    nSteps = nStepsForTest(gen, useEmbeddedWeights)
    err = [gen.errorDahlquist(lam_DAHLQUIST, u0_DAHLQUIST, tEnd_DAHLQUIST, nS, useEmbeddedWeights=useEmbeddedWeights) for nS in nSteps]
    order, rmse = numericalOrder(nSteps, err)
    expectedOrder = gen.getExpectedOrder(useEmbeddedWeights, linear=True)

    assert rmse < 0.02, f"rmse to high ({rmse}) for {scheme} : {err}"
    if nNodes < 4:
        eps = 0.1
    else:
        eps = 0.25  # less constraining conditions for higher order
    assert abs(order-expectedOrder) < eps, f"Expected order {expectedOrder:.2f}, but got {order:.2f} for {scheme} {err}"

    assert gen.rightIsNode == (gen.quadType in ['LOBATTO', 'RADAU-RIGHT'])


@pytest.mark.parametrize("useEmbeddedWeights", [True, False])
@pytest.mark.parametrize("scheme", SCHEMES.keys())
def testRiccati(scheme, useEmbeddedWeights):
    gen:QGenerator = SCHEMES[scheme].getInstance()

    if useEmbeddedWeights:
        try:
            gen.weightsEmbedded
        except NotImplementedError:
            return None

    expectedOrder = gen.getExpectedOrder(useEmbeddedWeights, linear=False)
    nSteps = nStepsForTest(gen, useEmbeddedWeights)
    err = [gen.errorRiccati(u0_RICCATI, tEnd_RICCATI, nS, useEmbeddedWeights=useEmbeddedWeights) for nS in nSteps]
    order, rmse = numericalOrder(nSteps, err)
    assert rmse < 0.02, f"rmse to high ({rmse}) for {scheme}"
    assert abs(order-expectedOrder) < 0.2, f"Expected order {expectedOrder:.2f}, but got {order:.2f} for {scheme}"


@pytest.mark.parametrize("quadType", QUAD_TYPES)
@pytest.mark.parametrize("nodesType", NODE_TYPES)
@pytest.mark.parametrize("nNodes", [2, 3, 4])
def testRiccatiCollocation(nNodes, nodesType, quadType, useEmbeddedWeights=False):
    gen:Collocation = SCHEMES["Collocation"](nNodes, nodesType, quadType)
    if useEmbeddedWeights:
        try:
            gen.weightsEmbedded
        except NotImplementedError:
            return None
    scheme = f"Collocation({nNodes}, {nodesType}, {quadType})"
    nSteps = nStepsForTest(gen, useEmbeddedWeights)
    err = [gen.errorRiccati(u0_RICCATI, tEnd_RICCATI, nS, useEmbeddedWeights=useEmbeddedWeights) for nS in nSteps]
    order, rmse = numericalOrder(nSteps, err)
    expectedOrder = gen.getExpectedOrder(useEmbeddedWeights, linear=False)

    assert rmse < 0.02, f"rmse to high ({rmse}) for {scheme} : {err}"
    if nNodes < 4:
        eps = 0.1
    else:
        eps = 0.25  # less constraining conditions for higher order
    assert abs(order-expectedOrder) < eps, f"Expected order {expectedOrder:.2f}, but got {order:.2f} for {scheme} {err}"

    assert gen.rightIsNode == (gen.quadType in ['LOBATTO', 'RADAU-RIGHT'])
