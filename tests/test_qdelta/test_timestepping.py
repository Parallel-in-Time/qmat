import pytest
import numpy as np

from qmat.qdelta import QDELTA_GENERATORS
import qmat.qdelta.timestepping as module
from qmat.utils import getClasses, numericalOrder
from qmat.qcoeff.collocation import Collocation
from qmat.nodes import NODE_TYPES, QUAD_TYPES
from qmat.sdc import errorDahlquistSDC, getOrderSDC

SCHEMES = getClasses(QDELTA_GENERATORS, module="timestepping")


@pytest.mark.parametrize("quadType", QUAD_TYPES)
@pytest.mark.parametrize("nodeType", NODE_TYPES)
@pytest.mark.parametrize("nNodes", [2, 3, 4, 5, 6])
def testBE(nNodes, nodeType, quadType):
    coll = Collocation(nNodes, nodeType, quadType)
    nodes = coll.nodes
    QDelta = module.BE(nodes).getQDelta()

    assert np.allclose(np.tril(QDelta), QDelta), \
        "QDelta is not lower triangular"
    assert np.allclose(QDelta.sum(axis=1), nodes), \
        "sum over the columns is not equal to nodes"


@pytest.mark.parametrize("quadType", QUAD_TYPES)
@pytest.mark.parametrize("nodeType", NODE_TYPES)
@pytest.mark.parametrize("nNodes", [2, 3, 4, 5, 6])
def testFE(nNodes, nodeType, quadType):
    coll = Collocation(nNodes, nodeType, quadType)
    nodes = coll.nodes
    QDelta = module.FE(nodes).getQDelta()

    assert np.allclose(np.tril(QDelta), QDelta), \
        "QDelta is not lower triangular"
    assert not np.diag(QDelta).any(), \
        "QDelta has not zero diagonal"
    assert np.allclose(QDelta.sum(axis=1)[1:], np.cumsum(np.diff(coll.nodes))), \
        "sum over the columns is not equal to cumsum of node differences"

    _, dTau = module.FE(nodes).genCoeffs(dTau=True)
    assert type(dTau) == np.ndarray, \
        f"dTau is not np.ndarray but {type(dTau)}"
    assert dTau.ndim == 1, \
        f"dTau is not 1D : {dTau}"
    assert dTau.size == nNodes, \
        f"dTau has not the correct size : {dTau}"
    assert np.allclose(dTau, coll.nodes[0]), \
        "dTau is not equal to nodes[0]"


@pytest.mark.parametrize("quadType", QUAD_TYPES)
@pytest.mark.parametrize("nodeType", NODE_TYPES)
@pytest.mark.parametrize("nNodes", [2, 3, 4, 5, 6])
def testTRAP(nNodes, nodeType, quadType):
    coll = Collocation(nNodes, nodeType, quadType)
    nodes = coll.nodes
    QDelta = module.TRAP(nodes).getQDelta()

    assert np.allclose(np.tril(QDelta), QDelta), \
        "QDelta is not lower triangular"

    QDeltaFE = module.FE(nodes).getQDelta()
    QDeltaBE = module.BE(nodes).getQDelta()
    assert np.allclose(QDelta, (QDeltaFE+QDeltaBE)/2), \
        "QDelta is not the mean of QDeltaBE and QDeltaFE"

    _, dTau = module.TRAP(nodes).genCoeffs(dTau=True)
    assert type(dTau) == np.ndarray, \
        f"dTau is not np.ndarray but {type(dTau)}"
    assert dTau.ndim == 1, \
        f"dTau is not 1D : {dTau}"
    assert dTau.size == nNodes, \
        f"dTau has not the correct size : {dTau}"
    assert np.allclose(dTau, coll.nodes[0]/2), \
        "dTau is not equal to node[0] divided by 2"


@pytest.mark.parametrize("quadType", QUAD_TYPES)
@pytest.mark.parametrize("nodeType", NODE_TYPES)
@pytest.mark.parametrize("nNodes", [2, 3, 4, 5, 6])
def testBEPAR(nNodes, nodeType, quadType):
    coll = Collocation(nNodes, nodeType, quadType)
    nodes = coll.nodes
    QDelta = module.BEPAR(nodes).getQDelta()

    assert np.allclose(np.tril(QDelta), QDelta), \
        "QDelta is not lower triangular"
    assert np.allclose(QDelta, np.diag(np.diag(QDelta))), \
        "QDelta is not diagonal"
    assert np.allclose(QDelta, np.diag(coll.nodes)), \
        "QDelta diagonal is not equal to nodes"


@pytest.mark.parametrize("quadType", QUAD_TYPES)
@pytest.mark.parametrize("nodeType", NODE_TYPES)
@pytest.mark.parametrize("nNodes", [2, 3, 4, 5, 6])
def testTRAPAR(nNodes, nodeType, quadType):
    coll = Collocation(nNodes, nodeType, quadType)
    nodes = coll.nodes
    QDelta = module.TRAPAR(nodes).getQDelta()

    assert np.allclose(np.tril(QDelta), QDelta), \
        "QDelta is not lower triangular"
    assert np.allclose(QDelta, np.diag(np.diag(QDelta))), \
        "QDelta is not diagonal"
    assert np.allclose(QDelta, np.diag(coll.nodes)/2), \
        "QDelta diagonal is not equal to nodes divided by 2"

    _, dTau = module.TRAPAR(nodes).genCoeffs(dTau=True)
    assert type(dTau) == np.ndarray, \
        f"dTau is not np.ndarray but {type(dTau)}"
    assert dTau.ndim == 1, \
        f"dTau is not 1D : {dTau}"
    assert dTau.size == nNodes, \
        f"dTau has not the correct size : {dTau}"
    assert np.allclose(dTau, coll.nodes/2), \
        "dTau is not equal to nodes divided by 2"


def nStepsForTest(order):
    nSteps = [1, 2, 4]  # default value (very high order methods)
    if order == 1:
        nSteps = [64, 128, 256]
    elif order == 2:
        nSteps = [32, 64, 128]
    elif order == 3:
        nSteps = [16, 32, 64]
    elif order in [4, 5]:
        nSteps = [8, 16, 32]
    elif order in [6, 7]:
        nSteps = [4, 8, 16]
    return nSteps

u0 = 1
lam = 1j
T = 2*np.pi

@pytest.mark.parametrize("quadType", QUAD_TYPES)
@pytest.mark.parametrize("nodeType", NODE_TYPES)
@pytest.mark.parametrize("nNodes", [2, 3, 4])
@pytest.mark.parametrize("qDelta", SCHEMES.keys())
@pytest.mark.parametrize("nSweeps", [0, 1, 2, 3])
def testSDCConvergenceQUADRATURE(nSweeps, qDelta, nNodes, nodeType, quadType):
    coll = Collocation(nNodes, nodeType, quadType)
    nodes, weights, Q = coll.genCoeffs()
    QDelta = SCHEMES[qDelta](nodes).getQDelta()

    orderSDC = getOrderSDC(coll, nSweeps, qDelta, "QUADRATURE")
    nSteps = nStepsForTest(orderSDC)

    # Edge cases
    if qDelta == "BEPAR":
        if nSweeps == 3 and nNodes == 2 and nodeType == "LEGENDRE" and quadType == "RADAU-RIGHT":
            nSteps = nStepsForTest(orderSDC+1)
        if nSweeps == 3 and nNodes == 3 and nodeType == "EQUID" and quadType == "RADAU-RIGHT":
            nSteps = nStepsForTest(orderSDC+1)

    err = [
        errorDahlquistSDC(
            lam, u0, T, nS, nSweeps, Q, QDelta, weights)
        for nS in nSteps
        ]

    order, rmse = numericalOrder(nSteps, err)
    assert rmse < 0.04, f"rmse to high ({rmse})"
    assert abs(order-orderSDC) < 0.5, f"wrong numerical order ({order})"


@pytest.mark.parametrize("quadType", ["LOBATTO", "RADAU-RIGHT"])
@pytest.mark.parametrize("nodeType", NODE_TYPES)
@pytest.mark.parametrize("nNodes", [2, 3, 4])
@pytest.mark.parametrize("qDelta", SCHEMES.keys())
@pytest.mark.parametrize("nSweeps", [1, 2, 3, 4])
def testSDCConvergenceLASTNODE(nSweeps, qDelta, nNodes, nodeType, quadType):
    coll = Collocation(nNodes, nodeType, quadType)
    nodes, _, Q = coll.genCoeffs()
    QDelta = SCHEMES[qDelta](nodes).getQDelta()

    orderSDC = getOrderSDC(coll, nSweeps, qDelta, "LASTNODE")
    nSteps = nStepsForTest(orderSDC)

    # Edge cases
    if qDelta == "BEPAR":
        if nSweeps == 4 and nNodes == 3 and nodeType == "EQUID" and quadType == "LOBATTO":
            nSteps = nStepsForTest(orderSDC-1)
        if nSweeps == 4 and nNodes == 3 and nodeType == "EQUID" and quadType == "RADAU-RIGHT":
            nSteps = nStepsForTest(orderSDC-1)
        if nSweeps == 4 and nNodes == 3 and nodeType in "LEGENDRE":
            nSteps = nStepsForTest(orderSDC-1)
        if nSweeps == 4 and nNodes == 3 and nodeType in "CHEBY-1" and quadType == "LOBATTO":
            nSteps = nStepsForTest(orderSDC-1)
        if nSweeps == 4 and nNodes == 3 and nodeType == "CHEBY-1" and quadType == "RADAU-RIGHT":
            nSteps = nStepsForTest(orderSDC-1)
        if nSweeps == 4 and nNodes == 3 and nodeType == "CHEBY-2" and quadType == "LOBATTO":
            nSteps = nStepsForTest(orderSDC-1)
        if nSweeps == 4 and nNodes == 3 and nodeType == "CHEBY-2" and quadType == "RADAU-RIGHT":
            nSteps = nStepsForTest(orderSDC-1)
        if nSweeps == 4 and nNodes == 3 and nodeType == "CHEBY-3":
            nSteps = nStepsForTest(orderSDC-1)
        if nSweeps == 4 and nNodes == 3 and nodeType == "CHEBY-4":
            nSteps = nStepsForTest(orderSDC-1)
        if nSweeps == 4 and nNodes == 4:
            nSteps = nStepsForTest(orderSDC-1)


    err = [
        errorDahlquistSDC(
            lam, u0, T, nS, nSweeps, Q, QDelta, weights=None)
        for nS in nSteps
        ]

    # import matplotlib.pyplot as plt
    # plt.plot(T/np.array(nSteps), err)
    # import pdb; pdb.set_trace()

    order, rmse = numericalOrder(nSteps, err)
    assert rmse < 0.05, f"rmse to high ({rmse}), err={err}"
    assert abs(order-orderSDC) < 0.5, f"wrong numerical order ({order}), err={err}"
