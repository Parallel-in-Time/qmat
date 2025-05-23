import pytest
import numpy as np

from qmat.qdelta import QDELTA_GENERATORS
import qmat.qdelta.diag as module
from qmat.qcoeff.collocation import Collocation
from qmat.nodes import NODE_TYPES, QUAD_TYPES


def stiffSR(Q, QDelta):
    I = np.eye(Q.shape[0])
    M = I - np.linalg.solve(QDelta, Q)
    return max(abs(np.linalg.eigvals(M)))

# TODO : stiff spectral radius is not the same on different system ... to investigate
margin = 2

STIFF_SR = {
    "MIN": 0.43,
    "VDHS": 0.026,
    "MIN3": 0.0082,
    "MIN-SR-S": 0.00025,
    }
STIFF_PARAMS = {
    "nNodes": 4,
    "nodeType": "LEGENDRE",
    "quadType": "RADAU-RIGHT",
    }

@pytest.mark.parametrize("name", STIFF_SR.keys())
def testStiff4LegendreRadauRight(name):
    coll = Collocation(**STIFF_PARAMS)
    gen = QDELTA_GENERATORS[name](Q=coll.Q, **STIFF_PARAMS)

    QDelta = gen.getQDelta()
    assert np.allclose(np.tril(QDelta), QDelta), \
        "QDelta is not lower triangular"
    assert np.allclose(QDelta, np.diag(np.diag(QDelta))), \
        "QDelta is not diagonal"

    sr = stiffSR(coll.Q, QDelta)
    assert sr < STIFF_SR[name] * margin, "spectral radius too high"


def nilpotencyNonStiff(Q, QDelta):
    nNodes = QDelta.shape[0]
    P = np.linalg.matrix_power(Q - QDelta, nNodes)
    return np.linalg.norm(P, ord=np.inf)

@pytest.mark.parametrize("quadType", QUAD_TYPES)
@pytest.mark.parametrize("nodeType", NODE_TYPES)
@pytest.mark.parametrize("nNodes", [2, 3, 4, 5, 6, 7])
def testNonStiff(nNodes, nodeType, quadType):
    coll = Collocation(nNodes=nNodes, nodeType=nodeType, quadType=quadType)
    nodes, Q = coll.nodes, coll.Q

    gen = module.MIN_SR_NS(nodes)
    QDelta = gen.getQDelta()

    assert np.allclose(np.tril(QDelta), QDelta), \
        "QDelta is not lower triangular"
    assert np.allclose(QDelta, np.diag(np.diag(QDelta))), \
        "QDelta is not diagonal"
    assert nilpotencyNonStiff(Q, QDelta) < 1e-15 * margin, \
        "nilpotency measure is to high"


def nilpotencyStiff(Q, QDelta):
    if QDelta[0, 0] == 0:
        Q = Q[1:, 1:]
        QDelta = QDelta[1:, 1:]
    nNodes = QDelta.shape[0]
    I = np.eye(nNodes)
    P = np.linalg.matrix_power(I - np.linalg.solve(QDelta, Q), nNodes)
    return np.linalg.norm(P, ord=np.inf)

@pytest.mark.parametrize("quadType", QUAD_TYPES)
@pytest.mark.parametrize("nodeType", NODE_TYPES)
@pytest.mark.parametrize("nNodes", [2, 3, 4])
def testStiff(nNodes, nodeType, quadType):
    coll = Collocation(nNodes=nNodes, nodeType=nodeType, quadType=quadType)
    Q = coll.Q

    gen = module.MIN_SR_S(nNodes=nNodes, nodeType=nodeType, quadType=quadType)
    QDelta = gen.getQDelta()

    assert np.allclose(np.tril(QDelta), QDelta), \
        "QDelta is not lower triangular"
    assert np.allclose(QDelta, np.diag(np.diag(QDelta))), \
        "QDelta is not diagonal"
    assert nilpotencyStiff(Q, QDelta) < 1e-11 * margin, \
        "nilpotency measure is to high"


@pytest.mark.parametrize("quadType", ["GAUSS", "RADAU-RIGHT"])
@pytest.mark.parametrize("nodeType", NODE_TYPES)
@pytest.mark.parametrize("nNodes", [2, 3, 4])
def testFlex(nNodes, nodeType, quadType):
    coll = Collocation(nNodes=nNodes, nodeType=nodeType, quadType=quadType)
    Q = coll.Q

    gen = module.MIN_SR_FLEX(nNodes=nNodes, nodeType=nodeType, quadType=quadType)
    gen2 = module.MIN_SR_FLEX(coll=coll)
    for i in range(nNodes):
        assert np.allclose(gen2.getQDelta(k=i+1), gen.getQDelta(k=i+1)), \
            "not the same coefficients using the coll parameter"

    I = np.eye(nNodes)
    P = I
    for k in range(nNodes):
        P = (I - np.linalg.solve(gen.getQDelta(k+1), Q)) @ P

    assert np.linalg.norm(P, ord=np.inf) < 1e-13 * margin, \
        "nilpotency measure is to high"

    genS = module.MIN_SR_S(nNodes=nNodes, nodeType=nodeType, quadType=quadType)

    QDeltaS = genS.getQDelta()
    QDeltaFlex = gen.getQDelta(nNodes+1)
    assert np.allclose(QDeltaS, QDeltaFlex), \
        "QDelta for k > nNodes is not equal to MIN-SR-S"

    QDelta0 = gen.getQDelta()
    QDelta1 = gen.getQDelta(1)
    assert np.allclose(QDelta0, QDelta1), \
        "default QDelta is not equal to k=1"


@pytest.mark.parametrize("quadType", ["GAUSS", "RADAU-RIGHT"])
@pytest.mark.parametrize("nodeType", NODE_TYPES)
@pytest.mark.parametrize("nNodes", [2, 3, 4])
def testJumper(nNodes, nodeType, quadType):
    coll = Collocation(nNodes=nNodes, nodeType=nodeType, quadType=quadType)
    nodes, Q = coll.nodes, coll.Q
    k = np.arange(nNodes)+1
    
    gen = module.Jumper(nodes=nodes)
    genFlex = module.MIN_SR_FLEX(coll=coll)
    QDeltas = gen.genCoeffs(k)
    QDeltasFlex = genFlex.genCoeffs(k)

    assert np.allclose(QDeltas, QDeltasFlex/2)

    gen2 = module.FlexJumper(nodes=nodes)
    QDeltas2 = gen2.genCoeffs(k)

    assert np.allclose(QDeltas2[0], QDeltasFlex[0])
    assert np.allclose(QDeltas2[1:], QDeltas[:-1])


@pytest.mark.parametrize("quadType", ["GAUSS", "RADAU-RIGHT"])
@pytest.mark.parametrize("nodeType", NODE_TYPES)
@pytest.mark.parametrize("nNodes", [2, 3, 4])
@pytest.mark.parametrize("divider", [1, 2, 3, 4, 5])
def testDNODES(divider, nNodes, nodeType, quadType):
    coll = Collocation(nNodes=nNodes, nodeType=nodeType, quadType=quadType)
    nodes = coll.nodes
    
    Generator = {
        1: module.DNODES,
        2: module.DNODES2,
        3: module.DNODES3,
        4: module.DNODES4,
        5: module.DNODES5,
    }[divider]
    gen:module.DNODES = Generator(nodes=nodes)
    QDeltas = gen.genCoeffs()

    assert np.allclose(QDeltas, np.diag(nodes)/divider)