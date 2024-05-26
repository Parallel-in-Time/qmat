import pytest
import numpy as np

from qmat.qdelta import QDELTA_GENERATORS, genQDeltaCoeffs
from qmat.utils import getClasses

# Test base functionnality only on algebraic QDelta-generators
GENERATORS = getClasses(QDELTA_GENERATORS, module="algebraic")


@pytest.mark.parametrize("nNodes", [2, 3, 4])
@pytest.mark.parametrize("name", GENERATORS.keys())
def testGeneration(name, nNodes):

    Q = np.random.rand(nNodes, nNodes)
    gen = GENERATORS[name](Q=Q)

    QD1 = gen.genCoeffs()
    assert type(QD1) == np.ndarray, \
        f"QDelta for {name} is not np.ndarray but {type(QD1)}"
    assert QD1.shape == (nNodes, nNodes), \
        f"QDelta for {name} has unconsistent shape : {QD1.shape}"
    if name != "Exact":
        assert np.allclose(np.tril(QD1), QD1), \
            f"QDelta for {name} is not lower triangular"

    QD2 = genQDeltaCoeffs(name, Q=Q)
    assert QD1.shape == QD2.shape, \
        f"OOP QDelta and PP QDelta have not the same shape for {name}"
    assert np.allclose(QD1, QD2), \
        f"OOP QDelta and PP QDelta are not equals for {name}"

    _, dTau1 = gen.genCoeffs(dTau=True)
    assert type(dTau1) == np.ndarray, \
        f"dTau for {name} is not np.ndarray but {type(dTau1)}"
    assert dTau1.ndim == 1, \
        f"dTau for {name} is not 1D : {dTau1}"
    assert dTau1.size == nNodes, \
        f"dTau for {name} has not the correct size : {dTau1}"

    _, dTau2 = genQDeltaCoeffs(name, Q=Q, dTau=True)
    assert np.allclose(dTau1, dTau2), \
        f"OOP dTau and PP dTau are not equals for {name}"


nNodes = 4
@pytest.mark.parametrize("nSweeps", [1, 2, 3])
@pytest.mark.parametrize("name", GENERATORS.keys())
def testSingleGenerationMultipleSweeps(name, nSweeps):
    Q = np.random.rand(nNodes, nNodes)
    gen = GENERATORS[name](Q=Q)

    QD1 = gen.genCoeffs(k=[k+1 for k in range(nSweeps)])
    assert QD1.shape == (nSweeps, nNodes, nNodes), \
        f"QDelta for {name} has unconsistent shape : {QD1.shape}"

    QD2 = genQDeltaCoeffs(name, nSweeps=nSweeps, Q=Q)
    assert QD1.shape == QD2.shape, \
        f"OOP QDelta and PP QDelta have not the same shape for {name}"
    assert np.allclose(QD1, QD2), \
        f"OOP QDelta and PP QDelta are not equals for {name}"

    _, dTau1 = gen.genCoeffs(k=[k+1 for k in range(nSweeps)], dTau=True)
    _, dTau2 = genQDeltaCoeffs(name, nSweeps=nSweeps, dTau=True, Q=Q)
    assert dTau1.shape == dTau2.shape, \
        f"OOP dTau and PP dTau have not the same shape for {name}"
    assert np.allclose(dTau1, dTau2), \
        f"OOP dTau and PP dTau are not equals for {name}"


@pytest.mark.parametrize("nSweeps", [1, 2, 3, 4])
def testMultipleGenerationMultipleSweeps(nSweeps):
    Q = np.random.rand(nNodes, nNodes)

    names = list(GENERATORS.keys())[:nSweeps]
    generators = [GENERATORS[name](Q=Q) for name in names]

    QD1 = np.array([gen.genCoeffs() for gen in generators])
    QD2 = genQDeltaCoeffs(names, Q=Q)
    assert QD1.shape == QD2.shape, \
        f"OOP QDelta and PP QDelta have not the same shape with K={nSweeps}"
    assert np.allclose(QD1, QD2), \
        f"OOP QDelta and PP QDelta are not equals with K={nSweeps}"

    QD3 = genQDeltaCoeffs(names, nSweeps=nSweeps+1, Q=Q)
    assert QD3.shape[0] == QD1.shape[0]+1 and QD3.shape[1:] == QD1.shape[1:],\
        f"sweep extension produces inconsistent shapes for K={nSweeps}"
    assert np.allclose(QD3[-1], QD3[-2]), \
        f"sweep extension don't re-use the same matrix for K={nSweeps}"

    _, dTau1 = genQDeltaCoeffs(names, dTau=True, Q=Q)
    _, dTau2 = genQDeltaCoeffs(names, nSweeps=nSweeps+1, dTau=True, Q=Q)
    assert dTau1.shape == dTau2.shape, \
        f"sweep extension produces inconsistent dTau shapes for K={nSweeps}"
    assert np.allclose(dTau1, dTau2), \
        f"sweep extension don't re-use the same dTau for K={nSweeps}"
