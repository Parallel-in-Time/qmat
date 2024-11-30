import pytest
import numpy as np

from qmat.lagrange import LagrangeApproximation

nNodeTests = [2, 3, 4, 5, 6, 7, 8]

def referenceWeights(pType, n):
    j = np.arange(n)
    if pType == 'CHEBY-1':
        j = j[-1::-1]
        points = np.cos((2*j+1)*np.pi/(2*n))
        weights = np.sin((2*j+1)*np.pi/(2*n)) * (-1)**j
    elif pType == 'CHEBY-2':
        j = j[-1::-1]
        points = np.cos(j*np.pi/(n-1))
        weights = np.array([0.5]+[1]*(n-2)+[0.5]) * (-1)**j
    weights /= np.max(np.abs(weights))
    return points, weights


@pytest.mark.parametrize("weightComputation", ["AUTO", "FAST", "STABLE", "CHEBFUN"])
@pytest.mark.parametrize("pType", ["CHEBY-1", "CHEBY-2"])
def testWeights(pType, weightComputation):
    for n in nNodeTests:
        points, weights = referenceWeights(pType, n)
        approx = LagrangeApproximation(points, weightComputation=weightComputation, scaleWeights=True)
        assert np.allclose(approx.weights, weights), f"discrepancy with reference weights for n={n}"


@pytest.mark.parametrize("weightComputation", ["AUTO", "CHEBFUN"])
@pytest.mark.parametrize("pType", ["CHEBY-1", "CHEBY-2"])
def testAsymptoticWeights(pType, weightComputation):
    points, weights = referenceWeights(pType, 8000)
    approx = LagrangeApproximation(points, weightComputation=weightComputation)
    assert np.allclose(approx.weights, weights)


@pytest.mark.parametrize("scaleRef", ["MAX", "ZERO"])
@pytest.mark.parametrize("pType", ["CHEBY-1", "CHEBY-2"])
def testAsymptoticWeightsSTABLE(pType, scaleRef):
    points, weights = referenceWeights(pType, 10000)
    approx = LagrangeApproximation(points, weightComputation="STABLE", scaleRef=scaleRef)
    assert np.allclose(approx.weights, weights)


@pytest.mark.parametrize("weightComputation", ["AUTO", "FAST", "STABLE", "CHEBFUN"])
@pytest.mark.parametrize("nNodes", nNodeTests)
def testInterpolation(nNodes, weightComputation):
    nodes = np.sort(np.random.rand(nNodes))
    approx = LagrangeApproximation(nodes, weightComputation=weightComputation)

    times = np.random.rand(nNodes*2)
    P = approx.getInterpolationMatrix(times)

    polyCoeffs = np.random.rand(nNodes)
    polyValues = np.polyval(polyCoeffs, nodes)
    refEvals = np.polyval(polyCoeffs, times)
    assert np.allclose(refEvals, P @ polyValues)


@pytest.mark.parametrize("weightComputation", ["AUTO", "FAST", "STABLE", "CHEBFUN"])
@pytest.mark.parametrize("nNodes", nNodeTests)
def testEvaluation(nNodes, weightComputation):
    nodes = np.sort(np.random.rand(nNodes))
    times = np.random.rand(nNodes*2)
    polyCoeffs = np.random.rand(nNodes)
    polyValues = np.polyval(polyCoeffs, nodes)

    approx = LagrangeApproximation(nodes, weightComputation=weightComputation, fValues=polyValues)
    P = approx.getInterpolationMatrix(times)
    refEvals = P @ polyValues

    polyEvals = approx(t=times, fValues=polyValues)
    assert np.allclose(polyEvals, refEvals)

    polyEvals = approx(t=times)
    assert np.allclose(polyEvals, refEvals)


@pytest.mark.parametrize("numQuad", ["LEGENDRE_NUMPY", "LEGENDRE_SCIPY", "FEJER"])
@pytest.mark.parametrize("weightComputation", ["AUTO", "FAST", "STABLE", "CHEBFUN"])
@pytest.mark.parametrize("nNodes", nNodeTests)
def testIntegration(nNodes, weightComputation, numQuad):
    nodes = np.sort(np.random.rand(nNodes))
    approx = LagrangeApproximation(nodes, weightComputation=weightComputation)

    times = np.random.rand(nNodes*2)
    P = approx.getIntegrationMatrix([(0, t) for t in times], numQuad=numQuad)

    polyCoeffs = np.random.rand(nNodes)
    polyNodes = np.polyval(polyCoeffs, nodes)
    polyInteg = np.polyval(np.polyint(polyCoeffs), times) - np.polyval(np.polyint(polyCoeffs), 0)

    assert np.allclose(polyInteg, P @ polyNodes)


@pytest.mark.parametrize("weightComputation", ["AUTO", "FAST", "STABLE", "CHEBFUN"])
@pytest.mark.parametrize("nNodes", nNodeTests)
def testDerivation(nNodes, weightComputation):
    nodes = np.sort(np.random.rand(nNodes))
    approx = LagrangeApproximation(nodes, weightComputation=weightComputation)

    D1, D2 = approx.getDerivationMatrix(order="ALL")

    assert np.allclose(D1, approx.getDerivationMatrix())
    assert np.allclose(D2, approx.getDerivationMatrix(order=2))

    polyCoeffs = np.random.rand(nNodes)
    polyNodes = np.polyval(polyCoeffs, nodes)
    polyDeriv1 = np.polyval(np.polyder(polyCoeffs), nodes)
    polyDeriv2 = np.polyval(np.polyder(polyCoeffs, m=2), nodes)

    assert np.allclose(polyDeriv1, D1 @ polyNodes)

    assert np.allclose(polyDeriv2, D2 @ polyNodes)
    assert np.allclose(polyDeriv2, D1 @ D1 @ polyNodes)
