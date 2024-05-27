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
        approx = LagrangeApproximation(points, weightComputation=weightComputation)
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
    polyNodes = np.polyval(polyCoeffs, nodes)
    polyTimes = np.polyval(polyCoeffs, times)
    assert np.allclose(polyTimes, P @ polyNodes)
    

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