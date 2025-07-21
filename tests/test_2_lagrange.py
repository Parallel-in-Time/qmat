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


@pytest.mark.parametrize("duplicates", ["USE_LEFT", "USE_RIGHT"])
@pytest.mark.parametrize("nCopy", [2, 3, 4])
@pytest.mark.parametrize("nPoints", [2, 3, 4])
def testDuplicates(nPoints, nCopy, duplicates):
    uniquePoints = np.sort(np.random.rand(nPoints))
    points = np.array((1 + nCopy)*uniquePoints.tolist())
    approx = LagrangeApproximation(points, duplicates=duplicates)

    assert approx.nPoints == points.size, "wrong nPoints"
    assert approx.nUniquePoints == uniquePoints.size, "wrong nUniquePoints"
    assert approx.hasDuplicates, "wrong hasDuplicates indicator"
    if duplicates == "USE_LEFT":
        assert np.allclose(approx._nnzIdx, np.arange(nPoints))
    if duplicates == "USE_RIGHT":
        assert np.allclose(approx._zerIdx, np.arange(nPoints*nCopy))

    approxUnique = LagrangeApproximation(uniquePoints)
    assert not approxUnique.hasDuplicates, "wrong hasDuplicates indicator for approxUnique"

    assert approxUnique.nPoints == approx.nUniquePoints, "discrepancy nUniquePoints"

    times = np.sort(np.random.rand(nPoints))
    P = approx.getInterpolationMatrix(times)
    P_noDuplicates = approx.getInterpolationMatrix(times, duplicates=False)
    P_ref = approxUnique.getInterpolationMatrix(times)

    assert np.allclose(P[:, approx._zerIdx], 0), "[P] zero indices have non-zero values"
    assert np.allclose(P[:, approx._nnzIdx], P_ref), "[P] nonzero values different from reference"
    assert np.allclose(P_noDuplicates, P_ref), "[P] no duplicates values different from reference"

    intervals = [(0, t) for t in times]
    Q = approx.getIntegrationMatrix(intervals)
    Q_noDuplicates = approx.getIntegrationMatrix(intervals, duplicates=False)
    Q_ref = approxUnique.getIntegrationMatrix(intervals)

    assert np.allclose(Q[:, approx._zerIdx], 0), "[Q] zero indices have non-zero values"
    assert np.allclose(Q[:, approx._nnzIdx], Q_ref), "[Q] nonzero values different from reference"
    assert np.allclose(Q_noDuplicates, Q_ref), "[Q] no duplicates values different from reference"

    D1, D2 = approx.getDerivationMatrix(order="ALL")
    D1_noDuplicates, D2_noDuplicates = approx.getDerivationMatrix(order="ALL", duplicates=False)
    D1_ref, D2_ref = approxUnique.getDerivationMatrix(order="ALL")

    assert np.allclose(D1[:, approx._zerIdx], 0), "[D1] zero indices have non-zero values"
    assert np.allclose(D1[:, approx._nnzIdx], D1_ref), "[D1] nonzero values different from reference"
    assert np.allclose(D1_noDuplicates, D1_ref), "[D1] no duplicates values different from reference"

    assert np.allclose(D2[:, approx._zerIdx], 0), "[D2] zero indices have non-zero values"
    assert np.allclose(D2[:, approx._nnzIdx], D2_ref), "[D2] nonzero values different from reference"
    assert np.allclose(D2_noDuplicates, D2_ref), "[D2] no duplicates values different from reference"


@pytest.mark.parametrize('inPoints', [np.linspace(0, 1, 15), np.linspace(0.3, 2.7, 13), np.linspace(0, 10, 4)])
@pytest.mark.parametrize('outPoints', [np.linspace(0, 1, 9), np.linspace(0.5, 2.2, 4), np.linspace(0.3, 2.7, 13)])
@pytest.mark.parametrize('order', [1, 2, 3, 4])
def testSparseInterpolation(inPoints, outPoints, order):
    from qmat.lagrange import getSparseInterpolationMatrix
    import scipy.sparse as sp

    np.random.seed(47)

    interpolationMatrix = getSparseInterpolationMatrix(inPoints, outPoints, order)
    assert isinstance(interpolationMatrix, sp.csc_matrix)

    polyCoeffs = np.random.randn(order)
    inPolynomial = np.polyval(polyCoeffs,inPoints)
    interpolated = interpolationMatrix @ inPolynomial
    error = np.linalg.norm(np.polyval(polyCoeffs, outPoints)- interpolated)
    assert error < 1e-11, f'Interpolation of order {order} polynomial is not exact with error {error:.2e}'

    testInexactInterpolation = True
    if len(inPoints) == len(outPoints):
        if np.allclose(inPoints, outPoints):
            testInexactInterpolation = False

    if testInexactInterpolation:
        polyCoeffs = np.random.randn(order+1)
        inPolynomial = np.polyval(polyCoeffs,inPoints)
        interpolated = interpolationMatrix @ inPolynomial
        assert not np.allclose(np.polyval(polyCoeffs, outPoints), interpolated), f'Interpolation of order {order+1} polynomial is unexpectedly exact'
