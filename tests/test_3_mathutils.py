import pytest
import numpy as np

import qmat.mathutils as mu

nNodeTests = [2, 3, 4, 5, 6, 7, 8]

@pytest.mark.parametrize("nNodes", nNodeTests)
def testRegression(nNodes):

    nodes = np.linspace(0, 1, num=nNodes, endpoint=False)

    for pOrder in range(1, nNodes):
        times = nodes + 1
        Pe = mu.getExtrapolationMatrix(nodes, times, pOrder)

        polyCoeffs = np.random.rand(pOrder+1)
        nodeValues = np.polyval(polyCoeffs, nodes)
        refValues = np.polyval(polyCoeffs, times)

        assert np.allclose(refValues, Pe @ nodeValues)


@pytest.mark.parametrize("nNodes", nNodeTests)
def testLDUFactorization(nNodes):
    Q = np.random.rand(nNodes, nNodes)
    L, D, U = mu.lduFactorization(Q)
    assert np.allclose(Q, L @ D @ U)
