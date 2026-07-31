import pytest
import numpy as np

from time import sleep

from qmat.utils import Timer
import qmat.utils.num as mu
from qmat.utils.sdc import solveDahlquistSDC
from qmat.qcoeff.collocation import Collocation
from qmat import QDELTA_GENERATORS


def testTimer():
    with Timer("test1"):
        pass

    clock = Timer("test2")
    clock.start()
    sleep(0.1)
    clock.stop()
    assert clock.tWall >= 0.1


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


@pytest.mark.parametrize("nNodes", [2, 3, 4])
@pytest.mark.parametrize("qDelta", ["BE", "FE"])
def testSweeps(qDelta, nNodes):

    coll = Collocation(nNodes=nNodes, nodeType="LEGENDRE", quadType="RADAU-RIGHT")
    gen = QDELTA_GENERATORS[qDelta](nodes=coll.nodes)

    runParams = dict(
        lam=1j, u0=1, tEnd=np.pi, nSteps=10, nSweeps=nNodes,
        Q=coll.Q,
    )

    QD1 = gen.getQDelta()
    uNum1 = solveDahlquistSDC(**runParams, QDelta=QD1)

    QD2 = gen.genCoeffs(k=[i+1 for i in range(nNodes)])
    uNum2 = solveDahlquistSDC(**runParams, QDelta=QD2)

    assert np.allclose(uNum1, uNum2), "solutions with 2D and 3D QDelta matrices are not the same"


@pytest.mark.parametrize("nNodes", [4, 6])
@pytest.mark.parametrize("nSteps", [10, 20])
@pytest.mark.parametrize("nSweeps", [2, 3, 4])
def testMonitors(nSweeps, nSteps, nNodes):
    coll = Collocation(nNodes=nNodes, nodeType="LEGENDRE", quadType="RADAU-RIGHT")
    gen = QDELTA_GENERATORS["BE"](nodes=coll.nodes)

    runParams = dict(
        lam=1j, u0=1, tEnd=np.pi, nSteps=nSteps, nSweeps=nSweeps,
        Q=coll.Q, QDelta=gen.getQDelta(),
    )

    uNum = solveDahlquistSDC(**runParams)
    uNum2, monitors = solveDahlquistSDC(**runParams, monitors=["errors", "residuals"])

    assert np.allclose(uNum, uNum2), "solution with and without monitors are not the same"

    for key in ["errors", "residuals"]:
        assert key in monitors, f"'{key}' not in monitors"
        values = monitors[key]

        assert values.shape == (nSweeps+1, nSteps, nNodes), f"inconsistent shape for '{key}' values"
        assert np.all(np.abs(values[-1]) < np.abs(values[-2])), f"no decreasing {key}"
