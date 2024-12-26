import pytest
import numpy as np

from qmat.sdc import solveDahlquistSDC
from qmat.qcoeff.collocation import Collocation
from qmat import QDELTA_GENERATORS


@pytest.mark.parametrize("nNodes", [2, 3, 4])
@pytest.mark.parametrize("qDelta", ["BE", "FE"])
def testSweeps(qDelta, nNodes):
    
    coll = Collocation(nNodes=nNodes, nodeType="LEGENDRE", quadType="RADAU-RIGHT")
    gen = QDELTA_GENERATORS[qDelta](nodes=coll.nodes)

    runParams = dict(
        lam=1j, u0=1, T=np.pi, nSteps=10, nSweeps=nNodes,
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
        lam=1j, u0=1, T=np.pi, nSteps=nSteps, nSweeps=nSweeps,
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
