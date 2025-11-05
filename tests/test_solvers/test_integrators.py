import pytest
import numpy as np

from qmat import Q_GENERATORS, QDELTA_GENERATORS
from qmat.solvers.generic import CoeffSolver, PhiSolver
from qmat.solvers.generic.integrators import ForwardEuler, BackwardEuler
from qmat.solvers.generic.diffops import DIFFOPS

EQUIVALENCES: dict[str, type[PhiSolver]] = {
    "FE": ForwardEuler,
    "BE": BackwardEuler,
}

@pytest.mark.parametrize("nNodes", [1, 4, 10])
@pytest.mark.parametrize("problem", ["Lorenz", "ProtheroRobinson"])
@pytest.mark.parametrize("scheme", EQUIVALENCES.keys())
def testPhiSolver(scheme, problem, nNodes):
    diffOp = DIFFOPS[problem]()
    tEnd = 0.1
    nSteps = 10*nNodes

    qGen = Q_GENERATORS[scheme].getInstance()

    refSolver = CoeffSolver(diffOp, tEnd=tEnd, nSteps=nSteps)
    ref = refSolver.solve(qGen.Q, qGen.weights)

    regNodes = np.linspace(0, 1, num=nNodes+1)[1:]

    phiSolver = EQUIVALENCES[scheme](diffOp, nodes=regNodes, tEnd=tEnd, nSteps=nSteps//nNodes)
    sol = phiSolver.solve()

    assert np.allclose(sol, ref[::nNodes]), \
        f"{phiSolver.__class__.__name__}-PhiSolver does not match equivalent CoeffSolver result"


@pytest.mark.parametrize("nSweeps", [1, 2, 4])
@pytest.mark.parametrize("quadType", ["RADAU-RIGHT", "LOBATTO"])
@pytest.mark.parametrize("nNodes", [2, 4, 8])
@pytest.mark.parametrize("problem", ["Lorenz", "ProtheroRobinson"])
@pytest.mark.parametrize("scheme", EQUIVALENCES.keys())
def testPhiSolverSDC(scheme, problem, nNodes, quadType, nSweeps):
    pParams = {}
    if problem == "ProtheroRobinson":
        pParams = {"epsilon": 0.01, "nonLinear": True}

    diffOp = DIFFOPS[problem](**pParams)
    tEnd = 0.1
    nSteps = 10

    coll = Q_GENERATORS["Collocation"](nNodes=nNodes, quadType=quadType, nodeType="LEGENDRE")
    approx = QDELTA_GENERATORS[scheme](qGen=coll)

    refSolver = CoeffSolver(diffOp, tEnd=tEnd, nSteps=nSteps)
    ref = refSolver.solveSDC(nSweeps, coll.Q, coll.weights, approx.getQDelta())

    phiSolver = EQUIVALENCES[scheme](diffOp, nodes=coll.nodes, tEnd=tEnd, nSteps=nSteps)

    sol = phiSolver.solveSDC(nSweeps, Q=coll.Q, weights=True)
    assert np.allclose(sol, ref), \
        f"{phiSolver.__class__.__name__}-PhiSolver SDC with given Q does not match equivalent CoeffSolver SDC result"

    sol = phiSolver.solveSDC(nSweeps, Q=None, weights=True)
    assert np.allclose(sol, ref), \
        f"{phiSolver.__class__.__name__}-PhiSolver SDC does not match equivalent CoeffSolver SDC result"

    ref = refSolver.solveSDC(nSweeps, coll.Q, None, approx.getQDelta())
    sol = phiSolver.solveSDC(nSweeps, weights=None)
    assert np.allclose(sol, ref), \
        f"{phiSolver.__class__.__name__}-PhiSolver SDC without weights does not match equivalent CoeffSolver SDC result"

    if scheme == "BE":
        original = BackwardEuler.phiSolve
        BackwardEuler.phiSolve = PhiSolver.phiSolve  # use default phiSolve
        sol = phiSolver.solveSDC(nSweeps, Q=coll.Q, weights=False)
        BackwardEuler.phiSolve = original
        assert np.allclose(sol, ref), \
            f"{phiSolver.__class__.__name__}-PhiSolver SDC with default phiSolve does not match equivalent CoeffSolver SDC result"
