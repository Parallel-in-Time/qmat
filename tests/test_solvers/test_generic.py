import pytest
import numpy as np

from qmat import Q_GENERATORS, QDELTA_GENERATORS
from qmat.nodes import QUAD_TYPES
from qmat.mathutils import numericalOrder
from qmat.solvers.sdc import solveDahlquistSDC

from qmat.solvers.generic import CoeffSolver
from qmat.solvers.generic.diffops import Dahlquist, Lorenz, ProtheroRobinson


@pytest.mark.parametrize("lam", [1j, -0.01, 1j-0.01])
@pytest.mark.parametrize("nSteps", [1, 5])
@pytest.mark.parametrize("tEnd", [1, 5])
@pytest.mark.parametrize("scheme", ["BE", "FE", "TRAP",  "RK4", "DIRK43",
                                    "ARK443ESDIRK", "ARK443ERK"])
def testLinearCoeffSolverDahlquist(scheme, tEnd, nSteps, lam):
    diffOp = Dahlquist(lam=lam)
    solver = CoeffSolver(diffOp=diffOp, tEnd=tEnd, nSteps=nSteps)

    qGen = Q_GENERATORS[scheme].getInstance()

    uRef = qGen.solveDahlquist(lam, 1, tEnd=tEnd, nSteps=nSteps)

    uNum = solver.solve(Q=qGen.Q, weights=qGen.weights)
    uNum = uNum[:, 0] + 1j*uNum[:, 1]

    assert np.allclose(uNum, uRef), \
        "generic CoeffSolver does not match reference solver for Dahlquist"

    if scheme.startswith("ARK443"):
        uNum = solver.solve(Q=qGen.Q, weights=None)
        uNum = uNum[:, 0] + 1j*uNum[:, 1]

        assert np.allclose(uNum, uRef), \
            "generic CoeffSolver without weights does not match reference solver for Dahlquist"


@pytest.mark.parametrize("quadType", QUAD_TYPES)
@pytest.mark.parametrize("nSweeps", [1, 2])
@pytest.mark.parametrize("nNodes", [1, 4])
@pytest.mark.parametrize("lam", [1j, -0.01, 1j-0.01])
@pytest.mark.parametrize("nSteps", [1, 2])
@pytest.mark.parametrize("tEnd", [1, 5])
@pytest.mark.parametrize("scheme", ["BE", "FE", "TRAP", "MIN-SR-FLEX"])
def testLinearCoeffSolverDahlquistSDC(
        scheme, tEnd, nSteps, lam, nNodes, nSweeps, quadType):
    if nNodes == 1 and quadType != "GAUSS":
        return

    diffOp = Dahlquist(lam=lam)
    solver = CoeffSolver(diffOp=diffOp, tEnd=tEnd, nSteps=nSteps)

    coll = Q_GENERATORS["Collocation"](
        nNodes=nNodes, quadType=quadType, nodeType="LEGENDRE")

    lastNode = np.allclose(coll.nodes[-1], 1)

    approx = QDELTA_GENERATORS[scheme](qGen=coll)
    kVals = [k+1 for k in range(nSweeps)]
    QDelta = approx.genCoeffs(k=kVals)

    for weights in [coll.weights, None]:
        if not lastNode:
            continue

        uRef = solveDahlquistSDC(
            lam, 1, tEnd=tEnd, nSteps=nSteps, nSweeps=nSweeps,
            Q=coll.Q, QDelta=QDelta, weights=weights)

        uNum = solver.solveSDC(
            nSweeps=nSweeps, Q=coll.Q, weights=weights, QDelta=QDelta)
        uNum = uNum[:, 0] + 1j*uNum[:, 1]

        details = " with weigths " if weights is not None else ""
        assert np.allclose(uNum, uRef), \
            f"generic CoeffSolver SDC {details} does not match reference solver for Dahlquist"


@pytest.fixture(scope="session")
def uRefLorentz():
    diffOp = Lorenz()
    tEnd = 0.1
    qGenRef = Q_GENERATORS["RK4"].getInstance()
    uRef = CoeffSolver(diffOp, tEnd=tEnd, nSteps=10000).solve(
        qGenRef.Q, qGenRef.weights)
    return {"tEnd": tEnd, "sol": uRef, "diffOp": diffOp}


@pytest.mark.parametrize("scheme", ["BE", "FE", "TRAP",  "RK4", "DIRK43"])
def testLinearCoeffSolverLorenz(scheme, uRefLorentz):
    diffOp = uRefLorentz["diffOp"]
    uRef = uRefLorentz["sol"]
    tEnd = uRefLorentz["tEnd"]

    nStepsVals = [10, 50, 100]
    err = []
    qGen = Q_GENERATORS[scheme].getInstance()
    for nSteps in nStepsVals:
        solver = CoeffSolver(diffOp, tEnd=tEnd, nSteps=nSteps)
        uNum = solver.solve(qGen.Q, qGen.weights)
        err.append(np.linalg.norm(uNum[-1] - uRef[-1]))

    expectedOrder = qGen.order
    order, rmse = numericalOrder(nStepsVals, err)
    assert rmse < 0.02, \
        f"rmse to high ({rmse}) for {scheme}"
    assert abs(order-expectedOrder) < 0.1, \
        f"expected order {expectedOrder:.2f}, but got {order:.2f} for {scheme}"


@pytest.mark.parametrize("quadType", QUAD_TYPES)
@pytest.mark.parametrize("nSweeps", [1, 2])
@pytest.mark.parametrize("nNodes", [3, 4])
@pytest.mark.parametrize("scheme", ["BE", "FE", "LU"])
def testLinearCoeffSolverLorenzSDC(scheme, nNodes, nSweeps, quadType, uRefLorentz):
    diffOp = Lorenz()
    uRef = uRefLorentz["sol"]
    tEnd = uRefLorentz["tEnd"]

    nStepsVals = [10, 50, 100]

    coll = Q_GENERATORS["Collocation"](
        nNodes=nNodes, nodeType="LEGENDRE", quadType=quadType)
    approx = QDELTA_GENERATORS[scheme](qGen=coll)
    nIters = [k+1 for k in range(nSweeps)]
    QDelta = approx.genCoeffs(k=nIters)

    err = []
    for nSteps in nStepsVals:
        solver = CoeffSolver(diffOp, tEnd=tEnd, nSteps=nSteps)
        uNum = solver.solveSDC(nSweeps, coll.Q, coll.weights, QDelta)
        err.append(np.linalg.norm(uNum[-1] - uRef[-1]))

    expectedOrder = nSweeps+1
    order, rmse = numericalOrder(nStepsVals, err)
    assert rmse < 0.02, \
        f"rmse to high ({rmse}) for {scheme}"
    assert abs(order-expectedOrder) < 0.1, \
        f"expected order {expectedOrder:.2f}, but got {order:.2f} for {scheme}"


@pytest.fixture(scope="session")
def uRefProtheroRobinson():
    diffOp = ProtheroRobinson(epsilon=0.5)
    tEnd = 0.5
    uRef = [diffOp.g(tEnd)]
    return {"tEnd": tEnd, "sol": uRef, "diffOp": diffOp}


@pytest.mark.parametrize("scheme", ["ARK4EDIRK", "ARK343ESDIRK"])
def testLinearCoeffSolverProtheroRobinson(scheme, uRefProtheroRobinson):
    diffOp = uRefProtheroRobinson["diffOp"]
    uRef = uRefProtheroRobinson["sol"]
    tEnd = uRefProtheroRobinson["tEnd"]

    nStepsVals = [20, 50, 100]
    err = []
    qGen = Q_GENERATORS[scheme].getInstance()
    for nSteps in nStepsVals:
        solver = CoeffSolver(diffOp, tEnd=tEnd, nSteps=nSteps)
        uNum = solver.solve(qGen.Q, qGen.weights)
        err.append(np.linalg.norm(uNum[-1] - uRef[-1]))

    expectedOrder = qGen.order
    order, rmse = numericalOrder(nStepsVals, err)

    import matplotlib.pyplot as plt
    plt.loglog(nStepsVals, err)
    plt.loglog(nStepsVals, np.array(nStepsVals, dtype=float)**(-expectedOrder), "--")

    assert rmse < 0.02, \
        f"rmse to high ({rmse}) for {scheme}"
    assert abs(order-expectedOrder) < 0.1, \
        f"expected order {expectedOrder:.2f}, but got {order:.2f} for {scheme}"


@pytest.mark.parametrize("quadType", QUAD_TYPES)
@pytest.mark.parametrize("nSweeps", [1, 2])
@pytest.mark.parametrize("nNodes", [3, 4])
@pytest.mark.parametrize("scheme", ["BE", "FE", "MIN-SR-FLEX"])
def testLinearCoeffSolverProtheroRobinsonSDC(scheme, nNodes, nSweeps, quadType, uRefProtheroRobinson):
    diffOp = uRefProtheroRobinson["diffOp"]
    uRef = uRefProtheroRobinson["sol"]
    tEnd = uRefProtheroRobinson["tEnd"]

    nStepsVals = [10, 50, 100]

    coll = Q_GENERATORS["Collocation"](
        nNodes=nNodes, nodeType="LEGENDRE", quadType=quadType)
    approx = QDELTA_GENERATORS[scheme](qGen=coll)
    nIters = [k+1 for k in range(nSweeps)]
    QDelta = approx.genCoeffs(k=nIters)

    err = []
    for nSteps in nStepsVals:
        solver = CoeffSolver(diffOp, tEnd=tEnd, nSteps=nSteps)
        uNum = solver.solveSDC(nSweeps, coll.Q, coll.weights, QDelta)
        err.append(np.linalg.norm(uNum[-1] - uRef[-1]))

    expectedOrder = nSweeps+1
    order, rmse = numericalOrder(nStepsVals, err)
    assert rmse < 0.02, \
        f"rmse to high ({rmse}) for {scheme}"
    assert abs(order-expectedOrder) < 0.1, \
        f"expected order {expectedOrder:.2f}, but got {order:.2f} for {scheme}"
