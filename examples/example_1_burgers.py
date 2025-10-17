#
# Example adopted from `test_4_sdc.py`
#


from itertools import product
import numpy as np
from itertools import product as _product
from burgers import Burgers
import pandas as pd


def solveBurgersSDC(
    burgers: Burgers, u0, T, nSteps: int, nSweeps: int, Q: np.ndarray, QDelta: np.ndarray, weights=None, monitors=None
) -> float:
    r"""
    Solve the Burgers' problem with SDC.

    Parameters
    ----------
    u0 : complex or float
        The initial solution :math:`u_0`.
    T : float
        Final time :math:`T`.
    nSteps : int
        Number of time-step for the whole :math:`[0,T]` interval.
    nSweeps : int
        Number of SDC sweeps.
    Q : np.ndarray
        Quadrature matrix :math:`Q` used for SDC.
    QDelta : np.ndarray
        Approximate quadrature matrix :math:`Q_\Delta` used for SDC.
        If three dimensional, use the first dimension for the sweep index.
    weights : np.ndarray, optional
        Quadrature weights to use for the prologation.
        If None, prolongation is not performed. The default is None.

    Returns
    -------
    uNum : np.ndarray
        Array containing the `nSteps+1` solutions :math:`\{u(0), ..., u(T)\}`.
    """
    nodes = Q.sum(axis=1)
    nNodes = Q.shape[0]
    dt = T / nSteps
    times = np.linspace(0, T, nSteps + 1)

    QDelta = np.asarray(QDelta)
    if QDelta.ndim == 3:
        assert QDelta.shape == (nSweeps, nNodes, nNodes), "inconsistent shape for QDelta"
    else:
        assert QDelta.shape == (nNodes, nNodes), "inconsistent shape for QDelta"
        QDelta = np.repeat(QDelta[None, ...], nSweeps, axis=0)

    # Preconditioner built for each sweeps
    P = np.eye(nNodes)[None, ...] - lam * dt * QDelta

    current_u = u0

    # Setup monitors if any
    if monitors:
        tmp = {}
        for key in monitors:
            assert key in ["residuals", "errors"], f"unknown key '{key}' for monitors"
            tmp[key] = np.zeros((nSweeps + 1, nSteps, nNodes), dtype=complex)
        monitors = tmp

    for i in range(nSteps):

        uNodes = np.ones(nNodes) * current_u

        for k in range(nSweeps):
            b = current_u + burgers.eval_f(current_u) * dt * nodes

            b = current_u + lam * dt * (Q - QDelta[k]) @ uNodes
            uNodes = np.linalg.solve(P[k], b)

            # Monitoring
            if monitors:
                if "residuals" in monitors:
                    monitors["residuals"][k + 1, i] = current_u + lam * dt * Q @ uNodes - uNodes
                if "errors" in monitors:
                    monitors["errors"][k + 1, i] = uNodes - u0 * np.exp(lam * (times[i] + dt * nodes))

        if weights is None:
            current_u = uNodes[-1]
        else:
            current_u = current_u + lam * dt * weights.dot(uNodes)

    if monitors:
        return current_u, monitors
    else:
        return current_u


from matplotlib import pyplot as plt

N = 128
nu = 0.2

T: float = 4.0
domain_size: float = 2.0 * np.pi
x = np.linspace(0, domain_size, N, endpoint=False)

burgers: Burgers = Burgers(N=N, nu=nu, domain_size=domain_size)
u0 = burgers.initial_u0("sine")

burgers.run_tests()

if 0:
    nt: int = 1000
    dt: float = T / nt

    if 0:
        for t in [_ * dt for _ in range(nt)]:
            ut = burgers.analytical_integration(u0, t=t)
            print(f"t={t:.3f}, ut[0]={ut[0]:.6f}")

            plt.plot(x, ut)
    else:
        ut = burgers.analytical_integration(u0, t=T)
        plt.plot(x, u0, label="u0")
        plt.plot(x, ut, label="ut")

    plt.legend()
    plt.show()


class SDCIntegration:
    def __init__(
        self, num_nodes: int = 3, node_type: str = "LOBATTO", quad_type: str = "LOBATTO"
    ):
        from qmat.qcoeff.collocation import Collocation
        import qmat.qdelta.timestepping as module

        coll = Collocation(nNodes=num_nodes, nodeType=node_type, quadType=quad_type)

        self.gen: module.FE = module.FE(coll.nodes)

        self.nodes, self.weights, self.q = coll.genCoeffs(form="N2N")

        self.q_delta: np.array = self.gen.getQDelta()
        self.d_tau: np.array = self.gen.dTau
        self.deltas: np.array = self.gen.deltas

        # Number of nodes
        self.N = len(self.nodes)


def sdc_integrate(u0: np.array, dt: float, num_timesteps, burgers: Burgers) -> np.array:
    sdci = SDCIntegration(num_nodes=3, node_type="LEGENDRE", quad_type="LOBATTO")

    if not np.isclose(sdci.nodes[0], 0.0):
        raise Exception("SDC nodes must include the left interval boundary.")

    if not np.isclose(sdci.nodes[-1], 1.0):
        raise Exception("SDC nodes must include the right interval boundary.")

    assert sdci.N == sdci.deltas.shape[0]

    n_sweeps = 2
    shape = (sdci.N,) + u0.shape
    u = np.zeros(shape=shape, dtype=float)

    u[0, :] = u0
    eval_f_k0 = np.empty_like(u)
    eval_f_k1 = np.empty_like(u)

    for _ in range(num_timesteps):

        # Propagate initial condition to all nodes
        for m in range(0, sdci.N):
            if m > 0:
                u[m] = u[m-1] + dt * sdci.deltas[m] * eval_f_k0[m-1]
            eval_f_k0[m] = burgers.eval_f(u[m])

        if 1:
            # Iteratively sweep over SDC nodes
            for _ in range(n_sweeps):
                for m in range(0, sdci.N):

                    if m > 0:
                        qeval = sdci.q[m] @ eval_f_k0
                        u[m] = u[m-1] + dt * (sdci.deltas[m] * (eval_f_k1[m-1] - eval_f_k0[m-1]) + qeval)

                    eval_f_k1[m] = burgers.eval_f(u[m])

                # Copy tendency arrays
                eval_f_k0[:] = eval_f_k1[:]

        if 0:
            # If we're using Radau-right, we can just use the last value
            u[0] = u[-1]
        else:
            # Compute new starting value with quadrature on tendencies
            u[0] = u[0] + dt * sdci.weights @ eval_f_k0

    return u[0]


time_integration = "sdc"

if 1:
    results = []

    u_analytical = burgers.analytical_integration(u0, t=T)

    for nt in range(4):

        num_timesteps = 2**nt * 1000
        print(f"Running simulation with num_timesteps={num_timesteps}")

        dt = T / num_timesteps

        burgers: Burgers = Burgers(N=N, nu=nu, domain_size=domain_size)
        u0 = burgers.initial_u0("sine")

        u = u0.copy()

        if time_integration == "rk1":
            u = burgers.step_rk1_n(u, dt, num_timesteps)

        elif time_integration == "rk2":
            u = burgers.step_rk2_n(u, dt, num_timesteps)

        elif time_integration == "rk4":
            u = burgers.step_rk4_n(u, dt, num_timesteps)

        elif time_integration == "sdc":
            u = sdc_integrate(u, dt, num_timesteps, burgers)

        else:
            raise Exception("TODO")

        plt.plot(x, u, label=f"numerical {num_timesteps}", linestyle="dashed")

        error = np.max(np.abs(u - u_analytical))
        results.append({"N": num_timesteps, "dt": dt, "error": error})

    prev_error = None
    for r in results:
        if prev_error is None:
            conv = None
        else:
            conv = np.log2(prev_error / r["error"])

        print(f"N={r["N"]}, dt={r["dt"]:.6e}, error={r["error"]:.6e}, conv={conv}")
        prev_error = r["error"]

    plt.plot(x, u_analytical, label="analytical", linestyle="dotted")

    plt.legend()
    plt.show()
