import numpy as np
from qmat.playground.diff_eqs.de_solver import DESolver


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

    def integrate(self, u0: np.array, dt: float, de_solver: DESolver) -> np.array:
        if not np.isclose(self.nodes[0], 0.0):
            raise Exception("SDC nodes must include the left interval boundary.")

        if not np.isclose(self.nodes[-1], 1.0):
            raise Exception("SDC nodes must include the right interval boundary.")

        assert self.N == self.deltas.shape[0]

        n_sweeps = 2
        shape = (self.N,) + u0.shape
        u = np.zeros(shape=shape, dtype=float)

        u[0, :] = u0
        eval_f_k0 = np.empty_like(u)
        eval_f_k1 = np.empty_like(u)

        # Propagate initial condition to all nodes
        for m in range(0, self.N):
            if m > 0:
                u[m] = u[m-1] + dt * self.deltas[m] * eval_f_k0[m-1]
            eval_f_k0[m] = de_solver.eval_f(u[m])

        if 1:
            # Iteratively sweep over SDC nodes
            for _ in range(n_sweeps):
                for m in range(0, self.N):

                    if m > 0:
                        qeval = self.q[m] @ eval_f_k0
                        u[m] = u[m-1] + dt * (self.deltas[m] * (eval_f_k1[m-1] - eval_f_k0[m-1]) + qeval)

                    eval_f_k1[m] = de_solver.eval_f(u[m])

                # Copy tendency arrays
                eval_f_k0[:] = eval_f_k1[:]

        if 0:
            # If we're using Radau-right, we can just use the last value
            u[0] = u[-1]
        else:
            # Compute new starting value with quadrature on tendencies
            u[0] = u[0] + dt * self.weights @ eval_f_k0

        return u[0]

    def integrate_n(self, u0: np.array, dt: float, num_timesteps, de_solver: DESolver) -> np.array:
        if not np.isclose(self.nodes[0], 0.0):
            raise Exception("SDC nodes must include the left interval boundary.")

        if not np.isclose(self.nodes[-1], 1.0):
            raise Exception("SDC nodes must include the right interval boundary.")

        u_value = u0

        for _ in range(num_timesteps):
            u_value = self.integrate(u_value, dt, de_solver)

        return u_value
