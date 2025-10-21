import numpy as np
from qmat.playgrounds.martin.diff_eqs.de_solver import DESolver


class SDCIntegration:
    def __init__(
        self, num_nodes: int = 3, node_type: str = "LOBATTO", quad_type: str = "LOBATTO", num_sweeps: int = None
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

        if num_sweeps is None:
            self.num_sweeps = len(self.nodes)
        else:
            self.num_sweeps = num_sweeps

        assert self.num_sweeps >= 1

    def integrate(self, u0: np.array, t: float, dt: float, de_solver: DESolver) -> np.array:
        if not np.isclose(self.nodes[0], 0.0):
            raise Exception("SDC nodes must include the left interval boundary.")

        if not np.isclose(self.nodes[-1], 1.0):
            raise Exception("SDC nodes must include the right interval boundary.")

        assert self.N == self.deltas.shape[0]

        shape = (self.N,) + u0.shape
        u = np.zeros_like(u0, shape=shape)

        u[0, :] = u0
        du_dt_k0 = np.empty_like(u)
        du_dt_k1 = np.empty_like(u)

        # Propagate initial condition to all nodes
        for m in range(0, self.N):
            if m > 0:
                u[m] = u[m-1] + dt * self.deltas[m] * du_dt_k0[m-1]
            du_dt_k0[m] = de_solver.du_dt(u[m], t + dt*self.nodes[m])

        # Iteratively sweep over SDC nodes
        for _ in range(1, self.num_sweeps):
            for m in range(0, self.N):

                if m > 0:
                    qeval = self.q[m] @ du_dt_k0
                    u[m] = u[m-1] + dt * (self.deltas[m] * (du_dt_k1[m-1] - du_dt_k0[m-1]) + qeval)

                du_dt_k1[m] = de_solver.du_dt(u[m], t + dt*self.nodes[m])

            # Copy tendency arrays
            du_dt_k0[:] = du_dt_k1[:]

        if 0:
            # If we're using Radau-right, we can just use the last value
            u[0] = u[-1]

        else:
            # Compute new starting value with quadrature on tendencies
            u[0] = u[0] + dt * self.weights @ du_dt_k0

        assert u0.shape == u[0].shape
        return u[0]

    def integrate_n(self, u0: np.array, t: float, dt: float, num_timesteps, de_solver: DESolver) -> np.array:
        if not np.isclose(self.nodes[0], 0.0):
            raise Exception("SDC nodes must include the left interval boundary.")

        if not np.isclose(self.nodes[-1], 1.0):
            raise Exception("SDC nodes must include the right interval boundary.")

        u_value = u0

        for n in range(num_timesteps):
            u_value = self.integrate(u_value, t + n * dt, dt, de_solver)

        return u_value
