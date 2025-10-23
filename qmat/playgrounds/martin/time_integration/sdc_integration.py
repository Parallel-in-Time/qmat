import numpy as np
from qmat.playgrounds.martin.diff_eqs.de_solver import DESolver


class SDCIntegration:
    def __init__(
        self,
        num_nodes: int = 3,
        node_type: str = "LOBATTO",  # Basic node types to generate SDC nodes
        quad_type: str = "LOBATTO",  # 'LOBATTO': Always include {0, 1} in quadrature points. Add them if they don't exist.
        num_sweeps: int = None,
        micro_time_integration: str = None,  # 'erk1' = explicit Euler, 'irk1' = implicit Euler, 'imex' = implicit-explicit
    ):
        from qmat.qcoeff.collocation import Collocation
        import qmat.qdelta.timestepping as module

        coll = Collocation(nNodes=num_nodes, nodeType=node_type, quadType=quad_type)

        self.gen: module.FE = module.FE(coll.nodes)

        self.nodes, self.weights, self.q = coll.genCoeffs(form="N2N")

        self.q_delta: np.ndarray = self.gen.getQDelta()
        # Deltas are the \tau
        self.deltas: np.ndarray = self.gen.deltas

        # Number of nodes
        self.N = len(self.nodes)

        print(f"self.nodes: {self.nodes}")
        print(f"self.deltas: {self.deltas}")

        if num_sweeps is None:
            self.num_sweeps = len(self.nodes)
        else:
            self.num_sweeps = num_sweeps

        # Time integration to be used within SDC sweeps
        # 'erk1' = explicit Euler
        # 'irk1' = implicit Euler
        # 'imex' = implicit-explicit => 'imex12'
        # 'imex12' = implicit-explicit: 1st term treated implicitly, 2nd term explicitly
        # 'imex21' = implicit-explicit: 2nd term treated implicitly, 1st term explicitly
        self.time_integration_method = micro_time_integration if micro_time_integration is not None else "erk1"

        if self.time_integration_method == "imex":
            self.time_integration_method = "imex12"

        assert self.num_sweeps >= 1

    def integrate_erk1(self, u0: np.array, t: float, dt: float, de_solver: DESolver) -> np.array:
        if not np.isclose(self.nodes[0], 0.0):
            raise Exception("SDC nodes must include the left interval boundary.")

        if not np.isclose(self.nodes[-1], 1.0):
            raise Exception("SDC nodes must include the right interval boundary.")

        assert self.N == self.deltas.shape[0]

        shape = (self.N,) + u0.shape
        u = np.zeros_like(u0, shape=shape)

        u[0, :] = u0
        evalF_k0 = np.empty_like(u)
        evalF_k1 = np.empty_like(u)

        #
        # Propagate initial condition to all nodes
        #
        for m in range(0, self.N):
            if m > 0:
                u[m] = u[m - 1] + dt * self.deltas[m] * evalF_k0[m - 1]
            evalF_k0[m] = de_solver.evalF(u[m], t + dt * self.nodes[m])

        #
        # Iteratively sweep over SDC nodes
        #
        for _ in range(1, self.num_sweeps):
            for m in range(0, self.N):

                if m > 0:
                    qeval = self.q[m] @ evalF_k0
                    u[m] = u[m - 1] + dt * (self.deltas[m] * (evalF_k1[m - 1] - evalF_k0[m - 1]) + qeval)

                evalF_k1[m] = de_solver.evalF(u[m], t + dt * self.nodes[m])

            # Copy tendency arrays
            evalF_k0[:] = evalF_k1[:]

        if 0:
            # If we're using Radau-right, we can just use the last value
            u[0] = u[-1]

        else:
            # Compute new starting value with quadrature on tendencies
            u[0] = u[0] + dt * self.weights @ evalF_k0

        assert u0.shape == u[0].shape
        return u[0]

    def integrate_irk1(self, u0: np.array, t: float, dt: float, de_solver: DESolver) -> np.array:
        if not np.isclose(self.nodes[0], 0.0):
            raise Exception("SDC nodes must include the left interval boundary.")

        if not np.isclose(self.nodes[-1], 1.0):
            raise Exception("SDC nodes must include the right interval boundary.")

        assert self.N == self.deltas.shape[0]

        shape = (self.N,) + u0.shape
        u = np.zeros_like(u0, shape=shape)

        u[0, :] = u0
        evalF_k0 = np.empty_like(u)
        evalF_k1 = np.empty_like(u)

        # Backup integrator contribution I[...] of previous iteration
        ISolves = np.empty_like(u)

        #
        # Propagate initial condition to all nodes
        #
        for m in range(0, self.N):
            if m > 0:
                #
                # Solve implicit step:
                #
                # u^n+1 = u^n + dt * delta * F(u^n+1)
                # <=> u^n+1 - dt * delta * F(u^n+1) = u^n
                # <=> (I - dt * delta * F) * u^n+1 = u^n
                #
                rhs = u[m - 1]
                u[m] = de_solver.fSolve(rhs, dt * self.deltas[m], t + dt * self.nodes[m])
                # Compute I[...] term
                # u^n+1 = u^n + dt * delta * F(u^n+1)
                # dt * delta * F(u^n+1) = u^n+1 - u^n
                ISolves[m] = u[m] - u[m - 1]

            evalF_k0[m] = de_solver.evalF(u[m], t + dt * self.nodes[m])

        #
        # Iteratively sweep over SDC nodes
        #
        for _ in range(1, self.num_sweeps):
            for m in range(0, self.N):
                if m > 0:
                    #
                    # Solve implicit step:
                    #
                    # u^n+1 = u^n + dt * delta * (F(u^n+1)) - I(u^n) + dt * Q * F(u^n)
                    # <=> u^n+1 - dt * delta * F(u^n+1) = u^n - I(u^n) + dt * Q * F(u^n)
                    # <=> (I - dt * delta * F) * u^n+1 = u^n - I(u^n) + dt * Q * F(u^n)
                    #
                    # rhs = u^n - I(u^n) + dt * Q * F(u^n)
                    #
                    qeval = self.q[m] @ evalF_k0
                    rhs = u[m - 1] - ISolves[m] + dt * qeval

                    u[m] = de_solver.fSolve(rhs, dt * self.deltas[m], t + dt * self.nodes[m])

                    # Update I[...] term for next correction
                    # <=> - dt * delta * F(u^n+1) = u^n - I(u^n) + dt * Q * F(u^n) - u^n+1
                    # <=> dt * delta * F(u^n+1) = u^n+1 - rhs
                    ISolves[m] = u[m] - rhs

                evalF_k1[m] = de_solver.evalF(u[m], t + dt * self.nodes[m])

            # Copy tendency arrays
            evalF_k0[:] = evalF_k1[:]

        if 0:
            # If we're using Radau-right, we can just use the last value
            u[0] = u[-1]

        else:
            # Compute new starting value with quadrature on tendencies
            u[0] = u[0] + dt * self.weights @ evalF_k0

        assert u0.shape == u[0].shape
        return u[0]

    def integrate_imex21(self, u0: np.array, t: float, dt: float, de_solver: DESolver) -> np.array:
        """
        IMEX SDC where the first term is treated implicitly and the second term explicitly.
        """
        if not np.isclose(self.nodes[0], 0.0):
            raise Exception("SDC nodes must include the left interval boundary.")

        if not np.isclose(self.nodes[-1], 1.0):
            raise Exception("SDC nodes must include the right interval boundary.")

        assert self.N == self.deltas.shape[0]

        shape = (self.N,) + u0.shape
        u = np.zeros_like(u0, shape=shape)

        u[0, :] = u0
        evalF_k0 = np.empty_like(u)
        evalF_k1 = np.empty_like(u)

        # Backup integrator contribution I[...] of previous iteration
        ISolves = np.empty_like(u)

        #
        # Propagate initial condition to all nodes
        #
        for m in range(0, self.N):
            if m > 0:
                #
                # Solve explicit step first
                # u* = u^n + dt * delta * F1(u^n)
                #
                rhs = u[m - 1] + dt * self.deltas[m] * de_solver.evalF1(u[m - 1], t + dt * self.nodes[m])

                #
                # Solve implicit step next (see integrate_irk1)
                # u^{n+1} = u* + dt * delta * F2(u^{n+1})
                #
                u[m] = de_solver.fSolve2(rhs, dt * self.deltas[m], t + dt * self.nodes[m])

                #
                # Compute I[...] term for implicit and explicit parts
                #
                # u^n+1 = u^n + dt * delta * F1(u^n) + dt * delta * F2(u^n+1)
                # dt * delta * F1(u^n) + dt * delta * F2(u^n+1) = u^n+1 - u^n
                #
                ISolves[m] = u[m] - u[m - 1]

            evalF_k0[m] = de_solver.evalF(u[m], t + dt * self.nodes[m])

        #
        # Iteratively sweep over SDC nodes
        #
        for _ in range(1, self.num_sweeps):
            for m in range(0, self.N):
                if m > 0:
                    #
                    # Solve IMEX step:
                    #
                    # u^n+1 = u^n + dt * delta * (F1(u^n) + F2(u^n+1)) - I(u^n) + dt * Q * F(u^n)
                    # <=> u^n+1 - dt * delta * F2(u^n+1) = u^n + dt * delta * F1(u^n) - I(u^n) + dt * Q * F(u^n)
                    # <=> (I - dt * delta * F) * u^n+1 = u^n + dt * delta * F1(u^n) - I(u^n) + dt * Q * F(u^n)
                    #
                    # rhs = u^n + dt * delta * F1(u^n) - I(u^n) + dt * Q * F(u^n)
                    #
                    qeval = self.q[m] @ evalF_k0
                    euler = dt * self.deltas[m] * de_solver.evalF1(u[m - 1], t + dt * self.nodes[m])
                    rhs = u[m - 1] + euler - ISolves[m] + dt * qeval

                    u[m] = de_solver.fSolve2(rhs, dt * self.deltas[m], t + dt * self.nodes[m])

                    #
                    # Update I[...] term for next correction
                    # <=> dt * delta * (F1(u^n) + F2(u^n+1)) = u^n+1 - u^n + I(u^n) - dt * Q * F(u^n)
                    #
                    ISolves[m] = u[m] - u[m - 1] + ISolves[m] - dt * qeval

                evalF_k1[m] = de_solver.evalF(u[m], t + dt * self.nodes[m])

            # Copy tendency arrays
            evalF_k0[:] = evalF_k1[:]

        if 0:
            # If we're using Radau-right, we can just use the last value
            u[0] = u[-1]

        else:
            # Compute new starting value with quadrature on tendencies
            u[0] = u[0] + dt * self.weights @ evalF_k0

        assert u0.shape == u[0].shape
        return u[0]

    def integrate_imex12(self, u0: np.array, t: float, dt: float, de_solver: DESolver) -> np.array:
        """
        IMEX SDC where the first term is treated implicitly and the second term explicitly.
        """
        if not np.isclose(self.nodes[0], 0.0):
            raise Exception("SDC nodes must include the left interval boundary.")

        if not np.isclose(self.nodes[-1], 1.0):
            raise Exception("SDC nodes must include the right interval boundary.")

        assert self.N == self.deltas.shape[0]

        shape = (self.N,) + u0.shape
        u = np.zeros_like(u0, shape=shape)

        u[0, :] = u0
        evalF_k0 = np.empty_like(u)
        evalF_k1 = np.empty_like(u)

        # Backup integrator contribution I[...] of previous iteration
        ISolves = np.empty_like(u)

        #
        # Propagate initial condition to all nodes
        #
        for m in range(0, self.N):
            if m > 0:
                #
                # Solve explicit step first
                # u* = u^n + dt * delta * F1(u^n)
                #
                rhs = u[m - 1] + dt * self.deltas[m] * de_solver.evalF2(u[m - 1], t + dt * self.nodes[m])

                #
                # Solve implicit step next (see integrate_irk1)
                # u^{n+1} = u* + dt * delta * F2(u^{n+1})
                #
                u[m] = de_solver.fSolve1(rhs, dt * self.deltas[m], t + dt * self.nodes[m])

                #
                # Compute I[...] term for implicit and explicit parts
                #
                # u^n+1 = u^n + dt * delta * F1(u^n) + dt * delta * F2(u^n+1)
                # dt * delta * F1(u^n) + dt * delta * F2(u^n+1) = u^n+1 - u^n
                #
                ISolves[m] = u[m] - u[m - 1]

            evalF_k0[m] = de_solver.evalF(u[m], t + dt * self.nodes[m])

        #
        # Iteratively sweep over SDC nodes
        #
        for _ in range(1, self.num_sweeps):
            for m in range(0, self.N):
                if m > 0:
                    #
                    # Solve IMEX step:
                    #
                    # u^n+1 = u^n + dt * delta * (F1(u^n) + F2(u^n+1)) - I(u^n) + dt * Q * F(u^n)
                    # <=> u^n+1 - dt * delta * F2(u^n+1) = u^n + dt * delta * F1(u^n) - I(u^n) + dt * Q * F(u^n)
                    # <=> (I - dt * delta * F) * u^n+1 = u^n + dt * delta * F1(u^n) - I(u^n) + dt * Q * F(u^n)
                    #
                    # rhs = u^n + dt * delta * F1(u^n) - I(u^n) + dt * Q * F(u^n)
                    #
                    qeval = self.q[m] @ evalF_k0
                    euler = dt * self.deltas[m] * de_solver.evalF2(u[m - 1], t + dt * self.nodes[m])
                    rhs = u[m - 1] + euler - ISolves[m] + dt * qeval

                    u[m] = de_solver.fSolve1(rhs, dt * self.deltas[m], t + dt * self.nodes[m])

                    #
                    # Update I[...] term for next correction
                    # <=> dt * delta * (F1(u^n) + F2(u^n+1)) = u^n+1 - u^n + I(u^n) - dt * Q * F(u^n)
                    #
                    ISolves[m] = u[m] - u[m - 1] + ISolves[m] - dt * qeval

                evalF_k1[m] = de_solver.evalF(u[m], t + dt * self.nodes[m])

            # Copy tendency arrays
            evalF_k0[:] = evalF_k1[:]

        if 0:
            # If we're using Radau-right, we can just use the last value
            u[0] = u[-1]

        else:
            # Compute new starting value with quadrature on tendencies
            u[0] = u[0] + dt * self.weights @ evalF_k0

        assert u0.shape == u[0].shape
        return u[0]

    def integrate(self, u0: np.array, t: float, dt: float, de_solver: DESolver) -> np.array:
        if self.time_integration_method == "erk1":
            return self.integrate_erk1(u0, t, dt, de_solver)
        elif self.time_integration_method == "irk1":
            return self.integrate_irk1(u0, t, dt, de_solver)
        elif self.time_integration_method == "imex12":
            return self.integrate_imex12(u0, t, dt, de_solver)
        elif self.time_integration_method == "imex21":
            return self.integrate_imex21(u0, t, dt, de_solver)
        else:
            raise Exception(f"Unsupported time integration within SDC: '{self.time_integration_method}'")

    def integrate_n(self, u0: np.array, t: float, dt: float, num_timesteps, de_solver: DESolver) -> np.array:
        if not np.isclose(self.nodes[0], 0.0):
            raise Exception("SDC nodes must include the left interval boundary.")

        if not np.isclose(self.nodes[-1], 1.0):
            raise Exception("SDC nodes must include the right interval boundary.")

        u_value = u0

        for n in range(num_timesteps):
            u_value = self.integrate(u_value, t + n * dt, dt, de_solver)

        return u_value
