from typing import List
import numpy as np
from qmat.playgrounds.martin.diff_eqs.de_solver import DESolver


class RKIntegration:

    # List of supported time integration methods
    supported_methods: List[str] = ["rk1", "rk2", "rk4", "irk1", "irk2"]

    def __init__(self, method: str):
        assert method in self.supported_methods, "Unsupported RK method"
        self.method = method

    def integrate(self, u0: np.array, t: float, dt: float, de_solver: DESolver) -> np.array:
        u = u0

        if self.method == "rk1":
            # RK1 (Forward Euler)
            return u + dt * de_solver.evalF(u, t)

        elif self.method == "rk2":
            # RK2: Heun's method
            k1 = de_solver.evalF(u, t)
            k2 = de_solver.evalF(u + 0.5 * dt * k1, t + 0.5 * dt)
            return u + dt * k2

        elif self.method == "rk4":
            # Classical RK4 method
            k1 = de_solver.evalF(u, t)
            k2 = de_solver.evalF(u + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = de_solver.evalF(u + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = de_solver.evalF(u + dt * k3, t + dt)

            # Update solution
            return u + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        elif self.method == "irk1":
            # IRK1 (Implicit/backward Euler)
            return de_solver.fSolve(u, dt, t)

        elif self.method == "irk2":
            # IRK2 (Crank-Nicolson method)
            # Implicit step
            u = de_solver.fSolve(u, 0.5*dt, t)
            # Forward step
            u += 0.5 * dt * de_solver.evalF(u, t + 0.5*dt)
            return u

        else:
            raise Exception("TODO")

        return u

    def integrate_n(self, u0: np.array, t: float, dt: float, num_timesteps, de_solver: DESolver) -> np.array:
        u_value = u0

        for n in range(num_timesteps):
            u_value = self.integrate(u_value, t + n * dt, dt, de_solver)

        return u_value
