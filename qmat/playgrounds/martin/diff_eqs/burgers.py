import numpy as np
from qmat.playgrounds.martin.diff_eqs.de_solver import DESolver


class Burgers(DESolver):
    """
    Class to handle the 1D viscous Burgers' equation.
    """

    def __init__(self, N: int, nu: float, domain_size: float = 2.0 * np.pi):
        # Resolution
        self._N: int = N

        # Viscosity
        self._nu: float = nu

        # Domain size
        self._domain_size: float = domain_size

        # Prepare spectral differentiation values
        self._d_dx_ = 1j * np.fft.fftfreq(self._N, d=1.0 / self._N) * 2.0 * np.pi / self._domain_size

    def _d_dx(self, u: np.ndarray) -> np.ndarray:
        """Compute the spatial derivative of `u` using spectral methods.

        Parameters
        ----------
        u : np.ndarray
            Array of shape (N,) representing the solution at the current time step.

        Returns
        -------
        du_dx : np.ndarray
            Array of shape (N,) representing the spatial derivative of `u`.
        """
        u_hat = np.fft.fft(u)
        du_dx_hat = u_hat * self._d_dx_
        du_dx = np.fft.ifft(du_dx_hat).real
        return du_dx

    def initial_u0(self, mode: str) -> np.ndarray:
        """Compute some initial conditions for the 1D viscous Burgers' equation."""

        if mode == "sine":
            x = np.linspace(0, self._domain_size, self._N, endpoint=False)
            u0 = np.sin(x)

        elif mode == "hat":
            x = np.linspace(0, self._domain_size, self._N, endpoint=False)
            u0 = np.where((x >= np.pi / 2) & (x <= 3 * np.pi / 2), 1.0, 0.0)

        elif mode == "random":
            np.random.seed(42)
            u0 = np.random.rand(self._N)

        else:
            raise ValueError(f"Unknown initial condition mode: {mode}")

        return u0

    def evalF(self, u: np.ndarray, t: float) -> np.ndarray:
        """
        Evaluate the right-hand side of the 1D viscous Burgers' equation.

        Parameters
        ----------
        u : np.ndarray
            Array of shape (N,) representing the solution at the current time step.

        Returns
        -------
        f : np.ndarray
            Array of shape (N,) representing the right-hand side evaluated at `u`.
        """
        # Compute spatial derivatives using spectral methods
        u_hat = np.fft.fft(u)
        du_dx_hat = self._d_dx_ * u_hat
        d2u_dx2_hat = (self._d_dx_**2) * u_hat

        du_dx = np.fft.ifft(du_dx_hat).real
        d2u_dx2 = np.fft.ifft(d2u_dx2_hat).real

        f = -u * du_dx + self._nu * d2u_dx2
        return f

    def u_solution(self, u0: np.ndarray, t: float) -> np.ndarray:
        """
        Compute the analytical solution of the 1D viscous Burgers' equation at time `t`.

        See
        https://gitlab.inria.fr/sweet/sweet/-/blob/6f20b19f246bf6fcc7ace1b69567326d1da78635/src/programs/_pde_burgers/time/Burgers_Cart2D_TS_ln_cole_hopf.cpp

        Parameters
        ----------
        u0 : np.ndarray
            Array of shape (N,) representing the initial condition.
        t : float
            Time at which to evaluate the analytical solution.

        Returns
        -------
        u_analytical : np.ndarray
            Array of shape (N,) representing the analytical solution at time `t`.
        """

        if self._nu < 0.05:
            print("Viscosity is very small which can lead to errors in analytical solution!")

        u0_hat = np.fft.fft(u0)

        # Divide by d/dx operator in spectral space
        tmp = np.zeros_like(u0_hat, dtype=complex)
        tmp[1:] = u0_hat[1:] / self._d_dx_[1:]

        # Back to physical space
        phi = np.fft.ifft(tmp).real

        # Apply exp(...)
        phi = np.exp(-phi / (2 * self._nu))

        phi_hat = np.fft.fft(phi)

        # Solve directly the heat equation in spectral space with exponential integration
        phi_hat = phi_hat * np.exp(self._nu * self._d_dx_**2 * t)

        phi = np.fft.ifft(phi_hat)
        phi = np.log(phi)

        phi_hat = np.fft.fft(phi)

        u1_hat = -2.0 * self._nu * phi_hat * self._d_dx_
        return np.fft.ifft(u1_hat).real

    def test(self):
        """
        Run test for currently set up Burgers instance.
        """
        x = np.linspace(0, self._domain_size, self._N, endpoint=False)
        w = 2.0 * np.pi / self._domain_size
        u0 = np.sin(x * w)

        u1_analytical = np.cos(x * w) * w
        u1_num = self._d_dx(u0)

        error: float = np.max(np.abs(u1_num - u1_analytical))

        if error > 1e-10:
            raise AssertionError(f"Test failed: error {error} too large for domain size {self._domain_size}.")

    def run_tests(self):
        """
        Run basic tests to verify the correctness of the implementation.

        This doesn't change the current instance, but will create new instances.
        """

        for domain_size in [2.0 * np.pi, 1.0, 9.0]:
            burgers = Burgers(N=128, nu=0.01, domain_size=domain_size)
            burgers.test()
