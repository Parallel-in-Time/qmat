import numpy as np

if 1:
    N = 500
    u0 = 1.0
    t: np.array = np.linspace(0, 4*np.pi, N, endpoint=False)
    lam1 = 10.j
    lam2 = 1.j

    def u(u0, t) -> np.array:
        return np.exp(t*lam1) * u0 + np.exp(t*lam2) * u0

    def du_dt(u0, t) -> np.array:
        return (lam1*np.exp(t*lam1) * u0 + lam2*np.exp(t*lam2)) * u0

    u_eval: np.array = u(u0, t)

    from matplotlib import pyplot as plt
    plt.plot(t, np.real(u_eval), label="Re(u)")
    plt.plot(t, np.imag(u_eval), label="Im(u)")
    plt.legend()
    plt.show()
