import numpy as np
from qmat.playground.diff_eqs.dahlquist2 import Dahlquist2


N = 500
u0 = np.array([1.0])
t: np.array = np.linspace(0, 4*np.pi, N, endpoint=False)

dahlquist = Dahlquist2(lam1=10.j, lam2=1.j)

u_eval = np.array([dahlquist.u_solution(u0, _) for _ in t])


if 1:
    from matplotlib import pyplot as plt
    plt.plot(t, np.real(u_eval), label="Re(u)")
    plt.plot(t, np.imag(u_eval), label="Im(u)")
    plt.legend()
    plt.show()
