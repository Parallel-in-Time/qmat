import numpy as np
from matplotlib import pyplot as plt
from qmat.playgrounds.martin.diff_eqs.dahlquist import Dahlquist


N = 500
u0 = np.array([1.0])
t: np.array = np.linspace(0, 4*np.pi, N, endpoint=False)

dahlquist: Dahlquist = Dahlquist(lam1=20.0j, lam2=1.0j)

u_eval = np.array([dahlquist.int_f(u0, _) for _ in t])


plt.plot(t, np.real(u_eval), label="Re(u)")
plt.plot(t, np.imag(u_eval), label="Im(u)")
plt.legend()
plt.show()
