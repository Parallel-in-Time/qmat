import numpy as np
from matplotlib import pyplot as plt
from qmat.playgrounds.martin.diff_eqs.two_freq import TwoFreq


N = 500
t: np.array = np.linspace(0, 4*np.pi, N, endpoint=False)

two_freq: TwoFreq = TwoFreq(lam1=1.0j, lam2=20.0j, lam3=0.5j)

u0 = two_freq.initial_u0()
u_eval = np.array([two_freq.u_solution(u0, _) for _ in t])

for i in range(2):
    plt.plot(t, np.real(u_eval[:, i]), label=f"Re(u[{i}])")
    plt.plot(t, np.imag(u_eval[:, i]), label=f"Im(u[{i}])")

plt.legend()
plt.show()
