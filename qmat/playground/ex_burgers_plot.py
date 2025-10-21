#
# Example adopted from `test_4_sdc.py`
#


import numpy as np
from diff_eqs.burgers import Burgers


from matplotlib import pyplot as plt

N = 128
nu = 0.2

T: float = 4.0
domain_size: float = 2.0 * np.pi
t = np.linspace(0, domain_size, N, endpoint=False)

burgers: Burgers = Burgers(N=N, nu=nu, domain_size=domain_size)
u0 = burgers.initial_u0("sine")

burgers.run_tests()

if 1:
    nt: int = 1000
    dt: float = T / nt

    if 0:
        for t in [_ * dt for _ in range(nt)]:
            ut = burgers.analytical_integration(u0, t=t)
            print(f"t={t:.3f}, ut[0]={ut[0]:.6f}")

            plt.plot(t, ut)
    else:
        ut = burgers.u_solution(u0, t=T)
        plt.plot(t, u0, label="u0")
        plt.plot(t, ut, label="ut")

    plt.legend()
    plt.show()
