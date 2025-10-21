#
# Example adopted from `test_4_sdc.py`
#


import numpy as np
from qmat.playground.diff_eqs.burgers import Burgers
from qmat.playground.time_integration.sdc_integration import SDCIntegration
from qmat.playground.time_integration.rk_integration import RKIntegration


from matplotlib import pyplot as plt

N = 128
nu = 0.2
T: float = 4.0
t: float = 0.0  # Starting point in time

domain_size: float = 2.0 * np.pi
t_ = np.linspace(0, domain_size, N, endpoint=False)

burgers: Burgers = Burgers(N=N, nu=nu, domain_size=domain_size)
u0 = burgers.initial_u0("sine")

burgers.run_tests()


time_integration = "rk2"

if 1:
    results = []

    u_analytical = burgers.u_solution(u0, t=T)

    for nt in range(4):

        num_timesteps = 2**nt * 1000
        print(f"Running simulation with num_timesteps={num_timesteps}")

        dt = T / num_timesteps

        burgers: Burgers = Burgers(N=N, nu=nu, domain_size=domain_size)
        u0 = burgers.initial_u0("sine")

        u = u0.copy()

        if time_integration in RKIntegration.supported_methods:
            rki = RKIntegration(method=time_integration)
            u = rki.integrate_n(u, t, dt, num_timesteps, burgers)

        elif time_integration == "sdc":
            sdci = SDCIntegration(num_nodes=3, node_type="LEGENDRE", quad_type="LOBATTO")
            u = sdci.integrate_n(u, t, dt, num_timesteps, burgers)

        else:
            raise Exception("TODO")

        plt.plot(t_, u, label=f"numerical {num_timesteps}", linestyle="dashed")

        error = np.max(np.abs(u - u_analytical))
        results.append({"N": num_timesteps, "dt": dt, "error": error})

    prev_error = None
    for r in results:
        if prev_error is None:
            conv = None
        else:
            conv = np.log2(prev_error / r["error"])

        print(f"N={r["N"]}, dt={r["dt"]:.6e}, error={r["error"]:.6e}, conv={conv}")
        prev_error = r["error"]

    if 1:
        plt.plot(t_, u_analytical, label="analytical", linestyle="dotted")

        plt.legend()
        plt.show()
