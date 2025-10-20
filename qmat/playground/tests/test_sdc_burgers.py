
import numpy as np
from qmat.playground.diff_eqs.burgers import Burgers
from qmat.playground.time_integration.sdc_integration import SDCIntegration


def test_sdc_burgers():
    N = 128
    nu = 0.2

    T: float = 4.0
    domain_size: float = 2.0 * np.pi
    t = np.linspace(0, domain_size, N, endpoint=False)

    burgers: Burgers = Burgers(N=N, nu=nu, domain_size=domain_size)
    u0 = burgers.initial_u0("sine")

    burgers.run_tests()

    for time_integration in ["rk1", "rk2", "rk4", "sdc"]:
        print("="*80)
        print(f"Time integration method: {time_integration}")
        print("="*80)
        results = []

        u_analytical = burgers.analytical_integration(u0, t=T)

        for nt in range(4):

            num_timesteps = 2**nt * 1000
            print(f"Running simulation with num_timesteps={num_timesteps}")

            dt = T / num_timesteps

            burgers: Burgers = Burgers(N=N, nu=nu, domain_size=domain_size)
            u0 = burgers.initial_u0("sine")

            u = u0.copy()

            if time_integration == "rk1":
                u = burgers.step_rk1_n(u, dt, num_timesteps)

            elif time_integration == "rk2":
                u = burgers.step_rk2_n(u, dt, num_timesteps)

            elif time_integration == "rk4":
                u = burgers.step_rk4_n(u, dt, num_timesteps)

            elif time_integration == "sdc":
                sdci = SDCIntegration(num_nodes=3, node_type="LEGENDRE", quad_type="LOBATTO")
                u = sdci.integrate_n(u, dt, num_timesteps, burgers)

            else:
                raise Exception("TODO")

            error = np.max(np.abs(u - u_analytical))
            results.append({"N": num_timesteps, "dt": dt, "error": error})

        prev_error = None
        for r in results:
            if prev_error is None:
                conv = None
            else:
                conv = np.log2(prev_error / r["error"])

            print(f" - N={r["N"]}, dt={r["dt"]:.6e}, error={r["error"]:.6e}, conv={conv}")
            prev_error = r["error"]

        if time_integration == "rk1":
            assert results[-1]["error"] < 1e-4
        elif time_integration == "rk2":
            assert results[-1]["error"] < 1e-8
        elif time_integration == "rk4":
            assert results[-1]["error"] < 1e-14
        elif time_integration == "sdc":
            assert results[-1]["error"] < 1e-14
