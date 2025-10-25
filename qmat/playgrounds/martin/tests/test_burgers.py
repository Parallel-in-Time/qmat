
import numpy as np
from qmat.playgrounds.martin.diff_eqs.burgers import Burgers
from qmat.playgrounds.martin.time_integration.sdc_integration import SDCIntegration
from qmat.playgrounds.martin.time_integration.rk_integration import RKIntegration


def test_burgers():
    N = 128
    nu = 0.2

    T: float = 4.0
    t: float = 0.0  # Starting point in time
    domain_size: float = 2.0 * np.pi

    burgers: Burgers = Burgers(N=N, nu=nu, domain_size=domain_size)
    u0 = burgers.initial_u0("sine")

    burgers.run_tests()

    for time_integration_method in ["rk1", "rk2", "rk4", "sdc"]:
        print("="*80)
        print(f"Time integration method: {time_integration_method}")
        print("="*80)
        results = []

        u_analytical = burgers.int_f(u0, t=T)

        for nt in range(2, 4):

            num_timesteps = 2**nt * 500
            print(f"Running simulation with num_timesteps={num_timesteps}")

            dt = T / num_timesteps

            burgers: Burgers = Burgers(N=N, nu=nu, domain_size=domain_size)
            u0 = burgers.initial_u0("sine")

            u = u0.copy()

            if time_integration_method in RKIntegration.supported_methods:
                rki = RKIntegration(method=time_integration_method)
                u = rki.integrate_n(u, t, dt, num_timesteps, burgers)

            elif time_integration_method == "sdc":
                sdci = SDCIntegration(num_nodes=3, node_type="LEGENDRE", quad_type="LOBATTO")
                u = sdci.integrate_n(u, t, dt, num_timesteps, burgers)

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
            r["conv"] = conv

        if time_integration_method == "rk1":
            assert results[-1]["error"] < 1e-4
            assert np.abs(results[-1]["conv"]-1.0) < 1e-3

        elif time_integration_method == "rk2":
            assert results[-1]["error"] < 1e-7
            assert np.abs(results[-1]["conv"]-2.0) < 1e-3

        elif time_integration_method == "rk4":
            assert results[-1]["error"] < 1e-14

        elif time_integration_method == "sdc":
            assert results[-1]["error"] < 1e-14

        else:
            raise Exception(f"TODO for {time_integration_method}")
