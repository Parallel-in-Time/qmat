import numpy as np
from qmat.playground.diff_eqs.dahlquist import Dahlquist
from time_integration.sdc_integration import SDCIntegration
from qmat.playground.time_integration.rk_integration import RKIntegration
from matplotlib import pyplot as plt


def test_dahlquist2():
    u0 = np.array([1.0])  # Initial condition
    T: float = 4 * np.pi  # Time interval
    T: float = 0.5  # Time interval
    t: float = 0.0  # Starting time

    dahlquist: Dahlquist = Dahlquist(lam1=1.0j, lam2=1.0j)

    for time_integration in ["rk1", "rk2", "rk4", "sdc"]:
        print("="*80)
        print(f"Time integration method: {time_integration}")
        print("="*80)
        results = []

        u_analytical = dahlquist.u_solution(u0, t=T)

        for nt in range(4):

            num_timesteps = 2**nt * 10
            print(f"Running simulation with num_timesteps={num_timesteps}")

            dt = T / num_timesteps

            u0 = dahlquist.initial_u0()

            u = u0.copy()

            if time_integration in RKIntegration.supported_methods:
                rki = RKIntegration(method=time_integration)

                u = rki.integrate_n(u, t, dt, num_timesteps, dahlquist)

            elif time_integration == "sdc":
                sdci = SDCIntegration(num_nodes=3, node_type="LEGENDRE", quad_type="LOBATTO", num_sweeps=1)
                u = sdci.integrate_n(u, t, dt, num_timesteps, dahlquist)

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

        if time_integration == "rk1":
            assert results[-1]["error"] < 1e-2
            assert np.abs(results[-1]["conv"]-1.0) < 1e-2

        elif time_integration == "rk2":
            assert results[-1]["error"] < 1e-4
            assert np.abs(results[-1]["conv"]-2.0) < 1e-3

        elif time_integration == "rk4":
            assert results[-1]["error"] < 1e-9
            assert np.abs(results[-1]["conv"]-4.0) < 1e-4

        elif time_integration == "sdc":
            assert results[-1]["error"] < 1e-4
            assert np.abs(results[-1]["conv"]-2.0) < 1e-3
