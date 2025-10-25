import numpy as np
from qmat.playgrounds.martin.diff_eqs.two_freq import TwoFreq
from time_integration.sdc_integration import SDCIntegration
from qmat.playgrounds.martin.time_integration.rk_integration import RKIntegration


def test_dahlquist2():
    T: float = 4 * np.pi  # Time interval
    T: float = 0.5  # Time interval
    t: float = 0.0  # Starting time

    two_freq: TwoFreq = TwoFreq(lam1=1.0j, lam2=0.1j)
    u0 = two_freq.initial_u0()

    for time_integration_method in ["rk1", "rk2", "rk4", "irk1", "irk2", "sdc"]:
        if time_integration_method == "sdc":
            micro_time_integration_ = ["erk1", "irk1", "imex12", "imex21"]
        else:
            micro_time_integration_ = ["-"]

        for micro_time_integration in micro_time_integration_:
            print("=" * 80)
            print(f"Time integration method: {time_integration_method} ({micro_time_integration})")
            print("=" * 80)
            results = []

            u_analytical = two_freq.int_f(u0, t=T)

            for nt in range(4):

                num_timesteps = 2**nt * 10
                print(f"Running simulation with num_timesteps={num_timesteps}")

                dt = T / num_timesteps

                u0 = two_freq.initial_u0()

                u = u0.copy()

                if time_integration_method in RKIntegration.supported_methods:
                    rki = RKIntegration(method=time_integration_method)

                    u = rki.integrate_n(u, t, dt, num_timesteps, two_freq)

                elif time_integration_method == "sdc":
                    sdci = SDCIntegration(
                        num_nodes=3,
                        node_type="LEGENDRE",
                        quad_type="LOBATTO",
                        num_sweeps=3,
                        micro_time_integration=micro_time_integration,
                    )
                    u = sdci.integrate_n(u, t, dt, num_timesteps, two_freq)

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

                print(f" - N={r["N"]}, dt={r["dt"]:.6e}, error={r["error"]:.6e}, conv={conv}, mti={micro_time_integration}")
                prev_error = r["error"]
                r["conv"] = conv

            if time_integration_method == "rk1":
                assert results[-1]["error"] < 1e-2
                assert np.abs(results[-1]["conv"] - 1.0) < 1e-2

            elif time_integration_method == "rk2":
                assert results[-1]["error"] < 1e-5
                assert np.abs(results[-1]["conv"] - 2.0) < 1e-3

            elif time_integration_method == "rk4":
                assert results[-1]["error"] < 1e-11
                assert np.abs(results[-1]["conv"] - 4.0) < 1e-2

            elif time_integration_method == "irk1":
                assert results[-1]["error"] < 1e-2
                assert np.abs(results[-1]["conv"] - 1.0) < 1e-2

            elif time_integration_method == "irk2":
                assert results[-1]["error"] < 1e-2
                assert np.abs(results[-1]["conv"] - 2.0) < 1e-2

            elif time_integration_method == "sdc":
                assert results[-1]["error"] < 1e-8
                assert np.abs(results[-1]["conv"] - 4.0) < 1e-3

            else:
                raise Exception(f"TODO for {time_integration_method}")
