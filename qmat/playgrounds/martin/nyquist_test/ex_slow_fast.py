import numpy as np
from qmat.playgrounds.martin.diff_eqs.two_freq import TwoFreq
from qmat.playgrounds.martin.diff_eqs.dahlquist import Dahlquist
from qmat.playgrounds.martin.time_integration.sdc_integration import SDCIntegration
from qmat.playgrounds.martin.time_integration.rk_integration import RKIntegration
from matplotlib import pyplot as plt


s: int = 1
T: float = 1.0 * np.pi * s  # Time interval
t: float = 0.0    # Starting time

time_integration: str = "sdc"
sdc_micro_time_integration: str = "irk1"
sdc_micro_time_integration: str = "erk1"
sdc_micro_time_integration: str = "imex21"
sdc_num_sweeps: int = 6

de_solver: Dahlquist = Dahlquist(lam1=1.0j, lam2=10.0j, ext_scalar=0.5)
u0 = de_solver.initial_u0()

results = []

plt.close()

if 1:
    num_timesteps = 1
    print(f"Running simulation with num_timesteps={num_timesteps}")

    dt = T / num_timesteps

    u_analytical_fin = de_solver.int_f(u0, dt=T, t=0)
    u0 = de_solver.initial_u0()

    u = u0.copy()

    u_: np.ndarray = np.array([u])
    t_: np.ndarray = np.array([t])

    if time_integration in RKIntegration.supported_methods:
        rki = RKIntegration(method=time_integration)

        for n in range(num_timesteps):
            u = rki.int_f(u, t + n * dt, dt, de_solver)

            u_ = np.concatenate((u_, np.expand_dims(u, axis=0)))
            t_ = np.append(t_, t + (n + 1) * dt)

    elif time_integration == "sdc":
        sdci = SDCIntegration(
            num_nodes=6,
            # node_type="LEGENDRE",
            node_type="EQUID",
            quad_type="LOBATTO",
            num_sweeps=sdc_num_sweeps,
            micro_time_integration=sdc_micro_time_integration,
            use_quadrature=False,
        )

        for n in range(num_timesteps):
            u = sdci.int_f(u, t + n * dt, dt, de_solver)

            u_ = np.concatenate((u_, np.expand_dims(u, axis=0)))
            t_ = np.append(t_, t + (n + 1) * dt)

    else:
        raise Exception(f"Unsupported time integration method '{time_integration}'")

    # for i in range(u_.shape[1]):
    for i in [0]:
        # plt.plot(t_, np.real(u_[:, i]), label=f"u[{i}].real", linestyle="dashed")
        plt.plot(t_, np.imag(u_[:, i]), label=f"u[{i}].imag", linestyle="solid")

    for t in t_:
        plt.vlines(t, ymin=-1, ymax=1, colors="gray", linewidth=1)

    error = np.max(np.abs(u - u_analytical_fin))
    results.append({"N": num_timesteps, "dt": dt, "error": error})


if 1:
    # Plot analytical solution
    t_ = np.linspace(0, T, 1000)
    u_analytical_fin = np.array([de_solver.int_f(u0, dt=t, t=0) for t in t_])
    # for i in range(u_analytical_fin.shape[1]):
    for i in [0]:
        # plt.plot(t_, np.real(u_analytical_fin[:, i]), linestyle="dotted", color="black")
        plt.plot(t_, np.imag(u_analytical_fin[:, i]), linestyle="dotted", color="black")


if 0:
    prev_error = None
    for r in results:
        if prev_error is None:
            conv = None
        else:
            conv = np.log2(prev_error / r["error"])

        print(f"N={r["N"]}, dt={r["dt"]:.6e}, error={r["error"]:.6e}, conv={conv}")
        prev_error = r["error"]

if 1:
    plt.legend()
    plt.show()
