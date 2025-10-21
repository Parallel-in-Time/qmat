import numpy as np
from qmat.playgrounds.martin.diff_eqs.dahlquist2 import Dahlquist2
#from qmat.playgrounds.martin.diff_eqs.dahlquist import Dahlquist
from time_integration.sdc_integration import SDCIntegration
from qmat.playgrounds.martin.time_integration.rk_integration import RKIntegration
from matplotlib import pyplot as plt


u0 = np.array([1.0])  # Initial condition
T: float = 2 * np.pi  # Time interval
t: float = 0.0  # Starting time

time_integration = "rk4"

dahlquist2: Dahlquist2 = Dahlquist2(lam1=20.0j, lam2=1.0j, s=0.1)

results = []

plt.close()

for nt in range(1):

    num_timesteps = 2**nt * 300
    print(f"Running simulation with num_timesteps={num_timesteps}")

    dt = T / num_timesteps

    u_analytical_fin = dahlquist2.u_solution(u0, t=T)
    u0 = dahlquist2.initial_u0()

    u = u0.copy()

    u_: np.ndarray = np.array([u])
    t_: np.ndarray = np.array([t])

    if time_integration in RKIntegration.supported_methods:
        rki = RKIntegration(method=time_integration)

        for n in range(num_timesteps):
            u = rki.integrate(u, t + n * dt, dt, dahlquist2)

            u_ = np.concatenate((u_, np.expand_dims(u, axis=0)))
            t_ = np.append(t_, t + (n + 1) * dt)

    elif time_integration == "sdc":
        sdci = SDCIntegration(num_nodes=3, node_type="LEGENDRE", quad_type="LOBATTO")

        for n in range(num_timesteps):
            u = sdci.integrate(u, t + n * dt, dt, dahlquist2)

            u_ = np.concatenate((u_, np.expand_dims(u, axis=0)))
            t_ = np.append(t_, t + (n + 1) * dt)

    else:
        raise Exception(f"Unsupported time integration method '{time_integration}'")

    plt.plot(t_, np.real(u_), label=f"ndt={num_timesteps}, real", linestyle="dashed")
    plt.plot(t_, np.imag(u_), label=f"ndt={num_timesteps}, imag", linestyle="solid")

    error = np.max(np.abs(u - u_analytical_fin))
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
    t_ = np.linspace(0, T, 1000)
    u_analytical_fin = np.array([dahlquist2.u_solution(u0, t) for t in t_])
    plt.plot(t_, np.real(u_analytical_fin), linestyle="dotted", color="gray")
    plt.plot(t_, np.imag(u_analytical_fin), linestyle="dotted", color="gray")

if 1:
    plt.ylim(-1.5, 1.5)

    plt.legend()
    plt.show()
