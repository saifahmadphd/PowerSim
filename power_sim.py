# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp
import jax.scipy.linalg                    # for linear solve
import numpy as np                         # keep NumPy for file I/O, randomness
import matplotlib.pyplot as plt
from jax import lax
from jax.scipy.sparse.linalg import gmres
import time
from jax import linearize
'''
Title: Code for Journal on Inertia compensated DVPP, simulation of all node dyn.
using RK4 and adaptive estimator using Euler Explicit
'''

ieee14_power_line_data = [
    ('Bus1','Bus2',0.05917), ('Bus2','Bus3',0.19797), ('Bus1','Bus5',0.22304),
    ('Bus2','Bus4',0.17632), ('Bus2','Bus5',0.17388), ('Bus3','Bus4',0.17103),
    ('Bus4','Bus5',0.04211), ('Bus4','Bus7',0.20912), ('Bus4','Bus9',0.55618),
    ('Bus5','Bus6',0.25202), ('Bus6','Bus11',0.19890), ('Bus6','Bus12',0.25581),
    ('Bus6','Bus13',0.13027), ('Bus7','Bus8',0.17615), ('Bus7','Bus9',0.11001),
    ('Bus9','Bus10',0.08450), ('Bus9','Bus14',0.27038), ('Bus10','Bus11',0.19207),
    ('Bus12','Bus13',0.19988), ('Bus13','Bus14',0.34802),
]

            
buses = {x for x, y, _ in ieee14_power_line_data} | {y for x, y, _ in ieee14_power_line_data} # union of from and to sets
n = len(buses)
x = np.arange(n)
bus_names_sorted = sorted(buses, key = lambda b: int(b[3:]))
bus_idx = dict(zip(bus_names_sorted, x))

bus_names = list(bus_names_sorted)

gen_buses = []            # synchronous machines
vpp_buses = ['Bus1','Bus2', 'Bus3','Bus6', 'Bus8'] # remaining are load buses
is_gen_np = np.array([name in gen_buses for name in bus_names])
is_vpp_np = np.array([name in vpp_buses for name in bus_names])

is_gen = jnp.asarray(is_gen_np)
is_vpp = jnp.asarray(is_vpp_np)

# Inertia (H) and damping (D) for generators vs loads
# Generators: H=2.0, D=5.0; Loads: H=0.0001, D=0.01
H = jnp.where(is_gen, 2.0, jnp.where(is_vpp, 0.01, 0.00001))
D = jnp.where(is_gen, 0.0, jnp.where(is_vpp, 0.1, 0.01))

# Secondary integral gain: non-zero value for Gens, 0 for vpps and loads
K_sec = jnp.where(is_gen, 0.5, 0.0)

w_bw = jnp.where(is_vpp, 10, 0.0)
BIG = 1e9
# Boiler reheater time-constant
T_mech = jnp.where(is_gen, 5, BIG)
T_bess = jnp.where(is_vpp, 1, BIG)
T_k_input = jnp.where(is_vpp, 0.01, 0.01)
T_uk_input = jnp.where(is_vpp, 0.01, 0.01)
# Primary droop constant R (p.u. change per Hz) for generators (inf for others)
R_val = 0.05
R = jnp.where(is_gen, R_val, BIG)
# Droop gain = 1/R (zero for non-generators)
droop_gain = 1.0 / R
P_droop_cap = 0.1  # 10 percant of total power capacity i.e. 1 p.u = 100 MVA or MW
# Load disturbance data 
t0, tf, h = 0.0, 100.0, 0.001
times = np.arange(t0, tf+h, h)
Pe = jnp.asarray(np.load("inputs.npy"))  # shape (n, Nt, 2)
no_of_states = 11 # (delta, omega, P_mech, P_s) 

# sparse edge based tie-line power flow
send = np.array([bus_idx[x] for (x, y, _) in ieee14_power_line_data])
recv = np.array([bus_idx[y] for (x, y, _) in ieee14_power_line_data])
w    = np.array([1.0/x for (_,_,x) in ieee14_power_line_data])

# convert to jax arrays
send = jnp.asarray(send)
recv = jnp.asarray(recv)
w    = jnp.asarray(w)

# compute P_tie
@jax.custom_jvp
def tie_power(delta, send, recv, w):
    d  = delta[send] - delta[recv]
    s  = w * jnp.sin(d)
    out = jnp.zeros_like(delta)
    out = out.at[send].add(s)
    out = out.at[recv].add(-s)
    return out

@tie_power.defjvp
def tie_power_jvp(primals, tangents):
    delta, send, recv, w = primals
    v,     _,    _,    _ = tangents  # only delta has a tangent

    d    = delta[send] - delta[recv]
    dv   = v[send]     - v[recv]
    ds   = w * jnp.cos(d) * dv       # d/d(delta) [w*sin(d)] @ v

    jout = jnp.zeros_like(delta)
    jout = jout.at[send].add(ds)
    jout = jout.at[recv].add(-ds)

    return tie_power(delta, send, recv, w), jout


# compute right hand side f(x)
def swing_rhs_sparse(x, k, Pe, send, recv, w, n, 
                     H, D, K_sec, droop_gain, P_droop_cap, 
                     T_mech, T_bess, T_k_input, T_uk_input, w_bw, no_of_states):
    delta, omega, P_mech, P_sec, hat_omega, hat_P_uk, hat_theta, u_bess, P_k_input, P_uk_input, e_omega_f = jnp.split(x, no_of_states)

    P_tie = tie_power(delta, send, recv, w)  # O(E) not O(n^2)

    u_central = - hat_P_uk - P_k_input - Pe[:, k, 2] - P_mech   # <-- make sure your Pe indexing matches its shape!
    regressor = P_k_input + hat_P_uk + P_mech - P_tie + u_bess - D * omega

    alpha = 0.01
    e_omega_hpf = (omega - hat_omega) - e_omega_f

    ddelta   = omega
    domega   = (P_k_input + P_uk_input + P_mech - P_tie + u_bess - D * omega) / (2.0 * H)
    dP_mech  = (P_sec - P_mech - droop_gain * P_droop_cap * omega)/T_mech
    dP_sec   = -K_sec * omega

    dhat_omega = regressor * hat_theta + 2 * w_bw * (omega - hat_omega)
    dhat_Pe_uk = w_bw**2 * (omega - hat_omega)
    dhat_theta = (w_bw**8) * regressor/(1+regressor**2) * jnp.clip(omega - hat_omega, -0.1, 0.1) - 0.001 * hat_theta
    du_bess    = (u_central - u_bess)/T_bess
    dP_k_input  = (Pe[:, k, 0] - Pe[:, k, 1] - P_k_input)/T_k_input
    dP_uk_input = (-Pe[:, k, 1] - P_uk_input)/T_uk_input
    de_omega_f  = (-e_omega_f + (omega - hat_omega)) / alpha

    return jnp.concatenate([ddelta, domega, dP_mech, dP_sec, dhat_omega, dhat_Pe_uk,
                            dhat_theta, du_bess, dP_k_input, dP_uk_input, de_omega_f])


def rosenbrock_euler_step_matfree(
    x, h, k, Pe, send, recv, w, n,
    H, D, K_sec, droop_gain, P_droop_cap,
    T_mech, T_bess, T_k_input, T_uk_input, w_bw, no_of_states
):
    # f returns f(x) as before (uses your sparse tie-line)
    f = lambda y: swing_rhs_sparse(
        y, k, Pe, send, recv, w, n,
        H, D, K_sec, droop_gain, P_droop_cap,
        T_mech, T_bess, T_k_input, T_uk_input, w_bw, no_of_states
    )

    fx, jvp = linearize(f, x)  # jvp(v) = J(x) @ v

    def mv(v):
        return v - h * jvp(v)

    # 
    k1, info = gmres(mv, fx, tol=1e-4, maxiter=50)
    return x + h * k1

# a single ros step

droop_ros_step = jax.jit(
    rosenbrock_euler_step_matfree,
    static_argnames=("no_of_states",)   # keep h as JAX value if it may change
)


# ---------------- Simulation setup & params ----------------
no_of_states = 11  # [delta, omega, P_mech, P_sec, hat_omega, hat_P_uk, hat_theta, u_bess, P_k_input, P_uk_input, e_omega_f]

# Time grid
T = jnp.arange(t0, tf + h, h)
Nt = int(T.size)

# ---------------------------------------------------------------------


# advance one step
def one_step(x, k):
    x_next = droop_ros_step(x, h, k, Pe, send, recv, w, n, H, D, K_sec, droop_gain,
                            P_droop_cap, T_mech, T_bess, T_k_input, T_uk_input, w_bw, no_of_states)
    return x_next, x_next




xs0 = jnp.zeros(no_of_states*n)
_, X = lax.scan(one_step, xs0, jnp.arange(Nt-1))
X = jnp.vstack([xs0, X])  # prepend initial


# compile time computation
y = droop_ros_step(xs0, h, jnp.int32(1), Pe, send, recv, w, n, H, D, K_sec,
                        droop_gain, P_droop_cap, T_mech, T_bess, 
                        T_k_input, T_uk_input, w_bw, no_of_states)
jax.block_until_ready(y)  # warm-up
t0 = time.perf_counter()
y = droop_ros_step(xs0, h, jnp.int32(2), Pe, send, recv, w, n, H, D, K_sec,
                        droop_gain, P_droop_cap, T_mech, T_bess, 
                        T_k_input, T_uk_input, w_bw, no_of_states)
jax.block_until_ready(y)
t1 = time.perf_counter()
print("Step time:", t1 - t0)

# --- Plots for VPP buses only ---
X_np = np.asarray(X)          # (Nt, no_of_states*n)
T_np = np.asarray(T)

vpp_indices = np.where(is_vpp_np)[0]   # indices of VPP buses
plt.figure(figsize=(9,4))
for i in vpp_indices:
    plt.plot(T_np, X_np[:, 1*n + i], label=bus_names[i])  # omega block starts at offset n

plt.title("VPP bus frequencies")
plt.xlabel("Time (s)")
plt.ylabel(r"$\omega$ (p.u.)")
plt.grid(True, alpha=0.4)
plt.legend(ncol=4, fontsize=8)
plt.xlim(T_np[0], T_np[-1])
plt.tight_layout()
plt.show()

