
#Imports ----------------------------------------------
import jax
import jax.numpy as jnp
import jax.scipy.linalg                    # for linear solve
import numpy as onp                         # keep NumPy for file I/O, randomness
import matplotlib.pyplot as plt

# Build network data (unchanged except jnp arrays) ---------------------
ieee14_power_line_data = [
    ('Bus1','Bus2',0.05917), ('Bus2','Bus3',0.19797), ('Bus1','Bus5',0.22304),
    ('Bus2','Bus4',0.17632), ('Bus2','Bus5',0.17388), ('Bus3','Bus4',0.17103),
    ('Bus4','Bus5',0.04211), ('Bus4','Bus7',0.20912), ('Bus4','Bus9',0.55618),
    ('Bus5','Bus6',0.25202), ('Bus6','Bus11',0.19890), ('Bus6','Bus12',0.25581),
    ('Bus6','Bus13',0.13027), ('Bus7','Bus8',0.17615), ('Bus7','Bus9',0.11001),
    ('Bus9','Bus10',0.08450), ('Bus9','Bus14',0.27038), ('Bus10','Bus11',0.19207),
    ('Bus12','Bus13',0.19988), ('Bus13','Bus14',0.34802),
]

# Map bus names to indices assigned first come first served in power_line_data
bus_idx = {}
for f, t, _ in ieee14_power_line_data:
    for b in (f, t):
        id_x= bus_idx.setdefault(b, len(bus_idx))
n = len(bus_idx)

# Build susceptance matrix
B = onp.zeros((n, n))
for f, t, x in ieee14_power_line_data:
    i, j = bus_idx[f], bus_idx[t]
    B[i, j] = B[j, i] = 1.0 / x
B = jnp.asarray(B)

# Ordered list of bus names
bus_names = [None] * n
for name, idx in bus_idx.items():
    bus_names[idx] = name

# Specify which buses are generators and which are VPPs
gen_buses = ['Bus1', 'Bus2', 'Bus3']            # synchronous machines
vpp_buses = ['Bus4', 'Bus5']                    # virtual power plants (example)

is_gen_np = onp.array([name in gen_buses for name in bus_names])
is_vpp_np = onp.array([name in vpp_buses for name in bus_names])

is_gen = jnp.asarray(is_gen_np)
is_vpp = jnp.asarray(is_vpp_np)

# Inertia (H) and damping (D) for generators vs loads
# Generators: H=2.0, D=5.0; Loads: H=0.0001, D=0.01
H = jnp.where(is_gen, 2.0, jnp.where(is_vpp, 0.01, 0.0001))
D = jnp.where(is_gen, 5.0, jnp.where(is_vpp, 0.01, 0.01))

# Integral gain: 1 for VPPs, 0 for generators and loads
K_I = jnp.where(is_vpp, 1.0, 0.0)

# Primary droop constant R (p.u. change per Hz) for generators (inf for others)
R_val = 0.05
R = jnp.where(is_gen, R_val, jnp.inf)
# Droop gain = 1/R (zero for non-generators)
droop_gain = 1.0 / R

# Load disturbance data (placeholder timeline)
t0, tf, h = 0.0, 500.0, 1.0
times = onp.arange(t0, tf+h, h)
Pe = jnp.asarray(onp.load("inputs.npy"))  # shape (n, Nt, 2)

#JAX version of swing_rhs ------------------------------
def swing_rhs(x, k, Pe, B, H, D, K_I, droop_gain):
    """Return time derivative at discrete index k."""
    delta, omega, z = jnp.split(x, 3)
    # electrical power flows
    diff   = delta[:, None] - delta[None, :]
    P_tie  = (B * jnp.sin(diff)).sum(axis=1)
    # control actions: integral + primary droop
    u_int    = -K_I * z
    u_droop  = -droop_gain * omega
    u        = u_int + u_droop
    # state derivatives
    ddelta   = omega
    domega   = (Pe[:, k, 0] - Pe[:, k, 1] + u - P_tie - D * omega) / (2.0 * H)
    dz       = omega
    return jnp.concatenate([ddelta, domega, dz])

#AD‑driven Rosenbrock–Euler step -----------------------
def rosenbrock_euler_step_ad(x, h, k, Pe, B, H, D, K_I, droop_gain):
    f = lambda y: swing_rhs(y, k, Pe, B, H, D, K_I, droop_gain)
    J = jax.jacobian(f)(x)  # 3n × 3n Jacobian
    k1 = jax.scipy.linalg.solve(jnp.eye(x.size) - h * J, f(x))
    return x + h * k1

# JIT‑compile for speed
droop_ros_step = jax.jit(
    rosenbrock_euler_step_ad,
    static_argnames=("h",)
)

#Time integration ---------------------------------------
T = onp.arange(t0, tf+h, h)
Nt = len(T)

X = jnp.zeros((Nt, 3*n))
x = jnp.zeros(3*n)
for k in range(Nt-1):
    x = droop_ros_step(x, h, k, Pe, B, H, D, K_I, droop_gain)
    X = X.at[k+1].set(x)

# ---------------- 8) Plot results -------------------------------------------
X_np = onp.array(X)
colors = [f"C{i%10}" for i in range(n)]
plt.figure(figsize=(8,4))
for i in range(n):
    plt.plot(T, X_np[:, n+i], color=colors[i], label=f"Bus {i+1}")
plt.title("IEEE14bus")
plt.xlabel("Time (s)")
plt.ylabel(r"$\omega$ (pu)")
plt.grid(True, alpha=0.4)
plt.legend(ncol=5, fontsize=8)
plt.xlim(0, tf)
# plt.tight_layout(rect=[0,0,0.85,1])
plt.show()
