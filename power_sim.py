#Imports ----------------------------------------------
import jax
import jax.numpy as jnp
import jax.scipy.linalg              # for linear solve
import numpy as onp                  # keep NumPy for file I/O, randomness
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
gen_buses = ['Bus1', 'Bus2', 'Bus3']        # synchronous machines
vpp_buses = ['Bus4', 'Bus5']                # virtual power plants (example)

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
t0, tf, h = 0.0, 5.0, 0.0001
# T = onp.arange(t0, tf+h, h) # T will be defined after Pe is loaded
Pe = jnp.asarray(onp.load("inputs.npy"))  # shape (n, 501, 2)

#JAX version of swing_rhs ------------------------------
def swing_rhs(x, k, Pe, B, H, D, K_I, droop_gain):
    """Return time derivative at discrete index k."""
    delta, omega, z = jnp.split(x, 3)
    
    # electrical power flows
    diff     = delta[:, None] - delta[None, :]
    P_tie    = (B * jnp.sin(diff)).sum(axis=1)
    
    # control actions: integral + primary droop
    u_int    = -K_I * z
    u_droop  = -droop_gain * omega
    u        = u_int + u_droop
    
    # state derivatives
    ddelta   = omega
    domega   = (Pe[:, k, 0] - Pe[:, k, 1] + u - P_tie - D * omega) / (2.0 * H)
    dz       = omega
    
    return jnp.concatenate([ddelta, domega, dz])

# --- NEW: Trapezoidal rule step solved with Newton's method ---

# Define f(x, step_idx) for convenience
f_k_func = lambda x, step_idx: swing_rhs(x, step_idx, Pe, B, H, D, K_I, droop_gain)
# Get the Jacobian of f w.r.t. the state vector x (arg 0)
J_f_func = jax.jacobian(f_k_func, argnums=0) 

def trapezoidal_step_ad(x_k, h, k, Pe, B, H, D, K_I, droop_gain, tol=1e-6, max_iter=10):
    """
    Performs one step of the Trapezoidal method using Newton's method.
    
    Solves the implicit equation for y = x_{k+1}:
    G(y) = y - x_k - (h/2) * (f(x_k, k) + f(y, k+1)) = 0
    """
    
    # --- Constants for the step ---
    f_k = f_k_func(x_k, k) # f(x_k, k)
    I = jnp.eye(x_k.size)

    # --- Initial Guess ---
    # Use explicit Euler as a simple, good guess for y (which is x_{k+1})
    y_guess = x_k + h * f_k 

    # --- Newton's Iteration (using a while_loop for JIT) ---
    
    # Loop state is (y_guess, iter_num, error)
    init_val = (y_guess, 0, jnp.inf)
    
    def cond_fun(val):
        """Continue if not converged AND max_iter not reached"""
        y_guess, iter_num, error = val
        return jnp.logical_and(error > tol, iter_num < max_iter)

    def body_fun(val):
        """Performs one Newton iteration"""
        y_guess, iter_num, error = val
        
        # G(y) = y_guess - x_k - (h/2) * (f_k + f(y_guess, k+1))
        G = y_guess - x_k - (h / 2.0) * (f_k + f_k_func(y_guess, k+1))
        
        # J_G(y) = I - (h/2) * J_f(y_guess, k+1)
        J_G = I - (h / 2.0) * J_f_func(y_guess, k+1)
        
        # Solve J_G * delta_y = -G
        delta_y = jax.scipy.linalg.solve(J_G, -G)
        
        # Update guess: y_new = y_guess + delta_y
        y_new = y_guess + delta_y
        
        new_error = jnp.linalg.norm(delta_y)
        
        return (y_new, iter_num + 1, new_error)

    # Run the while loop
    (x_k_plus_1, iters, final_error) = jax.lax.while_loop(cond_fun, body_fun, init_val)
    
    return x_k_plus_1

# JIT-compile the new step function
trapezoidal_step_jit = jax.jit(
    trapezoidal_step_ad,
    static_argnames=("h", "tol", "max_iter") 
)


#Time integration ---------------------------------------
T = onp.arange(t0, tf+h, h)
Nt = len(T) # Should be 501, matching Pe.shape[1]

# Check for shape mismatch
if Pe.shape[1] != Nt:
    print(f"Warning: Time vector length ({Nt}) does not match disturbance length ({Pe.shape[1]})!")
    # Adjust Nt to be the smaller of the two to prevent errors
    Nt = min(Nt, Pe.shape[1])
    T = T[:Nt]
    print(f"Adjusted simulation steps to {Nt}.")

# --- MODIFIED: Replaced Python for-loop with jax.lax.scan ---

# Define the function that jax.lax.scan will call at each step.
# Signature must be f(carry, x) -> (new_carry, output)
# carry = x_k (state from previous step)
# x = k (the time index we are scanning over)
def scan_body(x_k, k):
    """Performs one simulation step for scan."""
    # h, Pe, B, etc. are "closed over" from the outer scope.
    
    # --- OPTIMIZATION: Relax solver parameters for speed ---
    tol_realtime = 1e-4
    max_iter_realtime = 5
    # ----------------------------------------------------
    
    x_k_plus_1 = trapezoidal_step_jit(
        x_k, h, k, Pe, B, H, D, K_I, droop_gain,
        tol=tol_realtime, 
        max_iter=max_iter_realtime
    )
    
    # new_carry = x_{k+1}, output_to_stack = x_{k+1}
    return x_k_plus_1, x_k_plus_1

# JIT-compile the *entire* scan operation.
# This compiles the whole simulation loop into one optimized kernel.
@jax.jit
def run_simulation_scan(x_initial, k_indices):
    """Runs the full simulation using jax.lax.scan."""
    final_state, X_stacked = jax.lax.scan(scan_body, x_initial, k_indices)
    return X_stacked

print("Compiling and running JIT-compiled simulation (using jax.lax.scan)...")

# Set up initial state and time indices for the scan
x_initial = jnp.zeros(3*n)
k_indices = jnp.arange(Nt - 1) # Indices from 0 to Nt-2

# Run the compiled simulation
X_stacked = run_simulation_scan(x_initial, k_indices)

# Wait for computation to finish (important for accurate timing)
X_stacked.block_until_ready()
print("Simulation complete.")

# `X_stacked` has shape (Nt-1, 3*n), containing states x_1 through x_{Nt-1}
# We recreate the full state array X(Nt, 3*n) by stacking x_0 on top.
X = jnp.vstack([x_initial.reshape(1, -1), X_stacked])

# --- END OF MODIFICATION ---


# EXAMPLE OF OLD LOOP (for comparison):
# X = jnp.zeros((Nt, 3*n))
# x = jnp.zeros(3*n)
# print("Starting simulation...")
# for k in range(Nt-1): # Loop from k = 0 to Nt-2
#     x = trapezoidal_step_jit(x, h, k, Pe, B, H, D, K_I, droop_gain)
#     X = X.at[k+1].set(x)
# print("Simulation complete.")


# ---------------- 8) Plot results -------------------------------------------
X_np = onp.array(X)
colors = [f"C{i%10}" for i in range(n)]
plt.figure(figsize=(8,4))
for i in range(n):
    plt.plot(T, X_np[:, n+i], color=colors[i], label=f"Bus {i+1}")
plt.title("IEEE14bus (Trapezoidal Method + jax.lax.scan)")
plt.xlabel("Time (s)")
plt.ylabel(r"$\omega$ (pu)")
plt.grid(True, alpha=0.4)
plt.legend(ncol=5, fontsize=8)
plt.xlim(0, tf)
# plt.tight_layout(rect=[0,0,0.85,1])
plt.show()

