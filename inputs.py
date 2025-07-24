import numpy as np
import matplotlib.pyplot as plt
import random


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def inputs_res_load(
    bus_number,
    num_steps,
    delta_t=1.0,
    delta=1.0,
    threshold=0.1,
    decay_factor=0.05,
    start_index=0,
    prbs_levels=None,
    prbs_rates=None
):
    """
    Simulates 1D Brownian motion with threshold reset and custom PRBS noise.

    Parameters
    ----------
    bus_number : int or float
        Seed identifier for reproducibility.
    num_steps : int
        Number of time steps.
    delta_t : float, optional
        Time increment (default 1.0).
    delta : float, optional
        Scale of Brownian increments (default 1.0).
    threshold : float, optional
        Reset threshold (default 0.1).
    decay_factor : float, optional
        Decay applied when threshold exceeded (default 0.05).
    start_index : int, optional
        Step to begin motion (default 0).
    prbs_levels : list of float, optional
        Amplitudes for PRBS components (default 0.001/i for i in 1..6).
    prbs_rates : list of float, optional
        Toggle probabilities for PRBS components (default i/2000 for i in 1..6).

    Returns
    -------
    input : ndarray, shape (num_steps, 2)
        Column 0: Brownian motion; Column 1: summed PRBS signal.
    """
    seed = 300 - int((bus_number % 10) * 5)
    np.random.seed(seed)

    if prbs_levels is None:
        prbs_levels = [0.001*random.choice([-1,1]) / i for i in range(1, 7)]
    if prbs_rates is None:
        prbs_rates = [i * delta_t / 5 for i in range(1, 7)]
    if len(prbs_levels) != len(prbs_rates):
        raise ValueError("prbs_levels and prbs_rates must match in length")

    prbs_states = prbs_levels.copy()
    h_sqrt = np.sqrt(delta_t)
    inputs = np.zeros((num_steps, 2))
    cum_sum = 0.0

    for i in range(1, num_steps):
        if i > start_index:
            cum_sum += np.random.normal(0, delta * h_sqrt)
            if abs(cum_sum) > threshold:
                cum_sum *= (1 - decay_factor)
            for j, (level, rate) in enumerate(zip(prbs_levels, prbs_rates)):
                if np.random.rand() < rate:
                    prbs_states[j] = -prbs_states[j]
        inputs[i] = [cum_sum, sum(prbs_states)]

    return inputs

number_of_buses = 14        # no. of buses in the electrical system
num_steps = 500     # number of time steps 
h = 1                  # step size
delta = 0.00005             # for the random walk
threshold = 0.005           # for resetting random walk if it goes beyond 
decay_factor = 0.2          # factor of resetting
t_start_random = 0
start_index = t_start_random / h

t = np.linspace(0, h * num_steps, num_steps)
inputs = [
    inputs_res_load(i + 1, num_steps, h, delta, threshold, decay_factor, start_index)
    for i in range(number_of_buses)
]

fig1 = plt.figure(figsize=(10, 4))
ax1 = fig1.add_subplot(1,1,1)
fig2 = plt.figure(figsize=(10, 4))
ax2 = fig2.add_subplot(1,1,1)
for series in inputs:
    ax1.plot(t, series[:, 0])
    ax2.plot(t, series[:, 1])
ax1.set_xlim(0, h * num_steps)
ax2.set_xlim(0, h * num_steps)
plt.show()

np.save("inputs.npy", inputs)
