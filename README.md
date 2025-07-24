# PowerSim
Repository for simulating grid frequency dynamics in a large-scale transmission system. It makes use of the Rosenbrock-Euler implicit method (with AutomaticDifferentiation (from JAX) to compute the Jacobian) for numerical integration to allow selection of large simulation time-steps. By default, it uses the per-unit reactance data for transmission lines and connections for the [IEEE 14 bus](https://github.com/ITI/models/tree/master/electric-grid/physical/reference/ieee-14bus), but this can be customized.

To run the simulator, first run the "[inputs.py](inputs.py)" to generate the input and then the power_sim.py for the output.
