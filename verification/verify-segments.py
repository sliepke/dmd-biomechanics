import math
import numpy as np
from matplotlib import pyplot as plt

import zxy9_biomechanical_model.simulation as sim


RK4_TIMESTEP = "1e-2"


# -- define the analytical solution for r(t) -- #


k1 = (-1000 + (837998) ** (1/2)) / 2
k2 = (-1000 - (837998) ** (1/2)) / 2
c1 = 1/100  - 1/(100 - 100 * k2 / k1)
c2 = 1/(100 - 100 * k2 / k1)

def r_expected(t):
	return c1 * math.exp(k1 * t) + c2 * math.exp(k2 * t)


# -- construct simulated solution for r(t) -- #


# t values and r values recorded so far
times = []
r_values = []
# callback function that will fill in values, is an input to Simulation.run()
def callback(simu, done):
	global times, r_values
	times += [simu.sim_time]
	r_values += [simu.positions[2] - simu.positions[0] - 1.0]
# make a simulation of 2 point masses lying on the x axis
s = sim.Simulation( \
	"verification-inputs/segments/body.json", \
	"verification-inputs/segments/start-position.json", \
	rk4_timestep = float(RK4_TIMESTEP), \
	# no gravity or ground forces
	g=0.0, A_normal=0.0, B_normal=0.0, zero_velocity_threshold=0.0 \
)
# displace the 2nd point mass's x position by 1 cm
s.positions[2] += 0.01
# run the simulation with the callback function
s.run(stop_time=1.0, misc_callback=callback, misc_callback_period=0.01)


# plot the error between the analytical and simulated solutions


analytic_solution = []
abs_error = []

for i in range(len(times)):
	analytic_solution += [r_expected(times[i])]
	abs_error += [abs(analytic_solution[i] - r_values[i])]

plt.plot(times, abs_error)

plt.title(f"Error with RK4 timestep of {RK4_TIMESTEP}")
plt.xlabel("t (s)")
plt.ylabel("|Analytic r(t) - Numerical r(t)|    (m)")
plt.show()
