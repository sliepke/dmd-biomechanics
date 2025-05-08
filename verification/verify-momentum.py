import numpy as np
from matplotlib import pyplot as plt

import zxy9_biomechanical_model.simulation as sim


RK4_TIMESTEP = "5e-5"


# -- define the Simulation callback that will measure momentum -- #


# masses unfortunately aren't an attribute of Simulation so we just list them here
masses = np.array([
	24.9485, 30.8205,     \
	2.339, 1.0225,        \
	2.339, 1.0225,        \
	6.1765, 2.501, 0.577, \
	6.1765, 2.501, 0.577  \
])

def cross_product_2d(a, b):
	"""
	The 3rd component of the cross product between
	(a but with a 3rd component of 0) and (b but with a 3rd component of 0)
	where a, b are 2d vectors
	"""
	return a[0] * b[1] - a[1] * b[0]

times = []
x_mm = []
y_mm = []
angular_mm_about_origin = []
def measure_mm_callback(simu, done):
	global x_mm, y_mm, angular_mm_about_origin, times
	x_mm_sum = 0.0
	y_mm_sum = 0.0
	ang_mm_sum = 0.0
	for i in range(len(masses)):
		mm = masses[i] * simu.velocities[2 * i : 2 * i + 2]
		angular_mm = cross_product_2d(simu.positions[2 * i: 2 * i + 2], mm)
		x_mm_sum += mm[0]
		y_mm_sum += mm[1]
		ang_mm_sum += angular_mm
	times += [simu.sim_time]
	x_mm += [x_mm_sum]
	y_mm += [y_mm_sum]
	angular_mm_about_origin += [ang_mm_sum]


# -- define the Simulation callback that will set the activations -- #


relaxed_act = np.array([0.0] * 10)
flexion_act = np.array([1.0] * 10)
extension_act = np.array([-1.0] * 10)
def activations_callback(simu):
	if simu.sim_time < 1:
		return relaxed_act
	elif simu.sim_time < 2:
		return flexion_act
	return extension_act


# -- run simulation with the callbacks -- #


s = sim.Simulation( \
	# body and start position files
	"verification-inputs/mm/body.json", \
	"verification-inputs/mm/start-position.json", \
	# gravity and ground forces of 0
	g=0.0, A_normal=0.0, B_normal=0.0, zero_velocity_threshold=0.0, \
	rk4_timestep=float(RK4_TIMESTEP) \
)


s.run(stop_time=6.0, activations_callback=activations_callback, misc_callback=measure_mm_callback, misc_callback_period=0.01, \
# to watch it
speed=0.5, display=True \
)


# -- plot the error between the analytical and simulated solutions -- #


plt.plot(times, x_mm, label="x momentum (kg * m / s)")
plt.plot(times, y_mm, label="y momentum (kg * m / s)")
plt.plot(times, angular_mm_about_origin, label="angular momentum about " \
	"(0, 0) (kg * m^2 / s)")

plt.title(f"Momentums over time with RK4 timestep of {RK4_TIMESTEP}")
plt.xlabel("t (s)")
plt.ylabel("momentum")
plt.legend()
plt.show()
