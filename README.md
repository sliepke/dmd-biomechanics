# Overview

The biomechanical model is mostly written as a C extension to Python. To make it easier to build, it has been uploaded to the python package index and can be installed with:

`pip install zxy9_biomechanical_model`

Note that there is only a source distribution, and I had linkage complications when building on Windows (Linux was fine though). For some reason, running `python.exe -m pip install zxy9_biomechanical_model` on Windows from the directory python.exe was located in allowed it to build perfectly fine.

# Documentation

from python:

`>> import zxy9_biomechanical_model.simulation as sim`

`>> help(sim.Simulation)`

# Examples

## Keep the shoulder joint at an angle of 90 degrees, while floating in outer space

	import math
	import numpy as np
	import zxy9_biomechanical_model.simulation as sim
	
	# index of 'shoulder1' in the 'joints' object of body.json
	SHOULDER_JOINT_INDEX = 0
	
	def activations(simu):
		angle_error = (90 * math.pi / 180) - simu.thetas[SHOULDER_JOINT_INDEX]
		shoulder_act = min(1, max(-1, angle_error))
		acts = np.array([0.0] * 10)
		acts[SHOULDER_JOINT_INDEX] = shoulder_act
		return acts
	
	s = sim.Simulation( \
		"RL/RL-inputs/human/body.json", "RL/RL-inputs/human/start-position.json", \
		# no gravity or ground forces
		g=0, A_normal=0, B_normal=0, zero_velocity_threshold=0 \
	)
	
	s.run(display=True, speed=1, activations_callback=activations)

## Measure average height of 'top of torso' point, with all zero activations

	import math
	import numpy as np
	import zxy9_biomechanical_model.simulation as sim
	
	# activation function
	
	def activations(simu):
		return np.array([0] * 10, dtype=np.double)
	
	# define the function going into the 'misc callback' arg of Simulation.run()

	# index of 'top' in the 'point-masses' object of body.json
	TOP_POINT_INDEX = 0
	STOP_TIME       = 2
	NUM_TICKS       = 20
	
	height_sum      = 0.0
	
	def callback(simu, done):
		global height_sum
		# measure height at t = 0, 0.1, ..., 1.9
		if not done:
			# y position = 2 * point_index + 1
			height_sum += simu.positions[2 * TOP_POINT_INDEX + 1]
		# at t = 2.0, print average height
		else:
			average_height = height_sum / NUM_TICKS
			print(f"average height of 'top of torso': {average_height}")
	
	# run simulation
	
	s = sim.Simulation("RL/RL-inputs/human/body.json", "RL/RL-inputs/human/start-position.json",)
	s.run(
		display=True, speed=1, \
		activations_callback=activations, \
		misc_callback=callback, misc_callback_period = STOP_TIME / NUM_TICKS,
		stop_time=STOP_TIME \
	)
