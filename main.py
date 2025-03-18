import datetime

import time
import tkinter as tk

import numpy as np

import environment
import body
import activation


# Parameters to display canvas window
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
# Parameters / functions to translate positions into canvas units
CANVAS_WIDTH_METERS = 6
CANVAS_SCALE = CANVAS_WIDTH / CANVAS_WIDTH_METERS
def _translate(pos):
	x = (CANVAS_WIDTH / 2) + pos[0] * CANVAS_SCALE
	y = (CANVAS_HEIGHT / 2) - pos[1] * CANVAS_SCALE
	return (x, y)

STOP_TIME = None
debugged = False


def display_simulation(env, body, act, starting_positions, timestep=None, \
	speed=1.0, speednote=None):
	"""
	...
	default timestep is the timestep of the activation function, if it has one
	speed is the rate at which the simulation ticks relative to real time
	"""
	global debugged
	if timestep is None:
		timestep = act.get_timestep()
	if timestep is None:
		raise ValueError("No timestep was given, and activation function does not have one")
	
	# check that body's number of point masses correspond with starting position
	body_num_pts = body.get_num_point_masses()
	start_num_pts = len(starting_positions)
	if start_num_pts % 2 != 0:
		raise ValueException("The number of starting coordinates must be even, " + \
			"but was " + str(start_num_pts))
	start_num_pts /= 2
	if body_num_pts != start_num_pts:
		raise ValueException("Was given " + str(start_num_pts * 2) + " starting " + \
			" coordinates, but there are " + str(body_num_pts) + " point masses " + \
			"in this body")
	# check that body's number of joints correspond with activation function
	body_num_jts = body.get_num_joints()
	act_num_jts = act.get_num_joints()
	if body_num_jts != act_num_jts:
		raise ValueException("Activation function is on " + str(act_num_jts) +
			" joints, but there are " + str(body_num_jts) + " joints in this body")
	
	velocities = np.array([0.0] * (body_num_pts * 2))
	# y (the vector being integrated) is 2d, and is indexed as
	#     [0 for position / 1 for velocity]
	#     [2 * point mass index + (0 for x component / 1 for y component)]
	y = np.array([starting_positions, velocities])
	
	force_divisor = np.ones(body_num_pts * 2) / np.repeat(body.point_masses, 2)
	
	# dy/dt function
	def f(t, y):
		global debugged
		# bodily forces
		acc = body.forces(y[0], y[1], act.activations(t))
		# environment forces
		for point in range(body_num_pts):
			# get mass, position, velocity, and bodily forces on this point
			mass = body.point_masses[point]
			p_ind = 2 * point
			pos = y[0][p_ind : p_ind + 2]
			vel = y[1][p_ind : p_ind + 2]
			fc = acc[p_ind : p_ind + 2]
			# use mass, pos, vel, and bodily force to determine environment forces
			env_fc = env.force(mass, pos, vel, fc)
			acc[p_ind] += env_fc[0]
			acc[p_ind + 1] += env_fc[1]
		acc *= force_divisor
		
		# debugging stuff
		if 0==1 and not debugged:
			print("SPECIAL DEBUGGING START")
			print("t=", t)
			print("positions, velocities, forces:")
			for i in range(body_num_pts):
				print("\t", body.point_names[i], "(", i, ")")
				print("\t\t", y[0][2 * i], ",", y[0][2 * i + 1])
				print("\t\t", y[1][2 * i], ",", y[1][2 * i + 1])
				print("\t\t", acc[2 * i], ",", acc[2 * i + 1])
			print("FORCE CALCULATION START")
			force = body.forces(y[0], y[1], act.activations(t), debug=True)
			print("FORCE CALCULATION END")
			print("SPECIAL DEBUGGING END")
			debugged = True
		if 0==1 and (not(STOP_TIME is None)) and t >= STOP_TIME:
			print("\nEND OF SIMULATION")
			print("t:", t)
			print("positions, velocities, forces:")
			for i in range(body_num_pts):
				print("\t", body.point_names[i], "(", i, ")")
				print("\t\t", y[0][2 * i], ",", y[0][2 * i + 1])
				print("\t\t", y[1][2 * i], ",", y[1][2 * i + 1])
				print("\t\t", force[2 * i], ",", force[2 * i + 1])
			root.mainloop()
			exit()
		
		return np.array([y[1], acc])
	
	# -- keep integrating forward and painting body on canvas  -- #
	# -- until activation function throws or user ends program -- #
	
	# create canvas
	root = tk.Tk()
	root.geometry(str(CANVAS_WIDTH) + "x" + str(CANVAS_HEIGHT))
	canvas = tk.Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="white")
	canvas.pack()
	lines = [None] * len(body.segments) # id of each line on canvas
	
	# paint environment on canvas
	# ...
	
	# find duration of each real world timestep, and when next one is
	step_delta = datetime.timedelta(seconds = timestep / speed)
	start = datetime.datetime.today()
	now = start
	next_real_timestep = now + step_delta
	if not(speednote is None):
		speednote_delta = datetime.timedelta(seconds = speednote)
		next_speednote = now + speednote_delta
		last_spnt_t = 0.0
	
	t = 0.0
	while True:
		# paint body on canvas
		for i in range(len(lines)):
			line = lines[i]
			if not(line is None):
				canvas.delete(line)
			p1_ind = 2 * body.segments[i][0]
			p2_ind = 2 * body.segments[i][1]
			newline = canvas.create_line( \
				_translate(y[0][p1_ind : p1_ind + 2]), \
				_translate(y[0][p2_ind : p2_ind + 2]))
			lines[i] = newline
		root.update()
		
		# update positions and velocities
		try:
			if 0==1 and t > 1 and not debugged:
				rk4_step_fast2(f, t, y, timestep, debug=True)
				debugged = True
			else:
				y = rk4_step(f, t, y, timestep)
		# (activation function throws ValueError if not defined at t)
		except ValueError as e:
			print("Stopping simulation:", str(e))
			real_time_passed = (datetime.datetime.today() - start).total_seconds()
			print("Average speed:", t / real_time_passed)
			exit()
			#root.mainloop()
		
		now = datetime.datetime.today()
		
		# give a speed note, if asked for
		if not(speednote is None) and now >= next_speednote:
			sim_time_passed = t - last_spnt_t
			observed_speed = sim_time_passed / speednote
			print("Actual speed:", observed_speed)
			print("Simulation time:", t)
			next_speednote += speednote_delta
			last_spnt_t = t
		
		# wait until next real world time step has arrived
		sleeptime = (next_real_timestep - now).total_seconds()
		if sleeptime > 0:
			time.sleep(sleeptime)
		
		next_real_timestep += step_delta
		t += timestep


def rk4_step(f, t, y, h):
	k1 = f(t, y)
	k2 = f(t + h / 2, y + h * k1 / 2)
	k3 = f(t + h / 2, y + h * k2 / 2)
	k4 = f(t + h, y + h * k3)
	return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def euler_step(f, t, y, h):
	return y + h * f(t, y)


if __name__ == "__main__":
	# File names to load data from
	env_file = 2
	# env_file can be:
	#     - 1: empty environment
	#     - 2: line environment
	#     - string: complex environment loaded from filename
	body_file = "human-inputs/body.json"
	bodypos_file = "human-inputs/start-position.json"
	act_file = "human-inputs/activation.json"
	
	# Load file data into Environment/Body/Activation instances and starting position vector
	if env_file == 1:
		env = environment.EmptyEnvironment()
	elif env_file == 2:
		env = environment.LineEnvironment()
	else:
		env = environment.ComplexEnvironment(env_file)
	body = body.Body(body_file)
	start_pos = body.positions(bodypos_file)
	print("starting positions:", start_pos, sep="\n")
	act = activation.SavedActivation(act_file)
	
	# Run and display simulation
	display_simulation(env, body, act, start_pos, timestep=0.001, speed=1, speednote=4)

