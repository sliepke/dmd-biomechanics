from math import pi
from copy import deepcopy

import numpy as np

# imports for display
import datetime
import time
import tkinter as tk

# implements the gymnasium.Env class which means it applies an API letting
# reinforcement learning models train from it
from gymnasium import Env
from gymnasium import spaces
from random import random, randint

# user defined imports
from . import sim_c_util
from .sim_py_util import body

class Simulation(Env):
	"""
	2d biomechanical simulation of human motion, with the physics engine
	implemented in C which runs a lot faster than when written in Python.
			
	The attribute arrays 'positions', 'velocities', 'thetas', and 'thetadots'
	are always up to date with the last RK4 tick or reset(). They allow callback
	functions [see __init__(env_reward_functions=X) and
	run(activations_callback=X, misc_callback=X)] to read the movement of the
	body.
	
	All 6 state arrays can be modified (by changing elements, not the
	reference), and the Simulation will observe the new values. However,
	changing elements of thetas or thetadots has no effect.
	
	If this class's constructor is given the necessary arguments, then this
	class implements the gymnasium.Env API, which lets you use reinforcement
	learning libraries like stablebaselines3 to train agents on one or more
	tasks that you choose (see __init__'s env_* parameters).
	
	Attributes:
		                              STATE
		positions
			1d numpy array of all point mass positions, in the order:
				point1 x, point1 y, point2 x, point2 y, ...
			where the points are in the same order as they appear in the
			"point-masses" object of the JSON file given by 'body_filename'
			(see __init__). Has a dtype of double.
		velocities
			1d numpy array of all point mass velocities (in the same order
			as positions).
		
		thetas
			1d numpy array of all joint angles, where the order of the joints
			is the same as given to __init__ through the 'body_filename'
			JSON file (specifically, its 'joints' object). Is updated every
			time the positions and velocities update, including during reset().
		thetadots
			1d numpy array of all joint angular velocities. Is updated every
			time the positions and velocities update, including during reset().
		
		flexion_c1s
			1d numpy array of all C1F's, as described in the paper. Joints
			are ordered the same as with the 'thetas' attribute. If you write
			to the elements of this array , the simulation will observe the new
			values (change the elements, NOT the reference).
		extension_c1s
			1d numpy array of all C1E's, as described in the paper.
		
		                          GYMNASIUM API
		action_space
			Is a Box with:
				low   = -1.0           high  = 1.0
				shape = (# joints, )   dtype = np.float32
			Each action[i] given by an agent is the activation level of the
			corresponding joint as described in the paper. Joints are ordered
			the same as with the 'thetas' attribute.
		observation_space
			In the most basic case, is a Box concatenating the 'positions' and
			'velocities' attributes, cast to float32's instead of doubles for
			compatibility.
			However, the Box is further concatenated with 'flexion_c1s' and then
			'extension_c1s' if env_strength_low is less than env_strength_high
			(see __init__).
			And, if more than one reward functions are specified in __init__,
			then observation_space is a Tuple(Box, Discrete(# reward functions)),
			where the the value of the Discrete part corresponds to the reward
			function being used during this episode (specifically, its index
			inside the env_reward_functions array, see __init__).
	"""
	
	
	#### Init function ####
	
	
	def __init__(self, \
	# files containing parameters for body (required)
	body_filename, start_position_filename, \
	# gravity acceleration                 (optional)
	g=9.81, \
	# ground force parameters              (optional)
	A_normal=40500.5, B_normal=1e3, mu_s=0.209, mu_k=0.209, zero_velocity_threshold=1e-6, \
	# numerical solver parameters          (optional)
	rk4_timestep=5e-5, \
	# for gymnasium.Env API                (optional)
	env_step_ticks=100, env_stop_times=[], env_reward_functions=[], \
	env_potential_reward_num=None, env_strength_low=1.0, env_strength_high=1.0, \
	env_thetas_diff=0.0, env_velocities_diff=0.0):
		"""
		All parameters other than *_filename or env_* have the same meaning as
		in the paper. The env_* parameters are listed below:
		
		                            REQUIRED FOR GYM ENV
			- env_reward_functions
				list of callables, each with signature (Simulation, bool) -> float.
				The bool is True when it is the last time the reward function
				will be called this episode.
				Every reset(), a random reward function will be chosen for the
				episode. If there is more than one reward function, the
				observation space will include which reward function applies to
				the current episode (see the observation_space attribute).
				Reward functions can determine the amount of reward by reading
				the Simulation's state attributes.
			- env_stop_times
				list of floats, specifying how many seconds an episode lasts
				for, depending on which reward function is being used.
			
		                            OPTIONAL FOR GYM ENV
			
			- env_potential_reward_num
				How many reward functions there could be, so that future additions
				of reward functions don't have to change the observation space
				and potentially restart the entire model.
			- env_step_ticks
				how many RK4 timesteps each step() takes.
			- env_strength_low, env_strength_high
				Each joint strength (C1E or C1F) for an episode is chosen from
				a uniform distribution. The lowest possible value is (the C1
				given in body_filename) * env_strength_low, and the highest is
				is (the C1 given in body_filename) * env_strength_high.
				If env_strength_low < env_strength_high, then the strengths
				are included in the observation space (see the observation_space
				attribute).
			- env_thetas_diff
				If this is nonzero, then each starting joint angle (in degrees)
				after initialization and every reset() will be randomly chosen
				from a uniform distribution in [specified angle - env_thetas_diff,
				specified angle + env_thetas_diff]. Each specified angle is
				given in the start position file.
				More specifically, the 'root2-rotation' will also be similarly
				randomized, then the positions solved for, then all positions
				will be shifted upwards such that the lowest point touches
				the surface of the ground (has a y position of exactly 0).
			- env_velocities_diff
				If this is nonzero, then both components of every point mass's
				velocity (in meters / second) will be randomly chosen after
				initialization and every reset(). Each will be chosen randomly
				from a uniform distribution in
				[original x pos - env_p0_diff, original x pos + env_p0_diff].
		
		Note that the zero_velocity_threshold parameter is not currently
		implemented. The value might have to be carefully chosen to make static
		friction behave as intended, while at the same time not subtly messing
		up the ability of other forces to change the horizontal direction of
		point masses instead of just getting the velocity stuck at 0. Since we
		care about the case when mu_k = mu_s, it shouldn't make an asymptotic
		difference that it's not implemented.
		"""
		# -- convert all args not loaded from a file into the types they're supposed to be -- #
		
		g = float(g)
		A_normal = float(A_normal)
		B_normal = float(B_normal)
		mu_s = float(mu_s)
		mu_k = float(mu_k)
		zero_velocity_threshold = float(zero_velocity_threshold)
		rk4_timestep = float(rk4_timestep)
		
		# -- load body parameters from file into lists -- #
		
		self.body_obj = body.load_body_file(body_filename)
		
		# each m_p (as described in the paper)
		masses_ordered = list(self.body_obj["point-masses"].values())
		
		point_masses = list(self.body_obj["point-masses"])
		num_point_masses = len(point_masses)
		
		# each [a, b] in S
		# (but here, a and b are integer indexes, not strings)
		segments_ordered = [ [point_masses.index(segment[0]), point_masses.index(segment[1])] \
			for segment in self.body_obj["segments"]]
		# list of each l_(a, b) where (a, b) in S
		lengths_ordered = [ segment[2] for segment in self.body_obj["segments"]]
		# segment stiffness and damping
		A_segment = self.body_obj["segment-stiffness"]
		B_segment = self.body_obj["segment-damping"]
		
		# list of each [a, b, c] in J
		joints_ordered = [
			[	point_masses.index(joint["end"]), \
				point_masses.index(joint["center"]), \
				point_masses.index(joint["base"]) \
			] for joint in self.body_obj["joints"].values()]
		# list of parameters of each (a, b, c) in J EXCEPT C1's
		joint_params_ordered = [ \
			[	pi * joint["offset"] / 180,     \
				pi * joint["transition"] / 180, \
				joint["passive-torque-parameters"]["B1"], \
				joint["passive-torque-parameters"]["k1"], \
				joint["passive-torque-parameters"]["B2"], \
				joint["passive-torque-parameters"]["k2"], \
				joint["flexion-parameters"]["C2"], \
				joint["flexion-parameters"]["C3"], \
				joint["flexion-parameters"]["C4"], \
				joint["flexion-parameters"]["C5"], \
				joint["flexion-parameters"]["C6"], \
				joint["extension-parameters"]["C2"], \
				joint["extension-parameters"]["C3"], \
				joint["extension-parameters"]["C4"], \
				joint["extension-parameters"]["C5"], \
				joint["extension-parameters"]["C6"] \
			] for joint in self.body_obj["joints"].values()]
		# B3
		b3 = self.body_obj["joint-damping"]
		# for each joint, C1F
		c1fs_ordered = [joint["flexion-parameters"]["C1"] \
			for joint in self.body_obj["joints"].values()]
		# for each joint, C1E
		c1es_ordered = [joint["extension-parameters"]["C1"] \
			for joint in self.body_obj["joints"].values()]
		
		# -- load start position from file -- #
		# (this will be done again if this is a gymnasium.Env and thetas_diff > 0)
		
		self.start_obj = body.load_start_file(self.body_obj, start_position_filename)
		start_position = body.determine_positions(self.body_obj, self.start_obj)
		
		# -- convert lists into 1d and 2d numpy arrays for easier access in the c code -- #
		
		# parameters to the c code's "Sim_Entry" constructor
		masses_ordered       = np.array(masses_ordered, dtype=np.double)
		segments_ordered     = np.array(segments_ordered, dtype=np.intc)
		lengths_ordered      = np.array(lengths_ordered, dtype=np.double)
		joints_ordered       = np.array(joints_ordered, dtype=np.intc)
		joint_params_ordered = np.array(joint_params_ordered, dtype=np.double)
		# where we can update C1's (the c code will observe the update values)
		self.flexion_c1s     = np.array(c1fs_ordered, dtype=np.double)
		self.extension_c1s   = np.array(c1es_ordered, dtype=np.double)
		# where the c code will update positions + velocities
		self.positions       = np.array(start_position, dtype=np.double)
		self.velocities      = np.zeros(( 2 * num_point_masses ))
		# where the c code will update joint angle + angular velocities
		self.thetas          = np.zeros(( joints_ordered.shape[0] ))
		self.thetadots       = np.zeros(( joints_ordered.shape[0] ))
		
		# -- call the c code's "Sim_Entry" constructor -- #
		
		self.sim_entry = \
		sim_c_util.Sim_Entry(          \
			num_point_masses,          \
			masses_ordered,            \
			g,                         \
			segments_ordered.shape[0], \
			segments_ordered,          \
			lengths_ordered,           \
			A_segment,                 \
			B_segment,                 \
			joints_ordered.shape[0],   \
			joints_ordered,            \
			joint_params_ordered,      \
			b3,                        \
			self.flexion_c1s,          \
			self.extension_c1s,        \
			A_normal,                  \
			B_normal,                  \
			mu_s,                      \
			mu_k,                      \
			zero_velocity_threshold,   \
			self.positions,            \
			rk4_timestep,              \
			self.velocities,           \
			self.thetas,               \
			self.thetadots             \
		)
		
		#  -- keep track of some other things too -- #
		
		self.start_position = start_position.copy()
		self.timestep = rk4_timestep
		self.num_point_masses = num_point_masses
		self.segments = segments_ordered
		self.joints = joints_ordered
		
		# -- if using as a gymnasium.Env -- #
		
		num_tasks = len(env_reward_functions)
		if num_tasks >= 1:
			for r in env_reward_functions: assert callable(r), \
				"Simulation() received a noncallable reward function? what?"
			if len(env_stop_times) < len(env_reward_functions):
				raise ValueError("Must provide a stop time for every task." + \
					f" Was given {len(env_stop_times)} stop times and" + \
					f" {len(env_reward_functions)} reward functions.")
			self.is_gym = True
			#### copy env arguments into self ####
			self.env_step_ticks = env_step_ticks
			self.env_reward_functions = env_reward_functions
			self.env_last_ticks = [int(stop_time / self.timestep) for stop_time in env_stop_times]
			self.env_strength_low = env_strength_low
			self.env_strength_high = env_strength_high
			self.env_thetas_diff = env_thetas_diff
			self.env_velocities_diff = env_velocities_diff
			#### define attributes for env API ####
			# action space
			self.action_space = spaces.Box(low=-1.0, high=1.0, shape=self.thetas.shape, dtype=np.float32)
			# observation space
			num_box_obs = 2 * self.positions.shape[0]
			if env_strength_low < env_strength_high:
				num_box_obs += 2 * self.thetas.shape[0]
			box_obs_space = spaces.Box(-np.inf, np.inf, shape=(num_box_obs, ), dtype=np.float32)
			if num_tasks == 1:
				self.observation_space = box_obs_space
			else:
				if env_potential_reward_num is None:
					env_potential_reward_num = num_tasks
				self.observation_space = spaces.Dict({'state': box_obs_space, \
					'reward': spaces.Discrete(env_potential_reward_num)})
			#### start tracking # ticks elapsed to know when episode ended
			self.env_ticks_elapsed = 0
			#### initially set the random stuff
			# set self's RNG
			self.rng = np.random.default_rng()
			# store base strengths
			self.original_flexion_c1s = np.copy(self.flexion_c1s)
			self.original_extension_c1s = np.copy(self.extension_c1s)
			# randomize stuff (reward function, strengths, positions)
			self.__randomize()
		else:
			self.is_gym = False
		
		# -- debug: do 1 tick with force debugging on -- #
		#activations = np.array([0] * len(self.joints), dtype=np.double)
		#print("#1 positions:\n" + str(self.positions))
		#print("#1 velocities:\n" + str(self.velocities))
		#self.sim_entry.tick(activations, True)
		#print("#2 positions:\n" + str(self.positions))
		#print("#2 velocities:\n" + str(self.velocities))
	
	
	#### Reset function ####
	
	
	def reset(self, start_position=None, **kwargs):
		"""
		Implements gymnasium.Env's reset(), assuming __init__ was called
		with the necessary arguments.
		
		Positions are reset to the last starting position used, unless the
		start_position argument is given (which has to match the shape
		of the positions attribute). Velocities are reset to 0.
		
		Strengths are optionally reset to a random values (see the
		env_strength_low and env_strength_high arguments of __init__).
		"""
		# set new default start position, if specified
		if start_position is not None:
			np.copyto(self.start_position, start_position)
		
		#### stuff for gym environment
		if self.is_gym:
			# set ticks elapsed to 0
			self.env_ticks_elapsed = 0
			# randomize stuff for new episode
			# (reward function, strengths, positions, velocities)
			self.__randomize()
			# update thetas and thetadots by doing 1 tick - gotta do it somehow \( •_•)/
			self.sim_entry.tick(np.zeros(self.thetas.shape), False)
			
			return self.current_observation(), {}
		
		# set position to the default start position
		np.copyto(self.positions, self.start_position)
		# set velocities to 0
		self.velocities.fill(0.0)
		# update thetas and thetadots by doing 1 tick - gotta do it somehow \( •_•)/
		self.sim_entry.tick(np.zeros(self.thetas.shape), False)
	
	
	def step(self, activations, dbg=False):
		"""
		Implements gymnasium.Env's step() method. See the action_space and
		observation_space attributes.
		"""
		if not self.is_gym:
			raise ValueError("Cannot call Simulation.step() if the" + \
				"Simulation was not given a reward function in its constructor")
		# c code requires doubles.
		# stablebaselines3's PPO can only output floats32's.
		# so we have to cast the entire activations array every step 🤮🤮🤮
		if activations.dtype != np.double:
			activations = np.astype(activations, np.double)
		
		# run the rk4 ticks, possibly with force debugging on the first one
		ticks_to_do = self.env_step_ticks
		if dbg:
			self.sim_entry.tick(activations, True)
			ticks_to_do -= 1
		for i in range(ticks_to_do):
			self.sim_entry.tick(activations, False)
		
		self.env_ticks_elapsed += self.env_step_ticks
		
		truncated = (self.env_ticks_elapsed >= self.env_last_ticks[self.reward_index])
		return \
			self.current_observation(), \
			self.env_reward_functions[self.reward_index](self, truncated), \
			False, truncated, {}
	
	
	#### Run function ####
	
	
	def run(self, \
	# activations callback
	activations_callback=None, activations_callback_period=0.005, \
	# miscellaneous callback
	misc_callback=None, misc_callback_period=0.05,               \
	# whether to stop eventually, whether to debug when it stops
	stop_time=None, debug_at_stop=False,                         \
	# display arguments
	speed=1000,
	display=False, fps=40.0, width_px=600,             \
	height_px=1000, width_meters=2.5, point_radius_px=5):
		"""
		Runs the biomechanical simulation, with the following optional arguments:
			
			                    CALLBACK PARAMETERS
			
			- activations_callback
				function, will be periodically called to determine activation
				levels. must accept 1 argument of this Simulation object, and
				return a 1d numpy array whose length equals the number of
				joints in the body.
			
			- activations_callback_period
				float, number of simulated seconds between every call of
				activations_callback.
			
			- misc_callback
				function, will be periodically called. must accept 2 arguments:
				this Simulation object, followed by a bool which indicates
				whether the Simulation has just finished (which will be set to
				True at most once, and only if stop_time is set).
			
			- misc_callback_period
				float, number of simulated seconds between every call of
				misc_callback.
			
			                   STOP TIME PARAMETERS
			
			- stop_time
				float, number of simulated seconds to run (or None to
				never stop).
			
			- debug_at_stop
				bool, whether to print force computation at the moment the
				simulation stops.
			
			                    DISPLAY PARAMETERS
			
			- display
				bool, whether to display the movement of the body on a canvas.
			
			- speed
				float, the maximum number of simulated seconds to compute every
				real world second.
			
			- fps
				float, the maximum number of display updates to make every
				real world second.
			
			- width_px
				float, width that the canvas takes up on screen.
			
			- height_px
				float, height that the canvas takes up on screen.
			
			- width_meters
				float, the width in meters that the canvas is considered to be
				when drawing the body on screen.
			
			- point_radius_px
				float, how wide to draw the point masses on screen.
		"""
		# start time
		start_time = time.time()
		# determine rk4 tick period + set variable for time of next tick
		tick_period = self.timestep / speed
		next_tick_time = start_time
		# determine display update period + set variable for time of next display update
		display_period = 1 / fps
		next_display_time = start_time
		
		# sim time of next activations callback
		next_act_callback = 0.0
		# sim time of next misc callback
		next_misc_callback = 0.0
		
		activations = np.zeros(self.thetas.shape)
		
		# start display, if displaying
		if display:
			self.__start_display(width_px, height_px, width_meters, point_radius_px)
		
		self.ticks = 0
		display_updates = 0
		act_callbacks = 0
		misc_callbacks = 0
		while(True):
			self.sim_time = self.ticks * self.timestep
			# if reached stop time, notify misc then stop
			if (stop_time is not None) and self.sim_time >= stop_time:
				if misc_callback is not None:
					misc_callback(self, True)
				if debug_at_stop:
					self.sim_entry.tick(activations, True)
					self.__update_display()
					print( f"Real speed: " \
						f"{self.ticks * self.timestep / (time.time() - start_time)}")
				return
			# update activations, if it's time to
			if (activations_callback is not None) and self.sim_time >= next_act_callback:
				activations = activations_callback(self)
				# c code requires doubles (unfortunately, some ML models require floats)
				if activations.dtype != np.double:
					activations = np.astype(activations, np.double)
				act_callbacks += 1
				next_act_callback = act_callbacks * activations_callback_period
			# call misc, if it's time to
			if (misc_callback is not None) and self.sim_time >= next_misc_callback:
				misc_callback(self, False)
				misc_callbacks += 1
				next_misc_callback = misc_callbacks * misc_callback_period
			
			# update the simulation
			self.sim_entry.tick(activations, False)
			self.ticks += 1
			now = time.time()
			
			# update the display, if it's time to
			if display and (now >= next_display_time):
				self.__update_display()
				display_updates += 1
				next_display_time = start_time + display_updates * display_period
			
			# sleep until time of next rk4 tick
			sleeptime = next_tick_time - now
			if sleeptime > 0:
				time.sleep(sleeptime)
			next_tick_time = start_time + self.ticks * tick_period
	
	
	#### Helper functions for implementing gymnasium.Env ####
	
	
	def current_observation(self):
		"""
		Used as a helper function when this is a gymnasium.Env.
		See the observation_space attribute.
		"""
		if self.env_strength_low < self.env_strength_high:
			box = np.concatenate((self.positions, self.velocities, \
				self.flexion_c1s, self.extension_c1s), dtype=np.float32)
		else:
			box = np.concatenate((self.positions, self.velocities),
				dtype=np.float32)
		
		if len(self.env_reward_functions) > 1:
			return {'state': box, 'reward': self.reward_index}
		return box
	
	
	def __randomize(self):
		# initial reward function
		self.reward_index = self.rng.integers(0, len(self.env_reward_functions))
		
		# randomize initial positions, if applicable
		if self.env_thetas_diff > 0:
			new_start_obj = deepcopy(self.start_obj)
			# randomize root2 rotation
			new_start_obj['root2-rotation'] += \
				self.env_thetas_diff * (2  * self.rng.random() - 1)
			# randomize each flexion angle
			for key in new_start_obj['flexion-angles']:
				new_start_obj['flexion-angles'][key] += \
					self.env_thetas_diff * (2  * self.rng.random() - 1)
			# determine start positions from the random angles
			new_start_pos = body.determine_positions(self.body_obj, new_start_obj)
			# shift y positions to make the lowest point have y = 0
			new_start_pos[1::2] -= min(new_start_pos[1::2]) 
			np.copyto(self.positions, new_start_pos)
		else:
			# set position to the default start position
			np.copyto(self.positions, self.start_position)
		
		# randomize initial velocities, if applicable
		if self.env_velocities_diff > 0:
			self.rng.random(self.velocities.shape, dtype=np.double, out=self.velocities)
			self.velocities -= 0.5
			self.velocities *= 2 * self.env_velocities_diff
		else:
			# set velocities to 0
			self.velocities.fill(0.0)
		
		# randomize initial strengths, if applicable
		if self.env_strength_low < self.env_strength_high:
			# flexion c1s = original c1s * ((high - low) * rand(0.0, 1.0) + low)
			self.rng.random(self.flexion_c1s.shape, dtype=np.double, out=self.flexion_c1s)
			self.flexion_c1s *= self.env_strength_high - self.env_strength_low
			self.flexion_c1s += self.env_strength_low
			self.flexion_c1s *= self.original_flexion_c1s
			
			# extension c1s = original c1s * ((high - low) * rand(0.0, 1.0) + low)
			self.rng.random(self.extension_c1s.shape, dtype=np.double, out=self.extension_c1s)
			self.extension_c1s *= self.env_strength_high - self.env_strength_low
			self.extension_c1s += self.env_strength_low
			self.extension_c1s *= self.original_extension_c1s
	
	
	#### Display functions ####
	
	
	def __translate(self, x, y):
		xcanv = (self.width_px  / 2) + x * self.canvas_scale
		ycanv = (self.height_px / 2) - y * self.canvas_scale
		return (xcanv, ycanv)
	
	def __update_display(self):
		self.canvas.delete("all")
		# paint ground line on canvas
		self.canvas.create_line(self.__translate(-10, 0), self.__translate( 10, 0))
		# paint body segments on canvas
		for i in range(len(self.segments)):
			p1_ind = 2 * self.segments[i][0]
			p2_ind = 2 * self.segments[i][1]
			self.canvas.create_line( \
				self.__translate( \
					self.positions[p1_ind],     \
					self.positions[p1_ind + 1] \
				),
				self.__translate( \
					self.positions[p2_ind],     \
					self.positions[p2_ind + 1] \
				)
			)
		# paint body point masses on canvas
		for i in range(self.num_point_masses):
			x, y = self.__translate(     \
				self.positions[2 * i],   \
				self.positions[2 * i + 1] \
			)
			rect_p1 = (x - self.point_radius_px, y - self.point_radius_px)
			rect_p2 = (x + self.point_radius_px, y + self.point_radius_px)
			newpt = self.canvas.create_oval(rect_p1, rect_p2, fill="black")
		self.root.update()
	
	def __start_display(self, width_px, height_px, width_meters, point_radius_px):
		# set self's attributes to given arguments
		self.width_px = width_px
		self.height_px = height_px
		self.width_meters = width_meters
		self.point_radius_px = point_radius_px
		# set other attributes used for displaying
		self.canvas_scale = self.width_px / self.width_meters
		# create canvas
		self.root = tk.Tk()
		self.root.geometry( \
			str(self.width_px) + "x" + str(self.height_px) \
		)
		self.canvas = tk.Canvas( \
			self.root, width=self.width_px,   \
			height=self.height_px, bg="white" \
		)
		self.canvas.pack()
		# initially update the display
		self.__update_display()
