import math
import numpy as np

# imports for display
import datetime
import time
import tkinter as tk

# user defined imports
from . import sim_c_util
from .sim_py_util import body

class Simulation:
	"""
	2d biomechanical simulation of human motion.
	
	All attributes arrays are updated every tick, but stay as the same reference.
	They allow the callback functions given as arguments to run() to read the
	state of the body.
	
	Attributes
		positions
			1d numpy array of all point mass positions, in the order:
				point1 x, point1 y, point2 x, point2 y, ...
		velocities
			1d numpy array of all point mass velocities (in the same order
			as positions).
		thetas
			1d numpy array of all joint angles. Is updated every time the
			positions and velocities update, including during reset()
		thetadots
			1d numpy array of all joint angular velocities. Is updated every
			time the positions and velocities update, including during reset()
	"""
	
	
	#### Init function ####
	
	
	def __init__(self, \
	# files containing parameters for body (required)
	body_filename, start_position_filename, \
	# gravity acceleration                 (optional)
	g=9.81, \
	# ground force parameters              (optional)
	A_normal=1e4, B_normal=5e3, mu_s=0.209, mu_k=0.209, zero_velocity_threshold=1e-6, \
	# numerical solver parameters          (optional)
	rk4_timestep=8e-5):
		"""
		All parameters have the same meaning as in the paper. zero_velocity_threshold
		is the horizontal speed at which a point mass's horizontal velocity will round
		to 0.
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
		
		body_obj = body.load_body_file(body_filename)
		
		# each m_p (as described in the paper)
		masses_ordered = list(body_obj["point-masses"].values())
		
		point_masses = list(body_obj["point-masses"])
		num_point_masses = len(point_masses)
		
		# each [a, b] in S
		# (but here, a and b are integer indexes, not strings)
		segments_ordered = [ [point_masses.index(segment[0]), point_masses.index(segment[1])] \
			for segment in body_obj["segments"]]
		# list of each l_(a, b) where (a, b) in S
		lengths_ordered = [ segment[2] for segment in body_obj["segments"]]
		# segment stiffness and damping
		A_segment = body_obj["segment-stiffness"]
		B_segment = body_obj["segment-damping"]
		
		# list of each [a, b, c] in J
		joints_ordered = [
			[	point_masses.index(joint["end"]), \
				point_masses.index(joint["center"]), \
				point_masses.index(joint["base"]) \
			] for joint in body_obj["joints"].values()]
		# list of parameters of each (a, b, c) in J EXCEPT C1's
		joint_params_ordered = [ \
			[	math.pi * joint["offset"] / 180, \
				math.pi * joint["transition"] / 180, \
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
			] for joint in body_obj["joints"].values()]
		# B3
		b3 = body_obj["joint-damping"]
		# for each joint, C1F
		c1fs_ordered = [joint["flexion-parameters"]["C1"] \
			for joint in body_obj["joints"].values()]
		# for each joint, C1E
		c1es_ordered = [joint["extension-parameters"]["C1"] \
			for joint in body_obj["joints"].values()]
		
		# -- load start position from file -- #
		
		start_position = body.load_start_position(body_obj, start_position_filename)
		
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
		# how often to print force calculations (in ticks)
		# can be set to None
		self.debug_on = None # int(0.5 / self.timestep)
		self.ticks = 0
		
		# -- debug: do 1 tick with force debugging on -- #
		#activations = np.array([0] * len(self.joints), dtype=np.double)
		#print("#1 positions:\n" + str(self.positions))
		#print("#1 velocities:\n" + str(self.velocities))
		#self.sim_entry.tick(activations, True)
		#print("#2 positions:\n" + str(self.positions))
		#print("#2 velocities:\n" + str(self.velocities))
	
	
	#### Reset function ####
	
	
	def reset(self, start_position_filename=None):
		"""
		Reset the starting positions and velocities of all point masses.
		If start_position_filename is None, then the last starting positions
		will be used.
		"""
		if start_position_filename is not None:
			self.start_position = body.load_start_position(body_obj, start_position_filename)
		np.copyto(self.positions, self.start_position)
		self.velocities.fill(0.0)
		# have to update thetas + thetadots somehow \( •_•)/
		self.sim_entry.tick(np.zeros( (len(self.thetas)) ), False)
	
	
	#### Run function ####
	
	
	def run(self, \
	# activations callback
	activations_callback=None, activations_callback_period=0.01, \
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
				return a 1d numpy array with dtype numpy.double, and length
				equal to the number of joints in the body.
			
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
			
			- speed
				float, the maximum number of simulated seconds to
				compute every real world second.
		
			- display
				bool, whether to display the movement of the body on a canvas.
			
			- fps
				float, the maximum number of display updates to make every
				real world second.
			
			- width_px
				float, width that the canvas takes up on screen.
			
			- height_px
				float, height that the canvas takes up on screen.
			
			- width_meters
				float, number of meters that the canvas is considered to
				be when drawing the body on screen.
			
			- point_radius_px
				float, how wide to draw the point masses on screen.
		"""
		# start time
		start_time = time.time()
		# tick period, and variable for time of next tick
		tick_period = self.timestep / speed
		next_tick_time = start_time
		# display update period, and variable for time of next display update
		display_period = 1 / fps
		next_display_time = start_time
		
		# sim time of next activations callback
		next_act_callback = 0.0
		# sim time of next misc callback
		next_misc_callback = 0.0
		
		activations = np.zeros((len(self.thetas)))
		
		# start display, if displaying
		if display:
			self.__start_display(width_px, height_px, width_meters, point_radius_px)
		
		self.ticks = 0
		display_updates = 0
		act_callbacks = 0
		misc_callbacks = 0
		while(True):
			sim_time = self.ticks * self.timestep
			# if reached stop time, notify misc then stop
			if (stop_time is not None) and sim_time >= stop_time:
				if misc_callback is not None:
					misc_callback(self, True)
				if debug_at_stop:
					self.sim_entry.tick(activations, True)
					print( f"Real speed: " \
						f"{self.ticks * self.timestep / (time.time() - start_time)}")
				return
			# update activations, if it's time to
			if (activations_callback is not None) and sim_time >= next_act_callback:
				activations = activations_callback(self)
				act_callbacks += 1
				next_act_callback = act_callbacks * activations_callback_period
			# call misc, if it's time to
			if (misc_callback is not None) and sim_time >= next_misc_callback:
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
			
			# sleep until time of next tick
			sleeptime = next_tick_time - now
			if sleeptime > 0:
				time.sleep(sleeptime)
			next_tick_time = start_time + self.ticks * tick_period
	
	
	#### Display functions (private) ####
	
	
	def __translate(self, x, y):
		xcanv = (self.width_px  / 2) + x * self.canvas_scale
		ycanv = (self.height_px / 2) - y * self.canvas_scale
		return (xcanv, ycanv)
	
	def __update_display(self):
		self.canvas.delete("all")
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
