import math
import json
import numpy as np
from util import *


class SavedActivation:
	"""
	Represents a stepwise activation function loaded from a file.
	"""
	def __init__(self, filename):
		with open(filename) as file:
			obj = json.load(file)
		if type(obj) != dict:
			raise TypeError("Expected root element of " + filename + " to be an object")
		
		# validate outermost types
		self.__timestep = float( from_JSON_obj(obj, "root", "timestep", [float, int]) )
		self.__act_function = from_JSON_obj(obj, "root", "activation-function", [list])
		# validate nested types (what is inside activation function)
		self.__num_joints = None
		for i in range(len(self.__act_function)):
			# make sure activation-function[i] is a list with same length seen already
			acts_t = from_JSON_array(self.__act_function, "activation-function", i, [list])
			if self.__num_joints is None:
				self.__num_joints = len(acts_t)
			else:
				if self.__num_joints != len(acts_t):
					raise ValueError("Expected activation-function[" + str(i) + "] to be " \
						"a list of length " + str(self.__num_joints))
			# convert elements of activation-function[i] into floats
			for j in range(len(acts_t)):
				acts_t[j] = float( from_JSON_array(acts_t, "activation-function[" + \
					str(i) + "]", j, [float, int]) )
		
		self.__act_function = np.array(self.__act_function)
	
	
	def get_timestep(self):
		""" Period (s) at which activations change """
		return self.__timestep
	
	
	def get_num_joints(self):
		""" Number of joints that this SavedActivation uses """
		return self.__num_joints
	
	
	def activations(self, t):
		"""
		Returns the activation levels of each joint, as a 1d numpy array.
		
		Each element of the output array is a number from -1.0 (full exertion
		extension) to 1.0 (full exertion flexion) representing how that joint
		is activated at t seconds.
		"""
		index = math.floor(t / self.__timestep)
		if index < 0:
			raise ValueError("This activation function is not defined for t < 0")
		if index >= len(self.__act_function):
			raise ValueError("This activation function is not defined for t >= " + \
				str(self.__timestep * len(self.__act_function)) )
		return self.__act_function[index]
