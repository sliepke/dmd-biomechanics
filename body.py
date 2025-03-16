import json
import math
import numpy as np
from util import *


class Body:
	"""
	Model of the human body as a series of point masses in 2d space.
	Contains the following data:
		- Names and masses of the point masses
		- Length and points involved (2) in each body segment
		- Points involved (3) in each joint
		- Joint parameters (for passive and active joint torque model)
	Is able to:
		- Determine position of all point masses, as a function of all joint
		  flexion angles and the position of 2 root points masses
		- Estimate bodily forces exerted on the points masses at a moment in time,
		  as a function of the point masses' positions, velocities, and joint
		  activation levels
	"""
	
	
	def __init__(self, filename):
		"""
		"""
		with open(filename) as file:
			obj = json.load(file)
		
		segment_stiffness = float( from_JSON_obj(obj, "root", \
			"segment-stiffness", [float, int]) )
		segment_dampening = float( from_JSON_obj(obj, "root", \
			"segment-dampening", [float, int]) )
		
		self.point_masses = from_JSON_obj(obj, "root", "point-masses", [dict])
		# verify types in point mass dictionary
		for key in self.point_masses:
			self.point_masses[key] = float( from_JSON_obj(self.point_masses, \
				"point-masses", key, [float, int]) )
		
		# turn (point name -> mass dict) into list of names and list of masses
		self.point_names = list(self.point_masses)
		pm_cp = [0] * len(self.point_names)
		for i in range(len(self.point_names)):
			pm_cp[i] = self.point_masses[self.point_names[i]]
		self.point_masses = pm_cp
		
		# convert each 'segment' (element of "segments" list in json) into a list of
		# the form [point mass 1 index, point mass 2 index, _Body_Segment instance]
		self.segments = from_JSON_obj(obj, "root", "segments", [list])
		for i in range(len(self.segments)):
			# make sure file's 'segments' element is a list [string, string, float]
			entry = from_JSON_array(self.segments, "segments", i, [list])
			from_JSON_array(entry, "segments[" + str(i) + "]", 0, [str])
			from_JSON_array(entry, "segments[" + str(i) + "]", 1, [str])
			entry[2] = float( from_JSON_array(entry, \
				"segments[" + str(i) + "]", 2, [float, int]) )
			
			# make sure both segment endpoints are listed as point masses
			try:
				p1_index = self.point_names.index(entry[0])
			except KeyError:
				raise ValueError( entry[0] + \
					" is listed as a segment endpoint, but not a point mass")
			try:
				p2_index = self.point_names.index(entry[1])
			except KeyError:
				raise ValueError( entry[1] + \
					" is listed as a segment endpoint, but not a point mass")
			
			bs_instance = _Body_Segment(entry[2], segment_stiffness, \
				segment_dampening)
			self.segments[i] = [p1_index, p2_index, bs_instance]
		
		# convert each 'joint' (element of "joints" obj in json) into a list
		# of the form [base index, center index, end index, _Joint instance]
		self.joints = from_JSON_obj(obj, "root", "joints", [dict])
		for key in self.joints:
			# make sure each field of file's 'joints' is an object {
			#	base, center, end -> string
			#	offset -> float
			#	passive_torque_parameters -> object {
			#		B1, k1, B2, k2 -> float
			#	}
			#	(flexion/extension)_parameters -> object {
			#		C1, C2, C3, C4, C5, C6 -> float
			#	}
			entry = self.joints[key]
			from_JSON_obj(entry, "joints." + key, "base", [str])
			from_JSON_obj(entry, "joints." + key, "center", [str])
			from_JSON_obj(entry, "joints." + key, "end", [str])
			entry["offset"] = float( from_JSON_obj(entry, "joints." + key, \
				"offset", [float, int]) )
			
			def convert_parameters(params_name, param_list):
				from_JSON_obj(entry, "joints." + key, \
					params_name, [dict])
				param_float_list = []
				for p in param_list:
					param_float_list += [ float( from_JSON_obj(entry[params_name], \
						"joints." + key + "." + params_name, p, [float, int]) ) ]
				return param_float_list
			
			pp = convert_parameters("passive-torque-parameters", ["B1", "k1", "B2", "k2"])
			fp = convert_parameters("flexion-parameters", ["C1", "C2", "C3", "C4", "C5", "C6"])
			ep = convert_parameters("extension-parameters", ["C1", "C2", "C3", "C4", "C5", "C6"])
			
			# make sure the 3 joint endpoints are listed as point masses
			try:
				p1_index = self.point_names.index(entry["base"])
			except KeyError:
				raise ValueError( entry[0] + \
					" is listed as a segment endpoint, but not a point mass")
			try:
				p2_index = self.point_names.index(entry["center"])
			except KeyError:
				raise ValueError( entry[1] + \
					" is listed as a segment endpoint, but not a point mass")
			try:
				p3_index = self.point_names.index(entry["end"])
			except KeyError:
				raise ValueError( entry[2] + \
					" is listed as a segment endpoint, but not a point mass")
			
			j_instance = _Joint(entry["offset"], pp, fp, ep)
			self.joints[key] = [p1_index, p2_index, p3_index, j_instance]
	
	
	def positions(self, flexion_angles, root1, root1x, root1y, root2, theta):
		"""
		Given flexion angles for each joint in the body, the position of 1 point
		mass in the body, and the angle of another point relative to the known
		point, returns the positions of every joint in the body.
		
		Parameters:
			...
			flexion_angles are in degrees
			
			theta is in degrees, and is the counterclockwise angle of root2 relative
			to root1 (where 0 degrees means root2 is to the right of root1)
		"""
		
		theta *= math.pi / 180
		positions = np.array([[0.0, 0.0]] * len(self.point_names))
		
		# -- record positions of the 2 roots -- #
		
		# find length between root1 and root2
		r1_r2_length = None
		for segment_entry in self.segments:
			if root1 in segment_entry[:2] and root2 in segment_entry[:2]:
				r1_r2_length = segment_entry[2].resting_length
		if r1_r2_length is None:
			raise RuntimeError("There must be a segment between root1 and root2")
		# find position of root2
		root2x = root1x + r1_r2_length * math.cos(theta)
		root2y = root1y + r1_r2_length * math.sin(theta)
		# record root1, root2 positions
		positions[root1][0] = root1x
		positions[root1][1] = root1y
		positions[root2][0] = root2x
		positions[root2][1] = root2y
		
		# -- make list of constraints for each point -- #
		
		constraints = [[] for i in range(len(self.point_names))]
		joint_keys_ordered = list(self.joints)
		for i in range(len(joint_keys_ordered)):
			joint_entry = self.joints[joint_keys_ordered[i]]
			joint = joint_entry[3]
			b = joint_entry[0]
			c = joint_entry[1]
			e = joint_entry[2]
			
			# counterclockwise angle of ce relative to cb
			angle = (flexion_angles[i] - joint.offset) * math.pi / 180
			
			# find lengths of ce and cb
			ce_length = None
			cb_length = None
			for segment_entry in self.segments:
				if c in segment_entry[:2]:
					if e in segment_entry[:2]:
						ce_length = segment_entry[2].resting_length
					elif b in segment_entry[:2]:
						cb_length = segment_entry[2].resting_length
			if ce_length is None:
				raise RuntimeError("There is a joint " + str(b) + "-" + str(c) + \
					"-" + str(e) + " but no segment between " + str(c) + " and " + str(e))
			if cb_length is None:
				raise RuntimeError("There is a joint " + str(b) + "-" + str(c) + \
					"-" + str(e) + " but no segment between " + str(c) + " and " + str(b))
			
			# add constraint details to set of constraints
			b_constraint = [c, e, -angle, cb_length]
			e_constraint = [c, b, angle, ce_length]
			constraints[b].append(b_constraint)
			constraints[e].append(e_constraint)
		
		for point in range(len(self.point_names)):
			print(point)
			for constraint in constraints[point]:
				print("\t", constraint, sep="")
		print()
		
		# -- solve system of constraints, in a very inefficient way -- #
		
		known_pts = {root1, root2}
		
		pt_index = 0
		ct_index = 0
		
		while True:
			print(pt_index, ct_index, sep=", ")
			if pt_index >= len(self.point_names):
				break
			if pt_index in known_pts:
				pt_index += 1; ct_index = 0
				continue
			if ct_index >= len(constraints[pt_index]):
				pt_index += 1; ct_index = 0
				continue
			
			constraint = constraints[pt_index][ct_index]
			c = constraint[0] # center of joint
			o = constraint[1] # endpoint of joint other than pt_index
			
			if not (c in known_pts and o in known_pts):
				ct_index += 1
				continue
			
			# solve for pt_index's position
			co = positions[o] - positions[c]
			angle = constraint[2] + np.arctan2(co[1], co[0])
			cpt = np.array([constraint[3] * math.cos(angle), constraint[3] * math.sin(angle)])
			pos = positions[c] + cpt
			positions[pt_index] = pos
			
			known_pts.add(pt_index)
			pt_index = 0
			ct_index = 0
			print("\tsolved: ", pos)
		
		if len(known_pts) < len(self.point_names):
			raise RuntimeError("Could not determine all point mass positions. " + \
				"This may happen if the body is not connected together by " + \
				"segments and joints in a normal way.")
		
		# -- return positions as a 1d array -- #
		
		return positions.flatten()
	
	
	def forces(positions, vels, activations):
		"""
		Given the positions and velocities of every point mass in body, and the activation
		level of every joint in the body, determines the bodily forces exerted on every
		point mass in the body. The bodily forces come from:
			- The restoring forces of each body segment, which is modelled as a spring
			- The passive torques of each joint, which is modelled as described in
			  Journal of Biomechanics 40.14 (2007): 3105-3113.
			- The active torques of each joint, which is modelled as described in
			  Journal of Biomechanics 40.14 (2007): 3105-3113.
		
		Note that the point masses and joints are ordered. This contradicts the current
		JSON format being used, where the set of point masses and set of joints are given
		as objects rather than lists.
		
		Parameters:
			positions       1d numpy array of length (2 * number of point masses of this body)
				Positions of all the point masses, given in the following order:
				point mass #1 x, point mass #1 y, point mass #2 x, point mass #2 y, ...
			
			vels            1d numpy array of length (2 * number of point masses of this body)
				Velocities of all the point masses, given in the following order:
				point mass #1 x, point mass #1 y, point mass #2 x, point mass #2 y, ...
			
			activations     1d numpy array of length (number of joints of this body)
				Activation levels of each joint, each activation level being
				between -1.0 (full exertion extension) and 1.0 (full exertion flexion)
		"""
		force_list = np.array([0.0] * len(positions))
		
		# -- segment forces -- #
		
		for segment_entry in self.segments:
			p1index = segment_entry[0]; p2index = segment_entry[1]
			segment = segment_entry[2]
			
			# get positions and velocities of the 2 endpoints of the segment
			p1pos = np.array([positions[2 * p1index], positions[2 * p1index + 1]])
			p1vel = np.array([vels[2 * p1index], vels[2 * p1index + 1]])
			p2pos = np.array([positions[2 * p2index], positions[2 * p2index + 1]])
			p2vel = np.array([vels[2 * p2index], vels[2 * p2index + 1]])
			
			# determine forces on the 2 endpoints of the segment
			(p1f, p2f) = segment.forces(p1pos, p1vel, p2pos, p2vel)
			
			force_list[2 * p1index] = p1f[0]; force_list[2 * p1index + 1] = p1f[0]
			force_list[2 * p2index] = p2f[0]; force_list[2 * p2index + 1] = p2f[0]
		
		# -- joint forces -- #
		
		for key in self.joints:
			joint_entry = self.joints[key]
			bindex = joint_entry[0]; cindex = joint_entry[1]; eindex = joint_entry[2]
			joint = joint_entry[3]
			
			# get positions and velocities of the 3 endpoints of the joint
			bpos = np.array([positions[2 * bindex], positions[2 * bindex + 1]])
			bvel = np.array([vels[2 * bindex], vels[2 * bindex + 1]])
			cpos = np.array([positions[2 * cindex], positions[2 * cindex + 1]])
			cvel = np.array([vels[2 * cindex], vels[2 * cindex + 1]])
			epos = np.array([positions[2 * eindex], positions[2 * eindex + 1]])
			evel = np.array([vels[2 * eindex], vels[2 * eindex + 1]])
			
			# determine forces on the 3 endpoints of the joint
			(bf, cf, ef) = joint.forces(bpos, bvel, cpos, cvel, \
				epos, evel, activations[i])
			
			force_list[2 * bindex] = bf[0]; force_list[2 * bindex + 1] = bf[1]
			force_list[2 * cindex] = cf[0]; force_list[2 * cindex + 1] = cf[1]
			force_list[2 * eindex] = ef[0]; force_list[2 * eindex + 1] = ef[1]
		
		return force_list




class _Joint:
	"""
	Model of a joint of the body in 2d space. Is able to determine forces exerted
	on point masses using the point masses positions, velocities, and the joint
	activation level.
	
	Definitions:
		- Center    The point mass that the joint is located at
		- Base      A point mass which turns clockwise about the Center (relative to the
		            End) during flexion
		- End       A point mass which turns counterclockwise about the Center (relative
		            the Base) during flexion
	
	Attributes:
		- offset
			Number of degrees added to the counterclockwise angle (0 <= a < 360) of
			line segment CenterEnd relative to CenterBase, to result in the flexion angle
		- passive_params
			List of the 4 passive joint parameters B1, k1, B2, k2 as described in
			Journal of Biomechanics 40.14 (2007): 3105-3113.
		- active_flexion_params
			List of the 4 active joint parameters for flexion C1, C2, ..., C6 as
			described in Journal of Biomechanics 40.14 (2007): 3105-3113.
		- active_extension_params
			List of the 4 active joint parameters for extension C1, C2, ..., C6 as
			described in Journal of Biomechanics 40.14 (2007): 3105-3113.
	"""
	
	
	def __init__(self, offset, passive_params, active_flexion_params, active_extension_params):
		"""
		Create a new Joint with the given parameters. See Class description for the
		meaning of the parameters.
		"""
		self.offset = offset
		
		self.B1 = passive_params[0]
		self.k1 = passive_params[1]
		self.B2 = passive_params[2]
		self.k2 = passive_params[3]
		
		self.FC1 = active_flexion_params[0]
		self.FC2 = active_flexion_params[1]
		self.FC3 = active_flexion_params[2]
		self.FC4 = active_flexion_params[3]
		self.FC5 = active_flexion_params[4]
		self.FC6 = active_flexion_params[5]
		
		self.EC1 = active_extension_params[0]
		self.EC2 = active_extension_params[1]
		self.EC3 = active_extension_params[2]
		self.EC4 = active_extension_params[3]
		self.EC5 = active_extension_params[4]
		self.EC6 = active_extension_params[5]
	
	
	def forces(b, bvel, c, cvel, e, evel, activation):
		"""
		(See Class description for definitions of base, center, and end)
		
		Determines the torque T exerted on this joint when:
			- This joint's base, center, and end have the given positions (b, c, e) and
			  velocities (bvel, cvel, evel)
			- This joint has the given activation level (-1.0 is full exertion extension,
			  1.0 is full exertion flexion)
		Then, determines a set of forces to exert on the base, center, and end such that:
			- The base and end have a torque of T about the center
			- The base and end's torques are in opposite angular directions
			- Linear and angular momentum is conserved
		Returns the tuple (force on base, force on center, force on end)
		"""
		base = b - c
		end = e - c
		
		# -- determine flexion angle -- #
		
		dot = np.dot(base, end)
		det = base[0] * end[1] - base[1] * end[0]
		theta = np.arctan2(det, dot) + (math.pi * self.offset / 180)
		
		# -- determine flexion angular velocity -- #
		
		b_relative_vel = bvel - cvel
		b_angular_vel_direction = np.array([-base[1], base[0]])
		b_angular_vel = component(b_relative_vel, b_angular_vel_direction)
		
		e_relative_vel = evel - cvel
		e_angular_vel_direction = np.array([-end[1], end[0]])
		e_angular_vel = component(e_relative_vel, e_angular_vel_direction)
		
		thetha_prime = e_angular_vel - b_angular_vel
		
		# -- determine torque using passive / active torque model -- #
		
		t_passive = self.B1 * math.exp(self.k1 * theta) + \
		            self.B2 * math.exp(self.k2 * theta)
		
		# extension exertion
		if activation < 0:
			# (angular velocity positive in extension formula angular velocity is extending)
			theta_prime *= -1
			
			# concentric / stationary angular velocity
			if theta_prime >= 0:
				t_active = self.EC1 * math.cos(self.EC2 * (theta - self.EC3)) * \
				           (2 * self.EC4 * self.EC5 + theta_prime * (self.EC5 - 3 * self.EC4)) / \
						   (2 * self.EC4 * self.EC5 + theta_prime * (2 * self.EC5 - 4 * self.EC4))
			# eccentric angular velocity
			else:
				t_active = (1 - self.EC6 * theta_prime) * \
				           self.EC1 * math.cos(self.EC2 * (theta - self.EC3)) * \
				           (2 * self.EC4 * self.EC5 - theta_prime * (self.EC5 - 3 * self.EC4)) / \
						   (2 * self.EC4 * self.EC5 - theta_prime * (2 * self.EC5 - 4 * self.EC4))
		# flexion exertion
		if activation > 0:
			# concentric / stationary angular velocity
			if theta_prime >= 0:
				t_active = self.FC1 * math.cos(self.FC2 * (theta - self.FC3)) * \
				           (2 * self.FC4 * self.FC5 + theta_prime * (self.FC5 - 3 * self.FC4)) / \
						   (2 * self.FC4 * self.FC5 + theta_prime * (2 * self.FC5 - 4 * self.FC4))
			# eccentric angular velocity
			else:
				t_active = (1 - self.FC6 * theta_prime) * \
				           self.FC1 * math.cos(self.FC2 * (theta - self.FC3)) * \
				           (2 * self.FC4 * self.FC5 - theta_prime * (self.FC5 - 3 * self.FC4)) / \
						   (2 * self.FC4 * self.FC5 - theta_prime * (2 * self.FC5 - 4 * self.FC4))
		# no exertion
		if activation == 0:
			t_active = 0.0
		
		t_active *= activation
		t_total = t_passive + t_active
		
		# -- turn torques into forces on 3 point masses -- #
		
		f_b = - b_angular_vel_direction * t_total / \
			( np.linalg.norm(base) * np.linalg.norm(b_angular_vel_direction) )
		f_e = e_angular_vel_direction * t_total / \
			( np.linalg.norm(end) * np.linalg.norm(e_angular_vel_direction) )
		f_c = - f_b - f_e
		
		return (f_b, f_c, f_e)




class _Body_Segment:
	"""
	Model of an individual body segment (forearm, torso, foot, etc) as a spring
	which pushes the endpoints of the segment toward or away from each other,
	which keeps the segment at a similar length.
	"""
	
	
	def __init__(self, resting_length, stiffness, dampening):
		self.resting_length = resting_length
		self.stiffness = stiffness
		self.dampening = dampening
	
	
	def forces(p1, p1vel, p2, p2vel):
		"""
		Determines the forces exerted on the endpoints (p1, p2) of this body segment
		when p1 and p2 have the given positions and velocities. Returns the tuple
		(force on p1, force on p2)
		"""
		p1p2 = p2 - p1
		p1p2_norm = np.linalg.norm(p1p2)
		
		r = p1p2_norm - self.resting_length
		rp = component(p2vel - p1vel, p1p2)
		
		rf_mag = - r * stiffness - rp * dampening
		p1_feels = p1p2 * (rf_mag / p1p2_norm)
		p2_feels = - p1_feels
		return (p1_feels, p2_feels)
