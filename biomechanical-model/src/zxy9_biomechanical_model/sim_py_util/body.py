import math

# built in json module
import json

# our json helper module
from .json_util import *

def load_body_file(filename):
	"""
	Reads body parameters from the json-encoded file. Verifies that the
	resulting object contains all necessary parameters, with correct type,
	and is self-consistent (for example, a segment cannot be between point
	masses that were not defined in the 'point-masses' dict / object).
	
	If the file is in JSON format and its data is usable, returns the resulting
	object (with all parameters their expected type - for example, if a mass is
	given as an integer in the JSON, it will be converted to a float inside of
	the returned object). Otherwise, raises an Exception with a message
	explaining what is wrong with the format of the file.
	
	The exact JSON data that is needed in the file is not specified here, but
	the files inputs/human/body.json and inputs/bat/body.json are examples.
	
	Parameters:
		filename    the file to read body parameters from
	Returns:
		An object (as returned by json.load) containing all necessary body
		parameters, some of which may have undergone conversions (for example,
		int -> float)
	"""
	fn = filename
	with open(fn) as file:
		obj = json.load(file)
	if type(obj) != dict:
		raise ValueError("Filename: "+fn+", Problem: expected root of JSON to be object")
	
	convert_JSON_field(fn, obj, "root", "segment-stiffness", [float, int], float) 
	convert_JSON_field(fn, obj, "root", "segment-damping", [float, int], float)
	convert_JSON_field(fn, obj, "root", "joint-damping", [float, int], float)
	
	convert_JSON_field(fn, obj, "root", "point-masses", [dict], dict)
	# convert values of point mass dictionary (the masses in kg) to floats
	for key in obj["point-masses"]:
		convert_JSON_field(fn, obj["point-masses"], "point-masses", key, [float, int], float)
	
	# make sure root["segments"] is a list
	convert_JSON_field(fn, obj, "root", "segments", [list], list)
	
	# make sure each element of root["segments"] is an array [str, str, float]
	segments = obj["segments"]
	for i in range(len(segments)):
		# make sure this element of root["segments"] (i.e. this segment) is an array
		convert_JSON_el(fn, segments, "segments", i, [list], list)
		# make sure its elements from 0-2 have types [str, str, float]
		convert_JSON_el(fn, segments[i], "segments["+str(i)+"]", 0, [str], str)
		convert_JSON_el(fn, segments[i], "segments["+str(i)+"]", 1, [str], str)
		convert_JSON_el(fn, segments[i], "segments["+str(i)+"]", 2, [int, float], float)
		# make sure its elements 0-1 are names defined in root["point-masses"]
		if segments[i][0] not in obj["point-masses"]:
			raise ValueError("Filename: "+fn+", Problem: segments["+str(i)+"][0] ("+segments[i][0]+") was not found in point-masses")
		if segments[i][1] not in obj["point-masses"]:
			raise ValueError("Filename: "+fn+", Problem: segments["+str(i)+"][1] ("+segments[i][1]+") was not found in point-masses")
	
	# make sure root["joints"] is a dict
	convert_JSON_field(fn, obj, "root", "joints", [dict], dict)
	# make sure every field of root["joints"] is a dict with the right keys
	joints = obj["joints"]
	for key in joints:
		# make sure this field of root["joints"] (i.e. this joint) is a dict
		convert_JSON_field(fn, joints, "joints", key, [dict], dict)
		# make sure it contains all necessary keys (which are joint parameters)
		convert_JSON_field(fn, joints[key], "joints["+key+"]", "base", [str], str)
		convert_JSON_field(fn, joints[key], "joints["+key+"]", "center", [str], str)
		convert_JSON_field(fn, joints[key], "joints["+key+"]", "end", [str], str)
		convert_JSON_field(fn, joints[key], "joints["+key+"]", "offset", [int, float], float)
		convert_JSON_field(fn, joints[key], "joints["+key+"]", "transition", [int, float], float)
		convert_JSON_field(fn, joints[key], "joints["+key+"]", \
			"passive-torque-parameters", [dict], dict)
		convert_JSON_field(fn, joints[key], "joints["+key+"]", \
			"flexion-parameters", [dict], dict)
		convert_JSON_field(fn, joints[key], "joints["+key+"]", \
			"extension-parameters", [dict], dict)
		# make sure the base, center, and end fields are defined in root["point-masses"]
		if joints[key]["base"] not in obj["point-masses"]:
			raise ValueError("Filename: "+fn+", Problem: joints["+key+"][base] ("+joints[key]["base"]+") was not found in point-masses")
		if joints[key]["center"] not in obj["point-masses"]:
			raise ValueError("Filename: "+fn+", Problem: joints["+key+"][center] ("+joints[key]["center"]+") was not found in point-masses")
		if joints[key]["end"] not in obj["point-masses"]:
			raise ValueError("Filename: "+fn+", Problem: joints["+key+"][end] ("+joints[key]["end"]+") was not found in point-masses")
		# make sure base != center, and that either (base, center) or (center, base) is a segment
		if joints[key]["base"] == joints[key]["center"]:
			raise ValueError("Filename: "+fn+", Problem: joints["+key+"][base] and joints["+key+"][center] cannot be the same point mass ("+joints[key]["base"]+")")
		found_match = False
		for segment in segments:
			if segment[0] == joints[key]["base"] and segment[1] == joints[key]["center"]:
				found_match = True
				break
			if segment[1] == joints[key]["base"] and segment[0] == joints[key]["center"]:
				found_match = True
				break
		if not found_match:
			raise ValueError("Filename: "+fn+", Problem: joints["+key+"][base] and joints["+key+"][center] ("+joints[key]["base"]+ \
				" and "+joints[key]["center"]+") must be form a segment together") 
		# make sure end != center, and that either (end, center) or (center, end) is a segment
		if joints[key]["end"] == joints[key]["center"]:
			raise ValueError("Filename: "+fn+", Problem: joints["+key+"][end] and joints["+key+"][center] cannot be the same point mass ("+joints[key]["end"]+")")
		found_match = False
		for segment in segments:
			if segment[0] == joints[key]["end"] and segment[1] == joints[key]["center"]:
				found_match = True
				break
			if segment[1] == joints[key]["end"] and segment[0] == joints[key]["center"]:
				found_match = True
				break
		if not found_match:
			raise ValueError("Filename: "+fn+", Problem: joints["+key+"][end] and joints["+key+"][center] ("+joints[key]["end"]+ \
				" and "+joints[key]["center"]+") must be form a segment together") 
		# make sure base != end
		if joints[key]["base"] == joints[key]["end"]:
			raise ValueError("Filename: "+fn+", Problem: joints["+key+"][base] and joints["+key+"][end] cannot be the same point mass ("+joints[key]["base"]+")")
		# make sure the 3 torque parameter dicts each have their necessary keys
		for passive_key in ["B1", "k1", "B2", "k2"]:
			convert_JSON_field(fn, joints[key]["passive-torque-parameters"], \
				"joints["+key+"][passive-torque-parameters]", passive_key, \
				[int, float], float)
		for active_key in ["C1", "C2", "C3", "C4", "C5", "C6"]:
			convert_JSON_field(fn, joints[key]["flexion-parameters"], \
				"joints["+key+"][flexion-parameters]", active_key, \
				[int, float], float)
			convert_JSON_field(fn, joints[key]["extension-parameters"], \
				"joints["+key+"][extension-parameters]", active_key, \
				[int, float], float)
	return obj


def load_start_file(body_obj, filename):
	"""
	Load start position descriptor from the given json-encoded file. Verifies
	that all expected keys exist (the expected keys are given in the 'Keys of
	start_obj' part of the documentation for determine_start_positions()) are
	given, and that all keys of 'flexion-angles' are joint names in the body_obj.
	
	While getting the start position descriptor is as simple as json.load(),
	this function gives a helpful message to the user on failure, and also
	converts values to their expected types (such as int -> float).
	
	Parameters:
		body_obj    dict, as returned by load_body_file()
		filename    str, the file to read the starting position from
	Returns:
		An object (as returned by json.load) containing all values expected
		to be in a start position file, some of which may have undergone
		conversions (for example, int -> float).
	"""
	fn = filename
	with open(fn) as file:
		obj = json.load(file)
	if type(obj) != dict:
		raise ValueError(f"Filename: {fn}, Problem: Expected root of JSON to be object")
	
	# make sure root["root1"] is a string
	convert_JSON_field(fn, obj, "root", "root1", [str], str)
	# make sure root["root1"] is a point mass (i.e. a key in body_obj['point-masses'])
	if obj["root1"] not in body_obj["point-masses"]:
		raise ValueError(f"Filename: {fn}, Problem: root1 ({obj["root1"]}) is not in the body")
	# make sure root["root1-position"] is list with 2 floats
	convert_JSON_field(fn, obj, "root", "root1-position", [list], list)
	convert_JSON_el(fn, obj["root1-position"], "root[root1-position]", 0, [int, float], float)
	convert_JSON_el(fn, obj["root1-position"], "root[root1-position]", 1, [int, float], float)
	# make sure root["root2"] is a string
	convert_JSON_field(fn, obj, "root", "root2", [str], str)
	# make sure root["root2"] is a point mass (i.e. a key in body_obj['point-masses'])
	if obj["root2"] not in body_obj["point-masses"]:
		raise ValueError(f"Filename: {fn}, Problem: root2 ({obj["root2"]}) is not in the body")
	# make sure root["root2-rotation"] is a float
	convert_JSON_field(fn, obj, "root", "root2-rotation", [int, float], float)
	# make sure root["flexion-angles"] is dict with a key for each joint in body_obj
	convert_JSON_field(fn, obj, "root", "flexion-angles", [dict], dict)
	for joint_name in body_obj["joints"]:
		if joint_name not in obj["flexion-angles"]:
			raise ValueError(f"Filename: {fn}, Problem: The joint angle for" + \
				f" {joint_name} was not specified (must be a float or null)")
	
	return obj
	
	# this should go wherever u initially determine x, y start positions
	#except RuntimeError as e:
	#	raise RuntimeError("Filename: "+fn+", Problem: "+str(e))


def determine_positions(body_obj, start_obj):
	"""
	Given:
		- A body descriptor (as returned by load_body_file());
		- A start descriptor (as returned by load_start_file()), containing:
			- Each joint angle of the body (at least, enough of them);
			- The position of 1 point mass in the body;
			- The angle of another point mass relative to the first point (as
			  explained in 'Keys of start_obj' below);
	Returns the position of every point mass in the body.
	
	If this detects that there are multiple solutions to the set of constraints
	posed, then a RuntimeError is raised.
	
	Note that this implementation does not currently solve every solvable
	set of constraints. But generally, if:
		- The 2 roots (explained in 'Keys of start_obj') are endpoints of a
		  common segment;
		- The body is not weird (like having point masses that cannot reach
		  each other by travelling through a path of segments);
		- You specify every joint angle (i.e. no None values in flexion_angles);
	Then it will probably work.
	
	Keys of start_obj:
		flexion_angles	dict, mapping each joint name (i.e. each key of
		                body_obj["joints"]) to a float, which is that joint's
		                angle (in degrees) in the desired starting position. The
		                float can also be None, which leaves the joint's angle
		                unspecified and might result in an incomplete set of
		                constraints (which will raise a RuntimeError).
		root1           str, name of the root1 point mass, which is assumed to
		                be in body_obj["point-masses"].
		root1-position  list of 2 floats, giving the start position of root1.
		root2           str, name of the root2 point mass, which is assumed to
		                be in body_obj["point-masses"].
		root2-rotation  float, the angle that specifies the position of root2.
		                The angle is the counterclockwise rotation (deg) of the
		                displacement vector from root1 to root2, relative to the
		                horizontal ray starting at root1 and pointing towards +x.
	Returns:
		The position of every point mass in the body, as a 1d numpy array in
		the following order:
			point 1 x, point 1 y, point 2 x, point 2 y, ...
	"""
	
	# -- take keys out of start_obj -- #
	# (i initially wrote the function with the keys directly as parameters)
	
	root1 = start_obj['root1']
	root1x = start_obj['root1-position'][0]
	root1y = start_obj['root1-position'][1]
	root2 = start_obj['root2']
	theta = start_obj['root2-rotation']
	flexion_angles = start_obj['flexion-angles']
	
	# -- ... -- #
	
	theta *= math.pi / 180
	positions = np.array([[0.0, 0.0]] * len(body_obj["point-masses"]))
	
	# -- sanity check -- #
	
	if root1 == root2:
		raise RuntimeError("root1 and root2 must be different point masses")
	
	# -- record positions of the 2 roots -- #
	
	# find length between root1 and root2
	r1_r2_length = None
	for segment in body_obj["segments"]:
		if root1 in segment[:2] and root2 in segment[:2]:
			r1_r2_length = segment[2]
	if r1_r2_length is None:
		raise RuntimeError("There must be a segment between root1 and root2 (at least for now)")
	# find position of root2
	root2x = root1x + r1_r2_length * math.cos(theta)
	root2y = root1y + r1_r2_length * math.sin(theta)
	# record root1, root2 positions
	pts_ordered = list(body_obj["point-masses"])
	root1_ind = pts_ordered.index(root1)
	root2_ind = pts_ordered.index(root2)
	positions[root1_ind][0] = root1x
	positions[root1_ind][1] = root1y
	positions[root2_ind][0] = root2x
	positions[root2_ind][1] = root2y
	
	# -- make list of constraints for each point -- #
	
	# constraints[pt][i] is pt's i-th constraint, where:
	#     pt is a key in body_obj["point-masses"] (a str), and
	#     i is an array index (an integer)
	# (the format of each constraint isn't explained rn sorry)
	constraints = {pt : [] for pt in body_obj["point-masses"]}
	for joint_name in body_obj["joints"]:
		# if joint angle not specified, skip this iteration
		if flexion_angles[joint_name] is None:
			continue
		# identify the point mass names of the joint
		joint = body_obj["joints"][joint_name]
		b = joint["base"]
		c = joint["center"]
		e = joint["end"]
		# desired counterclockwise angle of ce relative to cb
		angle = ( flexion_angles[joint_name] + joint["offset"] ) * math.pi / 180
		# find lengths of segments ce and cb
		ce_length = None
		cb_length = None
		for segment in body_obj["segments"]:
			if c in segment[:2]:
				if e in segment[:2]:
					ce_length = segment[2]
				elif b in segment[:2]:
					cb_length = segment[2]
		if ce_length is None:
			raise RuntimeError("There is a joint (" + b + ", " + c + \
				", " + e + ") but no segment between " + c + " and " + e)
		if cb_length is None:
			raise RuntimeError("There is a joint (" + b + ", " + c + \
				", " + e + ") but no segment between " + c + " and " + b)
		# add constraint details to set of constraints
		b_constraint = [c, e, -angle, cb_length]
		e_constraint = [c, b, angle, ce_length]
		constraints[b].append(b_constraint)
		constraints[e].append(e_constraint)
	
	# -- solve system of constraints, in a very inefficient way -- #
	
	known_pts = {root1, root2}
	pts_ordered = list(body_obj["point-masses"])
	
	pt_index = 0
	ct_index = 0
	
	while True:
		if pt_index >= len(body_obj["point-masses"]):
			break
		pt = pts_ordered[pt_index]
		if pt in known_pts:
			pt_index += 1; ct_index = 0
			continue
		if ct_index >= len( constraints[pt] ):
			pt_index += 1; ct_index = 0
			continue
		
		constraint = constraints[pt][ct_index]
		c = constraint[0] # center of joint
		o = constraint[1] # endpoint of joint other than pt
		
		if not (c in known_pts and o in known_pts):
			ct_index += 1
			continue
		
		# solve for pt's position
		c_pos = positions[pts_ordered.index(c)]
		o_pos = positions[pts_ordered.index(o)]
		co = o_pos - c_pos
		# required angle of displacement vector from c to pt
		angle = constraint[2] + np.arctan2(co[1], co[0])
		# resulting displacement vector from c to pt
		cpt = np.array([constraint[3] * math.cos(angle), constraint[3] * math.sin(angle)])
		pt_pos = c_pos + cpt
		positions[pt_index] = pt_pos
		
		known_pts.add(pt)
		pt_index = 0
		ct_index = 0
	
	if len(known_pts) < len(pts_ordered):
		raise RuntimeError("Could not determine all point mass positions. " + \
			"This may happen if the body is not connected together by " + \
			"segments and joints in a normal way.")
	
	# -- return positions as a 1d array -- #
	
	return positions.flatten()

def futurestuff():
	# turn (point name -> mass dict) into list of names and list of masses
	self.point_names = list(self.point_masses)
	pm_cp = [0] * len(self.point_names)
	for i in range(len(self.point_names)):
		pm_cp[i] = self.point_masses[self.point_names[i]]
	self.point_masses = np.array(pm_cp)
