import numpy as np
import json
from util import *




# Constants




# once the component of a point mass's velocity parallel to the ground is less than this,
# static friction applies instead of kinetic friction. static friction differs from kinetic
# friction in that:
#	- its direction is opposite to other forces parallel to the ground that the point
#	  experiences, whereas kinetic friction opposes a point's velocity parallel to the ground
#	- its magnitude is capped by the magnitude of other forces parallel to the ground that a
#	  point experiences. kinetic friction is not capped by anything
#	- it may have a higher COF than kinetic friction
static_friction_threshold = 1e-6

# once the component of a point mass's velocity parallel to the ground is less than this,
# it is rounded to 0. This allows kinetic friction to completely stop an object rather
# than start moving it in the other direction (if static friction threshold is low), or
# be taken over by static friction while the point is still moving relative to the ground
# (if static friction threshold is high)
zero_velocity_threshold = 1e-6




# Classes




class EmptyEnvironment:
	"""
	Environment representing empty space. Points never experience gravity, normal forces, or friction forces
	"""
	
	# -- public methods -- #
	
	def in_ground(self, point):
		"""
		-1: in ground
		0: bordering ground surface
		1: out of ground
		"""
		return -1
	
	def environment_force(self, mass, position, velocity, other_force):
		return np.array([0., 0.])
	
	def round_velocity(self, position, velocity):
		return velocity

class LineEnvironment:
	"""
	Environment representing the ground as being the y <= 0 region. While there is an
	equivalent ComplexEnvironment, this computes ground forces more efficiently.
	
	In this model, the normal force a point mass experiences is always vertical (with
	default parameters, always upwards), with a magnitude of the normal force determined
	by the normal force coefficients (see Attributes). The friction force it experiences
	is always horizontal and proportional in magnitude to the normal force.
	
	Attributes:
		gravity				Acceleration due to gravity (m / s^2)
		static_COF			Coefficient of static friction (dimensionless)
		kinetic_COF			Coefficient of kinetic friction (dimensionless)
		nf_coefficients_1
			Coefficients A, B, C of normal force while a point is moving deeper into
			the ground. The magnitude of the normal force (which is in the direction
			of the surface of the ground) is:
				Ar + Br' + C
			where r is the (positive) amount of displacement (m) into the ground, and
			r' is the derivative of r with respect to time (m / s)
		nf_coefficients_2
			Coefficients A, B, C of the normal force while a point is moving closer to,
			or staying the same distance from, the surface of the ground. The magnitude
			of the normal force follows the same equation as nf_coefficients_1
	"""
	
	# -- constructor -- #
	
	def __init__(self):
		""" Create a new LineEnvironment instance """
		self.gravity = 9.8
		self.static_COF = 0.209
		self.kinetic_COF = 0.209
		self.nf_coefficients_1 = [50., 50., 50.]
		self.nf_coefficients_2 = [50., 0., 0.]
		pass
	
	# -- public methods -- #
	
	def in_ground(self, point):
		"""
		-1: in ground
		0: bordering ground surface
		1: out of ground
		"""
		if point[1] < 0:
			return -1
		if point[1] == 0:
			return 0
		return 1
	
	def environment_force(self, mass, position, velocity, other_force):
		if self.in_ground(position) == 1:
			return np.array([0., 0.])
		
		# determine gravity force magnitude (always downwards, right? RIGHT?)
		
		gmag = - mass * self.gravity
		
		# determine normal force magnitude (upwards -> positive, downwards -> negative)
		
		r = - point[1]
		rp = - velocity[1]
		if rp > 0:
			nf_cfs = self.nf_coefficients_1
		else:
			nf_cfs = self.nf_coefficients_2
		nmag = r * nf_cfs[0] + rp * nf_cfs[1] + nf_cfs[2]
		
		# determine friction force magnitude (towards +x -> positive, towards -x -> negative)
		
		using_static_friction = abs(velocity[0]) < static_friction_threshold
		if using_static_friction:
			fmag = min(abs(nmag) * self.static_COF, abs(other_force[1]))
			if other_force[0] > 0:
				f_mag *= -1
		else:
			fmag = abs(nmag) * self.kinetic_COF
			if velocity[0] > 0:
				fmag *= -1
		
		# return sum of forces
		
		return np.array([fmag, gmag + nmag])
	
	def round_velocity(self, position, velocity):
		vel = velocity
		if self.in_ground(position) != 1 and abs(vel[0]) < zero_velocity_threshold:
			vel[0] = 0
		return vel


class ComplexEnvironment:
	"""
	Environment representing the surface of the ground as a curve. The curve consists
	of 2 rays joined by a series of connected line segments, which start at the origin
	of one ray and end at the other's.
	
	In this model, the normal force a point mass experiences is in the direction of
	the nearest point on the surface of the ground curve, and with a magnitude determined
	by the normal force coefficients (see Attributes). The friction force it experiences
	is perpendicular to the normal force and proportional in magnitude to it.
	
	Attributes:
		gravity				Acceleration due to gravity (m / s^2)
		static_COF			Coefficient of static friction (dimensionless)
		kinetic_COF			Coefficient of kinetic friction (dimensionless)
		nf_coefficients_1
			Coefficients A, B, C of normal force while a point is moving deeper into
			the ground. The magnitude of the normal force (which is in the direction
			of the surface of the ground) is:
				Ar + Br' + C
			where r is the amount of displacement (m) into the ground, and r' is the
			derivative of r with respect to time (m / s)
		nf_coefficients_2
			Coefficients A, B, C of the normal force while a point is moving closer
			to the surface of the ground
		ground_point
			A point in the ground, determining which of the 2 regions the ground curve
			divides the ground is considered to be in
	"""
	
	
	# -- getters and setters -- #
	
	def get_num_points(self):
		""" Returns number of points used to describe ground curve """
		return len(self.__point_sequence)
	def get_point(self, index):
		""" Returns the point (tuple of 2 floats) at given index of the ground curve """
		return self.__point_sequence[index]	
	
	def set_point(self, index, point):
		""" Set point of ground curve at given index to given point """
		self.__point_sequence[index] = point
		self.__segments_constructed = False
	def insert_point(self, index, point):
		""" Insert given point into ground curve at given index """
		self.__point_sequence.insert(index, point)
		self.__segments_constructed = False
	def remove_point(self, index):
		"""
		Remove point at given index of ground curve. If the ground curve only contains
		2 points, then a point cannot be removed and a RuntimeError is thrown.
		"""
		if len(self.__point_sequence) <= 2:
			raise RuntimeError("Ground curve must contain at least 3 points to have a point removed")
		self.__point_sequence.remove(index)
		self.__segments_constructed = False
	
	
	# -- constructors -- #
	
	
	def __init__(self, filename=None):
		"""
		If filename is None, creates a new ComplexEnvironment with default parameters.
		
		Otherwise, loads a ComplexEnvironment from given file which contains
		ComplexEnvironment data in JSON format.
		Throws:
			- OSError if file cannot be read
			- JSONDecodeError or UnicodeDecodeError if the file does not contain JSON data
			  encoded in UTF8/16/32
			- ValueError (with an appropriate message) if the JSON data does not contain
			  all needed ComplexEnvironment data in the correct format
		"""
		if filename is None:
			gravity = 9.8
			static_COF = 0.209
			kinetic_COF = 0.209
			nf_coefficients_1 = [50., 50., 50.]
			nf_coefficients_2 = [50., 0., 50.]
			ground_point = np.array([0., -1.])
			__point_sequence = [np.array([-1, 0]), np.array([1, 0])]
			__segments = None
			__segments_constructed = False
			return
		
		with open(filename) as file:
			obj = json.load(file)
		
		if type(obj) != dict:
			raise ValueError("JSON root element must be object")
		
		# extract float fields
		self.gravity = float( from_JSON_obj(obj, "root", "gravity", [float, int]) )
		self.static_COF = float( from_JSON_obj(obj, "root", "static_COF", [float, int]) )
		self.kinetic_COF = float( from_JSON_obj(obj, "root", "kinetic_COF", [float, int]) )
		# extract list fields
		self.nf_coefficients_1 = from_JSON_obj(obj, "root", "normal_force_coefficients_1", [list])
		self.nf_coefficients_2 = from_JSON_obj(obj, "root", "normal_force_coefficients_2", [list])
		self.ground_point = from_JSON_obj(obj, "root", "ground_point", [list])
		self.__point_sequence = from_JSON_obj(obj, "root", "point_sequence", [list])
			
		# make sure normal force coefficients have exactly 3 floats
		if len(self.nf_coefficients_1) != 3 or len(self.nf_coefficients_2) != 3:
			raise ValueError("There must be exactly 3 of both types of normal force coefficients")
		try:
			for i in range(3):
				self.nf_coefficients_1[i] = float(self.nf_coefficients_1[i])
				self.nf_coefficients_2[i] = float(self.nf_coefficients_2[i])
		except Exception:
			raise ValueError("Normal force coefficients could not be interpreted as numbers")
		self.nf_coefficients_1 = np.array(self.nf_coefficients_1)
		self.nf_coefficients_2 = np.array(self.nf_coefficients_2)
		
		# make sure ground point has exactly 2 float coordinates
		if len(self.ground_point) != 2:
			raise ValueError("Ground point must have exactly 2 coordinates")
		try:
			for i in range(2):
				self.ground_point[i] = float(self.ground_point[i])
		except Exception:
			raise ValueError("Ground point coordinates could not be interpreted as numbers")
		self.ground_point = np.array(self.ground_point)
		
		# make sure point sequence has at least 2 elements, all of which are lists of 2 floats
		if len(self.__point_sequence) < 2:
			raise ValueError("Point sequence must have at least 2 points")
		for i in range(len(self.__point_sequence)):
			try:
				self.__point_sequence[i] = list(self.__point_sequence[i])
				if len(self.__point_sequence[i]) != 2: raise Exception
				self.__point_sequence[i][0] = float(self.__point_sequence[i][0])
				self.__point_sequence[i][1] = float(self.__point_sequence[i][1])
			except Exception:
				raise ValueError("Each element / point of the point sequence must be a list of 2 numbers")
			self.__point_sequence[i] = np.array(self.__point_sequence[i])


	# -- public methods -- #
	
	
	def merge_colinear(self):
		"""
		For every contiguous sequence of segments which is colinear, merges them all
		together into 1 segment
		"""
		pass
	
	def check_colinear(self):
		"""
		Check that ground point is not colinear with any segment. If it is, returns the index of
		the first point in point_sequence that is part of a colinear segment.
		"""
		pass
	
	def in_ground(self, position):
		"""
		-1: in ground
		0: bordering ground surface
		1: out of ground
		"""
		
		if not self.__segments_constructed:
			self.__construct_segments()
		
		# -- determine number of intersections between ground curve and pg -- #
		
		p = position
		g = self.ground_point
		intersections = 0
		
		for i in range(len(self.__segments)):
			segment = self.__segments[i]
			p1 = segment.p1
			p2 = segment.p2
			
			# side of p1p2's line that p is in
			p_region = side_of_line(segment.line, p)
			# side of p1p2's line that g is in
			g_region = side_of_line(segment.line, g)
			
			
			# p is on p1p2's line
			
			
			if p_region == 0:
				p_region3 = segment.region_of(p)
				
				# if p is on p1p2, then p is a border point
				if p_region3 in ( (0, 0), (1, 1), (2, 1) ):
					return 0
				
				# now, p is on p1p2's line, but not on p1p2
				
				# if g is not on p1p2's line, no intersection
				if g_region != 0:
					continue
				
				# now, g is on p1p2's line
				
				g_region3 = segment.region_of(g)
				
				# if g is on p1p2, this is an error
				if g_region3 in ( (0, 0), (1, 1), (2, 1) ):
					raise Exception("Ground point cannot be on the ground curve")
				
				# if g is in the same region3 as p, no intersection
				if g_region3 == p_region3:
					continue
				
				# now:
				#	- p and g are both on p1p2's line
				#	- p and g are not on p1p2
				#	- p and g are on opposite region3's of p1p2. this means p1p2 lies
				#	  between them, and both p1ray and p2ray are False
				
				# find the points q, r before and after segment
				q = point_sequence[i - 1]
				r = point_sequence[i + 2]
				pg_abc = line_coefficients(p, g)
				q_region = side_of_line(pg_abc, q)
				r_region = side_of_line(pg_abc, r)
				
				# make sure that q, r do not lie on pg's line (which equals p1p2's line)
				if q_region == 0 or r_region == 0:
					raise Exception("Adjacent ground curve segments must not be colinear")
				
				# if q, r are on same side of (pg / p1p2)'s line, no intersection
				if q_region == r_region:
					continue
				
				# if q, r are on different sides of (pg / p1p2)'s line, there is an intersection
				intersections += 1
				continue
			
			
			# g is on p1p2's line, p is not
			
			
			if g_region == 0:
				# if g is on the segment, this is an error
				g_region3 = segment.region_of(g)
				if g_region3 in ( (0, 0), (1, 1), (2, 1) ):
					raise Exception("Ground point cannot be on the ground curve")
				# if g is not on p1p2, no intersection
				continue
			
			
			# p, g are both in the same non-line region of segment 
			
			
			if p_region != 0 and p_region == g_region:
				# no intersection
				continue
			
			
			# p, g are on opposite sides of the segment line
			
			
			# coefficients A, B, C of pg's line equation
			pg_abc = line_coefficients(p, g)
			
			# region of pg's line that p1, p2 are in
			p1_region = side_of_line(pg_abc, segment.p1)
			p2_region = side_of_line(pg_abc, segment.p2)
			
			# if p1, p2 are on the same non-line regions of pg, then no intersection
			if p1_region != 0 and p2_region == p1_region:
				continue
			
			# if p1, p2 are on opposite non-line regions of pg, there is an intersection
			if p1_region != 0 and p2_region != 0:
				intersections += 1
				continue
			
			# now, either p1 or p2 is on pg's line and therefore pg
			# (p1 and p2 cannot both be - we already checked if p, g are both on p1p2's segment line)
			
			# find which one is 'it' (which of p1 or p2 is on pg's segment)
			it = i
			if p2_region == 0:
				it += 1
			
			# if 'it' is a ray representative, there is an intersection
			if it == i and segment.p1ray:
				intersections += 1
				continue
			elif segment.p2ray:
				intersections += 1
				continue
			
			# now, 'it' is an endpoint
			# find its adjacent point on the ground curve other than p1/p2
			
			if it == i:
				adjacent = point_sequence[it - 1]
				not_it_region = p1_region
			else:
				adjacent = point_sequence[it + 1]
				not_it_region = p2_region
			adj_region = line_region_of(pg_abc, adjacent)
			
			# compare region of pg's line that not_it and adjacent are in
			
			# if adjacent is on pg's line, we did or will deal with that another iteration
			#	(note: the way we determine if it-adjacent and p-g are colinear is different here than
			#	 on the other iteration, so it may be unreliable. a solution is to avoid choosing a
			#	 ground point that is close to colinear with a segment of the ground curve)
			if adj_region == 0:
				continue
			
			# if adjacent and not_it are on the same non-line side of pg, there is no intersection
			if adj_region == not_it_region:
				continue
			
			# if adjacent and not_it are on different sides of pg, then there is an intersection
			intersections += 1
			continue
		
		# even number of intersections -> in ground
		if (intersections % 2) == 0:
			return -1
		# odd number -> out of ground
		return 1
	
	
	def environment_force(self, mass, position, velocity, other_force):
		# start with gravity
		gravity_force = np.array([0, - self.gravity * mass])
		non_ground_force = other_force + gravity_force
		
		# if position not in ground, gravity is all
		if self.in_ground(position) == 1:
			return gravity
		
		# determine normal force vector
		
		to_ground = self.__vector_to_ground(position)
		to_ground_dist = np.linalg.norm(nf)
		
		r = to_ground_dist
		rp = component(velocity, to_ground)
		if rp > 0:
			nf_cfs = self.nf_coefficients_1
		else:
			nf_cfs = self.nf_coefficients_2
		nmag = r * nf_cfs[0] + rp * nf_cfs[1] + nf_cfs[2]
		
		normal_force = to_ground * (nmag / np.linalg.norm(to_ground))
		
		# determine friction force vector
		
		friction_direction = np.array([normal_force[1], - normal_force[0]])
		
		parallel_velocity_mag = component(velocity, friction_direction)
		using_static_friction = abs(parallel_velocity_mag) < static_friction_threshold
		if using_static_friction:
			parallel_force_mag = component(non_ground_force, friction_direction)
			friction_mag = min(abs(nmag) * self.static_COF, abs(parallel_force_amt))
			if parallel_force_mag > 0:
				friction_mag *= -1
		else:
			friction_mag = abs(nmag) * self.kinetic_COF
			if parallel_velocity_mag > 0:
				friction_mag *= -1
		
		friction_force = friction_direction * (friction_mag / np.linalg.norm(friction_direction))
		
		# return sum of forces
		
		return gravity_force + normal_force + friction_force
	
	
	def round_velocity(self, position, velocity):
		if self.in_ground(position) == 1:
			return velocity
		
		to_ground = self.__vector_to_ground(position)
		friction_direction = np.array([to_ground[1], - to_ground[0]])
		parallel_velocity_mag = component(velocity, friction_direction)
		if abs(parallel_velocity_mag) < zero_velocity_threshold:
			return velocity + friction_direction * (- parallel_velocity_mag / np.linalg.norm(friction_direction) )
		return velocity
	
	
	def save(self, filename):
		"""
		Saves ComplexEnvironment data to given file in JSON format. The resulting file can
		later be passed to the ComplexEnvironment constructor to recreate this Environment.
		"""
		obj = {}
		obj['gravity'] = self.gravity
		obj['static_COF'] = self.static_COF
		obj['kinetic_COF'] = self.kinetic_COF
		obj['normal_force_coefficients_1'] = self.nf_coefficients_1
		obj['normal_force_coefficients_2'] = self.nf_coefficients_2
		obj['ground_point'] = self.ground_point
		obj['point_sequence'] = self.__point_sequence
		with open(filename, 'r') as file:
			json.dump(obj, file)
	
	
	# -- private methods -- #
	
	
	def __construct_segments(self):
		self.__segments = [None] * (len(self.__point_sequence) - 1)
		for i in range(len(self.__point_sequence) - 1):
			p1 = self.__point_sequence[i]
			p2 = self.__point_sequence[i + 1]
			p1ray = (i == 0)
			p2ray = (i == len(self.__point_sequence) - 2)
			
			segment = Segment(p1, p2, p1ray, p2ray)
			self.__segments[i] = segment
		self.__segments_constructed = True
	
	def __vector_to_ground(self, position):
		"""
		Computes and returns the displacement vector from given position to a
		nearest point on the surface of the ground
		"""
		if not self.__segments_constructed:
			self.__construct_segments()
		
		# find lowest distance between position and a segment / ray of the ground curve
		
		min_distance = -1	# (-1 represents no segments / rays checked yet)
		best_segment = -1
		best_region = 0
		for segment in self.__segments:
			
			region = segment.region_of(position)
			if region == 0:
				dist = segment.perpendicular_distance(position)
			elif region == 1:
				dist = np.linalg.norm(segment.p1 - position)
			elif region == 2:
				dist = np.linalg.norm(segment.p2 - position)
			
			if dist < min_distance or min_distance == -1:
				min_distance = dist
				best_segment = segment
				best_region = region
		
		# knowing the closest segment and region of segment, return displacement vector
		
		if best_region == 0:
			return best_segment.perpendicular(position)
		if best_region == 1:
			return segment.p1 - position
		return segment.p2 - position


class Segment:
	"""
	Represents a line segment or ray of the ground curve. Contains preprocessed data
	about the segment which speeds up normal force and friction calculations.
	"""
	
	# endpoints of segment
	p1 = None
	p2 = None
	
	# whether each endpoint is a representative of a ray
	p1ray = False
	p2ray = False
	
	# coefficients A, B, C of line equation (the line that p1, p2 fall on)
	# (x, y) is on line <===> Ax + By + C = 0
	line = (0, 0, 0)
	
	# if p1 is a segment endpoint (not a ray representative), then region 1
	# is defined as the region of points whos closest point to the segment
	# is p1. similarly for p2
	
	# coefficients A, B, C of region 1 equation (if p1ray is False)
	# (x, y) is in region 1 <===> Ax + By + C >= 0
	r1 = (0, 0, 0)
	
	# coefficients A, B, C of region 2 equation (or None if p2ray is False)
	# (x, y) is in region 2 <===> Ax + By + C >= 0
	r2 = (0, 0, 0)
	
	
	# -- public methods -- #
	
	
	def __init__(self, p1, p2, p1ray, p2ray):
		self.p1 = p1
		self.p2 = p2
		self.p1ray = p1ray
		self.p2ray = p2ray
		
		# set coefficients of line and region equations
		
		self.line = line_coefficients(p1, p2)
		if not p1ray:
			self.r1 = region_coefficients(p2, p1)
		if not p2ray:
			self.r2 = region_coefficients(p1, p2)
	
	def region_of(self, position):
		"""
		Returns the tuple (r, b), where:
			r = 1 <==>	p1 is a segment endpoint (instead of a ray representative),
						and p1 is the closest point on this segment from given position.
			r = 2 <==>	p2 is a segment endpoint (instead of a ray representative),
						and p2 is the closest point on this segment from given position.
			b = 1 <==>	the given position is on the boundary of one of these regions
		And r and b default to 0.
		"""
		
		if not self.p1ray:
			val = self.r1[0] * position[0] + self.r1[1] * position[1] + self.r1[2]
			if val > 0:
				return (1, 0)
			if val == 0:
				return (1, 1)
		if not self.p2ray:
			val = self.r2[0] * position[0] + self.r2[1] * position[1] + self.r2[2]
			if val > 0:
				return (2, 0)
			if val == 0:
				return (2, 1)
		return (0, 0)
	
	def perpendicular_distance(self, position):
		"""
		Finds the shortest distance between given position and the line that this
		segment lies on. Note that this is NOT necessarily the same as the shortest
		distance between given position and this line segment.
		"""
		
		line = self.line
		return abs(line[0] * position[0] + line[1] * position[1] + line[2]) / \
		       (line[0] ** 2 + line[1] ** 2) ** (1 / 2)
	
	def perpendicular(self, position):
		"""
		Finds the displacement vector from given position and the line that this
		segment lies on.
		"""
		
		displacement = np.array([self.p2[1] - self.p1[1], self.p1[0] - self.p2[0]])
		displacement *= self.perpendicular_distance(position) / np.linalg.norm(displacement)
		
		p3 = self.p2 + displacement
		line = self.line
		
		# which side of (the line this segment lies on) is position on (+ / -)
		position_side = line[0] * position[0] + line[1] * position[1] + line[2] > 0
		# which side of (the line this segment lies on) is p3 on (+ / -)
		p3_side = line[0] * p3[0] + line[1] * p3[1] + line[2] > 0
		
		if position_side == p3_side:
			displacement *= -1
		
		return displacement




# Helper functions




def line_coefficients(z1, z2):
	"""
	For points z1 != z2, finds the equation of the line that z1, z2 fall on.
	This equation is of the form Ax + By + C = 0. Returns the tuple (A, B, C).
	"""
	
	abc = [0, 0, 0]
	abc[0] = z1[1] - z2[1]
	abc[1] = z2[0] - z1[0]
	abc[2] = z1[0] * (z2[1] - z1[1]) + z1[1] * (z1[0] - z2[0])
	
	return tuple(abc)


def region_coefficients(z1, z2):
	"""
	For points z1 != z2, finds the equation of the region of points whos nearest
	point on the line segment from z1 to z2 is z2. This equation is of the form
	Ax + By + C >= 0. Returns the tuple (A, B, C).
	"""
	
	displacement = z2 - z1
	bounding_point = z2 + np.array([displacement[1], - displacement[0]])
	
	# find equation of the line (Ax + By + C = 0) that z2 and bounding_point lie on
	
	abc = line_coefficients(z2, bounding_point)
	
	# if z1 is in region Ax + By + C > 0, then negate A, B, C so that it is not
	
	if abc[0] * z1[0] + abc[1] * z1[1] + abc[2] > 0:
		abc[0] *= -1
		abc[1] *= -1
		abc[2] *= -1
	
	return abc


def side_of_line(abc, z):
	"""
	For a point z, finds the side of the line given by coefficients A, B, C
	(in the tuple abc) that z falls on.
	Returns:
		-1		if Ax + By + C < 0
		0		if Ax + By + C = 0
		1		if Ax + By + C > 0
	where x, y are the components of z, and A, B, C are the elements of the tuple abc.
	"""
	
	val = abc[0] * z[0] + abc[1] * z[1] + abc[2]
	if val < 0:
		return -1
	if val == 0:
		return 0
	return 1

