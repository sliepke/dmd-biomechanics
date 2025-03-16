import copy
import json
import numpy as np


def component(v1, v2):
	"""
	Returns the component of v1 along v2. Meaning, the magnitude of the projection
	of v2 along v2.
	"""
	
	return np.dot(v1, v2) / np.linalg.norm(v2)


def from_JSON_obj(obj, objname, fieldname, accepted_types):
	"""
	Parameters:
		- obj				dictionary
		- objname           string
		- fieldname         string
		- accepted_types	list of types
	Returns:
		obj[fieldname]
	Raises:
		ValueError if:
			- obj does not contain fieldname as a key
			- obj[fieldname] is not one of the accepted types
		The ValueError will contain an appropriate message, in the
		context of extracting values of a JSON file containing obj
	"""
	if fieldname not in obj:
		raise ValueError("File's '" + objname + "' object is missing '" + fieldname + "' field")
	field = obj[fieldname]
	if type(field) not in accepted_types:
		types_str = ""
		for i in range(len(accepted_types)):
			if i != 0:
				types_str += ", "
			typ = accepted_types[i]
			types_str += typ.__name__
		raise ValueError("File's " + objname + "[" + fieldname + "] field must " +
			"be one of the following types: " + types_str)
	return field


def from_JSON_array(arr, arrname, index, accepted_types):
	"""
	Parameters:
		- arr               list
		- arrname           string
		- index             int
		- accepted_types	list of types
	Returns:
		arr[index]
	Raises:
		ValueError if:
			- index is out of bounds
			- arr[index] is not one of the accepted types
		The ValueError will contain an appropriate message, in the
		context of extracting values of a JSON file containing arr
	"""
	if index not in range(len(arr)):
		raise ValueError("File's '" + arrname + "' array is missing index " + str(index))
	val = arr[index]
	if type(val) not in accepted_types:
		types_str = ""
		for i in range(len(accepted_types)):
			if i != 0:
				types_str += ", "
			typ = accepted_types[i]
			types_str += typ.__name__
		raise ValueError("File's " + arrname + "[" + str(index) + "] value must " +
			"be one of the following types: " + types_str)
	return val


def validate_JSON_structure(val, valname, structure):
	"""
	Makes sure that given JSON value follows the given structure (see examples).
	
	If it does, returns a copy of val, but possibly with some values converted
	using native conversions, if that is necessary to make it follow the expected
	structure.
	
	Otherwise, raises:
		- TypeError, if val is not a JSON value as described below
		- ValueError, if val does not conform to given structure
	
	Parameters:
		val, one of the following:
			- instance of str, int, float, bool, None
			- a list whos elements are one of these
			- a dict whos keys are strings and values are one of these
		structure, one of the following:
			- str, int, float, bool, None (the types, not instances of them)
			- a list whos elements are one of these
			- a dict whos keys are strings and values are one of these
	
	Example:
		val = ["7.3", 4]
		structure = [int, int]
		result = [7, 4]
	Example:
		val = [[1]]
		structure = [int]
		(ValueError: Expected (valname)[0] to be int
	Example:
		val = { { "num": 7 }, "num": 8 }
		structure = { { "num": str }, "num": str }
		result = { { "num": "7" }, "num": "8" }
	Example:
		val = { "x": 20, "y": 25, "z": 30 }
		structure = { "x": 20, "y": 25 }
		(ValueError: Did not expect (valname) to have field 'z')
	"""
	
	def validate_subvalue(obj, structure):
		# Upon error, returns a tuple with:
		#     - error code (int) (see indexing for 'value' meaning)
		#           1: value is not the expected type
		#           2: value is a dict without an expected field
		#           3: value is a dict with an unexpected field
		#           4: value is a non-json type
		#           5: value (but with structure being indexed) is a non-json type
		#     - indexing (string)
		#           eval("obj" + indexing) is the value that the error applies to
		#           (for example, if indexing is "[3]['hello']", then the error
		#           applies to obj[3]['hello'])
		#     - arg (string)
		#           if error code 1: expected type
		#           if error code 2: expected field
		#           if error code 3: unexpected field
		#           if error code 4: the non-json type found
		#           if error code 5: the non-json type found
		# If there is no error, does one or the other:
		#     - If obj is a list or dict, converts any values within obj
		#       in place, and returns None
		#     - If obj is some other json type, returns obj converted
		#       to the expected type
		
		if type(structure) not in [str, int, float, bool, None, list, dict]:
			return (5, "", type(structure).__name__)
		
		obj_type = type(obj)
		if obj_type in [str, int, float, bool, None]:
			try:
				converted = structure(obj)
			except Exception:
				return (1, "", structure.__name__)
			return converted
		if obj_type in [dict, list]:
			# make sure dict / list was expected
			if type(structure) != list:
				return (1, "", structure.__name__)
		if obj_type == list:
			# make sure the lists obj and structure are the same length
			if len(obj) != len(structure):
				return ( 1, "", "list of length " + str(len(structure)) )
			# validate each element of obj
			for i in range(obj):
				result = validate_subvalue(obj[i], structure[i])
				if type(result) is tuple:
					indexing = "[" + str(i) + "]" + result[1]
					return (result[0], indexing, result[2])
				if type(obj[i]) not in [list, dict]:
					obj[i] = result
		if obj_type == dict:
			# make sure all keys of obj are expected
			for obj_key in obj:
				if obj_key not in structure:
					return (3, "", obj_key)
			# make sure all expected keys are in obj
			for st_key in structure:
				if st_key not in obj:
					return (2, "", st_key)
			# validate each value of obj
			for key in obj:
				result = validate_subvalue(obj[key], structure[key])
				if type(result) is tuple:
					indexing = "[" + key + "]" + result[1]
					return (result[0], indexing, result[2])
				if type(obj[key]) not in [list, dict]:
					obj[key] = result
		else:
			return (4, "", obj_type.__name__)
	
	cp = copy.deepcopy(val)
	result = validate_subvalue(cp, structure)
	
	if type(result) is tuple:
		code = result[0]
		indexing = result[1]
		arg = result[2]
		src = valname + indexing
		if code  == 1:
			raise ValueError("Expected " + src + " to be " + arg)
		if code  == 2:
			raise ValueError("Expected " + src + " to have field '" + arg + "'")
		if code  == 3:
			raise ValueError("Did not expect " + src + " to have field '" + arg + "'")
		if code == 4:
			raise TypeError(src + " is the non-json type " + arg)
		raise TypeError("structure" + indexing + " is the non-json type " + arg)
	if type(result) is None:
		return cp
	return result
