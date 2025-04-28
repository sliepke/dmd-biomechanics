import numpy as np


def convert_JSON_field(filename, obj, objname, fieldname, accepted_types, convert_to):
	"""
	If obj[fieldname] exists and its type is in accepted_types, then simply
	converts obj[fieldname] in-place to the type convert_to.
	
	Otherwise, raises an exception with a message that assumes objname represents
	where obj was found in the JSON file given by filename. For example, objname
	could start with "root" as in "root[0]['names']". This helps the exception
	message indicate where in the JSON file the error is.
	
	Parameters:
		- obj               dict
		- objname           string
		- fieldname         string
		- accepted_types    list of types
		- convert_to        type
	Returns:
		nothing
	Raises:
		- ValueError, if obj does not contain fieldname as a key
		- TypeError, if obj[fieldname] is not one of the accepted types
	"""
	if fieldname not in obj:
		raise ValueError("JSON file: "+filename+", Problem: the object '" + objname + "' is missing the '" + fieldname + "' field")
	field = obj[fieldname]
	if type(field) not in accepted_types:
		types_str = ""
		for i in range(len(accepted_types)):
			if i != 0:
				types_str += ", "
			typ = accepted_types[i]
			types_str += typ.__name__
		raise ValueError("JSON file: "+filename+", Problem: the field '" + objname + "["+fieldname+"]'" + \
			" must be one of the following types: " + types_str)
	obj[fieldname] = convert_to(field)


def convert_JSON_el(filename, arr, arrname, index, accepted_types, convert_to):
	"""
	If arr[index] is in bounds and its type is in accepted_types, then simply
	converts arr[index] in-place to the type convert_to.
	
	Otherwise, raises an exception with a message that assumes arrname represents
	where arr was found in the JSON file given by filename. For example, arrname
	could start with "root" as in "root[0]['mass-list']". This helps the exception
	message indicate where in the JSON file the error is.
	
	Parameters:
		- filename          filename that arr was json-decoded from
		- arr               list
		- arrname           string
		- index             int
		- accepted_types    list of types
		- convert_to        type
	Returns:
		nothing
	Raises:
		- ValueError, if arr[index] is out of bounds
		- TypeError, if arr[index] is not one of the accepted types

	"""
	if index not in range(len(arr)):
		raise ValueError("JSON file: "+filename+", Problem: the array '" + arrname + "' is missing the index '" + str(index))
	val = arr[index]
	if type(val) not in accepted_types:
		types_str = ""
		for i in range(len(accepted_types)):
			if i != 0:
				types_str += ", "
			typ = accepted_types[i]
			types_str += typ.__name__
		raise ValueError("JSON file: "+filename+", Problem: the element " + arrname + "["+str(index)+"]" + \
			" must be one of the following types: " + types_str)
	arr[index] = convert_to(val)
