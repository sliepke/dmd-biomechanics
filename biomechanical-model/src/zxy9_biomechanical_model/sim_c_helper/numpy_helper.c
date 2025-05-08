// // //
//
// Defines
//   - A helper function for requiring a numpy array argument to be a certain
//     type and shape
//   - A helper function for copying elements of a 1d or 2d numpy array into
//     a certain struct field throughout an array of structs
//   - A helper function for printing a 1d numpy array of doubles
//
// // //

#ifndef NP_HELPER
#define NP_HELPER

#include <stdbool.h> // for true and false




/*
parameters
	np_array
		numpy array to validate the type and shape of
	type
		NPY_INT: require NPY_INT
			i'm pretending this is the same as a c int, which some sources say it is.
			however, the API (maybe mistakenly?) only says it is 32 bit. it seems like
			the website might be setting the definition of NPY_INT depending on which
			platform is reading the page (without mentioning that it depends on the
			platform and is the same as a c int). if that is the case, then that is ridiculous.
				https://numpy.org/doc/stable/reference/c-api/dtype.html#c.NPY_TYPES.NPY_INT
		NPY_DOUBLE: require NPY_DOUBLE (again, pretending this is the same as double)
	ndim
		1: require a 1d array with shape (d1)
		2: require a 2d array with shape (d1, d2)
		EXCEPTION TO THIS RULE: if ndim is 2 and d1 is 0, allows a 1d, 0-length array
		(shouldn't be anything else)
	d1
		explained with ndim
	d2
		explained with ndim
returns
	true: success
	false: failure (then this will printf the argument's name, and set python's error to a less specific message)
*/
int validate_np_arg(const char* name, PyArrayObject* np_array, int type, int ndim, npy_intp d1, npy_intp d2) {
	// validate type arg
	if (type != NPY_INT && type != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "validate_np_arg called with a type other than NPY_INT or NPY_DOUBLE\n");
		goto validate_fail;
	}
	// validate ndim arg
	if (ndim != 1 && ndim != 2) {
        PyErr_SetString(PyExc_ValueError, "validate_np_arg called with an ndim other than 1 or 2\n");
		goto validate_fail;
	}
	// validate type
	if (PyArray_TYPE(np_array) != type) {
		if (type == NPY_INT)
			PyErr_SetString(PyExc_ValueError, "A numpy array argument did not have type NPY_INT\n");
		if (type == NPY_DOUBLE)
			PyErr_SetString(PyExc_ValueError, "A numpy array argument did not have type NPY_DOUBLE\n");
		goto validate_fail;
	}
	
	//// validate shape
	int array_ndim = PyArray_NDIM(np_array);
	npy_intp *array_shape = PyArray_SHAPE(np_array);
	// allow shape of (0) when ndim = 2, d1 = 0
	if (ndim == 2 && d1 == 0 && array_ndim == 1 && array_shape[0] == 0)
		return true;
	// validate number of dimensions
	if (array_ndim != ndim) {
        PyErr_SetString(PyExc_ValueError, "A numpy array argument did not have the correct number of dimensions\n");
		goto validate_fail;
	}
	// validate shape
	if (array_shape[0] != d1) {
        PyErr_SetString(PyExc_ValueError, "A numpy array argument did not have the correct length of dimension 1\n");
		goto validate_fail;
	}
	if (ndim == 2 && array_shape[1] != d2) {
        PyErr_SetString(PyExc_ValueError, "A numpy array argument did not have the correct length of dimension 2\n");
		goto validate_fail;
	}
	
	return true;
validate_fail:
	printf("failing arg: %s\n", name);
	return false;
}

/* checks that PyArray_ISBEHAVED(the given numpy array) returns true. necessary
   if you want to directly access the data in a simple way. if not behaved,
   sets an appropriate exception message and returns false. otheriwse,
   returns true */
int validate_np_arg_behaved(const char* name, PyArrayObject* np_array) {
	if (!PyArray_ISBEHAVED(np_array)) {
        PyErr_SetString(PyExc_ValueError, "A numpy array argument was not contiguous or aligned\n");
		printf("failing arg: %s\n", name);
		return false;
	}
	return true;
}

/*
parameters
	np_arr
		a numpy array to copy elements from
	ndim
		1: data sources are np_arr[0], np_arr[1], ... (num_steps times)
		2: data sources are np_arr[0][i1], np_arr[1][i1], ... (num_steps times)
		(shouldn't be anything else)
	i1
		explained with ndim
	type
		NPY_INT: copies into int array (then numpy array should have type NPY_INT)
		NPY_DOUBLE: copies into double array (then numpy array should have type NPY_DOUBLE)
	dst_base
		data destinations are dst_base, dst_base + dst_step, ... (num_steps times)
	dst_step
		explained with dst_base
	num_steps
		explained with ndim and dst_base
returns
	true: success
	false: failure
*/
int copy_from_np(PyArrayObject* np_arr, int ndim, int i1, int type, void* dst_base, size_t dst_step, size_t num_steps) {
	// validate type arg
	if (type != NPY_INT && type != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "validate_np_arg called with a type other than NPY_INT or NPY_DOUBLE\n");
        return false;
	}
	// validate ndim arg
	if (ndim != 1 && ndim != 2) {
        PyErr_SetString(PyExc_ValueError, "validate_np_arg called with an ndim other than 1 or 2\n");
        return false;
	}
	// copy from numpy to native array
	for (size_t i = 0; i < num_steps; i++) {
		char* dst = ((char*) dst_base) + i * dst_step;
		// get element of the numpy array
		PyObject *val_obj;
		if (ndim == 1)
			val_obj = PyArray_GETITEM(np_arr, PyArray_GETPTR1(np_arr, i));
		else
			val_obj = PyArray_GETITEM(np_arr, PyArray_GETPTR2(np_arr, i, i1));
		
		if (val_obj == NULL)
			return false;
		
		// set value at dst
		if (type == NPY_INT)
			// set an int
			*((int*) dst) = PyLong_AsLong(val_obj);
		else
			// set a double
			*((double*) dst) = PyFloat_AsDouble(val_obj);
		Py_DECREF(val_obj);
		// if PyX_AsY() failed, then return false
		if (PyErr_Occurred() != NULL) {
			return false;
		}
	}
	return true;
}


void print_1d_np_arr(PyArrayObject* np_arr) {
	if (PyArray_NDIM(np_arr) != 1 || !PyArray_ISBEHAVED(np_arr) || PyArray_TYPE(np_arr) != NPY_DOUBLE)
		return;
	
	printf("[   ");
	
	npy_intp size = PyArray_DIMS(np_arr)[0];
	double* data = PyArray_DATA(np_arr);
	
	npy_intp index = 0;
	for (; index < size; index++) {
		if (index > 0 && (index % 6 == 0))
			printf("\n    ");
		printf("%12.3e", data[index]);
	}
	
	if (index % 6 == 0)
		printf("   ]\n");
	else
		printf("%*c   ]\n", (int) ( 12 * (6 - (index % 6))), ' ');
}




#endif
