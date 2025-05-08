// // //
//
// Defines the Sim_Entry type object (Sim_Entry_Type), and methods of the
// Sim_Entry class.
//
// // //

#ifndef SIM_ENTRY_TYPE
#define SIM_ENTRY_TYPE

#include <Python.h>
#include <numpy/arrayobject.h>

// user defined imports
#include "Sim_Entry_Object.c"
#include "math_helper.c"
#include "numpy_helper.c"

static void print_arr(char* name, double* arr, size_t length);




// -- Sim_Entry methods -- //




static char *new_kwlist[] = {
	"num_point_masses",
	"masses",
	"g",
	"num_segments",
	"segments",
	"lengths",
	"A_segment",
	"B_segment",
	"num_joints",
	"joints",
	"joint_params",
	"b3",
	"flexion_c1s",
	"extension_c1s",
	"A_normal",
	"B_normal",
	"mu_s",
	"mu_k",
	"zero_velocity_threshold",
	"start_position",
	"rk4_timestep",
	"v",
	"thetas",
	"thetadots",
	NULL
};


PyObject* Sim_Entry_New(PyTypeObject* type, PyObject* args, PyObject* kwds) {
	// -- allocate self, set self's native + numpy array pointers to NULL in case __new__ fails -- //
	
	
	Sim_Entry_Object* self = (Sim_Entry_Object*) (type->tp_alloc(type, 0));
	if (self == NULL) {
		return NULL;
	}
	
	Parameters* params = &self->parameters;
	
	// if __new__ fails, it will DECREF(self) which calls the deallocation function
	// the deallocation function free all native arrays
	// (except ones that are NULL, or wrapped by a numpy array)
	params->masses = NULL;
	params->segments = NULL;
	params->joints = NULL;
	self->k1dp = NULL;
	
	
	// -- read arguments (not fun code) -- //
	
	
	PyArrayObject* masses;
	PyArrayObject* segments;
	PyArrayObject* lengths;
	PyArrayObject* joints;
	PyArrayObject* joint_params;
	int args_status = PyArg_ParseTupleAndKeywords(
		args, kwds, "iO!diO!O!ddiO!O!dO!O!dddddO!dO!O!O!", new_kwlist,
		// (expected type if object argument, argument destination)
		&params->num_point_masses,
		&PyArray_Type, &masses, 
		&params->g,
		&params->num_segments,
		&PyArray_Type, &segments,
		&PyArray_Type, &lengths,
		&params->A_segment,
		&params->B_segment,
		&params->num_joints,
		&PyArray_Type, &joints,
		&PyArray_Type, &joint_params,
		&params->b3,
		&PyArray_Type, &params->c1fs_np,
		&PyArray_Type, &params->c1es_np,
		&params->A_normal,
		&params->B_normal,
		&params->mu_s,
		&params->mu_k,
		&params->zero_velocity_threshold,
		&PyArray_Type, &self->p_np,
		&self->rk4_timestep,
		&PyArray_Type, &self->v_np,
		&PyArray_Type, &self->thetas_np,
		&PyArray_Type, &self->thetadots_np
	);
	// obtain a reference to all non-null numpy arrays just copied into self.
	// do this BEFORE potentially calling decref(self), because decref(self) will
	// xdecref all of them (and we don't want to decref an array we don't own
	// a reference to)
	Py_XINCREF(params->c1fs_np);
	Py_XINCREF(params->c1es_np);
	Py_XINCREF(self->p_np);
	Py_XINCREF(self->v_np);
	Py_XINCREF(self->thetas_np);
	Py_XINCREF(self->thetadots_np);
	// if arg parsing failed, deallocate self and return null
	if (!args_status) {
		Py_DECREF(self);
		return NULL;
	}
	
	
	// -- verify that numpy array args are correct type and shape -- //
	
	
	if (
		   !validate_np_arg("masses", masses, NPY_DOUBLE, 1, params->num_point_masses, 0)
		|| !validate_np_arg("segments", segments, NPY_INT, 2, params->num_segments, 2)
		|| !validate_np_arg("lengths", lengths, NPY_DOUBLE, 1, params->num_segments, 0)
		|| !validate_np_arg("joints", joints, NPY_INT, 2, params->num_joints, 3)
		|| !validate_np_arg("joint_params", joint_params, NPY_DOUBLE, 2, params->num_joints, 16)
		|| !validate_np_arg("flexion_c1s", params->c1fs_np, NPY_DOUBLE, 1, params->num_joints, 0)
		|| !validate_np_arg("extension_c1s", params->c1es_np, NPY_DOUBLE, 1, params->num_joints, 0)
		|| !validate_np_arg("start_position", self->p_np, NPY_DOUBLE, 1, 2 * params->num_point_masses, 0)
		|| !validate_np_arg("v", self->v_np, NPY_DOUBLE, 1, 2 * params->num_point_masses, 0)
		|| !validate_np_arg("thetas", self->thetas_np, NPY_DOUBLE, 1, params->num_joints, 0)
		|| !validate_np_arg("thetadots", self->thetadots_np, NPY_DOUBLE, 1, params->num_joints, 0)
	) {
		// (the call to validate_np_arg that failed will set the exception + message)
		Py_DECREF(self);
		return NULL;
	}
	
	
	// -- store pointers to the native arrays wrapped by the Sim_Entry's numpy arrays -- //
	
	
	// first, make sure they are contiguous and aligned
	if (
		   !validate_np_arg_behaved("flexion_c1s", params->c1fs_np)
		|| !validate_np_arg_behaved("extension_c1s", params->c1es_np)
		|| !validate_np_arg_behaved("start_position", self->p_np)
		|| !validate_np_arg_behaved("v", self->v_np)
		|| !validate_np_arg_behaved("thetas", self->thetas_np)
		|| !validate_np_arg_behaved("thetadots", self->thetadots_np)
	) {
		// (the call to validate_np_arg_behaved that failed will set the exception + message)
		Py_DECREF(self);
		return NULL;
	}
	// now, store pointers to their data
	params->c1fs = PyArray_DATA(params->c1fs_np);
	params->c1es = PyArray_DATA(params->c1es_np);
	self->p = PyArray_DATA(self->p_np);
	self->v = PyArray_DATA(self->v_np);
	self->thetas = PyArray_DATA(self->thetas_np);
	self->thetadots = PyArray_DATA(self->thetadots_np);
	
	
	// -- allocate the Sim_Entry's native arrays (other than the ones wrapped by a numpy array) -- //
	
	
	/* in the future, might point all arrays to one contiguous block
	   of memory, which may improve performance because of caching   */
	
	// size of the self->p array, which holds the 2d position of each point mass
	size_t p_size = 2 * params->num_point_masses * sizeof(double);
	// number of p-sized buffers being used in rk4
	size_t num_ps = 9;
	
	if (
		// allocate arrays storing masses, segment parameters, and joint parameters
		   !( params->masses = (double*) malloc(sizeof(double) * params->num_point_masses) )
		|| !( params->segments = (Segment*) malloc(sizeof(Segment) * params->num_segments) )
		|| !( params->joints = (Joint*) malloc(sizeof(Joint) * params->num_joints)         )
		// allocate the block containing the rk4 buffers (which are not initialized in __new__)
		|| !( self->k1dp = (double*) malloc(num_ps * p_size)                               )
	) {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate memory\n");
		Py_DECREF(self);
		return NULL;
	}
	
	// point each of the 9 rk4 buffers to the block just allocated (at self->k1dp)
	self->k1dv = self->k1dp + (2 * params->num_point_masses);
	self->k2dp = self->k1dp + 2 * (2 * params->num_point_masses);
	self->k2dv = self->k1dp + 3 * (2 * params->num_point_masses);
	self->k3dp = self->k1dp + 4 * (2 * params->num_point_masses);
	self->k3dv = self->k1dp + 5 * (2 * params->num_point_masses);
	self->k4dp = self->k1dp + 6 * (2 * params->num_point_masses);
	self->k4dv = self->k1dp + 7 * (2 * params->num_point_masses);
	self->p_temp = self->k1dp + 8 * (2 * params->num_point_masses);
	
	
	// -- copy from numpy array arguments into Sim_Entry's native arrays (the best code ever written :D) -- //
	
	
	// signature of copy_from_np:
	// int copy_from_np(PyArrayObject* np_arr, int ndim, int i1, int type, void* dst_base, size_t dst_step, size_t num_steps)
	
	if (
		//// Masses   ////
		   !copy_from_np(masses,       1, 0,  NPY_DOUBLE, params->masses,                sizeof(double),  params->num_point_masses)
		//// Segments ////
		// a, b
		|| !copy_from_np(segments,     2, 0,  NPY_INT,    &params->segments[0].a,        sizeof(Segment), params->num_segments)
		|| !copy_from_np(segments,     2, 1,  NPY_INT,    &params->segments[0].b,        sizeof(Segment), params->num_segments)
		// length
		|| !copy_from_np(lengths,      1, 0,  NPY_DOUBLE, &params->segments[0].length,   sizeof(Segment), params->num_segments)
		//// Joints   ////
		// a, b, c
		|| !copy_from_np(joints,       2, 0,  NPY_INT,    &params->joints[0].a,          sizeof(Joint),   params->num_joints)
		|| !copy_from_np(joints,       2, 1,  NPY_INT,    &params->joints[0].b,          sizeof(Joint),   params->num_joints)
		|| !copy_from_np(joints,       2, 2,  NPY_INT,    &params->joints[0].c,          sizeof(Joint),   params->num_joints)
		// offset, transition
		|| !copy_from_np(joint_params, 2, 0,  NPY_DOUBLE, &params->joints[0].offset,     sizeof(Joint),   params->num_joints)
		|| !copy_from_np(joint_params, 2, 1,  NPY_DOUBLE, &params->joints[0].transition, sizeof(Joint),   params->num_joints)
		// B1, k1, B2, k2
		|| !copy_from_np(joint_params, 2, 2,  NPY_DOUBLE, &params->joints[0].b1,         sizeof(Joint),   params->num_joints)
		|| !copy_from_np(joint_params, 2, 3,  NPY_DOUBLE, &params->joints[0].k1,         sizeof(Joint),   params->num_joints)
		|| !copy_from_np(joint_params, 2, 4,  NPY_DOUBLE, &params->joints[0].b2,         sizeof(Joint),   params->num_joints)
		|| !copy_from_np(joint_params, 2, 5,  NPY_DOUBLE, &params->joints[0].k2,         sizeof(Joint),   params->num_joints)
		// C2F, C3F, C4F, C5F, C6F
		|| !copy_from_np(joint_params, 2, 6,  NPY_DOUBLE, &params->joints[0].c2f,        sizeof(Joint),   params->num_joints)
		|| !copy_from_np(joint_params, 2, 7,  NPY_DOUBLE, &params->joints[0].c3f,        sizeof(Joint),   params->num_joints)
		|| !copy_from_np(joint_params, 2, 8,  NPY_DOUBLE, &params->joints[0].c4f,        sizeof(Joint),   params->num_joints)
		|| !copy_from_np(joint_params, 2, 9,  NPY_DOUBLE, &params->joints[0].c5f,        sizeof(Joint),   params->num_joints)
		|| !copy_from_np(joint_params, 2, 10, NPY_DOUBLE, &params->joints[0].c6f,        sizeof(Joint),   params->num_joints)
		// C2E, C3E, C4E, C5E, C6E
		|| !copy_from_np(joint_params, 2, 11, NPY_DOUBLE, &params->joints[0].c2e,        sizeof(Joint),   params->num_joints)
		|| !copy_from_np(joint_params, 2, 12, NPY_DOUBLE, &params->joints[0].c3e,        sizeof(Joint),   params->num_joints)
		|| !copy_from_np(joint_params, 2, 13, NPY_DOUBLE, &params->joints[0].c4e,        sizeof(Joint),   params->num_joints)
		|| !copy_from_np(joint_params, 2, 14, NPY_DOUBLE, &params->joints[0].c5e,        sizeof(Joint),   params->num_joints)
		|| !copy_from_np(joint_params, 2, 15, NPY_DOUBLE, &params->joints[0].c6e,        sizeof(Joint),   params->num_joints)
	) {
		// (if it failed, then the exception + message will already be set)
		Py_DECREF(self);
		return NULL;
	}
	
	
	// -- set starting joint angles and angular velocities (angular velocities should be 0) -- //
	
	
	set_thetas_and_thetadots_easy(self);
	
	
	// -- if debugging: print parameters of Sim_Entry, call f() with all debug flags set (with activations of 0) -- //
	
	
	/*
	printf("\n== Parameters ==\n");
	print_parameters(self);
	
	double* activations = (double*) malloc( params->num_joints * sizeof(double) );
	for (int i = 0; i < params->num_joints; i++) {
		activations[i] = 0;
	}
	f(params, self->p, self->v, self->k1dp, self->k1dv, activations, ~0);
	*/
	
	
	return (PyObject*) self;
}


static void Sim_Entry_Dealloc(Sim_Entry_Object* self) {
	Parameters* params = &self->parameters;
	
	//// free native arrays allocated with malloc()
	// masses
	if (params->masses != NULL)
		free(params->masses);
	// segments
	if (params->segments != NULL)
		free(params->segments);
	// joints
	if (params->joints != NULL)
		free(params->joints);
	// rk4 buffer
	if (self->k1dp != NULL)
		free(self->k1dp);
	
	//// release references to numpy arrays passed into constructor
	Py_XDECREF(params->c1fs_np);
	Py_XDECREF(params->c1es_np);
	Py_XDECREF(self->p_np);
	Py_XDECREF(self->v_np);
	Py_XDECREF(self->thetas_np);
	Py_XDECREF(self->thetadots_np);
	
	Py_TYPE(self)->tp_free((PyObject*) self);
}


static void print_arr(char* name, double* arr, size_t length) {
	printf("%s:", name);
	for (size_t i = 0; i < length; i++) {
		if (i % 4 == 0)
			printf("\n");
		printf(" %- .8e", arr[i]);
	}
	printf("\n");
}


static PyObject* tick(Sim_Entry_Object* self, PyObject* args) {
	// validate arguments (activations + debug)
	PyArrayObject* activations_np;
	int debug;
	if (!PyArg_ParseTuple(args, "O!p", &PyArray_Type, &activations_np, &debug))
		return NULL;
	if (!PyArray_ISBEHAVED(activations_np)) {
        PyErr_SetString(PyExc_ValueError, "activations array is not contiguous or aligned\n");
		return NULL;
	}
	if (PyArray_TYPE(activations_np) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "activations array must have type double\n");
		return NULL;
	}
	if (
		   !(PyArray_NDIM(activations_np) == 1)
		|| !(PyArray_DIMS(activations_np)[0] == self->parameters.num_joints)
	) {
        PyErr_SetString(PyExc_ValueError, "activations array is not the correct shape\n");
		return NULL;
	}
	
	// get pointer to data of activations array
	double *activations = PyArray_DATA(activations_np);
	
	// do rk4 (not fun code)
	
	int dbg = debug ? ~0 : 0;
	Parameters* params = &self->parameters;
	double h = self->rk4_timestep;
	size_t length = 2 * params->num_point_masses;
	
	// k1 = f(y)
	f(params, self->p, self->v, self->k1dp, self->k1dv, activations, dbg);
	
	// <p_temp, k2dp> = y + h k1 / 2
	linear_comb2(self->p_temp, self->p, self->k1dp, h / 2, length);
	linear_comb2(self->k2dp,   self->v, self->k1dv, h / 2, length);
	// k2 = f(y + h k1 / 2)
	f(params, self->p_temp, self->k2dp, self->k2dp, self->k2dv, activations, 0);
	
	// <p_temp, k3dp> = y + h k2 / 2
	linear_comb2(self->p_temp, self->p, self->k2dp, h / 2, length);
	linear_comb2(self->k3dp,   self->v, self->k2dv, h / 2, length);
	// k3 = f(y + h k2 / 2)
	f(params, self->p_temp, self->k3dp, self->k3dp, self->k3dv, activations, 0);
	
	// <p_temp, k4dp> = y + h k3
	linear_comb2(self->p_temp, self->p, self->k3dp, h, length);
	linear_comb2(self->k4dp,   self->v, self->k3dv, h, length);
	// k4 = f(y + h k3)
	f(params, self->p_temp, self->k4dp, self->k4dp, self->k4dv, activations, 0);
	
	// <p, v> += h k1 / 6 + h k2 / 3 + h k3 / 3 + h k4 / 6
	linear_comb4(self->p, self->k1dp, h / 6, self->k2dp, h / 3, self->k3dp, h / 3, self->k4dp, h / 6, length);
	linear_comb4(self->v, self->k1dv, h / 6, self->k2dv, h / 3, self->k3dv, h / 3, self->k4dv, h / 6, length);
	
	// now that p and v are updated, compute thetas + thetadots
	//     this will be done again at next tick(), but we want them now. computing
	//     them only once would indeed be slightly faster, but we'd have to change f()
	set_thetas_and_thetadots_easy(self);
	
	Py_RETURN_NONE;
}




// -- Sim_Entry_Type -- //




static PyMethodDef Sim_Entry_Methods[] = {
	{"tick", (PyCFunction) tick, METH_VARARGS,
	"Perform one tick of rk4, which updates the position, velocity, thetas, "
	"and thetadots arrays (by reference) that were passed to this Sim_Entry's "
	"constructor."
	},
	{NULL}
};


static PyTypeObject Sim_Entry_Type = {
	.ob_base = PyVarObject_HEAD_INIT(NULL, 0)
	.tp_name = "sim_c_util.Sim_Entry",
    .tp_doc = PyDoc_STR(
		"An instance of this class is held by a Simulation object. The main"
		" feature of this class is the tick() method, which is written in"
		" C and so runs faster than a tick() method written in Python."
	),
	.tp_basicsize = sizeof(Sim_Entry_Object),
    .tp_new = Sim_Entry_New,
	.tp_dealloc = (destructor) Sim_Entry_Dealloc,
	.tp_methods = Sim_Entry_Methods,
};




#endif
