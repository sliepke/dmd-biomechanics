// python
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// numpy
#include <numpy/arrayobject.h>

// c library
#include <stdlib.h> // for malloc

// user defined imports
#include "sim_c_helper/Sim_Entry_Object.c"
#include "sim_c_helper/Sim_Entry_Type.c"





// Module def
static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "sim_c_util",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
};

// Module init
PyMODINIT_FUNC
PyInit_sim_c_util(void)
{
	// create module
    PyObject *m = PyModule_Create(&module_def);
	if (m == NULL)
		return NULL;
	
	// initialize numpy API
    import_array();
	
	// add Sim_Entry class to module
    if (PyType_Ready(&Sim_Entry_Type) < 0) {
        Py_DECREF(m);
        return NULL;
	}
    if (PyModule_AddObjectRef(m, "Sim_Entry", (PyObject *) &Sim_Entry_Type) < 0) {
        Py_DECREF(m);
        return NULL;

    }
	
	// return
    return m;
}
