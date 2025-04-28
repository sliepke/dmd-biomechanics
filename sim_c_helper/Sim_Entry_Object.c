// // //
//
// Defines
//     - The Sim_Entry_Object struct and its substructs (Segment, Joint, Parameters)
//     - Helper functions for printing the parameters and state of a Sim_Entry_Object
//
// // //

#ifndef SIM_ENTRY_OBJECT
#define SIM_ENTRY_OBJECT

#include "numpy_helper.c"




// -- Sim_Entry_Object struct and substructs -- //




// Segment struct
typedef struct {
	int a;
	int b;
	double length;
} Segment;


// Joint struct
typedef struct {
	int a;
	int b;
	int c;
	double offset;
	double transition;
	double b1, k1, b2, k2;
	/* note that each joint's c1f and c1e are stored in a separate numpy array
	   so that they can be given by reference to python code */
	double c2f, c3f, c4f, c5f, c6f;
	double c2e, c3e, c4e, c5e, c6e;
} Joint;


// Parameters struct (holds parameters of a Sim_Entry)
typedef struct {
	int num_point_masses;
	/* all dynamically allocated data not directly in the field of a Sim_Entry_Object
	   struct, or inside a field of the its parameters field, is in one contiguous
	   block. any pointers in the Sim_Entry_Object struct point to this block, with
	   Sim_Entry_Object.parameters.masses pointing to the start of the block */
	double* masses;
	
	// acceleration of gravity
	double g;
	
	// information about each segment (a, b) in S
	int num_segments;
	Segment* segments;
	// segment parameters
	double A_segment;
	double B_segment;
	
	// information about each joint (a, b, c) in J EXCEPT C1's
	int num_joints;
	Joint* joints;
	// numpy arrays holding C1's (so they can be given by reference to python code)
	PyArrayObject* c1fs_np;
	PyArrayObject* c1es_np;
	// native arrays wrapped by the numpy arrays
	double* c1fs;
	double* c1es;
	// b3
	double b3;
	
	// normal force parameters
	double A_normal;
	double B_normal;
	
	// friction parameters
	double mu_s;
	double mu_k;
	double zero_velocity_threshold;
} Parameters;


// Sim_Entry struct (which extends the PyObject struct)
typedef struct {
	PyObject_HEAD
	
	// -- parameters -- //
	
	Parameters parameters;
	
	// -- state -- //
	
	// positions and velocities of each point mass
	// (in numpy arrays so they can be given by reference to python code)
	PyArrayObject* p_np;
	PyArrayObject* v_np;
	// native arrays wrapped by the numpy arrays
	double* p;
	double* v;
	
	// each joint angle and angular velocity
	// (in numpy arrays so they can be given by reference to python code)
	PyArrayObject* thetas_np;
	PyArrayObject* thetadots_np;
	// native arrays wrapped by the numpy arrays
	double* thetas;
	double* thetadots;
	
	// used for rk4
	double rk4_timestep;
	double* k1dp, *k1dv;
	double* k2dp, *k2dv;
	double* k3dp, *k3dv;
	double* k4dp, *k4dv;
	double* p_temp;
} Sim_Entry_Object;




// -- Print helper functions (just for debugging) -- //




void print_parameters(Sim_Entry_Object* self) {
	Parameters* params = &self->parameters;
	printf("num point masses: %d\n", params->num_point_masses);
	printf("num segments: %d\n", params->num_segments);
	printf("num joints: %d\n", params->num_joints);
	printf("g: %f\n", params->g);
	printf("A_segment: %f\n", params->A_segment);
	printf("B_segment: %f\n", params->B_segment);
	printf("b3: %f\n", params->b3);
	printf("A_normal: %f\n", params->A_normal);
	printf("B_normal: %f\n", params->B_normal);
	printf("mu_s: %f\n", params->mu_s);
	printf("mu_k: %f\n", params->mu_k);
	printf("zero_velocity_threshold: %f\n", params->zero_velocity_threshold);
	printf("rk4_timestep: %f\n", self->rk4_timestep);
	
	printf("masses: ");
	for (int i = 0; i < params->num_point_masses; i++) {
		printf("%f, ", params->masses[i]);
	}
	printf("\n");
	
	printf("segments:\n");
	for (int i = 0; i < params->num_segments; i++) {
		printf("\t");
		printf("a: %d, ", params->segments[i].a);
		printf("b: %d, ", params->segments[i].b);
		printf("length: %f, ", params->segments[i].length);
		printf("\n");
	}
	
	printf("joints:\n");
	for (int i = 0; i < params->num_joints; i++) {
		printf("\t");
		printf("a: %d, ", params->joints[i].a);
		printf("b: %d, ", params->joints[i].b);
		printf("c: %d, ", params->joints[i].c);
		
		printf("offset: %f, ", params->joints[i].offset);
		printf("transition: %f, ", params->joints[i].transition);
		
		printf("B1: %f, ", params->joints[i].b1);
		printf("k1: %f, ", params->joints[i].k1);
		printf("B2: %f, ", params->joints[i].b2);
		printf("k2: %f, ", params->joints[i].k2);
		
		printf("c1f: %f, ", params->c1fs[i]);
		printf("c2f: %f, ", params->joints[i].c2f);
		printf("c3f: %f, ", params->joints[i].c3f);
		printf("c4f: %f, ", params->joints[i].c4f);
		printf("c5f: %f, ", params->joints[i].c5f);
		printf("c6f: %f, ", params->joints[i].c6f);
		
		printf("c1e: %f, ", params->c1es[i]);
		printf("c2e: %f, ", params->joints[i].c2e);
		printf("c3e: %f, ", params->joints[i].c3e);
		printf("c4e: %f, ", params->joints[i].c4e);
		printf("c5e: %f, ", params->joints[i].c5e);
		printf("c6e: %f, ", params->joints[i].c6e);
		printf("\n");
	}
}


/* prints
       - Position and velocity of every point mass
       - Angle and angular velocity of every joint
*/
void print_state(Sim_Entry_Object* self) {
	printf("positions\n");
	print_1d_np_arr(self->p_np);
	printf("velocities\n");
	print_1d_np_arr(self->v_np);
	printf("joint angles\n");
	print_1d_np_arr(self->thetas_np);
	printf("joint angular velocities\n");
	print_1d_np_arr(self->thetadots_np);
}




#endif
