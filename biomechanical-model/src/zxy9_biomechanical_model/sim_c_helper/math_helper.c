// // //
//
// Defines:
//     - set_thetas_easy, a function that initializes the values of joint angles
//     - f, a function that computes all forces exerted on point masses at a given time
//     - linearcomb2 and linearcomb4, which set an array to a linear combination of
//       other arrays, which helps with rk4
//
// // //

#ifndef MATH_HELPER
#define MATH_HELPER

#include <math.h>
#include "Sim_Entry_Object.c"

// doubles on this system might have a really high precision okay!! trust me bro!!
#define PI 3.14159265358979323846264338327950288419716939937510

#define DBG_POS            1
#define DBG_VEL            (1 << 1)
#define DBG_NETFORCE       (1 << 2)
#define DBG_GRAVITY        (1 << 3)
#define DBG_SEGMENT        (1 << 4)
#define DBG_JOINT          (1 << 5)
#define DBG_GROUND         (1 << 6)




// declare helper functions for calculating joint forces
static double get_theta(double *p, int a, int b, int c, double offset, double transition);
static double get_angular_vel(
	double ax, double ay,
	double avx, double avy,
	double bx, double by,
	double bvx, double bvy);
static double get_thetadot(double *p, double *v, int a, int b, int c);
static double get_t_max_active_flexion(Joint* j, double theta, double thetadot, double c1f);
static double get_t_max_active_extension(Joint* j, double theta, double thetadot, double c1e);
static double get_t_passive(Joint* j, double theta, double thetadot, double damping);

// declare functions for computing each type of force (gravity, segments, joints, ground)
static void f_gravity(Parameters* params, double* f_out, int debug);
static void f_segment(Parameters* params, double* p_in, double* v_in, double* f_out, int debug);
static void f_joint(Parameters* params, double* p_in, double* v_in, double* f_out, double* activations, int debug);
static void f_ground(Parameters* params, double* p_in, double* v_in, double* f_out, int debug);





/*
determine the derivative of the positions and velocities of all point masses
with respect to time, at a moment in time.

parameters
	params
		address of the Parameters struct giving masses, segments, segment lengths, and so on
	p_in
		array of the current positions of all point masses, in the order: point 1 x, point 1 y, point 2 x, etc
	v_in
		array of the current velocities of all point masses, in the same order as p_in
	dp_out
		array to write the position derivatives into. this is the same as copying v_in into dp_out.
		however, dp_out may be equal to v_in - in that case, f will not write to dp_out
	dv_out
		array to write the velocity derivatives into
	debug
		determines what intermediate results to print. to set this, 'or' together zero or more
		of the following flags (or just set it to ~0 to print everything):
			DBG_POS             DBG_VEL             DBG_NETFORCE
			DBG_GRAVITY         DBG_SEGMENT         DBG_JOINT
			DBG_GROUND
*/
static void f(Parameters* params, double* p_in, double* v_in, double* dp_out, double* dv_out, double* activations, int debug) {
	// debug positions
	if (debug & DBG_POS) {
		printf("\n\n%12sPOSITIONS\n", "");
		printf("%-12s%12s%12s\n", "point", "x position", "y position");
		for (int i = 0; i < params->num_point_masses; i++) {
			printf("%- 12d%12.4f%12.4f\n", i, p_in[2 * i], p_in[2 * i + 1]);
		}
	}
	// debug velocities
	if (debug & DBG_VEL) {
		printf("\n\n%12sVELOCITIES", "");
		printf("\n%-12s%12s%12s\n", "point", "x velocity", "y velocity");
		for (int i = 0; i < params->num_point_masses; i++) {
			printf("%- 12d%12.4f%12.4f\n", i, v_in[2 * i], v_in[2 * i + 1]);
		}
	}
	
	// determine net force on each point, written to the dv_out buffer
	f_gravity(params, dv_out, debug);
	f_segment(params, p_in, v_in, dv_out, debug);
	f_joint(params, p_in, v_in, dv_out, activations, debug);
	f_ground(params, p_in, v_in, dv_out, debug);
	
	// debug net forces
	if (debug & DBG_NETFORCE) {
		printf("\n\n%12sNET FORCE\n", "");
		printf("%-12s%12s%12s\n", "point", "x force", "y force");
		for (int i = 0; i < params->num_point_masses; i++) {
			printf("%- 12d%12.4f%12.4f\n", i, dv_out[2 * i], dv_out[2 * i + 1]);
		}
	}
	
	// determine acceleration of each point, by dividing its net force by its mass
	for (int i = 0; i < params->num_point_masses; i++) {
		dv_out[2 * i]     /= params->masses[i];
		dv_out[2 * i + 1] /= params->masses[i];
	}
	
	// if v_in != dp_out, then copy v_in to dp_out
	if (v_in != dp_out) {
		for (int i = 0; i < 2 * params->num_point_masses; i++) {
			dp_out[i] = v_in[i];
		}
	}
}




// -- define functions for computing each type of force (gravity, segments, joints, ground) -- //




// must be called first, since it sets values of f_out (instead of adding to them)
static void f_gravity(Parameters* params, double* f_out, int debug) {
	if (debug & DBG_GRAVITY) {
		printf("\n\n%12sGRAVITY\n", "");
		printf("%-12s%12s\n", "point", "+y force");
	}
	
	double g = params->g;
	for (int i = 0; i < params->num_point_masses; i++) {
		double y_fc = - g * params->masses[i];
		f_out[2 * i + 1] = y_fc;
		f_out[2 * i] = 0;
		
		// debug
		if (debug & DBG_GRAVITY)
			printf("%- 12d%12.4f\n", i, y_fc);
	}
}


static void f_segment(Parameters* params, double* p_in, double* v_in, double* f_out, int debug) {
	if (debug & DBG_SEGMENT) {
		printf("\n\n%12sSEGMENTS\n", "");
		printf("%-12s%12s%12s%12s%12s%12s%12s\n", "( a, b)", "r", "r'",
			"x fc on a", "y fc on a", "x fc on b", "y fc on b");
	}
	
	for (int i = 0; i < params->num_segments; i++) {
		//// find a's position and velocity
		int a = params->segments[i].a;
		double ax = p_in[2 * a];
		double ay = p_in[2 * a + 1];
		double avx = v_in[2 * a];
		double avy = v_in[2 * a + 1];
		
		//// find b's position and velocity
		int b = params->segments[i].b;
		double bx = p_in[2 * b];
		double by = p_in[2 * b + 1];
		double bvx = v_in[2 * b];
		double bvy = v_in[2 * b + 1];
		
		//// find unit vector pointing from a to b
		// displacement ab
		double abx = bx - ax;
		double aby = by - ay;
		// norm of displacement ab
		double ab_norm = sqrt(abx * abx + aby * aby);
		// ab / norm(ab)
		double ab_unitx = abx / ab_norm;
		double ab_unity = aby / ab_norm;
		
		//// find r, the displacement of the spring
		double r = ab_norm - params->segments[i].length;
		
		//// find r'
		// velocity of b in a's inertial frame
		double bvelx = bvx - avx;
		double bvely = bvy - avy;
		// component of this velocity along the displacement ab (which is r')
		// = dot(bvel, ab) / norm(ab)
		double r_prime = (bvelx * abx + bvely * aby) / ab_norm;
		
		//// find magnitude of segment force
		double f_mag = (params->A_segment * r + params->B_segment * r_prime);
		
		//// apply forces to points
		double fc_on_ax  =   ab_unitx * f_mag;
		double fc_on_ay  =   ab_unity * f_mag;
		double fc_on_bx  = - ab_unitx * f_mag;
		double fc_on_by  = - ab_unity * f_mag;
		f_out[2 * a]     +=  fc_on_ax;
		f_out[2 * a + 1] +=  fc_on_ay;
		f_out[2 * b]     +=  fc_on_bx;
		f_out[2 * b + 1] +=  fc_on_by;
		
		if (debug & DBG_SEGMENT) {
			printf("(%2d,%2d)     %12.4f%12.4f%12.4f%12.4f%12.4f%12.4f\n",
				a, b, r, r_prime, fc_on_ax, fc_on_ay, fc_on_bx, fc_on_by);
		}
	}
}


static void f_joint(Parameters* params, double* p_in, double* v_in, double* f_out, double* activations, int debug) {
	if (debug & DBG_JOINT) {
		printf("\n\n%12sJOINTS\n", "");
		printf("( a, b, c)         theta    thetadot   T passive    T active  activation\n");
		printf("               x fc on a   y fc on a   x fc on b   y fc on b   x fc on c   y fc on c\n");
	}
	for (int i = 0; i < params->num_joints; i++) {
		Joint* joint = &params->joints[i];
		int a = joint->a;
		int b = joint->b;
		int c = joint->c;
		double theta = get_theta(p_in, a, b, c, joint->offset, joint->transition);
		double thetadot = get_thetadot(p_in, v_in, a, b, c);
		
		//// determine joint torque to apply
		// passive torque
		double t_passive = get_t_passive(joint, theta, thetadot, params->b3);
		// active (flexion or extension) torque
		double t_active = 0;
		double act = activations[i];
		if (act > 0) {
			t_active = act * get_t_max_active_flexion(joint, theta, thetadot, params->c1fs[i]);
		} else if (act < 0) {
			t_active = act * get_t_max_active_extension(joint, theta, thetadot, params->c1es[i]);
		}
		
		// total torque
		double t = t_passive + t_active;
		
		//// determine force to apply on a
		// displacement ba
		double bax = p_in[2 * a] - p_in[2 * b];
		double bay = p_in[2 * a + 1] - p_in[2 * b + 1];
		// norm(ba) ** 2
		double ba_norm_sq = bax * bax + bay * bay;
		// force on a
		double fc_on_ax = - bay * t / ba_norm_sq;
		double fc_on_ay =   bax * t / ba_norm_sq;
		// apply force on a
		f_out[2 * a]     += fc_on_ax;
		f_out[2 * a + 1] += fc_on_ay;
		
		//// determine force to apply on c
		// displacement bc
		double bcx = p_in[2 * c] - p_in[2 * b];
		double bcy = p_in[2 * c + 1] - p_in[2 * b + 1];
		// norm(bc) ** 2
		double bc_norm_sq = bcx * bcx + bcy * bcy;
		// force on c
		double fc_on_cx =   bcy * t / bc_norm_sq;
		double fc_on_cy = - bcx * t / bc_norm_sq;
		// apply force on c
		f_out[2 * c]     += fc_on_cx;
		f_out[2 * c + 1] += fc_on_cy;
		
		/// determine force to apply on b
		double fc_on_bx = - fc_on_ax - fc_on_cx;
		double fc_on_by = - fc_on_ay - fc_on_cy;
		// apply force on b
		f_out[2 * b]     += fc_on_bx;
		f_out[2 * b + 1] += fc_on_by;
		
		if (debug & DBG_JOINT) {
			printf("\n(%2d,%2d,%2d)  %12.4f%12.4f%12.4f%12.4f%12.4f\n",
				a, b, c, theta, thetadot, t_passive, t_active, act);
			printf("            %12.4f%12.4f%12.4f%12.4f%12.4f%12.4f\n",
				fc_on_ax, fc_on_ay, fc_on_bx, fc_on_by, fc_on_cx, fc_on_cy);
		}
	}
}


// must be called last since it depends on the values of other forces
// does normal and friction at the same time, since friction depends on normal force
static void f_ground(Parameters* params, double* p_in, double* v_in, double* f_out, int debug) {
	if (debug & DBG_GROUND) {
		printf("\n\n%12sGROUND\n", "");
		printf("%-12s%24s%24s\n", "point", "+x friction force", "+y normal force");
	}
	
	// ground force parameters
	double a_normal = params->A_normal;
	double b_normal = params->B_normal;
	double mu_s = params->mu_s;
	double mu_k = params->mu_k;
	for (int i = 0; i < params->num_point_masses; i++) {
		// only apply ground forces if point is in the ground
		double y_pos = p_in[2 * i + 1];
		if (y_pos > 0)
			continue;
		
		// determine normal force towards +y to exert on point i
		double y_fc = 0;
		y_fc = - a_normal * y_pos - b_normal * v_in[2 * i + 1];
		if (y_fc < 0)
			y_fc = 0;
		f_out[2 * i + 1] += y_fc;
		
		// determine friction force towards +x to exert on point i
		double x_fc = 0;
		double x_vel = v_in[2 * i];
		// static friction
		if (x_vel == 0) {
			double h = f_out[2 * i];
			if (h != 0) {
				double normal_norm = y_fc;
				if (normal_norm < 0)
					normal_norm *= -1;
				
				double h_abs  = (h > 0) ? h : -h;
				x_fc = (mu_s * normal_norm < h_abs) ? (mu_s * normal_norm) : h_abs;
				
				double h_sign = (h > 0) ? 1 : -1;
				x_fc *= - h_sign;
			}
		}
		// kinetic friction
		else {
			double normal_norm = y_fc;
			if (normal_norm < 0)
				normal_norm *= -1;
			
			double x_vel_sign = (x_vel > 0) ? 1 : -1;
			x_fc = - x_vel_sign * mu_k * normal_norm;
		}
		f_out[2 * i] += x_fc;
		
		// debug
		if (debug & DBG_GROUND)
			printf("%-12d%24.4f%24.4f\n", i, x_fc, y_fc);
	}
}


static void set_thetas_and_thetadots_easy(Sim_Entry_Object *self) {
	Parameters* params = &self->parameters;
	for (int i = 0; i < params->num_joints; i++) {
		Joint* joint = &params->joints[i];
		double theta = get_theta(
			self->p,  joint->a,      joint->b,
			joint->c, joint->offset, joint->transition
		);
		self->thetas[i] = theta;
		double thetadot = get_thetadot(
			self->p,  self->v,
			joint->a, joint->b, joint->c
		);
		self->thetadots[i] = thetadot;
	}
}




// -- define helper functions for calculating joint forces -- //




// returns the angle of given joint (assumes a, b, c are valid point mass indices)
static double get_theta(double *p, int a, int b, int c, double offset, double transition) {
	// positions of point masses (a, b, c)
	double ax = p[2 * a];
	double ay = p[2 * a + 1];
	
	double bx = p[2 * b];
	double by = p[2 * b + 1];
	
	double cx = p[2 * c];
	double cy = p[2 * c + 1];
	
	// displacements ba and bc
	double bax = ax - bx;
	double bay = ay - by;
	
	double bcx = cx - bx;
	double bcy = cy - by;
	
	// counterclockwise angle C of ba from bc
	double theta = atan2( bcx*bay - bcy*bax, bcx*bax + bcy*bay );
	// angle of joint (a, b, c)
	theta -= offset;
	if (theta <= transition - 2 * PI)
		theta += 2 * PI;
	else if (theta > transition)
		theta -= 2 * PI;
	
	return theta;
}


// the counterclockwise angular velocity of A about B, measured in B's inertial frame
static double get_angular_vel(
	double ax, double ay,
	double avx, double avy,
	double bx, double by,
	double bvx, double bvy)
{
	// let V = velocity of A, in B's inertial frame
	double vx = avx - bvx;
	double vy = avy - bvy;
	// let R = the displacement BA rotated by 90 deg counterclockwise
	double ba_x = ax - bx;
	double ba_y = ay - by;
	double rx = - ba_y;
	double ry = ba_x;
	// let C = scalar component of V along R
	//       = dot(V, R) / norm(R)
	double r_norm = sqrt(rx * rx + ry * ry);
	double c = (vx * rx + vy * ry) / r_norm;
	// then the answer is C / || BA ||
	double ba_norm = sqrt(ba_x * ba_x + ba_y * ba_y);
	return c / ba_norm;
}


// returns the angular velocity of given joint
static double get_thetadot(double *p, double *v, int a, int b, int c) {
	// counterclockwise angular velocity of a about b, measured in b's inertial frame
	double vel_a = get_angular_vel(
		p[2 * a], p[2 * a + 1],
		v[2 * a], v[2 * a + 1],
		p[2 * b], p[2 * b + 1],
		v[2 * b], v[2 * b + 1]
	);
	// counterclockwise angular velocity of c about b, measured in b's inertial frame
	double vel_c = get_angular_vel(
		p[2 * c], p[2 * c + 1],
		v[2 * c], v[2 * c + 1],
		p[2 * b], p[2 * b + 1],
		v[2 * b], v[2 * b + 1]
	);
	return vel_a - vel_c;
}


// returns passive torque exerted by a joint
static double get_t_passive(Joint* j, double theta, double thetadot, double damping) {
	return
		  j->b1 * exp(j->k1 * theta)
		+ j->b2 * exp(j->k2 * theta)
		- damping * thetadot;
}


// returns maximum voluntary active flexion torque exerted by a joint
static double get_t_max_active_flexion(Joint* j, double theta, double thetadot, double c1f) {
	// the maximum active torque equals C1 *
	//     (thing that gets smaller as theta gets further away from the optimal angle C3) *
	//     (thing that gets smaller as the magnitude of thetadot gets larger)
	
	// note that here we restrict both scaling terms to be nonnegative, whereas
	// in the paper the maximum torque was restricted to be nonnegative. it's
	// a subtle difference that prevents torque from being allowed when it
	// shouldn't. i didn't have time to change the equations in the paper
	
	double angle_scale = cos(j->c2f * (theta - j->c3f));
	if (angle_scale < 0)
		angle_scale = 0;
	
	double angular_vel_scale;
	if (thetadot >= 0) {
		angular_vel_scale =
			(2 * j->c4f * j->c5f + thetadot * (j->c5f - 3 * j->c4f)) /
			(2 * j->c4f * j->c5f + thetadot * (2 * j->c5f - 4 * j->c4f));
	} else {
		angular_vel_scale = (1 - j->c6f * thetadot) *
			(2 * j->c4f * j->c5f - thetadot * (j->c5f - 3 * j->c4f)) /
			(2 * j->c4f * j->c5f - thetadot * (2 * j->c5f - 4 * j->c4f));
	}
	if (angular_vel_scale < 0)
		angular_vel_scale = 0;
	
	return c1f * angle_scale * angular_vel_scale;
}


// returns maximum voluntary active extension torque exerted by a joint
static double get_t_max_active_extension(Joint* j, double theta, double thetadot, double c1e) {
	/* in the formula for active extension torque, the other people take thetadot > 0 to
	   mean rotation towards extension, while we take it to mean rotation towards flexion  */
	thetadot *= -1;
	
	double angle_scale = cos(j->c2e * (theta - j->c3e));
	if (angle_scale < 0)
		angle_scale = 0;
	
	double angular_vel_scale;
	if (thetadot >= 0) {
		angular_vel_scale =
			(2 * j->c4e * j->c5e + thetadot * (j->c5e - 3 * j->c4e)) /
			(2 * j->c4e * j->c5e + thetadot * (2 * j->c5e - 4 * j->c4e));
	} else {
		angular_vel_scale = (1 - j->c6e * thetadot) *
			(2 * j->c4e * j->c5e - thetadot * (j->c5e - 3 * j->c4e)) /
			(2 * j->c4e * j->c5e - thetadot * (2 * j->c5e - 4 * j->c4e));
	}
	if (angular_vel_scale < 0)
		angular_vel_scale = 0;
	
	return c1e * angle_scale * angular_vel_scale;
}




// -- linear combination helper functions, which help with rk4 -- //




// u = v + kw
// where each u, v, w are arrays of doubles, and k is a scalar
static void linear_comb2(double *u, double *v, double *w, double k, size_t length) {
	for (size_t i = 0; i < length; i++) {
		u[i] = v[i] + k * w[i];
	}
}

// dst += k1 v1 + k2 v2 + k3 v3 + k4 v4
// where dst and each vi are arrays of doubles, and each ki is a scalar
static void linear_comb4(
	double* dst,
	double* v1, double k1,
	double* v2, double k2,
	double* v3, double k3,
	double* v4, double k4,
	size_t length
) {
	for (size_t i = 0; i < length; i++) {
		dst[i] +=
			  k1 * v1[i]
			+ k2 * v2[i]
			+ k3 * v3[i]
			+ k4 * v4[i];
	}
}




#endif
