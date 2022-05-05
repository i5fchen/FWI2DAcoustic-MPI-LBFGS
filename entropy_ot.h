#include <stdlib.h>

typedef struct _eot_msf_par {

	int nt, nr, max_iter;
	float **M, *u, *v;
	float eps, tor, dt;

} eot_msf_par;

typedef struct _eot_log_msf_par {
	float bar;	
	float **M, *u, *v;		
	float *logp, *logq, *delta_u, *delta_v, vmax0;	

} eot_log_msf_par;

void eot_msf_destroy(eot_msf_par *par);

void eot_msf_init(eot_msf_par *par);

void eot_log_msf_init(eot_msf_par *par, eot_log_msf_par *par_exd);

void entropy_ot(float *p, float *q, eot_msf_par *par, float *msf, float *res);

void entropy_ot_2d_trace_by_trace(float **obs, float **syn, eot_msf_par *par, float *msf, float **res);

void entropy_ot_log(float *p, float *q, eot_msf_par *par, eot_log_msf_par *par_exd, float *msf, float *res);

void entropy_ot_log_2d_trace_by_trace(float **obs, float **syn, eot_msf_par *par, eot_log_msf_par *par_exd, float *msf, float **res);
