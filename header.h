#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

#include <gsl/gsl_filter.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>

#include <fftw3.h>

#include <omp.h>

#define MAX2(x, y) ((x)>(y)?(x):(y))
#define MIN2(x, y) ((x)<(y)?(x):(y))
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

typedef struct {

	const size_t dim, mm, npml, nx, nz, nt, nt_mem, free_surface;

	const float dx, dz, dt; 

} fwi_base;

typedef struct {

	size_t mnx, mnz;
	size_t pnx, pnz;
	size_t unx, unz;
	size_t vnx, vnz;

} fwi_shape;

typedef struct {

	fwi_base *base;

	fwi_shape *f, *b;

} fwi_specs;

typedef struct {

	float **md, **kp, **ekp, *btry, alpha_btry;
	int *ibtry;

} fwi_model;

typedef struct {

	float **m0, **m1, **kp_total, **kp_0, **ekp_total, **ekp_0, *btry, alpha_btry;
	int *ibtry;

} rwi_model;

typedef struct {

	const size_t nt, nr, jump,rectx, rectt, times;
	const int alpha_fbt0, alpha_fbt1;
	const float dr, dt, os0, os1, alpha_msf;

	float **obs0, **obs, **obs_t, **syn, **syn0, **syn_t, *fbt, *val;
	int ios0, ios1, *ifbt;

} fwi_msf;

typedef struct {
	
	float **pold, **pnow, **pnew, **u, **v, **ux, **vz;

} fwi_state; 	

/*
typedef struct {
	
	float **pold, **pnow, **pnew, **u, **v, **ux, **vz;

} rwi_state; 	
*/
typedef struct {
	
	const size_t dim;
	const float maxF, maxV, coefR;

} pml_base;

typedef struct {
	
	int npml;
	
	float *a0[2], *b0[2], *a1[2], *b1[2];
	
	float **u0, **u1, **v0, **v1,**ux0, **ux1, **vz0, **vz1;
 
} pml_profile;

typedef struct {

	float c0x, c1x, ix;	
	float c0z, c1z, iz;
	float iz840, iz24;	

} fwi_fdcoe;

typedef struct {

	float iarea, *xyz_sr;

	int *ixyz_sr;

	float *wav0, *wav, **res; 

} fwi_sou;

typedef struct {
	size_t id;
	const float end_t, start_t, alpha_grad, perc;
	const size_t jump;
	const size_t rectx1, rectz1, repeat1;
	const size_t rectx2, rectz2, repeat2;
	float *body;	

} fwi_grad;

typedef struct {
	
	char *common;

	char *shots;
	char *fbt;
	char *xyz_sr;

	char *init_mdl;
	char *btry;
	char *wavelet;

} fwi_files_path;

typedef struct {
	
	char *common;

	char *shots;
	char *fbt;
	char *xyz_sr;

	char *init_mdl_0;
	char *init_mdl_1;
	char *btry;
	char *wavelet;

} rwi_files_path;

typedef struct {

	fwi_specs *grid;

	fwi_model *model;

	fwi_state *fw;
	fwi_state *bw;
		
	fwi_sou *sou;
	
	fwi_msf *msf;

	pml_profile *pml;

	fwi_fdcoe *coe;	
	
	fwi_files_path *path;
	
	fwi_grad *grad;
	
	struct {	
		char path_bd_wr[300];
		float ***bc_ud, ***bc_lr; 
		int offset;
	};

} fwi_plan;

typedef struct {

	fwi_specs *grid;

	rwi_model *model;

	fwi_state *fw_inc;
	fwi_state *bw_inc;
		
	fwi_state *fw_per;
	fwi_state *bw_per;
	
	fwi_sou *sou;
	
	fwi_msf *msf;

	pml_profile *pml_inc;
	pml_profile *pml_per;

	fwi_fdcoe *coe;	
	
	rwi_files_path *path;
	
	fwi_grad *grad;
	
	struct {	
		char path_bd0_wr[300];
		char path_bd1_wr[300];
		float ***bc_ud_inc, ***bc_lr_inc; 
		float ***bc_ud_per, ***bc_lr_per; 
		int offset;
	};

} rwi_plan;


void fwi_init(fwi_plan *, int ishot);
void fwi_init_wav(fwi_plan *, int ishot,int);
void rwi_init(rwi_plan *, int ishot);

void cpml_coe(float* restrict a0, float* restrict b0, const pml_base *base, const int npml, const float dx, const float dt);

void time_integral(float* restrict * restrict pnew, 
			  float* restrict * restrict pnow, 
		      float* restrict * restrict pold, 
		      float* restrict * restrict pxx, 
		      float* restrict * restrict pzz, 
		      float* restrict * restrict ekp, 
			  const size_t m, const size_t n);

void write_bdr_fs_no(float***  bc_ud, 
			  float***  bc_lr, 
			  float * restrict * restrict p, 
			  const fwi_shape *gb, 
			  const fwi_base *base,
			  FILE *fp,
			  size_t it);

void write_bdr(float***  bc_ud, 
			  float***  bc_lr, 
			  float * restrict * restrict p, 
			  const fwi_shape *gb, 
			  const fwi_base *base,
			  FILE *fp,
			  size_t it);

void read_bdr(float *** bc_ud, 
			  float *** bc_lr, 
			  float * restrict * restrict p, 
			  const fwi_shape *gb, 
			  const fwi_base *base,
			  FILE *fp,
			  const int offset,
			  size_t it);

void pml_var_init(pml_profile *pml, fwi_shape *grd_f) ;
void state_var_init(fwi_state *var, fwi_shape *shape) ;
void ricker(float *wav);
void cpmlcoex(float *a, float *b, float maxvel, bool flag_half);
void cpmlcoez(float *a, float *b, float maxvel, bool flag_half);
/*
void free2d(float **array);
void free3d(float ***array);
*/
float ***fcalloc3d(int n3, int n2, int n1);

void model_ext(float **model, int n1, int n2, float **emodel, int en1, int en2);
void model_ext_fs_no(float **model, int n1, int n2, float **emodel, int en1, int en2);

void boxconv(int nb, int nx, float *xx, float *yy);
void triangle(int nr, int n12, float *uu, float *vv);
void triangle2(int rect1, int rect2, int nt, int n1, int n2, float *uu, float *vv);
void grad_precond_new4(fwi_grad *, fwi_shape *, int *);

void newclip(float *x, int n, int type, int rect, int, float v0, float v1, float *bottom);
void windfdiy(float *win, int n1, int wh0, int wh1);
void grad4_1shot(fwi_plan *, int);
void grad_wavest(fwi_plan *, int,int);
void grad4_1shot_rfl(rwi_plan *, int);

typedef struct _epsdf1 
{
    int nr, nt, nmodel, npad4Ax, npad4Atr, niter;

    char *conv_how_to_slice, *reg_type;
    
    float tor0, eps0, Tdepressor;  

} epsdf1;

void fun_Ax(fftwf_complex *fftA, fftwf_complex *fftx, float *ret, int s, fftwf_plan *plan);

void Ax_slicing(float *ret, epsdf1 *ipar, float *r0);

void Atr_slicing(float *Atr, epsdf1 *ipar, float *z0);

void fun_Dx(float *x, epsdf1 *ipar, float *r1);

void fun_Dtr(float *r1, epsdf1 *ipar, float *z0);

void MF1D(float **obs, float **syn, void *mpars, float *misfit, float **res);

typedef struct _par_xcorr1d 
{
    int nr, nt, npad4Ab;

    char *conv_how_to_slice;  

} par_xcorr1d;
void XCORR1D(float **obs, float **syn, float *, float, float, void *mpars, float *misfit, float **res);

void a_lap_opt_2d_pml_fs_no(

	float *restrict *restrict p, 
	float *restrict *restrict u, //dpdx
	float *restrict *restrict v, //dpdz
	float *restrict *restrict f, //dudx=dp2dx2
	float *restrict *restrict g, //dvdz=dp2dz2
 	
	const fwi_fdcoe *restrict coe, 
	const fwi_shape *restrict grd, 
	const pml_profile *restrict pml);


 void a_lap_opt_2d_pml(

	float *restrict *restrict p, 
	float *restrict *restrict u, //dpdx
	float *restrict *restrict v, //dpdz
	float *restrict *restrict f, //dudx=dp2dx2
	float *restrict *restrict g, //dvdz=dp2dz2
 	
	const fwi_fdcoe *restrict coe, 
	const fwi_shape *restrict grd, 
	const pml_profile *restrict pml);

void a_lap_opt_2d(

	float *restrict *restrict p, 
	float *restrict *restrict u, //dpdx
	float *restrict *restrict v, //dpdz
	float *restrict *restrict f, //dudx=dp2dx2
	float *restrict *restrict g, //dvdz=dp2dz2
 	
	const fwi_fdcoe *restrict coe, 
	const fwi_shape *restrict grd);
/*
float** allocate2d(const int n1, const int n2);
float ***allocate3d(const int n1, const int n2, const int n3);
*/
void fileread(char *filename, int size, int count, void *dest);
void grad_precond_new5(fwi_grad *g, fwi_shape *bw, int *bottom);
 void shrinkage(float *x, int n, float lambda);
void ricker_wavelet(int nt, float dt, float t0, float fm, float *out) ;
void upper_off_srcood(float **in, float *sr, int nr, int nt, float dt, float t0, float v0, float K, float **out);
void lower_off_srcood(float **in, float *sr, int nr, int nt, float dt, float t0, float v0, float K, float **out);
void fbt_muter(float *fbt, float *obs, int nt, int nr, float alpha, float dt, float tune, bool early_gone) ;	
void sfbandpass(float *indat, int nt,  int nr, float dt,
    bool phase, int nphi, int nplo, float fhi, float flo, 
    float *outdat);
void normalized_lcc_misfit_phase(float **obs, float **syn, int nt, int nr, int nl, int nw, float alpha_win, float alpha_argmax, char *penalty, int offsetfrom, int offsetto, float zero_devision_bar, float barrier_sim, float barrier_amp, int barrier_offset, float *msf, float **res);
void nlcc_auto_mute(float **obs, float **syn, int nt, int nx, int nl, int nw, float alpha_win, int offsetfrom, int offsetto, int *t_start, int *t_end, int K, int median, int tune, float **nobs, float**nsyn);
 float soft_argmax0(float *arg, int nt, float alpha, float vmax, float *res);

void normalized_lcc_misfit_adaptive(float **obs, float **syn, int nt, int nx, int nl, int nw, float alpha_win, float alpha_argmax, char *penalty, int offsetfrom, int offsetto, float barrier, float *msf, float **res);
 void nlcc_auto_mute2(float **obs, float **syn, int nt, int nx, int nl, int nw, float alpha_win, int offsetfrom, int offsetto, int *t_start, int *t_end, int K, int median, int *time);

void XCORR1D_t(float *obs, float *syn, void *mpars, float *misfit, float *res);

void err_matrix_lcc(float **obs, float **syn, int *t0, int *t1, int offset0, int offset1, int nt, int nx, int nl, int nw, float alpha_win, float zero_devision_bar, float barrier_sim, int barrier_lag, float *msf, float **res);

void linear_upper_off(float **in, float *sr, int nr, int nt, float dt, float t0, float v0, int N);
void linear_lower_off(float **in, float *sr, int nr, int nt, float dt, float t0, float v0, int N);

void quadratic_upper_off(float **in, float *sr, int nr, int nt, float dt, float t0, float v0, int N);
void quadratic_lower_off(float **in, float *sr, int nr, int nt, float dt, float t0, float v0, int N);


