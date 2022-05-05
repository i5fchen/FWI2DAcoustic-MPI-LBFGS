#include "spline_interp.h"

void Minterp1d(double *x, double *fx, const int N0,
  double *xx, const int N1, const gsl_interp_type *itype,
  double *fxx){

  gsl_interp_accel *acc = gsl_interp_accel_alloc();
  gsl_spline *spline = gsl_spline_alloc(itype, N0);

  gsl_spline_init(spline, x, fx, N0);
  for (int i = 1; i < N1-1; ++i)
    fxx[i] = gsl_spline_eval(spline, xx[i], acc);
  fxx[0] = fx[0]; fxx[N1-1] = fx[N0-1];
  gsl_spline_free(spline);
  gsl_interp_accel_free(acc);
}


void trcinp(float **obs0, int nt0, float dt0, float **obs, int nt, float dt, int howmany){
  double *x0, *x1, *tmp0, *tmp1;
  x0 = malloc(nt0*sizeof(double)); tmp0 = malloc(nt0*sizeof(double));
  x1 = malloc(nt*sizeof(double));  tmp1 = malloc(nt*sizeof(double));
  for (int i2=0; i2<nt; i2++)
    x1[i2] = i2*dt;
  for (int i2=0; i2<nt0; i2++)
    x0[i2] = i2*dt0;

  for (int i1=0; i1<howmany; i1++){
    for (int i2=0; i2<nt0; i2++){
      tmp0[i2] = obs0[i1][i2];
    }
    Minterp1d(x0, tmp0, nt0, x1, nt, gsl_interp_cspline, tmp1);
    for (int i2=0; i2<nt; i2++){
      obs[i1][i2] = tmp1[i2];
    }
  }
  free(tmp0); free(tmp1); free(x0); free(x1);
}


