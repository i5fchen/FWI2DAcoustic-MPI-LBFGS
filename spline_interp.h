#include <gsl/gsl_math.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_blas.h>

void Minterp1d(double *x, double *fx, const int N0,
  double *xx, const int N1, const gsl_interp_type *itype,
  double *fxx);


void trcinp(float **obs0, int nt0, float dt0, float **obs, int nt, float dt, int howmany);
