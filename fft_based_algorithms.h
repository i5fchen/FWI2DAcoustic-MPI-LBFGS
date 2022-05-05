#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>

typedef struct {

	float *g_pad, *f_pad, *ret;
	
	fftwf_plan fwd, rwd;
	
	fftwf_complex *fft_g, *fft_f;
	
	int ng, nf, ns, ns_pad;

} fft_corr_utility;

void fftcorrelate1d_init(fft_corr_utility *fft, int ng, int nf);

void fftcorrelate1d(float *g, float *f, fft_corr_utility *fft);

void fftcorrelate1d_free(fft_corr_utility *fft);

void fftconvolve1d(float *g, float *f, fft_corr_utility *fft);
void ddd_msf(float **obs, float **syn, int nt, int nr, int nstep, float msf[], float **res);
