#include "fft_based_algorithms.h"
#include "read_write.h"
static int good_size_real(int n) {

	// translated version of good_size_real in scipy/fft/_pocketfft
	
	if (n<=6) return n;
	
	int bestfac = 2*n;
	
	for (int f5=1; f5<bestfac; f5*=5){

		int x=f5;
		
		while (x<n) x *= 2;
		
		for (;;){
		
			if (x<n) x*=3;
		
			else if (x>n) {
		
				if (x<bestfac) bestfac = x;
				if (x&1) break;
				x>>=1;
		
			} else {
			
				return n;
			
			}
		}

	}

	return bestfac;
	
}

void fftcorrelate1d_init(fft_corr_utility *fft, int ng, int nf){

	fft->ng = ng;
	fft->nf = nf;

	fft->ns = fft->nf+fft->ng-1;

	fft->ns_pad = good_size_real(fft->ns);

	fft->g_pad = (float *)calloc(fft->ns_pad, sizeof(float));
	fft->f_pad = (float *)calloc(fft->ns_pad, sizeof(float));
	fft->ret   = (float *)calloc(fft->ns_pad, sizeof(float));

	fft->fft_g  =  (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (fft->ns_pad/2+1) );
	fft->fft_f  =  (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (fft->ns_pad/2+1) );

	fft->fwd = fftwf_plan_dft_r2c_1d(fft->ns_pad, fft->g_pad,  fft->fft_g,  FFTW_ESTIMATE);  // plan to do fft
    fft->rwd = fftwf_plan_dft_c2r_1d(fft->ns_pad, fft->fft_g,  fft->g_pad,  FFTW_ESTIMATE);  // plan to do ifft
	

}

void fftcorrelate1d_free(fft_corr_utility *fft){
		
	free(fft->g_pad);
	free(fft->f_pad);
	free(fft->ret);
	fftwf_free(fft->fft_g);
	fftwf_free(fft->fft_f);

	fftwf_destroy_plan(fft->fwd);
	fftwf_destroy_plan(fft->rwd);
}

void fftcorrelate1d(float *g, float *f, fft_corr_utility *fft){

	float tt1, tt2;

	for(int i=0; i<fft->ng; i++) {
		fft->g_pad[i] = g[i] ;
	}

	for(int i=0; i<fft->nf; i++) {
		fft->f_pad[i] = f[fft->nf-1-i] ;
	}

	fftwf_execute_dft_r2c(fft->fwd, fft->g_pad, fft->fft_g); 
	fftwf_execute_dft_r2c(fft->fwd, fft->f_pad, fft->fft_f); 
	
	for (int i=0; i<fft->ns_pad/2+1; i++){

        tt1 = (fft->fft_g[i][0]*fft->fft_f[i][0]-fft->fft_g[i][1]*fft->fft_f[i][1]);
        tt2 = (fft->fft_g[i][0]*fft->fft_f[i][1]+fft->fft_g[i][1]*fft->fft_f[i][0]);
        
		fft->fft_g[i][0] = tt1/fft->ns_pad; fft->fft_g[i][1] = tt2/fft->ns_pad;
            
    }
        
    fftwf_execute_dft_c2r(fft->rwd, fft->fft_g, fft->ret);

}

void fftconvolve1d(float *g, float *f, fft_corr_utility *fft){

	float tt1, tt2;

	for(int i=0; i<fft->ng; i++) {
		fft->g_pad[i] = g[i] ;
	}

	for(int i=0; i<fft->nf; i++) {
		fft->f_pad[i] = f[i] ;
	}

	fftwf_execute_dft_r2c(fft->fwd, fft->g_pad, fft->fft_g); 
	fftwf_execute_dft_r2c(fft->fwd, fft->f_pad, fft->fft_f); 
	
	for (int i=0; i<fft->ns_pad/2+1; i++){

        tt1 = (fft->fft_g[i][0]*fft->fft_f[i][0]-fft->fft_g[i][1]*fft->fft_f[i][1]);
        tt2 = (fft->fft_g[i][0]*fft->fft_f[i][1]+fft->fft_g[i][1]*fft->fft_f[i][0]);
        
		fft->fft_g[i][0] = tt1/fft->ns_pad; fft->fft_g[i][1] = tt2/fft->ns_pad;
            
    }
        
    fftwf_execute_dft_c2r(fft->rwd, fft->fft_g, fft->ret);

}

