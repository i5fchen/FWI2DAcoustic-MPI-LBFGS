#include "header.h"
#include "fft_based_algorithms.h"
	
void XCORR1D(float **obs, float **syn, float *sr_xyz, float min, float max, void *mpars, float *misfit, float **res){
    
    // syn xcorr obs trace by trace.

    par_xcorr1d *ipar = (par_xcorr1d *) mpars;

    int s0 = ipar->nt + ipar->nt - 1 + ipar->npad4Ab; 

    float *T, *A, *b, *x, tt1, tt2, alpha;

    fftwf_complex *fftA, *fftb; 

    fftwf_plan fwd, bwd;
	
	float dr, ds, e1, e2;
    
    A = (float *)calloc(s0, sizeof(float) );
    b = (float *)calloc(s0, sizeof(float) );
    x = (float *)calloc(ipar->nt, sizeof(float) );

    //fftwf_init_threads();

    fftA  =  (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (s0/2+1) );
    fftb  =  (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (s0/2+1) );

    //fftwf_plan_with_nthreads(1);

    fwd  = fftwf_plan_dft_r2c_1d(s0, A,  fftA,  FFTW_ESTIMATE); // plan to do fft(a) and fft(x)

    bwd  = fftwf_plan_dft_c2r_1d(s0, fftA,  x,  FFTW_ESTIMATE); // plan to do ifft(fft(a*x)) 

    if (strncmp(ipar->conv_how_to_slice, "middle", 6) * strncmp(ipar->conv_how_to_slice, "leading", 6) != 0) {printf("incorrete conv mode.\n"); exit(0);}

    T = malloc(sizeof(float) * ipar->nt);

    for (int i=0; i<ipar->nt; i++){
        
        if (strncmp(ipar->conv_how_to_slice, "middle", 6) == 0) {
            
            if (i>ipar->nt/2)
                T[i] = i-ipar->nt/2;
            else
                T[i] = ipar->nt/2-i;
        
        } else {

            T[i] = i;

        }

    }

    misfit[0] = 0.0; 
    misfit[1] = 0.0; 
   			

	for (int ir=0; ir<ipar->nr; ir++){
		e1=0; e2=0;
		for(int it=0; it<ipar->nt; it++){
			res[ir][it] = 0.0f;
			e1 += fabs(obs[ir][it]);		
			e2 += fabs(syn[ir][it]);		
		
		}
	
		ds = sr_xyz[2+2*ir]-sr_xyz[0];
		ds *= ds;

		dr = sr_xyz[2+2*ir+1]-sr_xyz[1];
		dr *= dr;

		dr = sqrt(ds+dr);

		if (e2/(e1+1e-12)>10 || e1/(e2+1e-12)>10 || dr<min || dr>max)
			continue;

        for (int it=0; it<s0; it++){
            A[it] = 0;
        }

        for (int it=0; it<ipar->nt; it++){
            A[it] = obs[ir][ipar->nt-1-it];
            b[it] = syn[ir][it];
			misfit[0] += fabs(A[it]-b[it]);
        }
        
        fftwf_execute_dft_r2c(fwd, A, fftA); 
        fftwf_execute_dft_r2c(fwd, b, fftb);   

        for (int i=0; i<s0/2+1; i++){

            tt1 = (float)(fftA[i][0]*fftb[i][0]-fftA[i][1]*fftb[i][1]);
            tt2 = (float)(fftA[i][0]*fftb[i][1]+fftA[i][1]*fftb[i][0]);
            fftA[i][0] = tt1/s0; fftA[i][1] = tt2/s0;
            
        }
        
        fftwf_execute_dft_c2r(bwd, fftA, A);
        
        if (strncmp(ipar->conv_how_to_slice, "middle", 6) == 0){

            for (int i=ipar->nt/2; i<ipar->nt/2+ipar->nt; i++) 
                x[i-ipar->nt/2] = A[i];    

        }else {

            for (int i=0; i<ipar->nt; i++) x[i] = A[i];  

        }
       
        int j=0; float tmp = -1.0e20;
        for (int i=0; i<ipar->nt; i++) {

            if (tmp<x[i]){

                tmp = x[i]; j = i;
            
            } 

        }

        for (int i=0; i<ipar->nt; i++) b[i] = T[i]*x[i]; 
        
        tt1 = cblas_sdot(ipar->nt,  b, 1,  b, 1);
        tt2 = cblas_sdot(ipar->nt,  x, 1,  x, 1);
        
        alpha = tt1/tt2;
        misfit[1] += alpha;

        for (int i=0; i<ipar->nt; i++)
            x[i] = (T[i]*b[i]-x[i]*alpha)/tt2*2.0;
        

        for (int it=0; it<s0; it++){
            A[it] = 0;
        }

        for (int it=0; it<ipar->nt; it++){
            A[it] = obs[ir][it];
            b[it] = x[it];
        }
        
        fftwf_execute_dft_r2c(fwd, A, fftA); 
        fftwf_execute_dft_r2c(fwd, b, fftb);   

        for (int i=0; i<s0/2+1; i++){

            tt1 = (float)(fftA[i][0]*fftb[i][0]-fftA[i][1]*fftb[i][1]);
            tt2 = (float)(fftA[i][0]*fftb[i][1]+fftA[i][1]*fftb[i][0]);
            fftA[i][0] = tt1/s0; fftA[i][1] = tt2/s0;
            
        }
        
        fftwf_execute_dft_c2r(bwd, fftA, A);
        
        if (strncmp(ipar->conv_how_to_slice, "middle", 6) == 0){

            for (int i=ipar->nt/2; i<ipar->nt/2+ipar->nt; i++) 
                res[ir][i-ipar->nt/2] = A[i];    

        }else {

            for (int i=0; i<ipar->nt; i++) 
                res[ir][i] = A[i];  

        }


    }

       
    //fftwf_cleanup_threads();
    free(A); free(x); free(b);
    fftwf_free(fftA); fftwf_free(fftb);
    fftwf_destroy_plan(fwd); fftwf_destroy_plan(bwd);  
    free(T);

}

