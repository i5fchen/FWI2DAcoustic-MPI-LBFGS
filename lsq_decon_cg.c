#include "header.h"
#include "fft_based_algorithms.h"
void fun_Ax(fftwf_complex *fftA, fftwf_complex *fftx, float *ret, int s, fftwf_plan *plan) {
            
    float tt1, tt2;

    for (int i=0; i<s/2+1; i++){

        tt1 = (float)(fftA[i][0]*fftx[i][0]-fftA[i][1]*fftx[i][1]);
        tt2 = (float)(fftA[i][0]*fftx[i][1]+fftA[i][1]*fftx[i][0]);
        fftx[i][0] = tt1/s; fftx[i][1] = tt2/s;
            
    }
        
    fftwf_execute_dft_c2r(*plan, fftx, ret);

}

void Ax_slicing(float *ret, epsdf1 *ipar, float *r0) {

    if (strncmp(ipar->conv_how_to_slice, "middle", 6) == 0)

        for (int i=ipar->nmodel/2; i<ipar->nmodel/2+ipar->nt; i++) r0[i-ipar->nmodel/2] = ret[i];    

    else {

        for (int i=0; i<ipar->nt; i++) r0[i] = ret[i];  

        }

}

void Atr_slicing(float *Atr, epsdf1 *ipar, float *z0) {

    if (strncmp(ipar->conv_how_to_slice, "middle",6) == 0)

        for (int i=ipar->nt-ipar->nmodel/2-1; i<ipar->nmodel/2+ipar->nt; i++) z0[i-(ipar->nt-ipar->nmodel/2-1)] = Atr[i];    

    else {

        for (int i=ipar->nt-1; i<ipar->nt-1+ipar->nmodel; i++) z0[i-(ipar->nt-1)] = Atr[i];  

    }

}

void fun_Dx(float *x, epsdf1 *ipar, float *r1) {

    if (strncmp(ipar->reg_type, "diff", 3) == 0){

        for (int i=0; i<ipar->nmodel-1; i++) {r1[i] = x[i+1]-x[i];}; r1[ipar->nmodel-1] = x[ipar->nmodel-1];   

    }else {

        for (int i=0; i<ipar->nmodel; i++) r1[i] = x[i];  

    }

    for (int i=0; i<ipar->nmodel; i++) r1[i] = ipar->eps0 * r1[i];

}

void fun_Dtr(float *r1, epsdf1 *ipar, float *z0) {

    if (strncmp(ipar->reg_type, "diff",3) == 0){

        for (int i=1; i<ipar->nmodel-1; i++) {
            z0[i] += (r1[i-1]-r1[i])*ipar->eps0;
        }; 
        z0[0] -= r1[0]*ipar->eps0; 
        z0[ipar->nmodel-1] += (r1[ipar->nmodel-1]+r1[ipar->nmodel-2])*ipar->eps0;   

    }else {

        for (int i=0; i<ipar->nmodel; i++) z0[i] += r1[i]*ipar->eps0;  

    }

}


void MF1D(float **obs, float **syn, void *mpars, float *misfit, float **res){
    

    
    epsdf1 *ipar = (epsdf1 *) mpars;

    int s0 = ipar->nt + ipar->nmodel - 1 + ipar->npad4Ax;  // the length of padded a and padded x (to calculate A.dot(x) or np.convolve(a,x))
    int s1 = ipar->nt + ipar->nt     - 1 + ipar->npad4Atr; // the length of padded a.reverse and padded r (to calculate A^t.dot(r) or np.correlate(r, a))
    
    float *T, *A, *b, *x, *r0, *r1, *z0, *w0, *w1, *p0, *Ax, *At, *Atr;

    float *cgne_r0, *cgne_b, *cgne_x0, *cgne_x1, *cgne_p0, *cgne_p1; 

    float rnorm0, rnorm1, beta, alpha, alpha_num, alpha_den;

    fftwf_complex *fftA, *fftx, *fftAt, *fftr0; 

    fftwf_plan fwd_Ax, fwd_Atr, bwd_Ax, bwd_Atr;
    
    A = (float *)calloc(s0, sizeof(float) );
    x = (float *)calloc(s0, sizeof(float) );
    b = (float *)calloc(ipar->nt, sizeof(float) );

    At = (float *)calloc(s1, sizeof(float) );
    r0 = (float *)calloc(s1, sizeof(float) );
    r1 = (float *)calloc(ipar->nmodel, sizeof(float) );

    z0 = (float *)calloc(ipar->nmodel, sizeof(float) );
    p0 = (float *)calloc(s0, sizeof(float) );
    
    w0 = (float *)calloc(ipar->nt, sizeof(float) );
    w1 = (float *)calloc(ipar->nmodel, sizeof(float) );

    Ax  = (float *)calloc(s0, sizeof(float) );
    Atr = (float *)calloc(s1, sizeof(float) );
    
    //fftwf_init_threads();

    fftA  =  (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (s0/2+1) );
    fftx  =  (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (s0/2+1) );

    fftAt =  (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (s1/2+1) );
    fftr0 =  (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (s1/2+1) );

    //fftwf_plan_with_nthreads(1);

    fwd_Ax   = fftwf_plan_dft_r2c_1d(s0, A,  fftA,  FFTW_ESTIMATE); // plan to do fft(a) and fft(x)
    fwd_Atr  = fftwf_plan_dft_r2c_1d(s1, At, fftAt, FFTW_ESTIMATE); // plan to do fft(a.reverse()) and fft(r)

    bwd_Ax   = fftwf_plan_dft_c2r_1d(s0, fftA,  Ax,  FFTW_ESTIMATE); // plan to do ifft(fft(a*x)) 
    bwd_Atr  = fftwf_plan_dft_c2r_1d(s1, fftAt, Atr, FFTW_ESTIMATE); // plan to do ifft(fft(a.xcorr.r)) 

    cgne_r0 = (float *)calloc(s0,           sizeof(float) ); // padded version of nmodel
    cgne_b  = (float *)calloc(ipar->nmodel, sizeof(float) );
    cgne_x0 = (float *)calloc(s1,           sizeof(float) );
    cgne_x1 = (float *)calloc(ipar->nmodel, sizeof(float) );
    cgne_p0 = (float *)calloc(s1,           sizeof(float) );
    cgne_p1 = (float *)calloc(ipar->nmodel, sizeof(float) );

    if (strncmp(ipar->conv_how_to_slice, "middle", 6) * strncmp(ipar->conv_how_to_slice, "leading", 6) != 0) {printf("incorrete conv mode.\n"); exit(0);}

    if (strncmp(ipar->reg_type, "diff", 3) * strncmp(ipar->reg_type, "ide", 3) != 0) {printf("incorrete reg type.\n"); exit(0);}

    T = malloc(sizeof(float) * ipar->nmodel);
     
    for (int i=0; i<ipar->nmodel; i++){
        
        if (strncmp(ipar->conv_how_to_slice, "middle", 6) == 0) {
            
            if (i>ipar->nmodel/2)
                T[i] = i-ipar->nmodel/2;
            else
                T[i] = ipar->nmodel/2-i;
        
        } else {

            T[i] = i;

        }

        T[i] *= ipar->Tdepressor;
        
    }
    
    misfit[0] = 0.0;
	
    for (int i=0; i<ipar->nt*ipar->nr; i++){
        
        misfit[0] += pow((&obs[0][0])[i]-(&syn[0][0])[i], 2);
    
    }
    
    misfit[1] = 0.0; 
    
    for (int j=0; j<ipar->nr; j++)
    for (int i=0; i<ipar->nt; i++)
            res[j][i] = 0.0;
        
    for (int ix=0; ix<ipar->nr; ix++){
	//	printf("%d\n",ix);	
		for (int it=0; it<ipar->nt; it++){
            A[it] = obs[ix][it]*100.0;
            b[it] = syn[ix][it]*100.0;
        }
        
		for (int i=0; i<ipar->nmodel; i++) x[i] = 0.0;
        
        for (int i=0; i<ipar->nt; i++) At[i] = A[ipar->nt-1-i]; 
        //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        fftwf_execute_dft_r2c(fwd_Ax, A, fftA); 
        fftwf_execute_dft_r2c(fwd_Ax, x, fftx);   

        fun_Ax(fftA, fftx, Ax, s0, &bwd_Ax);

        Ax_slicing(Ax, ipar, r0); // the first nt address unit reffered by r0 holds the properly sliced result   

        for (int i=0; i<ipar->nt; i++){
            r0[i] = b[i] - r0[i];
        }

        fun_Dx(x, ipar, r1); // r1 is recycled to hold the result of reg(x)
        
        for (int i=0; i<ipar->nmodel; i++) r1[i] = -r1[i];
        //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        fftwf_execute_dft_r2c(fwd_Atr, At, fftAt); 
        fftwf_execute_dft_r2c(fwd_Atr, r0, fftr0); 

        fun_Ax(fftAt, fftr0, Atr, s1, &bwd_Atr);
        
        Atr_slicing(Atr, ipar, z0); fun_Dtr(r1, ipar, z0);

        for (int i=0; i<ipar->nmodel; i++){ p0[i] = z0[i]; }
        for (int i=ipar->nmodel; i<s0; i++) { p0[i] = 0.0; }

        //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        for (int it=0; it<ipar->niter; it++){

            rnorm0 = sqrt(cblas_sdot(ipar->nt, r0, 1, r0, 1) + cblas_sdot(ipar->nmodel, r1, 1, r1, 1));
            
            //printf("%d.......%f\n", it,rnorm0);
            //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            // calculate w = Ap
            
            fftwf_execute_dft_r2c(fwd_Ax, p0, fftx);   

            fun_Ax(fftA, fftx, Ax, s0, &bwd_Ax);

            Ax_slicing(Ax, ipar, w0); // the first nt address u
            
            fun_Dx(p0, ipar, w1);

            alpha_num = cblas_sdot(ipar->nmodel, z0, 1, z0, 1);
            alpha_den = cblas_sdot(ipar->nt, w0, 1, w0, 1) + cblas_sdot(ipar->nmodel, w1, 1, w1, 1);
            alpha = alpha_num/alpha_den;
            
            //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

            for (int i=0; i<ipar->nmodel; i++){
                x[i] = x[i] + alpha*p0[i];
            }

            for (int i=0; i<ipar->nt; i++){
                r0[i] = r0[i] - alpha*w0[i];
            }

            for (int i=0; i<ipar->nmodel; i++){
                r1[i] = r1[i] - alpha*w1[i];
            }
            
            //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            // calculate Atr
            fftwf_execute_dft_r2c(fwd_Atr, r0, fftr0); 

            fun_Ax(fftAt, fftr0, Atr, s1, &bwd_Atr);
        
            Atr_slicing(Atr, ipar, z0); fun_Dtr(r1, ipar, z0);

            //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

            beta = cblas_sdot(ipar->nmodel, z0, 1, z0, 1)/alpha_num;

            for (int i=0; i<ipar->nmodel; i++){
                p0[i] = z0[i] + beta*p0[i];
            }

            rnorm1 = sqrt(cblas_sdot(ipar->nt, r0, 1, r0, 1) + cblas_sdot(ipar->nmodel, r1, 1, r1, 1));

            if (fabs(rnorm1-rnorm0) < ipar->tor0)
                break;
        }
        

        for (int i=0; i<s1; i++) cgne_x0[i] = 0;
        for (int i=0; i<ipar->nmodel; i++) cgne_x1[i] = 0;    

        for (int i=0; i<ipar->nmodel; i++) z0[i] = T[i]*x[i]; 
        
        alpha_num = cblas_sdot(ipar->nmodel, z0, 1, z0, 1);
        alpha_den = cblas_sdot(ipar->nmodel,  x, 1,  x, 1);
        
        alpha = alpha_num/alpha_den;
        
        misfit[1] += alpha;

        for (int i=0; i<ipar->nmodel; i++)
            cgne_b[i] = (T[i]*z0[i]-x[i]*alpha)/alpha_den*2.0;

        //00000000000000000000000000000000000000000000000000000000000000000000000000//
        //00000000000000000000000000000000000000000000000000000000000000000000000000//
        //00000000000000000000000000000000000000000000000000000000000000000000000000//
        //00000000000000000000000000000000000000000000000000000000000000000000000000//

        fftwf_execute_dft_r2c(fwd_Atr, cgne_x0, fftr0); 

        fun_Ax(fftAt, fftr0, Atr, s1, &bwd_Atr);
        
        Atr_slicing(Atr, ipar, z0); fun_Dtr(cgne_x1, ipar, z0);

        for (int i=0; i<ipar->nmodel; i++)
            cgne_r0[i] = cgne_b[i] - z0[i];
           

        //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        fftwf_execute_dft_r2c(fwd_Ax, cgne_r0, fftx);   

        fun_Ax(fftA, fftx, Ax, s0, &bwd_Ax);

        Ax_slicing(Ax, ipar, cgne_p0); 

        fun_Dx(cgne_r0, ipar, cgne_p1); 

        for (int it=0; it<ipar->niter; it++){

            rnorm0 = cblas_snrm2(ipar->nmodel, cgne_r0, 1);

            //printf("%d++++++++++++++%f\n", it,rnorm0);

            alpha_num = rnorm0*rnorm0;
            alpha_den = cblas_sdot(ipar->nt, cgne_p0, 1, cgne_p0, 1)+cblas_sdot(ipar->nmodel, cgne_p1, 1, cgne_p1, 1); 

            alpha = alpha_num / alpha_den;

            for (int i=0; i<ipar->nt; i++) 
                cgne_x0[i] += alpha*cgne_p0[i];
            for (int i=0; i<ipar->nmodel; i++)
                cgne_x1[i] += alpha*cgne_p1[i];

            fftwf_execute_dft_r2c(fwd_Atr, cgne_p0, fftr0); 

            fun_Ax(fftAt, fftr0, Atr, s1, &bwd_Atr);
        
            Atr_slicing(Atr, ipar, z0); fun_Dtr(cgne_p1, ipar, z0);

            for (int i=0; i<ipar->nmodel; i++)
                cgne_r0[i] -= alpha*z0[i];

            alpha_den = cblas_sdot(ipar->nmodel, cgne_r0, 1, cgne_r0, 1); beta = alpha_den/alpha_num;

            fftwf_execute_dft_r2c(fwd_Ax, cgne_r0, fftx);   

            fun_Ax(fftA, fftx, Ax, s0, &bwd_Ax);

            Ax_slicing(Ax, ipar, r0); 

            fun_Dx(cgne_r0, ipar, r1); 
            
            for (int i=0; i<ipar->nt; i++)     cgne_p0[i] = beta*cgne_p0[i] + r0[i];
            for (int i=0; i<ipar->nmodel; i++) cgne_p1[i] = beta*cgne_p1[i] + r1[i];

            rnorm1 = sqrt(alpha_den);

            if (fabs(rnorm1-rnorm0) < ipar->tor0)
                break;

        }  

        //////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////


        j=0; tmp = 0.0;
        for (int i=0; i<ipar->nt; i++) {

            if (tmp<fabs(cgne_x0[i])){
                tmp = fabs(cgne_x0[i]); j = i;
            } 

        }

        for (int it=0; it<ipar->nt; it++){
            res[ix][it] = cgne_x0[it];
        }

        //////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////

    } 
    //fftwf_cleanup_threads();
    free(A); free(x); free(b); free(At); free(r0); free(r1); free(z0); free(p0); free(w0); free(w1); free(Ax); free(Atr);
    fftwf_free(fftA); fftwf_free(fftx); fftwf_free(fftAt); fftwf_free(fftr0);
    fftwf_destroy_plan(fwd_Ax); fftwf_destroy_plan(fwd_Atr); fftwf_destroy_plan(bwd_Ax); fftwf_destroy_plan(bwd_Atr); 
    free(T);
    free(cgne_r0); free(cgne_b); free(cgne_x0); free(cgne_x1); free(cgne_p0); free(cgne_p1);

}

