#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include "Mpar.h"
#include "fwi_sys.h"

/*
Copy from Madagascar
*/
float **sf_floatalloc2 (size_t n1, size_t n2)
{
    size_t i2;
    float **ptr;
    
    ptr = (float**) calloc (n2,sizeof(float*));
    ptr[0] = calloc (n1*n2, sizeof(float));
    for (i2=1; i2 < n2; i2++) {
    	ptr[i2] = ptr[0]+i2*n1;
    }
    return ptr;
}

struct Sf_Butter {
    bool low;
    int nn;
    float **den, mid;
};

typedef struct Sf_Butter *sf_butter;

sf_butter sf_butter_init(bool low, float cutoff, int nn)
{
    int j;
    float arg, ss, sinw, cosw, fact;
    sf_butter bw;

    arg = 2.*PI*cutoff;
    sinw = sin(arg);
    cosw = cos(arg);

    bw = (sf_butter) calloc (1,sizeof(*bw));
    bw->nn = nn;
    bw->low = low;
    bw->den = sf_floatalloc2(2,(nn+1)/2);

    if (nn%2) {
    if (low) {
        fact = (1.+cosw)/sinw;
        bw->den[nn/2][0] = 1./(1.+fact);
        bw->den[nn/2][1] = 1.-fact;
    } else {
        fact = sinw/(1.+cosw);
        bw->den[nn/2][0] = 1./(fact+1.);
        bw->den[nn/2][1] = fact-1.;
    }
    }

    fact = low? sin(0.5*arg): cos(0.5*arg);
    fact *= fact;
    
    for (j=0; j < nn/2; j++) {
    ss = sin(PI*(2*j+1)/(2*nn))*sinw;
    bw->den[j][0] = fact/(1.+ss);
    bw->den[j][1] = (1.-ss)/fact;
    }
    bw->mid = -2.*cosw/fact;

    return bw;
}

void sf_reverse (int n1, float* trace)
{
    int i1;
    float t;

    for (i1=0; i1 < n1/2; i1++) { 
        t=trace[i1];
        trace[i1]=trace[n1-1-i1];
        trace[n1-1-i1]=t;
    }
}

void sf_butter_apply (const sf_butter bw, int nx, float *x /* data [nx] */)
{
    int ix, j, nn;
    float d0, d1, d2, x0, x1, x2, y0, y1, y2;

    d1 = bw->mid;
    nn = bw->nn;

    if (nn%2) {
    d0 = bw->den[nn/2][0];
    d2 = bw->den[nn/2][1];
    x0 = y1 = 0.;
    for (ix=0; ix < nx; ix++) { 
        x1 = x0; x0 = x[ix];
        y0 = (bw->low)? 
        (x0 + x1 - d2 * y1)*d0:
        (x0 - x1 - d2 * y1)*d0;
        x[ix] = y1 = y0;
    }
    }

    for (j=0; j < nn/2; j++) {
    d0 = bw->den[j][0];
    d2 = bw->den[j][1];
    x1 = x0 = y1 = y2 = 0.;
    for (ix=0; ix < nx; ix++) { 
        x2 = x1; x1 = x0; x0 = x[ix];
        y0 = (bw->low)? 
        (x0 + 2*x1 + x2 - d1 * y1 - d2 * y2)*d0:
        (x0 - 2*x1 + x2 - d1 * y1 - d2 * y2)*d0;
        y2 = y1; x[ix] = y1 = y0;
    }
    }
}


void sf_butter_close(sf_butter bw)
{
    free(bw->den[0]);
    free(bw->den);
    free(bw);
}

void sfbandpass(float *indat, int nt,  int nr, float dt,
    bool phase, int nphi, int nplo, float fhi, float flo, 
    float *outdat){

	float *trace = calloc(nt, sizeof(float));

    	const float eps=0.0001;
    	sf_butter blo=NULL, bhi=NULL;


    	if (0. > flo) {

       		printf("Negative flo=%g",flo); exit(0);
    
    	} else {
       
       		flo *= dt;
    
    	}

    	fhi *= dt;  

    	if (flo > fhi) 
        	printf("Need flo < fhi, got flo=%g, fhi=%g",flo/dt,fhi/dt);

    	if (0.5 < fhi)
        	printf("Need fhi < Nyquist, got fhi=%g, Nyquist=%g",fhi/dt,0.5/dt);
    
    
    	if (nplo < 1)            nplo = 1;
    	if (nplo > 1 && !phase)  nplo /= 2; 

    	if (nphi < 1)            nphi = 1;
    	if (nphi > 1 && !phase)  nphi /= 2; 


    	if (flo > eps)     blo = sf_butter_init(false, flo, nplo);
    	if (fhi < 0.5-eps) bhi = sf_butter_init(true,  fhi, nphi);

       	for (int j1=0; j1<nr; j1++) {
       		
		for (int i1=0; i1<nt; i1++)
			trace[i1] = indat[i1+j1*nt];

    		if (NULL != blo) {
        		sf_butter_apply (blo, nt, trace); 

        		if (!phase) {
        			sf_reverse (nt, trace);
        			sf_butter_apply (blo, nt, trace); 
        			sf_reverse (nt, trace); 
        		}
    		}

    		if (NULL != bhi) {
        		sf_butter_apply (bhi, nt, trace); 

        		if (!phase) {
        			sf_reverse (nt, trace);
        			sf_butter_apply (bhi, nt, trace); 
        			sf_reverse (nt, trace);     
        		}
    		}
 
    		for (int i1=0; i1<nt; i1++) outdat[i1+j1*nt] = trace[i1];

    	}

    if (NULL != blo) sf_butter_close(blo);
    if (NULL != bhi) sf_butter_close(bhi);

    free(trace);
    
}

