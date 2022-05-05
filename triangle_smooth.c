// Copy from Madagascar

#include "triangle_smooth.h"

void boxconv(int nb, int nx, float *xx, float *yy){
	/*
  	inputs:       nx,  xx(i), i=1,nx      the data
        	      nb                      the box length
  	output:       yy(i),i=1,nx+nb-1       smoothed data
  	*/
  	int ny = nx+nb-1;
  	float *bb;
  	bb = calloc(ny, sizeof(float));

  	bb[0] = xx[0];
  	for (int i=1; i<nx; i++)
    		bb[i] = bb[i-1] + xx[i]; // make B(Z) = X(Z)/(1-Z)
  	
	for (int i=nx; i<ny; i++)
    		bb[i] = bb[i-1];
  	
	for (int i=0; i<nb; i++)
    		yy[i] = bb[i];
  	
	for (int i=nb; i<ny; i++)
    		yy[i] = bb[i]-bb[i-nb]; // make Y(Z) = B(Z)*(1-Z**nb)
  	
	//for (int i=0; i<ny; i++)
    	//	yy[i] = yy[i]/nb;
	for (int i=0; i<ny; i++)
    		yy[i] = yy[i];

  	free(bb);
}

void triangle(int nr, int n12, float *uu, float *vv){
	/*
  	input:        nr      rectangle width (points)
  	input:        uu(i2),i2=1,n12
  	output:       vv(i2),i2=1,n12
  	*/
  	int np = nr+n12-1, nq = nr+np-1;
  	float *pp, *qq, *tt;
  	pp = calloc(np, sizeof(float)); qq = calloc(nq, sizeof(float));
  	tt = calloc(n12, sizeof(float));
  	for (int i=0; i<n12; i++)
    		qq[i] = uu[i];
  	
	boxconv(nr, n12, qq, pp);
  	boxconv(nr, np, pp, qq);

  	for (int i=0; i<n12; i++)
    	tt[i] = qq[i+nr-1];

  	for (int i=0; i<nr-1; i++)
    	tt[i] = tt[i] + qq[nr-(i+2)];
  	
	for (int i=0; i<nr-1; i++)
    	tt[n12-(i+2)+1] = tt[n12-(i+2)+1] + qq[np+i];
  	nr *= nr;
	for (int i=0; i<n12; i++)
    	vv[i] = tt[i]/nr;

  	free(pp);
  	free(tt);
  	free(qq);

}

void triangle2(int rect1, int rect2, int nt, int n1, int n2, float *uu, float *vv){

	float *rr, *ss, *tmp_r, *tmp_s;

  	rr = (float *)calloc(n1, sizeof(float)); tmp_r = (float *)calloc(n1, sizeof(float));
  	ss = (float *)calloc(n2, sizeof(float)); tmp_s = (float *)calloc(n2, sizeof(float));
  	
	for (int i=0; i<n1*n2; i++) vv[i] = uu[i];
	
	for (int it=0; it<nt; it++){
		
		for (int i2=0; i2<n2; i2++){
			
			for (int i1=0; i1<n1; i1++)
			rr[i1] = vv[i2*n1+i1];		
		
    			triangle(rect1, n1, rr, tmp_r);
		
			for (int i1=0; i1<n1; i1++)
			vv[i2*n1+i1] = tmp_r[i1];
		}
		
  	}
	
	for (int it=0; it<nt; it++){
		
		for (int i1=0; i1<n1; i1++){
      	
			for (int i2=0; i2<n2; i2++)
        		ss[i2] = vv[i2*n1+i1];
      
			triangle(rect2, n2, ss, tmp_s);
      		
			for (int i2=0; i2<n2; i2++)
        		vv[i2*n1+i1] = tmp_s[i2];
    		}
  	}	
	
	free(rr); free(tmp_r);
	free(ss); free(tmp_s);
}


