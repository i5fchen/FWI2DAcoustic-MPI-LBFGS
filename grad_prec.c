#include "header.h"
#include "alloc.h"
#include "min_max.h"

void grad_precond_new5(fwi_grad *g, fwi_shape *bw, int *bottom){

	float **tmp_grad, **smgrad, pclip=g->perc;

  	tmp_grad = allocate2d(bw->mnx, bw->mnz); 
  	smgrad = allocate2d(bw->mnx, bw->mnz); 

  	for (int iz=0; iz<bw->mnz; iz++)
    	for (int ix=0; ix<bw->mnx; ix++)
      		tmp_grad[iz][ix] = g->body[bw->mnx*iz+ix];

  	float vmin, vmax;
  	size_t i0;  
  	i0 = argmin(tmp_grad[0], sizeof(float), bw->mnz*bw->mnx, flt_cmp );
  	vmin = *(tmp_grad[0] + i0);

	i0 = argmax(tmp_grad[0], sizeof(float), bw->mnz*bw->mnx, flt_cmp );
	vmax=*(tmp_grad[0] + i0);

	if (pclip != 0){
  
    	for (int iz=0; iz<bw->mnz; iz++)
      	for (int ix=0; ix<bw->mnx; ix++){

        	if (tmp_grad[iz][ix] > vmax*pclip) 
          		tmp_grad[iz][ix] = vmax*pclip;
        	if (tmp_grad[iz][ix] < vmin*pclip) 
          		tmp_grad[iz][ix] = vmin*pclip;

      	}
  	}

  	triangle2(g->rectx1, g->rectx1, g->repeat1, bw->mnx, bw->mnz, &tmp_grad[0][0], &smgrad[0][0]);

  	for (int iz=0; iz<bw->mnz; iz++)
    	for (int ix=0; ix<bw->mnx; ix++)
      		if (iz < bottom[ix])
        		smgrad[iz][ix] = 0;

  	triangle2(g->rectx2, g->rectx2, g->repeat2, bw->mnx, bw->mnz, &smgrad[0][0], &tmp_grad[0][0]);
  
	for (int iz=0; iz<bw->mnz; iz++)
    	for (int ix=0; ix<bw->mnx; ix++)
      		g->body[iz*bw->mnx+ix] = tmp_grad[iz][ix]*g->alpha_grad;
    
  	free2d(tmp_grad); free2d(smgrad); 

}


