#include "header.h"
#include "Mpar.h"
#include "fwi_sys.h"
#include "fwi_files_path.h"
#include "alloc.h"
#include "read_write.h"

void state_var_init(fwi_state *var, fwi_shape *shape) {
		
	memset(var->pnew[0], 0, shape->pnx*shape->pnz*sizeof(float));
	memset(var->pnow[0], 0, shape->pnx*shape->pnz*sizeof(float));
	memset(var->pold[0], 0, shape->pnx*shape->pnz*sizeof(float));
	
}	

void pml_var_init(pml_profile *pml, fwi_shape *grd_f) {
	
	memset(pml->u0[0],  0, (pml->npml+1) * grd_f->unz*sizeof(float));
	memset(pml->u1[0],  0, (pml->npml+1) * grd_f->unz*sizeof(float));
	memset(pml->v0[0],  0, (pml->npml+1) * grd_f->vnx*sizeof(float));
	memset(pml->v1[0],  0, (pml->npml+1) * grd_f->vnx*sizeof(float));
	
	memset(pml->ux0[0], 0, pml->npml * grd_f->pnz*sizeof(float));	
	memset(pml->ux1[0], 0, pml->npml * grd_f->pnz*sizeof(float));
	memset(pml->vz0[0], 0, pml->npml * grd_f->pnx*sizeof(float));
	memset(pml->vz1[0], 0, pml->npml * grd_f->pnx*sizeof(float));
}

void model_ext_fs_no(float **model, int n1, int n2, float **emodel, int en1, int en2){

	int gap1, gap2;
  	gap1 = (en1-n1)/2;
  	gap2 = (en2-n2)/2;

  	for (int i2=gap2; i2<en2-gap2; i2++){
    	for (int i1=gap1; i1<en1-gap1; i1++){
      		emodel[i2][i1] = model[i2-gap2][i1-gap1];
    	}
    
    	for (int i1=0; i1<gap1; i1++){
      		emodel[i2][i1] = model[i2-gap2][0];
    	}
    	for (int i1=en1-gap1; i1<en1; i1++){
      		emodel[i2][i1] = model[i2-gap2][n1-1];
    	}
  	}
  
  for (int i2=0; i2 <gap2; i2++){
    for (int i1=gap1; i1<en1-gap1; i1++){
      emodel[i2][i1] = model[0][i1-gap1];
      emodel[i2+en2-gap2][i1] = model[n2-1][i1-gap1];
    }
  }

  for (int i2=0; i2 < gap2; i2++){
    for (int i1=0; i1 <gap1; i1++){
      emodel[i2][i1] = model[0][0];
      emodel[i2][i1+en1-gap1] = model[0][n1-1];
      emodel[i2+en2-gap2][i1] = model[n2-1][0];
      emodel[i2+en2-gap2][i1+en1-gap1] = model[n2-1][n1-1];
    }
  }
	
 
}

void model_ext(float **model, int n1, int n2, float **emodel, int en1, int en2){
  int gap1, gap2;
  gap1 = (en1-n1)/2;
  gap2 = (en2-n2);

  //printf("%d %d\n", gap1, gap2);

  for (int i2=0; i2<en2-gap2; i2++){
    for (int i1=gap1; i1<en1-gap1; i1++){
      emodel[i2][i1] = model[i2][i1-gap1];
    }
    
    for (int i1=0; i1<gap1; i1++){
      emodel[i2][i1] = model[i2][0];
    }
    for (int i1=en1-gap1; i1<en1; i1++){
      emodel[i2][i1] = model[i2][n1-1];
    }
    
  }
  
  for (int i2=en2-gap2; i2 < en2; i2++){
    for (int i1=gap1; i1<en1-gap1; i1++){
      emodel[i2][i1] = model[n2-1][i1-gap1];
    }
  }

  for (int i2=en2-gap2; i2 < en2; i2++){
    for (int i1=en1-gap1; i1 < en1; i1++){
      emodel[i2][i1] = model[n2-1][n1-1];
    }
  }

  for (int i2=en2-gap2; i2 < en2; i2++){
    for (int i1=0; i1 < gap1; i1++){
      emodel[i2][i1] = model[n2-1][0];
    }
  }
  
  
}


void cpml_coe(float* restrict a0, float* restrict b0, const pml_base *base, const int npml, const float dx, const float dt){
 	
	const float ndx = dx*0.5; 
	int npower = base->dim;
  	
  	float *d, *alpha, thickness_pml, PIf0, slope, d0;
  
  	d     = (float *)calloc(2*npml+1, sizeof(float));
  	alpha = (float *)calloc(2*npml+1, sizeof(float));
  
  	thickness_pml = npml*dx; PIf0 = PI*base->maxF; slope = PIf0/thickness_pml;

  	d0 = -(npower + 1)*base->maxV*log(base->coefR)/(2.0*thickness_pml);
 		
    for (int i = 0; i < 2*npml+1; i++){
    
	  	d[i] = (1.0-i*ndx/thickness_pml); d[i] *= d[i]; d[i] *= d0;
      	
		alpha[i] = slope*(i*ndx);
     	
		b0[i] = exp(-(d[i]+alpha[i])*dt);
      	a0[i] = d[i]/(d[i]+alpha[i])*(b0[i]-1.0);
    
	}
 
  	free(d); free(alpha);

}

void fwi_init(fwi_plan* plan, int ishot){

	static fwi_base base = {
	
		.dim=DIM, .mm=MM, .npml=NPML, .nx=NX, .nz=NZ, .nt=NT, .nt_mem=NT_MEM, .dx=DX, .dz=DZ, .dt=DT, .free_surface=FREE_SURFACE
	};
	
	static fwi_shape gfw, gbw;
	
	gbw.mnx = base.nx;  gbw.mnz = base.nz;
	
	gbw.pnx = gbw.mnx; 	   gbw.pnz = gbw.mnz;
	gbw.unx = gbw.pnx + 1; gbw.unz = gbw.pnz;
	gbw.vnx = gbw.pnx; 	   gbw.vnz = gbw.pnz + 1;

	gfw.mnx = gbw.mnx + 2*base.npml; gfw.mnz = gbw.mnz + base.npml;
	gfw.pnx = gbw.pnx + 2*base.npml; gfw.pnz = gbw.pnz + base.npml;
	gfw.unx = gbw.unx + 2*base.npml; gfw.unz = gbw.unz + base.npml;
	gfw.vnx = gbw.vnx + 2*base.npml; gfw.vnz = gbw.vnz + base.npml;
	
	if (0==base.free_surface) {

		gfw.mnz = gbw.mnz + 2*base.npml;
        gfw.pnz = gbw.pnz + 2*base.npml;
		gfw.unz = gbw.unz + 2*base.npml;
        gfw.vnz = gbw.vnz + 2*base.npml;
	}
	
	plan->grid = (fwi_specs *) malloc(sizeof(fwi_specs));

	plan->grid->base = &base;

	plan->grid->f = &gfw;
	plan->grid->b = &gbw;

	/* 0000000000000000000000000000000000000000000000000000000000000 */
	/* 0000000000000000000000000000000000000000000000000000000000000 */

	fwi_model *model_t = (fwi_model *)malloc(sizeof(fwi_model));
	
	model_t->md = allocate2d(gbw.mnx, gbw.mnz);
	model_t->kp = allocate2d(gbw.mnx, gbw.mnz);
	
	model_t->ekp   = allocate2d(gfw.mnx, gfw.mnz);
	model_t->btry  = (float *)malloc(gbw.mnx*sizeof(float)); 
	
	model_t->ibtry  = (int *)malloc(gbw.mnx*sizeof(int)); 
	
	model_t->alpha_btry = TUNE_BTRY;

	plan->model = model_t;
		
	/* 0000000000000000000000000000000000000000000000000000000000000 */
	/* 0000000000000000000000000000000000000000000000000000000000000 */

	static fwi_msf msf_t = {.nt=NT0, .nr=NR, .jump=DAT_JUMP, .rectx=DAT_RECT_X, .rectt=DAT_RECT_T, .times=DAT_SM_TIMES, .dr=DR, .dt=DT0, .os0=OFFSET0, .os1=OFFSET1, .alpha_fbt0=FBT_UP, .alpha_fbt1=FBT_DOWN, .alpha_msf=MSF_SCALE};
	if (( (msf_t.nt-1)*msf_t.jump != base.nt-1 || fabs(msf_t.dt - base.dt*msf_t.jump)>1e-3       )  )	{exit(0);}
	msf_t.obs0 = allocate2d(msf_t.nt, msf_t.nr);
	
	msf_t.obs   = allocate2d(msf_t.nt, msf_t.nr);
	msf_t.syn   = allocate2d(msf_t.nt, msf_t.nr);
	
	msf_t.obs_t = allocate2d(msf_t.nt, msf_t.nr);
	msf_t.syn_t = allocate2d(msf_t.nt, msf_t.nr);
	
	msf_t.fbt  = (float *)malloc(msf_t.nr*sizeof(float));
	msf_t.ifbt = (int   *)malloc(msf_t.nr*sizeof(int  ));

	msf_t.val = (float *)malloc(2 * sizeof(float));	

	plan->msf = &msf_t;
	
	/* 0000000000000000000000000000000000000000000000000000000000000 */
	/* 0000000000000000000000000000000000000000000000000000000000000 */

	fwi_sou *sou_t = (fwi_sou *)malloc(sizeof(fwi_sou));

	sou_t->iarea   = 1.0/(base.dx*base.dz);	
	sou_t->ixyz_sr   =  (int *) malloc((1+msf_t.nr)*base.dim*sizeof(int));
	sou_t->xyz_sr  = (float *)malloc((1+msf_t.nr)*base.dim*sizeof(float));
	sou_t->wav0 = (float *)malloc(base.nt*sizeof(float));
	sou_t->wav  = (float *)malloc(base.nt*sizeof(float));
	
	sou_t->res = allocate2d(base.nt, msf_t.nr);

	plan->sou = sou_t;
	
	/* 0000000000000000000000000000000000000000000000000000000000000 */
	/* 0000000000000000000000000000000000000000000000000000000000000 */
	static fwi_state fw, bw; 
	
	bw.pold = allocate2d(gbw.pnx, gbw.pnz);
	bw.pnow = allocate2d(gbw.pnx, gbw.pnz);
	bw.pnew = allocate2d(gbw.pnx, gbw.pnz);
	
	bw.u = allocate2d(gbw.unx, gbw.unz);
	bw.v = allocate2d(gbw.vnx, gbw.vnz);
	
	bw.ux = allocate2d(gbw.pnx, gbw.pnz);
	bw.vz = allocate2d(gbw.pnx, gbw.pnz);
	
	fw.pold = allocate2d(gfw.pnx, gfw.pnz);
	fw.pnow = allocate2d(gfw.pnx, gfw.pnz);
	fw.pnew = allocate2d(gfw.pnx, gfw.pnz);
	
	fw.u = allocate2d(gfw.unx, gfw.unz);
	fw.v = allocate2d(gfw.vnx, gfw.vnz);
	
	fw.ux = allocate2d(gfw.pnx, gfw.pnz);
	fw.vz = allocate2d(gfw.pnx, gfw.pnz);

	plan->bw = &bw;
	plan->fw = &fw;
	
	/* 0000000000000000000000000000000000000000000000000000000000000 */
	/* 0000000000000000000000000000000000000000000000000000000000000 */
	int npml = base.npml;

	pml_base meta = {2, MAX_F, MAX_V, COEF_R};

	pml_profile *pml = (pml_profile *)malloc(sizeof(pml_profile));

	pml->npml = npml;
	
	for (int i=0; i<2; i++) {
		
		pml->a0[i] = (float *) malloc((2*npml+1)*sizeof(float));
		pml->a1[i] = (float *) malloc((2*npml+1)*sizeof(float));
		pml->b0[i] = (float *) malloc((2*npml+1)*sizeof(float));
		pml->b1[i] = (float *) malloc((2*npml+1)*sizeof(float));
	
	}
	
	cpml_coe(pml->a0[0], pml->b0[0], &meta, npml, base.dx, base.dt); //a0x b0x
	cpml_coe(pml->a0[1], pml->b0[1], &meta, npml, base.dz, base.dt); //a0z b0z

	for (int i=0; i<2*npml+1; i++) {

		pml->a1[0][i] = pml->a0[0][2*npml-i];
		pml->b1[0][i] = pml->b0[0][2*npml-i];
		                 
		pml->a1[1][i] = pml->a0[1][2*npml-i];
		pml->b1[1][i] = pml->b0[1][2*npml-i];

	}

	/* 0000000000000000000000000000000000000000000000000000000000000 */
	/* 0000000000000000000000000000000000000000000000000000000000000 */
	
	pml->u0 = allocate2d(npml+1, gfw.unz); 
	pml->u1 = allocate2d(npml+1, gfw.unz);
  	pml->v0 = allocate2d(gfw.vnx, npml+1);
  	pml->v1 = allocate2d(gfw.vnx, npml+1);

  	pml->ux0 = allocate2d(npml, gfw.pnz); 
	pml->ux1 = allocate2d(npml, gfw.pnz);
  	pml->vz0 = allocate2d(gfw.pnx, npml);
  	pml->vz1 = allocate2d(gfw.pnx, npml);
	
	plan->pml = pml;
	/* 0000000000000000000000000000000000000000000000000000000000000 */
	/* 0000000000000000000000000000000000000000000000000000000000000 */

	static fwi_fdcoe coe;
	
	coe.ix = 1.0/base.dx;
	coe.iz = 1.0/base.dz;

	coe.c0x =  0.1129042e1 *coe.ix;
	coe.c1x = -0.4301412e-1*coe.ix;

	coe.c0z =  0.1129042e1 *coe.iz;
	coe.c1z = -0.4301412e-1*coe.iz;

	coe.iz840 = coe.iz/840.0;
	coe.iz24  = coe.iz/ 24.0;

	plan->coe = &coe;
	
	/* 0000000000000000000000000000000000000000000000000000000000000 */
	/* 0000000000000000000000000000000000000000000000000000000000000 */
	fwi_files_path path;
	
	path.common = (char *)malloc(300*sizeof(char));
	
	path.shots  = (char *)malloc(300*sizeof(char));
	path.fbt    = (char *)malloc(300*sizeof(char));
	path.xyz_sr = (char *)malloc(300*sizeof(char));
	
	path.init_mdl  = (char *)malloc(300*sizeof(char));
	path.btry      = (char *)malloc(300*sizeof(char));
	path.wavelet   = (char *)malloc(300*sizeof(char));
		
	sprintf(path.common, COMM);
	
	sprintf(path.xyz_sr, XYZ_SR, path.common, ishot);
	sprintf(path.shots,  SHOTS,  path.common, ishot);
	sprintf(path.fbt,    FBT,    path.common, ishot);
	
	sprintf(path.init_mdl, INIT_MDL, path.common);
	sprintf(path.btry,     BTRY,     path.common);
	sprintf(path.wavelet,   WAVELET_est,  path.common, (ishot/STEP)*STEP, STEP*(ishot/STEP)+STEP-1);
	if (!(0 == SHOTS_AVAIL))
	read_from_file(msf_t.obs0[0], sizeof(float), msf_t.nr * msf_t.nt, path.shots, 1);
	read_from_file(sou_t->xyz_sr, sizeof(float), base.dim*(msf_t.nr+1), path.xyz_sr, 1);
	
	memset(msf_t.fbt, 0, sizeof(float)*msf_t.nr);

	if (!(0 == FBT_AVAIL))
	read_from_file(msf_t.fbt, sizeof(float), msf_t.nr, path.fbt, 1);
	
	read_from_file(model_t->md[0], sizeof(float), gbw.mnz*gbw.mnx, path.init_mdl,1);
	read_from_file(model_t->btry,  sizeof(float), gbw.mnx, path.btry, 1);
	read_from_file(sou_t->wav0, sizeof(float), base.nt, path.wavelet, 1);
	for (int i=0; i<base.nt; i++) sou_t->wav0[i] *= WAV_SCALE;
	/* 0000000000000000000000000000000000000000000000000000000000000 */
	/* 0000000000000000000000000000000000000000000000000000000000000 */

	for (int i=0; i<base.nx; i++)
		model_t->ibtry[i] = model_t->btry[i]/base.dz-0.5;	
	
	for (int i=0; i<msf_t.nr; i++)
		msf_t.ifbt[i] = msf_t.fbt[i]/msf_t.dt;	
	
	sou_t->ixyz_sr[0] = (int)round(sou_t->xyz_sr[0]/base.dx);
	sou_t->ixyz_sr[1] = (int)round(sou_t->xyz_sr[1]/base.dz-0.5);

	for(int i=0; i<msf_t.nr; i++){

		sou_t->ixyz_sr[base.dim+2*i]   = (int)round(sou_t->xyz_sr[base.dim+2*i]  / base.dx );
		sou_t->ixyz_sr[base.dim+2*i+1] = (int)round(sou_t->xyz_sr[base.dim+2*i+1]/base.dz - 0.5);

	}

	/* 0000000000000000000000000000000000000000000000000000000000000 */
	/* 0000000000000000000000000000000000000000000000000000000000000 */
	
	static fwi_grad grad_t = {.end_t=G_END_T, .start_t=G_START_T, .alpha_grad = G_SCALE, .perc=G_PERC, .jump=G_JUMP, .rectx1=G_SM_RECTX1, .rectz1=G_SM_RECTZ1, .repeat1=G_SM_TIMES1, .rectx2=G_SM_RECTX2, .rectz2=G_SM_RECTZ2, .repeat2=G_SM_TIMES2};
	
	grad_t.id=ishot; 
	
	grad_t.body = (float *)malloc(gbw.mnx*gbw.mnz*sizeof(float));	
	plan->grad = &grad_t;

	/* 0000000000000000000000000000000000000000000000000000000000000 */
	/* 0000000000000000000000000000000000000000000000000000000000000 */
	plan->bc_ud = allocate3d(gbw.pnx, base.dim*(2*base.mm-1), base.nt_mem);
	plan->bc_lr = allocate3d(base.dim*(2*base.mm-1), gbw.pnz, base.nt_mem);
	sprintf(plan->path_bd_wr, PATH_BD_WR, ishot);	

	static int offset;
	
	offset  = gbw.pnx*base.dim*(2*base.mm-1) * base.nt_mem * sizeof(float);
    offset += gbw.pnz*base.dim*(2*base.mm-1) * base.nt_mem * sizeof(float);
	
	plan->offset = offset;

}

void rwi_init(rwi_plan* plan, int ishot){

	static fwi_base base = {
	
		.dim=DIM, .mm=MM, .npml=NPML, .nx=NX, .nz=NZ, .nt=NT, .nt_mem=NT_MEM, .dx=DX, .dz=DZ, .dt=DT, .free_surface=FREE_SURFACE
	};
	
	static fwi_shape gfw, gbw; //grid for forward and backward propagation
	
	gbw.mnx = base.nx;  gbw.mnz = base.nz;
	
	gbw.pnx = gbw.mnx; 	   gbw.pnz = gbw.mnz;
	gbw.unx = gbw.pnx + 1; gbw.unz = gbw.pnz;
	gbw.vnx = gbw.pnx; 	   gbw.vnz = gbw.pnz + 1;

	gfw.mnx = gbw.mnx + 2*base.npml; gfw.mnz = gbw.mnz + base.npml;
	gfw.pnx = gbw.pnx + 2*base.npml; gfw.pnz = gbw.pnz + base.npml;
	gfw.unx = gbw.unx + 2*base.npml; gfw.unz = gbw.unz + base.npml;
	gfw.vnx = gbw.vnx + 2*base.npml; gfw.vnz = gbw.vnz + base.npml;

	if (0==base.free_surface) {

		gfw.mnz = gbw.mnz + 2*base.npml;
        gfw.pnz = gbw.pnz + 2*base.npml;
		gfw.unz = gbw.unz + 2*base.npml;
        gfw.vnz = gbw.vnz + 2*base.npml;
	}

	plan->grid = (fwi_specs *) malloc(sizeof(fwi_specs));

	plan->grid->base = &base;

	plan->grid->f = &gfw;
	plan->grid->b = &gbw;

	/* 0000000000000000000000000000000000000000000000000000000000000 */
	/* 0000000000000000000000000000000000000000000000000000000000000 */

	rwi_model *model_t = (rwi_model *)malloc(sizeof(rwi_model));
	
	model_t->m0 = allocate2d(gbw.mnx, gbw.mnz);
	model_t->m1 = allocate2d(gbw.mnx, gbw.mnz);
	
	model_t->kp_0 	  = allocate2d(gbw.mnx, gbw.mnz);
	model_t->kp_total = allocate2d(gbw.mnx, gbw.mnz);
	
	model_t->ekp_0 	   = allocate2d(gfw.mnx, gfw.mnz);
	model_t->ekp_total = allocate2d(gfw.mnx, gfw.mnz);
	
	model_t->btry  = (float *)malloc(gbw.mnx*sizeof(float)); 
	
	model_t->ibtry  = (int *)malloc(gbw.mnx*sizeof(int)); 
	
	model_t->alpha_btry = TUNE_BTRY;

	plan->model = model_t;
		
	/* 0000000000000000000000000000000000000000000000000000000000000 */
	/* 0000000000000000000000000000000000000000000000000000000000000 */

	static fwi_msf msf_t = {.nt=NT0, .nr=NR, .jump=DAT_JUMP, .rectx=DAT_RECT_X, .rectt=DAT_RECT_T, .times=DAT_SM_TIMES, .dr=DR, .dt=DT0, .os0=OFFSET0, .os1=OFFSET1, .alpha_fbt0=FBT_UP, .alpha_fbt1=FBT_DOWN, .alpha_msf=MSF_SCALE};
	
	if (!((msf_t.nt-1)*msf_t.jump == base.nt-1 && fabs(msf_t.dt - base.dt*msf_t.jump)<1e-7))	{exit(0);}

	msf_t.obs0 = allocate2d(msf_t.nt, msf_t.nr);
	
	msf_t.obs   = allocate2d(msf_t.nt, msf_t.nr);
	msf_t.syn   = allocate2d(msf_t.nt, msf_t.nr);
	msf_t.syn0   = allocate2d(msf_t.nt, msf_t.nr);
	
	msf_t.obs_t = allocate2d(msf_t.nt, msf_t.nr);
	msf_t.syn_t = allocate2d(msf_t.nt, msf_t.nr);
	
	msf_t.fbt  = (float *)malloc(msf_t.nr*sizeof(float));
	msf_t.ifbt = (int   *)malloc(msf_t.nr*sizeof(int  ));

	msf_t.val = (float *)malloc(2 * sizeof(float));	

	plan->msf = &msf_t;
	
	/* 0000000000000000000000000000000000000000000000000000000000000 */
	/* 0000000000000000000000000000000000000000000000000000000000000 */

	fwi_sou *sou_t = (fwi_sou *)malloc(sizeof(fwi_sou));

	sou_t->iarea   = 1.0/(base.dx*base.dz);	
	sou_t->ixyz_sr   =  (int *) malloc((1+msf_t.nr)*base.dim*sizeof(int));
	sou_t->xyz_sr  = (float *)malloc((1+msf_t.nr)*base.dim*sizeof(float));
	sou_t->wav0 = (float *)malloc(base.nt*sizeof(float));
	sou_t->wav  = (float *)malloc(base.nt*sizeof(float));
	
	sou_t->res = allocate2d(base.nt, msf_t.nr);

	plan->sou = sou_t;
	
	/* 0000000000000000000000000000000000000000000000000000000000000 */
	/* 0000000000000000000000000000000000000000000000000000000000000 */
	fwi_state *fw_inc_t = (fwi_state *) malloc(sizeof(fwi_state));
	fwi_state *bw_inc_t = (fwi_state *) malloc(sizeof(fwi_state));
	
	fwi_state *fw_per_t = (fwi_state *) malloc(sizeof(fwi_state));
	fwi_state *bw_per_t = (fwi_state *) malloc(sizeof(fwi_state));
	/////////////////////////////////////////////////////////
	bw_inc_t->pold = allocate2d(gbw.pnx, gbw.pnz);
	bw_inc_t->pnow = allocate2d(gbw.pnx, gbw.pnz);
	bw_inc_t->pnew = allocate2d(gbw.pnx, gbw.pnz);
	
	bw_inc_t->u = allocate2d(gbw.unx, gbw.unz);
	bw_inc_t->v = allocate2d(gbw.vnx, gbw.vnz);
	
	bw_inc_t->ux = allocate2d(gbw.pnx, gbw.pnz);
	bw_inc_t->vz = allocate2d(gbw.pnx, gbw.pnz);
	/////////////////////////////////////////////////////////
	fw_inc_t->pold = allocate2d(gfw.pnx, gfw.pnz);
	fw_inc_t->pnow = allocate2d(gfw.pnx, gfw.pnz);
	fw_inc_t->pnew = allocate2d(gfw.pnx, gfw.pnz);
	
	fw_inc_t->u = allocate2d(gfw.unx, gfw.unz);
	fw_inc_t->v = allocate2d(gfw.vnx, gfw.vnz);
	
	fw_inc_t->ux = allocate2d(gfw.pnx, gfw.pnz);
	fw_inc_t->vz = allocate2d(gfw.pnx, gfw.pnz);
	/////////////////////////////////////////////////////////
	bw_per_t->pold = allocate2d(gbw.pnx, gbw.pnz);
	bw_per_t->pnow = allocate2d(gbw.pnx, gbw.pnz);
	bw_per_t->pnew = allocate2d(gbw.pnx, gbw.pnz);
	
	bw_per_t->u = allocate2d(gbw.unx, gbw.unz);
	bw_per_t->v = allocate2d(gbw.vnx, gbw.vnz);
	
	bw_per_t->ux = allocate2d(gbw.pnx, gbw.pnz);
	bw_per_t->vz = allocate2d(gbw.pnx, gbw.pnz);
	/////////////////////////////////////////////////////////
	fw_per_t->pold = allocate2d(gfw.pnx, gfw.pnz);
	fw_per_t->pnow = allocate2d(gfw.pnx, gfw.pnz);
	fw_per_t->pnew = allocate2d(gfw.pnx, gfw.pnz);
	
	fw_per_t->u = allocate2d(gfw.unx, gfw.unz);
	fw_per_t->v = allocate2d(gfw.vnx, gfw.vnz);
	
	fw_per_t->ux = allocate2d(gfw.pnx, gfw.pnz);
	fw_per_t->vz = allocate2d(gfw.pnx, gfw.pnz);
	//////////////////////////////////////////////////////////

	plan->fw_inc = fw_inc_t;
	plan->fw_per = fw_per_t;
	plan->bw_inc = bw_inc_t;
	plan->bw_per = bw_per_t;
	
	/* 0000000000000000000000000000000000000000000000000000000000000 */
	/* 0000000000000000000000000000000000000000000000000000000000000 */
	int npml = base.npml;

	pml_base meta = {2, MAX_F, MAX_V, COEF_R};

	pml_profile *pml_inc = (pml_profile *)malloc(sizeof(pml_profile));

	pml_inc->npml = npml;
	
	for (int i=0; i<2; i++) {
		
		pml_inc->a0[i] = (float *) malloc((2*npml+1)*sizeof(float));
		pml_inc->a1[i] = (float *) malloc((2*npml+1)*sizeof(float));
		pml_inc->b0[i] = (float *) malloc((2*npml+1)*sizeof(float));
		pml_inc->b1[i] = (float *) malloc((2*npml+1)*sizeof(float));
	
	}
	
	cpml_coe(pml_inc->a0[0], pml_inc->b0[0], &meta, npml, base.dx, base.dt); //a0x b0x
	cpml_coe(pml_inc->a0[1], pml_inc->b0[1], &meta, npml, base.dz, base.dt); //a0z b0z

	for (int i=0; i<2*npml+1; i++) {

		pml_inc->a1[0][i] = pml_inc->a0[0][2*npml-i];
		pml_inc->b1[0][i] = pml_inc->b0[0][2*npml-i];
		                 
		pml_inc->a1[1][i] = pml_inc->a0[1][2*npml-i];
		pml_inc->b1[1][i] = pml_inc->b0[1][2*npml-i];

	}

	/* 0000000000000000000000000000000000000000000000000000000000000 */
	/* 0000000000000000000000000000000000000000000000000000000000000 */
	
	pml_inc->u0 = allocate2d(npml+1, gfw.unz); 
	pml_inc->u1 = allocate2d(npml+1, gfw.unz);
  	pml_inc->v0 = allocate2d(gfw.vnx, npml+1);
  	pml_inc->v1 = allocate2d(gfw.vnx, npml+1);

  	pml_inc->ux0 = allocate2d(npml, gfw.pnz); 
	pml_inc->ux1 = allocate2d(npml, gfw.pnz);
  	pml_inc->vz0 = allocate2d(gfw.pnx, npml);
  	pml_inc->vz1 = allocate2d(gfw.pnx, npml);
	
	pml_profile *pml_per = (pml_profile *)malloc(sizeof(pml_profile));
	
	pml_per->npml = npml;
	
	for (int i=0; i<2; i++) {
		
		pml_per->a0[i]=pml_inc->a0[i]; 
		pml_per->a1[i]=pml_inc->a1[i];
		pml_per->b0[i]=pml_inc->b0[i];
		pml_per->b1[i]=pml_inc->b1[i];
	
	}

	pml_per->u0 = allocate2d(npml+1, gfw.unz); 
	pml_per->u1 = allocate2d(npml+1, gfw.unz);
 	pml_per->v0 = allocate2d(gfw.vnx, npml+1);
 	pml_per->v1 = allocate2d(gfw.vnx, npml+1);

  	pml_per->ux0 = allocate2d(npml, gfw.pnz); 
	pml_per->ux1 = allocate2d(npml, gfw.pnz);
  	pml_per->vz0 = allocate2d(gfw.pnx, npml);
  	pml_per->vz1 = allocate2d(gfw.pnx, npml);

	plan->pml_per = pml_per;
	plan->pml_inc = pml_inc;
/* 0000000000000000000000000000000000000000000000000000000000000 */
	/* 0000000000000000000000000000000000000000000000000000000000000 */

	static fwi_fdcoe coe;
	
	coe.ix = 1.0/base.dx;
	coe.iz = 1.0/base.dz;

	coe.c0x =  0.1129042e1 *coe.ix;
	coe.c1x = -0.4301412e-1*coe.ix;

	coe.c0z =  0.1129042e1 *coe.iz;
	coe.c1z = -0.4301412e-1*coe.iz;

	coe.iz840 = coe.iz/840.0;
	coe.iz24  = coe.iz/ 24.0;

	plan->coe = &coe;
	
	/* 0000000000000000000000000000000000000000000000000000000000000 */
	/* 0000000000000000000000000000000000000000000000000000000000000 */
	rwi_files_path path;
	
	path.common = (char *)malloc(300*sizeof(char));
	
	path.shots  = (char *)malloc(300*sizeof(char));
	path.fbt    = (char *)malloc(300*sizeof(char));
	path.xyz_sr = (char *)malloc(300*sizeof(char));
	
	path.init_mdl_0  = (char *)malloc(300*sizeof(char));
	path.init_mdl_1  = (char *)malloc(300*sizeof(char));
	path.btry      = (char *)malloc(300*sizeof(char));
	path.wavelet   = (char *)malloc(300*sizeof(char));
		
	sprintf(path.common, COMM);
	
	sprintf(path.xyz_sr, XYZ_SR, path.common, ishot);
	sprintf(path.shots,  SHOTS,  path.common, ishot);
	sprintf(path.fbt,    FBT,    path.common, ishot);
	
	sprintf(path.init_mdl_0, INIT_MDL_0, path.common);
	sprintf(path.init_mdl_1, INIT_MDL_1, path.common);
	
	sprintf(path.btry,     BTRY,     path.common);
	sprintf(path.wavelet,   WAVELET,  path.common);
	if (!(0 == SHOTS_AVAIL))
	read_from_file(msf_t.obs0[0], sizeof(float), msf_t.nr * msf_t.nt, path.shots, 1);
	read_from_file(sou_t->xyz_sr, sizeof(float), base.dim*(msf_t.nr+1), path.xyz_sr, 1);
	if (!(0 == FBT_AVAIL))
	read_from_file(msf_t.fbt, sizeof(float), msf_t.nr, path.fbt, 1);
	
	read_from_file(model_t->m0[0], sizeof(float), gbw.mnz*gbw.mnx, path.init_mdl_0,1);
	read_from_file(model_t->m1[0], sizeof(float), gbw.mnz*gbw.mnx, path.init_mdl_1,1);
	read_from_file(model_t->btry,  sizeof(float), gbw.mnx, path.btry, 1);
	read_from_file(sou_t->wav0, sizeof(float), base.nt, path.wavelet, 1);
	for (int i=0; i<base.nt; i++) sou_t->wav0[i] *= WAV_SCALE;
	/* 0000000000000000000000000000000000000000000000000000000000000 */
	/* 0000000000000000000000000000000000000000000000000000000000000 */

	for (int i=0; i<base.nx; i++)
		model_t->ibtry[i] = model_t->btry[i]/base.dz-0.5;	
	
	for (int i=0; i<msf_t.nr; i++)
		msf_t.ifbt[i] = msf_t.fbt[i]/msf_t.dt;	
	
	sou_t->ixyz_sr[0] = (int)round(sou_t->xyz_sr[0]/base.dx);
	sou_t->ixyz_sr[1] = (int)round(sou_t->xyz_sr[1]/base.dz-0.5);

	for(int i=0; i<msf_t.nr; i++){

		sou_t->ixyz_sr[base.dim+2*i]   = (int)round(sou_t->xyz_sr[base.dim+2*i]  / base.dx );
		sou_t->ixyz_sr[base.dim+2*i+1] = (int)round(sou_t->xyz_sr[base.dim+2*i+1]/base.dz - 0.5);

	}

	/* 0000000000000000000000000000000000000000000000000000000000000 */
	/* 0000000000000000000000000000000000000000000000000000000000000 */
	
	static fwi_grad grad_t = {.end_t=G_END_T, .start_t=G_START_T, .alpha_grad = G_SCALE, .perc=G_PERC, .jump=G_JUMP, .rectx1=G_SM_RECTX1, .rectz1=G_SM_RECTZ1, .repeat1=G_SM_TIMES1, .rectx2=G_SM_RECTX2, .rectz2=G_SM_RECTZ2, .repeat2=G_SM_TIMES2};
	
	grad_t.id=ishot; 
	
	grad_t.body = (float *)malloc(gbw.mnx*gbw.mnz*sizeof(float));	
	plan->grad = &grad_t;

	/* 0000000000000000000000000000000000000000000000000000000000000 */
	/* 0000000000000000000000000000000000000000000000000000000000000 */
	plan->bc_ud_inc = allocate3d(gbw.pnx, base.dim*(2*base.mm-1), base.nt_mem);
	plan->bc_lr_inc = allocate3d(base.dim*(2*base.mm-1), gbw.pnz, base.nt_mem);
	sprintf(plan->path_bd0_wr, PATH_BD0_WR, ishot);	

	plan->bc_ud_per = allocate3d(gbw.pnx, base.dim*(2*base.mm-1), base.nt_mem);
	plan->bc_lr_per = allocate3d(base.dim*(2*base.mm-1), gbw.pnz, base.nt_mem);
	sprintf(plan->path_bd1_wr, PATH_BD1_WR, ishot);	

	static int offset;
	
	offset  = gbw.pnx*base.dim*(2*base.mm-1) * base.nt_mem * sizeof(float);
    offset += gbw.pnz*base.dim*(2*base.mm-1) * base.nt_mem * sizeof(float);
	
	plan->offset = offset;

}


void a_lap_opt_2d(

	float *restrict *restrict p, 
	float *restrict *restrict u, 	//dpdx
	float *restrict *restrict v,	//dpdz
	float *restrict *restrict ux, 	//dudx=dp2dx2
	float *restrict *restrict vz, 	//dvdz=dp2dz2
 	
	const fwi_fdcoe *restrict coe, 
	const fwi_shape *restrict grd)

{   
	
	#pragma omp parallel for  
    for (int iz = 2; iz < grd->unz-2;   iz++)
    for (int ix = 2; ix < grd->unx-2; ix++)
    	u[iz][ix] = coe->c0x*(p[iz][ix] - p[iz][ix-1]) + coe->c1x*(p[iz][ix+1] - p[iz][ix-2]);
	    
	    
	#pragma omp parallel for 
    for (int iz = 2; iz < grd->vnz-2; iz++)
    for (int ix = 2; ix < grd->vnx-2;   ix++)
    	v[iz][ix] = coe->c0z*(p[iz][ix] - p[iz-1][ix])+coe->c1z*(p[iz+1][ix] - p[iz-2][ix]);
    
	   
    #pragma omp parallel for 
    for (int iz = 3; iz < grd->pnz-3;   iz++)    
	for (int ix = 3; ix < grd->pnx-3; ix++)
	    ux[iz][ix] = coe->c0x*(u[iz][ix+1] - u[iz][ix])+coe->c1x*(u[iz][ix+2] - u[iz][ix-1]);
	

    #pragma omp parallel for  
    for (int iz = 3; iz < grd->pnz-3; iz++)
    for (int ix = 3; ix < grd->pnx-3;   ix++)
    	vz[iz][ix] = coe->c0z*(v[iz+1][ix] - v[iz][ix])+coe->c1z*(v[iz+2][ix] - v[iz-1][ix]);
     

}

void a_lap_opt_2d_pml(

	float *restrict *restrict p, 
	float *restrict *restrict u, 	//dpdx
	float *restrict *restrict v,	//dpdz
	float *restrict *restrict ux, 	//dudx=dp2dx2
	float *restrict *restrict vz, 	//dvdz=dp2dz2
 	
	const fwi_fdcoe *restrict coe, 
	const fwi_shape *restrict grd, 
	const pml_profile   *restrict pml)

{   
	
	#pragma omp parallel for  
    for (int iz = 0; iz < grd->unz;   iz++)
    for (int ix = 2; ix < grd->unx-2; ix++)
    	u[iz][ix] = coe->c0x*(p[iz][ix] - p[iz][ix-1]) + coe->c1x*(p[iz][ix+1] - p[iz][ix-2]);
	    
	#pragma omp parallel for  
    for (int iz = 0; iz < grd->unz; iz++){
		u[iz][1]           = coe->ix*(p[iz][1]         -p[iz][0]         );		
		u[iz][grd->unx-2]  = coe->ix*(p[iz][grd->pnx-1]-p[iz][grd->pnx-2]);		
	}
    
	#pragma omp parallel for  
    for (int iz = 0; iz < grd->unz;    iz++)
    for (int ix = 0; ix < pml->npml+1; ix++){
        
    	pml->u0[iz][ix] = pml->b0[0][2*ix]*pml->u0[iz][ix]+pml->a0[0][2*ix] * u[iz][ix];       
		u[iz][ix] += pml->u0[iz][ix];
        
        pml->u1[iz][ix] = pml->b1[0][2*ix]*pml->u1[iz][ix]+pml->a1[0][2*ix] * u[iz][ix+grd->unx-pml->npml-1]; 
		u[iz][ix+grd->unx-pml->npml-1] += pml->u1[iz][ix];
    }
    
	#pragma omp parallel for 
    for (int iz = 2; iz < grd->vnz-2; iz++)
    for (int ix = 0; ix < grd->vnx;   ix++)
    	v[iz][ix] = coe->c0z*(p[iz][ix] - p[iz-1][ix])+coe->c1z*(p[iz+1][ix] - p[iz-2][ix]);
    
	#pragma omp parallel for 
    for (int ix = 0; ix < grd->vnx; ix++){
        
     	v[0][ix] = ( 3675.0*p[0][ix]-1225.0*p[1][ix]+441.0*p[2][ix]-75.0*p[3][ix])*coe->iz840;       
        
        v[1][ix] = (-1085.0*p[0][ix]+1015.0*p[1][ix]- 63.0*p[2][ix] +5.0*p[3][ix])*coe->iz840;
    
		v[grd->vnz-2][ix] = (p[grd->pnz-1][ix]-p[grd->pnz-2][ix])*coe->iz;
 	} 
    
    #pragma omp parallel for 
    for (int iz = 0; iz < pml->npml+1; iz++)
    for (int ix = 0; ix < grd->vnx;    ix++){
          
    	pml->v1[iz][ix] = pml->b1[1][2*iz]*pml->v1[iz][ix] + pml->a1[1][2*iz] * v[iz+grd->vnz-pml->npml-1][ix]; 
		v[iz+grd->vnz-pml->npml-1][ix] += pml->v1[iz][ix];
          
    }

    #pragma omp parallel for 
    for (int iz = 0; iz < grd->pnz;   iz++)    
	for (int ix = 1; ix < grd->pnx-1; ix++)
	    ux[iz][ix] = coe->c0x*(u[iz][ix+1] - u[iz][ix])+coe->c1x*(u[iz][ix+2] - u[iz][ix-1]);
	

    #pragma omp parallel for  
    for (int iz = 0; iz < grd->pnz; iz++){
		ux[iz][0] = (u[iz][1]-u[iz][0])*coe->ix;

		ux[iz][grd->pnx-1] = (u[iz][grd->unx-1]-u[iz][grd->unx-2])*coe->ix;
    }

    #pragma omp parallel for 
    for (int iz = 0; iz < grd->pnz; iz++)
      for (int ix = 0; ix < pml->npml; ix++){
       
          pml->ux0[iz][ix] = pml->b0[0][2*ix+1]*pml->ux0[iz][ix] + pml->a0[0][2*ix+1]*ux[iz][ix];
          ux[iz][ix] += pml->ux0[iz][ix];
       
          pml->ux1[iz][ix] = pml->b1[0][2*ix+1]*pml->ux1[iz][ix] + pml->a1[0][2*ix+1]*ux[iz][ix+grd->pnx-pml->npml];
          ux[iz][ix+grd->pnx-pml->npml] += pml->ux1[iz][ix];
      }
      
	
    #pragma omp parallel for  
    for (int ix = 0; ix < grd->pnx; ix++){
      vz[0][ix] = (-22.*v[0][ix]+17.*v[1][ix]+9.*v[2][ix]-5.*v[3][ix]+v[4][ix])*coe->iz24;
      vz[grd->pnz-1][ix] = (v[grd->vnz-1][ix]-v[grd->vnz-2][ix])*coe->iz;
	}         

    #pragma omp parallel for  
    for (int iz = 1; iz < grd->pnz-1; iz++)
    for (int ix = 0; ix < grd->pnx;   ix++)
    	vz[iz][ix] = coe->c0z*(v[iz+1][ix] - v[iz][ix])+coe->c1z*(v[iz+2][ix] - v[iz-1][ix]);
     
    #pragma omp parallel for  
    for (int iz = 0; iz < pml->npml; iz++)
    for (int ix = 0; ix < grd->pnx;  ix++){
    	pml->vz1[iz][ix] = pml->b1[1][2*iz+1]*pml->vz1[iz][ix] + pml->a1[1][2*iz+1]*vz[iz+grd->pnz-pml->npml][ix];
        vz[iz+grd->pnz-pml->npml][ix] += pml->vz1[iz][ix];
    }
 
}

void time_integral(float* restrict * restrict pnew, 
			  float* restrict * restrict pnow, 
		      float* restrict * restrict pold, 
		      float* restrict * restrict pxx, 
		      float* restrict * restrict pzz, 
		      float* restrict * restrict ekp, 
			  const size_t m, const size_t n){


	#pragma omp parallel for
	for (int i=0; i<m; i++)
	for (int j=0; j<n; j++)
		pnew[i][j] = 2.0*pnow[i][j] -pold[i][j] + ekp[i][j]*(pxx[i][j] + pzz[i][j]);

}

void write_bdr(float***  bc_ud, 
			   float***  bc_lr,
			   float * restrict * restrict p, 
			   const fwi_shape *gb, 
			   const fwi_base *base,
			   FILE *fp,
			   size_t it)
{
			
	it = it % base->nt_mem;
	
	#pragma omp parallel for
	for (int i=0; i<2*base->mm-1; i++)
	for (int j=0; j<gb->pnx;      j++){
		
		bc_ud[it][i][j] = p[i][j+base->npml];

		bc_ud[it][i+2*base->mm-1][j] = p[(gb->pnz-i-1)][j+base->npml];		
	}
	
	#pragma omp parallel for	
	for (int i=0; i<gb->pnz; i++)
	for (int j=0; j<2*base->mm-1;  j++){
			
		bc_lr[it][i][j] = p[i][j+base->npml];

		bc_lr[it][i][j+2*base->mm-1] = p[i][gb->pnx+base->npml-j-1];
		
	}

	if (it+1 == base->nt_mem){ 	
		fwrite(bc_ud[0][0], sizeof(float), gb->pnx*base->dim*(2*base->mm-1)*base->nt_mem, fp); 
		fwrite(bc_lr[0][0], sizeof(float), gb->pnz*base->dim*(2*base->mm-1)*base->nt_mem, fp);		
	}

}

void read_bdr(float*** bc_ud, 
			  float*** bc_lr,
      		  float * restrict * restrict p, 
			  const fwi_shape *gb, 
			  const fwi_base *base,
			  FILE *fp,
			  const int offset,
			  size_t it)
{
			
	it = it % base->nt_mem;
	int rtn;	
	if (it+1 == base->nt_mem){ 	

		fseek(fp, -(offset), SEEK_CUR);
		rtn = fread(bc_ud[0][0], sizeof(float), gb->pnx*base->dim*(2*base->mm-1)*base->nt_mem, fp); 
		rtn += fread(bc_lr[0][0], sizeof(float), gb->pnz*base->dim*(2*base->mm-1)*base->nt_mem, fp);		
		fseek(fp, -(offset), SEEK_CUR);
		if (rtn*sizeof(float) != offset) {printf("readding error.\n"); exit(-1);}
	}

	#pragma omp parallel for
	for (int i=0; i<2*base->mm-1; i++)
	for (int j=0; j<gb->pnx;      j++){
		
		p[i][j]             = bc_ud[it][i][j];

		p[(gb->pnz-i-1)][j] = bc_ud[it][i+2*base->mm-1][j];		
		
	}
	
	#pragma omp parallel for	
	for (int i=0; i<gb->pnz;      i++)
	for (int j=0; j<2*base->mm-1; j++){
			
		p[i][j] 	      = bc_lr[it][i][j];

		p[i][gb->pnx-j-1] = bc_lr[it][i][j+2*base->mm-1];	
	}


}

void write_bdr_fs_no(float***  bc_ud, 
			   float***  bc_lr,
			   float * restrict * restrict p, 
			   const fwi_shape *gb, 
			   const fwi_base *base,
			   FILE *fp,
			   size_t it)
{
			
	it = it % base->nt_mem;
	
	#pragma omp parallel for
	for (int i=0; i<2*base->mm-1; i++)
	for (int j=0; j<gb->pnx;      j++){
		
		bc_ud[it][i][j] = p[i+base->npml][j+base->npml];

		bc_ud[it][i+2*base->mm-1][j] = p[(gb->pnz-i-1)+base->npml][j+base->npml];		
	}
	
	#pragma omp parallel for	
	for (int i=0; i<gb->pnz; i++)
	for (int j=0; j<2*base->mm-1;  j++){
			
		bc_lr[it][i][j] = p[i+base->npml][j+base->npml];

		bc_lr[it][i][j+2*base->mm-1] = p[i+base->npml][gb->pnx+base->npml-j-1];
		
	}

	if (it+1 == base->nt_mem){ 	
		fwrite(bc_ud[0][0], sizeof(float), gb->pnx*base->dim*(2*base->mm-1)*base->nt_mem, fp); 
		fwrite(bc_lr[0][0], sizeof(float), gb->pnz*base->dim*(2*base->mm-1)*base->nt_mem, fp);		
	}

}


