//valina implementation and not considering the upper and lower bound along offset and time dimensions.

#include "header.h"
#include "read_write.h"
#include "alloc.h"
#include "min_max.h"

void gaussian_win(const int n, float delta, float *win){
    
    float tt1 = 0;
    
    int nn = n/2;
    
    for (int i=-nn; i<nn+1; i++){
        tt1 = i/delta;
        win[i+nn] = exp(-0.5*tt1*tt1);
    }
    
}

static inline float MAX3(float x, float y, float z){

	if (x>y) 
		return MAX2(x,z);
	else 
		return MAX2(y,z);

}

void ac_3pts_fw(float **err, float **ddd, int nl, int nt){

	for (int i=1; i<nl+1; i++)	
		ddd[0][i] = err[0][i];
	
	for (int i=1; i<nt;   i++)
	for (int j=1; j<nl+1; j++)		
		ddd[i][j] = err[i][j] + MAX3(ddd[i-1][j], ddd[i-1][j-1], ddd[i-1][j+1]);

}

void ac_5pts_fw(float **err, float **ddd, int nl, int nt){

	register float t1, t2, t3;

	for (int i=1; i<nl+1; i++){	
		ddd[0][i] = err[0][i];
		ddd[1][i] = err[1][i] + MAX3(ddd[0][i], ddd[0][i-1], ddd[0][i+1]);
	}

	for (int i=2; i<nt; i++)
		for (int j=1; j<nl+1; j++){
			
			t1=ddd[i-2][j-1]+err[i-1][j-1];
			t2=ddd[i-1][j];
			t3=ddd[i-2][j+1]+err[i-1][j+1];
			
			ddd[i][j] = err[i][j] + MAX3(t1,t2,t3);

		}

}

void ac_3pts_bw(float **err, float **ddd, int nl, int nt){

	for (int i=1; i<nl+1; i++)		
		ddd[nt-1][i] = err[nt-1][i];
	
	for (int i=nt-2; i>=0;  i--)
	for (int j=1; j<nl+1; j++)	
		ddd[i][j] = err[i][j] + MAX3(ddd[i+1][j], ddd[i+1][j+1], ddd[i+1][j-1]);

}

void ac_5pts_bw(float **err, float **ddd, int nl, int nt){

	register float t1, t2, t3;

	int i1=nt-2;
	
	for (int i=1; i<nl+1; i++){		
		ddd[nt-1][i] = err[nt-1][i];
		ddd[i1][i] = err[i1][i] + MAX3(ddd[i1+1][i], ddd[i1+1][i+1], ddd[i1+1][i-1]);
	}

	for (int i=nt-3; i>=0;   i--)
	for (int j=1; 	 j<nl+1; j++){

		t1=ddd[i+2][j-1]+err[i+1][j-1];
		t2=ddd[i+1][j];
		t3=ddd[i+2][j+1]+err[i+1][j+1];
			
		ddd[i][j] = err[i][j] + MAX3(t1,t2,t3);	
	
	}	
}

void bt_fw_3pts(float **ddd, int nl, int nt, float *path){

	int ll;
	
	path[0] = argmax(ddd[0], sizeof(float), nl+2, flt_cmp);

	for (int i=1; i<nt;  i++){
		ll = (int)path[i-1]	;
		if (MAX2(ddd[i][ll+1], ddd[i][ll]) > ddd[i][ll-1]){

			if (ddd[i][ll+1]>ddd[i][ll])
				path[i] = path[i-1]+1;
			else
				path[i] = path[i-1];
	
		}else{

			path[i] = path[i-1]-1;

		}
	
	}
	
}

void bt_fw_5pts(float **ddd, float **err, int nl, int nt, float *path){

	path[0] = argmax(ddd[0], sizeof(float), nl+2, flt_cmp);

	int ll, i=1;

	float a, b, c;
	
	while(i<nt-1){

		ll = (int)path[i-1];
		
		c = ddd[i+1][ll-1]+err[i][ll-1];
		b = ddd[i][ll];
		a = ddd[i+1][ll+1]+err[i][ll+1];
		
		if (MAX2(a,b) > c){

			if (a>b){
				path[i] = path[i-1]+0.5;
				path[i+1] = path[i-1]+1;
				i+=2;
			}else
				{path[i] = path[i-1];i+=1;}
	
		}else{

			path[i] = path[i-1]-0.5;
			path[i+1] = path[i-1]-1;
			i+=2;

		}
	
	}	
	
	if (i==nt-1){
		
		ll = (int)path[i-1];

		if (MAX2(ddd[i][ll+1], ddd[i][ll]) > ddd[i][ll-1]){

			if (ddd[i][ll+1]>ddd[i][ll])
				path[i] = path[i-1]+1;
			else
				path[i] = path[i-1];
	
		}else{

			path[i] = path[i-1]-1;

		}
	}	

}

void bt_bw_5pts(float **ddd, float **err, int nl, int nt, float *path){

	int ll, i;

	float a, b, c;
	
	i = nt-1;

	path[i] = argmax(ddd[i], sizeof(float), nl+2, flt_cmp);

	i -= 1;

	while(i>=1){
		
		ll = (int)path[i+1];
		
		c = ddd[i-1][ll-1]+err[i][ll-1];
		b = ddd[i][ll];
		a = ddd[i-1][ll+1]+err[i][ll+1];
		
		if (MAX2(a,b) > c){

			if (a>b){
				path[i] = path[i+1]+0.5;
				path[i-1] = path[i+1]+1;
				i-=2;
			}else
				{path[i] = path[i+1];i-=1;}
	
		}else{

			path[i] = path[i+1]-0.5;
			path[i-1] = path[i+1]-1;
			i-=2;

		}
	}	

	if (i==0){
		
		ll = (int)path[i+1];
		
		if (MAX2(ddd[i][ll+1], ddd[i][ll]) > ddd[i][ll-1]){

			if (ddd[i][ll+1]>ddd[i][ll])
				path[i] = path[i+1]+1;
			else
				path[i] = path[i+1];
	
		}else{

			path[i] = path[i+1]-1;

		}
	}	
}

void lcc_err(float **obs, float **syn, float *sr_xyz, float offset_range[], float eps, int nt, int nr, int nl, int nw, float alpha_win, float ***err){
 
    if ((1 != nw%2) || (1 != nl%2) ){
        printf("the odd length for window is preferred\n"); exit(0);
    }
	
	register float e1, e2, tt1, tt2, tt3, tt4;
	
	float min=offset_range[0], max=offset_range[1], *offset=(float*)malloc(nr*sizeof(float));

	for(int i=0; i<nr; i++){

		for(int j=0; j<nt; j++){

			e1 += fabs(obs[i][j]); e2 += fabs(syn[i][j]);		
		
		}
	
		tt1 = sr_xyz[2+2*i]-sr_xyz[0];
		tt1 *= tt1;

		tt2 = sr_xyz[2+2*i+1]-sr_xyz[1];
		tt2 *= tt2;

		tt2 = sqrt(tt1+tt2);
		offset[i] = tt2;

			
	}

	int nw2=nw/2, nl2=nl/2;

    float *b = calloc(nt+nw-1, sizeof(float)), *a = calloc(nt+nw-1+nl-1, sizeof(float)), *win = calloc(nw, sizeof(float));

	gaussian_win(nw, alpha_win, win);
	
	int it0, it1;
	for (int ix=0; ix<nr; ix++){
		
		if (offset[ix] > max || offset[ix] < min) continue;
		memset(b, 0, sizeof(float)*(nt+nw-1));
		memset(a, 0, sizeof(float)*(nt+nw+nl-2));

		for (int it=0; it<nt; it++){
   			 
			b[it+nw2]     = syn[ix][it]; 
            a[it+nw2+nl2] = obs[ix][it];
		}	   
 		
		for (int it=0; it<nt; it++){
   			 
 			if (syn[ix][it]>eps) {it0=it; break;}
		}		 
		
		for (int it=0; it<nt; it++){
   			 
			if (syn[ix][nt-1-it]>eps) {it1=nt-1-it;break;}
		}		 
		
		for (int it=it0; it<it1; it++){    
            for (int il=1; il<nl+1; il++){
        
                tt1 = 0.0; tt2 = 0.0; tt3 = 0.0;

                for (int iw=0; iw<nw; iw++){
            	
                	e1 = b[it+iw     ] * win[iw];
                    e2 = a[it+iw+il-1] * win[iw];
           	      
					tt1 += e2*e1;
                    tt2 += e2*e2;
                    tt3 += e1*e1;
                }

                tt4 = sqrt(tt2*tt3);
                err[ix][it][il] = tt1/(tt4+1e-12);

            }
		}
	
	}
	
	write_to_file(err[0][0], sizeof(float), nt*nr*(nl+2), "err.bin");
	free(b); free(a); free(win);
	free(offset);
}


