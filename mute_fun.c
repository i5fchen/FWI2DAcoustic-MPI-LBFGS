#include <math.h>
#include <stdbool.h>
#include "fwi_sys.h"

void fbt_muter(float *fbt, float *obs, int nt, int nr, float alpha, float dt, float tune, bool early_gone) {
	
	float tt, ttt;
	for (int ix=0; ix<nr; ix++)
		for (int it=0; it<nt; it++){
		
		tt = it*dt - (fbt[ix]+tune);
	 
		if (false == early_gone) tt=-tt;

			if (tt<=0){ 
				ttt = alpha*tt;
				if (ttt>-10.0)
					obs[it+ix*nt] *= exp(ttt);
				else 
					obs[it+ix*nt]  = 0.0;
			}
		} 
}

void win_apply(float *win, float *obs, int nt, int nr, float dt, float alpha, float *wobs) {
	
	float ss, tt, sss, ttt;

	for (int i=0; i<nt*nr; i++) wobs[i] = obs[i];

	for (int ix=0; ix<nr; ix++){

		ss = -win[2*ix]; tt = win[2*ix+1];

		for (int it=0; it<nt; it++){

			ss += dt;
			if (ss<=0.0){

				sss = alpha *ss; 

				if (sss>=-20.0)
					wobs[it+ix*nt] *= exp(sss);
				else    
					wobs[it+ix*nt]  = 0.0;
			}
			tt -= dt;
			if (tt<=0.0){

				ttt = alpha*tt;
				if ( ttt>=-20.0) 
					wobs[it+ix*nt] *= exp(ttt);
				else    
					wobs[it+ix*nt]  = 0.0;
			}
						
		}
	}
}


void upper_off_srcood(float **in, float *sr, int nr, int nt, float dt, float t0, float v0, float K, float **out){

  float srdist = 0, tt0=0;
  for (int i1=0; i1<nr; i1++){

    srdist = pow((sr[2+2*i1]-sr[0]),2) + pow((sr[2+2*i1+1]-sr[1]),2);
    tt0 = t0 + sqrt(srdist)/v0;
  
    for (int i2=0; i2<nt; i2++){
      in[i1][i2] = in[i1][i2]*0.5*(1+tanh(K*(i2*dt-tt0)));
    }
  
  }
}

void lower_off_srcood(float **in, float *sr, int nr, int nt, float dt, float t0, float v0, float K, float **out){

  float srdist = 0, tt0=0;
  for (int i1=0; i1<nr; i1++){

    srdist = pow((sr[2+2*i1]-sr[0]),2) + pow((sr[2+2*i1+1]-sr[1]),2);
    tt0 = t0 + sqrt(srdist)/v0;
  
    for (int i2=0; i2<nt; i2++){
      in[i1][i2] = in[i1][i2]*(1.0-(tanh(K*(i2*dt-tt0))));
    }
  
  }
}

void linear_upper_off(float **in, float *sr, int nr, int nt, float dt, float t0, float v0, int N){

	float srdist=0.0f, tt0=0.0f, tt1=0.0f;
	for (int i=0; i<nr; i++) {
		
		srdist =  pow(sr[2+2*i]  -sr[0], 2);
		srdist += pow(sr[2+2*i+1]-sr[1], 2);

		tt1 = t0 + sqrt(srdist)/v0;
		tt0 = tt1 - N*dt;

		for (int j=0; j<nt; j++){
			
			if (j*dt < tt1){
					
				if (j*dt >tt0)
					in[i][j] *= 0.5*(1-cos(PI*(j*dt-tt0)/(N*dt)));
				else
					in[i][j]=0.0f;
			}	
		}
	}
}

void linear_lower_off(float **in, float *sr, int nr, int nt, float dt, float t0, float v0, int N){

	float srdist=0.0f, tt0=0.0f, tt1=0.0f;
	for (int i=0; i<nr; i++) {
		
		srdist =  pow(sr[2+2*i]  -sr[0], 2);
		srdist += pow(sr[2+2*i+1]-sr[1], 2);

		tt1 = t0 + sqrt(srdist)/v0;
		tt0 = tt1 + N*dt;

		for (int j=0; j<nt; j++){
			
			if (j*dt > tt1){
					
				if (j*dt <tt0)
					in[i][j] *= 0.5*(1-cos(PI*(j*dt-tt0)/(N*dt)));
				else
					in[i][j]=0.0f;
			}	
		}
	}
}

void quadratic_upper_off(float **in, float *sr, int nr, int nt, float dt, float t0, float v0, int N){

	float srdist=0.0f, tt0=0.0f, tt1=0.0f;
	v0 *= v0;
	t0 *= t0;
	for (int i=0; i<nr; i++) {
		
		srdist =  pow(sr[2+2*i]  -sr[0], 2);
		srdist += pow(sr[2+2*i+1]-sr[1], 2);

		tt1 = sqrt(t0 + srdist/v0);
		tt0 = tt1 - N*dt;

		for (int j=0; j<nt; j++){
			
			if (j*dt < tt1){
					
				if (j*dt >tt0)
					in[i][j] *= 0.5*(1-cos(PI*(j*dt-tt0)/(N*dt)));
				else
					in[i][j]=0.0f;
			}	
		}
	}
}

void quadratic_lower_off(float **in, float *sr, int nr, int nt, float dt, float t0, float v0, int N){

	float srdist=0.0f, tt0=0.0f, tt1=0.0f;
	v0 *= v0;
	t0 *= t0;
	for (int i=0; i<nr; i++) {
		
		srdist =  pow(sr[2+2*i]  -sr[0], 2);
		srdist += pow(sr[2+2*i+1]-sr[1], 2);

		tt1 = sqrt(t0 + srdist/v0);
		tt0 = tt1 + N*dt;

		for (int j=0; j<nt; j++){
			
			if (j*dt > tt1){
					
				if (j*dt <tt0)
					in[i][j] *= 0.5*(1-cos(PI*(j*dt-tt0)/(N*dt)));
				else
					in[i][j]=0.0f;
			}	
		}
	}
}


