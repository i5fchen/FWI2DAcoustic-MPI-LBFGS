#include "alloc.h"

float** allocate2d(const size_t n1, const size_t n2){
    
	float *arr_t, **arr;
 
    arr   = (float **)calloc(   n2, sizeof (float*));
 
    arr_t = (float  *)calloc(n1*n2, sizeof (float )); 
    
	for (int i = 0; i < n2; i++)
        arr[i] = arr_t + i * n1;
 
    return arr;
}

void free2d(float** p) {
    free(*p);
    free(p);
} 

float ***allocate3d(const size_t n1, const size_t n2, const size_t n3){

	float ***arr = (float ***) malloc(n3*sizeof(float **));
	
	float **arr2_t = (float **)malloc(n3*n2*   sizeof(float*));
	float  *arr1_t = (float  *)malloc(n3*n2*n1*sizeof(float ));

	for (int i=0; i<n3; i++){
		
		arr[i] = arr2_t + n2*i;

		for (int j=0; j<n2; j++)
			arr[i][j] = arr1_t + j*n1 + i*n1*n2;

	}

	return arr;

}

void free3d(float ***p){
	
	free(**p);
	free(*p);
	free(p);

}



