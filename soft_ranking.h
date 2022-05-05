#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define offset_size_double_int (sizeof(double)-sizeof(int))

enum sortOrder_enum 
{
    SORT_ASCENDING = 0,  
    SORT_DESCENDING = 1, 
};

struct float2d_struct
{
    float val;
    int indx;
    char pad4[offset_size_double_int];
};

struct int2d_struct
{
    int val;
    int indx;
};

typedef struct _soft_ranking_vars {
	
	float *sorted, *sol;
	float *soft_rank;
	int *hard_rank, *perm, *target;

} soft_ranking_vars;


static int cmp_float_array(const void *x, const void *y);

static int cmp_int_array(const void *x, const void *y); 

int sorting_flt_argsort(const int n,
                    	const float *__restrict__ a,
                    	const enum sortOrder_enum order,
                    	int *__restrict__ perm,
						int *__restrict__ rank,
                    	float *__restrict__ sorted_a);

void isotonic_l2(int n, float *y, float *sol, int *target);

void soft_ranking_init(int n, soft_ranking_vars *base);

void soft_ranking_free(soft_ranking_vars *base) ;

void soft_ranking(int n, float *a, float regularization_strength, soft_ranking_vars *base) ;

void soft_ranking_msf(float *obs, float *syn, int n, float reg, soft_ranking_vars *obs_ranking, soft_ranking_vars *syn_ranking, float *msf, float *res);

void soft_ranking_2d_trace_by_trace(float **obs, float **syn, float *, float, float, soft_ranking_vars *par[], float reg, int nt, int nr, float msf[], float **res);


