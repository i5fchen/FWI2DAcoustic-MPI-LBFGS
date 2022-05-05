#include "soft_ranking.h"
#include <math.h>
static int cmp_float_array(const void *x, const void *y)
{
    struct float2d_struct xx = *(struct float2d_struct *) x;
    struct float2d_struct yy = *(struct float2d_struct *) y; 
    if (xx.val < yy.val) return -1; 
    if (xx.val > yy.val) return  1;
    return 0;
}

static int cmp_int_array(const void *x, const void *y) 
{
    struct int2d_struct xx = *(struct int2d_struct *) x;
    struct int2d_struct yy = *(struct int2d_struct *) y;  
    if (xx.val < yy.val) return -1; 
    if (xx.val > yy.val) return  1;  
    return 0;
}

int sorting_flt_argsort(const int n,
                    	const float *__restrict__ a,
                    	const enum sortOrder_enum order,
                    	int *__restrict__ perm,
						int *__restrict__ rank,
                    	float *__restrict__ sorted_a)
{
    struct float2d_struct *vals;
    int i;
       
	vals = (struct float2d_struct *)calloc((size_t) n, sizeof(struct float2d_struct));

    for (i=0; i<n; i++)
    {
        vals[i].val = a[i];
        vals[i].indx = i;
    }
    qsort((void *) vals, (size_t) n,
          sizeof(struct float2d_struct), cmp_float_array);
    
	if (order == SORT_ASCENDING)
    {   
        for (i=0; i<n; i++)
        {
            perm[i] = vals[i].indx;
        }
    }   
    else
    {   
        for (i=0; i<n; i++)
        {
            perm[i] = vals[n-1-i].indx;
        }
    }

	for (i=0; i<n; i++) {
		
		rank[perm[i]] = i+1;
		sorted_a[i] = a[perm[i]];	
		perm[i] += 1;
	}

    free(vals);
    return 0;
}

void isotonic_l2(int n, float *y, float *sol, int *target){
	
	int i=0, k=0;
	float sum_y, sum_c, prev_y;
	float *c=malloc(sizeof(float)*n), *sums=malloc(sizeof(float)*n);
	
	for (int i=0; i<n; i++) {

		target[i] = i; c[i] = 1; sums[i] = 0;
		sol[i] = y[i];
		sums[i] = y[i];
	}

	while (i<n) {
		
		k = target[i] + 1;
		
		if (k==n) break;

		if (sol[i] > sol[k]){ i=k; continue; }

		sum_y = sums[i];
		sum_c = c[i];
		
		while (true) {

			prev_y = sol[k]; sum_y += sums[k]; sum_c += c[k]; k = target[k] + 1;

			if (k==n || prev_y > sol[k]) {

				sol[i] = sum_y / sum_c; sums[i] = sum_y; c[i] = sum_c; target[i] = k-1; target[k-1] = i;
				
				if (i>0) i = target[i-1];

				break;

			}
		}

	}

	i=0; 

	while (i<n) {

		k = target[i] + 1;
		
		for (int j=i+1; j<k; j++) 
			sol[j] = sol[i];

		i=k;
	}
	
	free(c); free(sums);
}

void soft_ranking_init(int n, soft_ranking_vars *base) {

	base->sorted = (float *) malloc(sizeof(float)*n);
	
	base->sol = (float *) malloc(sizeof(float)*n);
	
	base->soft_rank = (float *) malloc(sizeof(float)*n);

	base->hard_rank = (int *) malloc(sizeof(int) * n);
	base->perm = (int *) malloc(sizeof(int) * n);
	base->target = (int *) malloc(sizeof(int) * n);

}

void soft_ranking_free(soft_ranking_vars *base) {
	
	free(base->sorted);
	free(base->sol);
	free(base->soft_rank);
	free(base->hard_rank);
	free(base->perm);
	free(base->target);

	free(base);
}


