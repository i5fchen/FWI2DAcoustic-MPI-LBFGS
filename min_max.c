#include "min_max.h"

size_t argmax(void *arr, size_t size, size_t n, int (*cmp)(const void *, const void *)){

	char *base = arr;
	void *max = base;
	size_t i0=0;
	for (size_t i=1; i<n; i++){

		if (cmp(&base[i*size], max)) {max = &base[i*size];i0=i;}
	
	}

	return i0;
}

size_t argmin(void *arr, size_t size, size_t n, int (*cmp)(const void *, const void *)){

	char *base = arr;
	void *min = base;
	size_t i0=0;
	for (size_t i=1; i<n; i++){

		if (cmp(min, &base[i*size])) {min = &base[i*size];i0=i;}	
	
	}

	return i0;
}

int int_cmp(const void *a, const void *b){

	int i1 = *(int *)a;
	int i2 = *(int *)b;

	return (i1>i2);

}

int flt_cmp(const void *a, const void *b){

	float i1 = *(float *)a;
	float i2 = *(float *)b;

	return (i1>i2);

}
