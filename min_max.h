#include <stdio.h>
#include <stdlib.h>

int int_cmp(const void *a, const void *b);
int flt_cmp(const void *a, const void *b);

size_t argmax(void *arr, size_t size, size_t n, int (*cmp)(const void *, const void *));
size_t argmin(void *arr, size_t size, size_t n, int (*cmp)(const void *, const void *));


