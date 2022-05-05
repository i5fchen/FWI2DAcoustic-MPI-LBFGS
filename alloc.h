#include <stdlib.h>

float** 	allocate2d(const size_t n1, const size_t n2);
float*** allocate3d(const size_t n1, const size_t n2, const size_t n3);

void free2d(float ** p);
void free3d(float ***p);


