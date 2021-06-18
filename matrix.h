#include <stdio.h>

typedef struct matrix{
	int row;
	int column;
	float** data;
} Matrix;


void initialize_matrix(Matrix* matrix, int row, int column);
void dot(Matrix* a, Matrix* b, Matrix* ans);
float determinant(Matrix* a);
void inverse(Matrix* a, Matrix* ans);
void add(Matrix* a, Matrix* b, Matrix* ans);
void sub(Matrix* a, Matrix* b, Matrix* ans);
int rank(Matrix* a);

