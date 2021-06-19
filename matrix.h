#ifndef _MATRIX_H_

typedef struct vector{
    int size;
    float* data;
} Vector;

typedef struct matrix{
	int row;
	int column;
	float** data;
} Matrix;

void initialize_vector(Vector* vector, int size);
void free_vector(Vector *vector);
void print_vector(Vector *vector);
void copy_vector_from_array(Vector *vector, float* array);
void copy_vector(Vector *copy, Vector *org);

void row_vector_to_matrix(Matrix *dst, Vector *org);
void column_vector_to_matrix(Matrix *dst, Vector *org);

void vector_add(Vector *ans, Vector *a, Vector *b);
void vector_sub(Vector *ans, Vector *a, Vector *b);
float vector_dot(Vector *a, Vector *b);

void initialize_matrix(Matrix* matrix, int row, int column);
void free_matrix(Matrix* matrix);
void print_matrix(Matrix* matrix);
void copy_matrix_from_array(Matrix* matrix, float* array);
void copy_matrix(Matrix* copy, Matrix* org);

void matrix_dot(Matrix* ans, Matrix* a, Matrix* b);
float matrix_determinant(Matrix* a);
void matrix_transpose(Matrix* ans, Matrix* a);
void matrix_cMul(Matrix* ans, Matrix* a, float c);
void matrix_inverse(Matrix* ans, Matrix* a);
void matrix_add(Matrix* ans, Matrix* a, Matrix* b);
void matrix_sub(Matrix* ans, Matrix* a, Matrix* b);
float matrix_trace(Matrix* a);
int matrix_rank(Matrix* a);
void matrix_eigenvalues(Matrix* eigenvalues, Matrix* a);

#endif
