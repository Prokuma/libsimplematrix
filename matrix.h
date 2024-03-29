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

void print_compile_info();

void initialize_vector(Vector* vector, int size);
void free_vector(Vector *vector);
void print_vector(Vector *vector);
void copy_vector_from_array(Vector *vector, float* array);
void copy_vector(Vector *copy, Vector *org);

void row_vector_to_matrix(Matrix *dst, Vector *org);
void column_vector_to_matrix(Matrix *dst, Vector *org);

float vector_get(Vector *a, int n);
void vector_add(Vector *ans, Vector *a, Vector *b);
void vector_sub(Vector *ans, Vector *a, Vector *b);
float vector_dot(Vector *a, Vector *b);

void initialize_matrix(Matrix* matrix, int row, int column);
void free_matrix(Matrix* matrix);
void print_matrix(Matrix* matrix);
void copy_matrix_from_array(Matrix* matrix, float* array);
void copy_matrix(Matrix* copy, Matrix* org);

float matrix_get(Matrix* a, int n, int m);
void matrix_dot(Matrix* ans, Matrix* a, Matrix* b);
float matrix_determinant(Matrix* a);
void matrix_transpose(Matrix* ans, Matrix* a);
void matrix_cMul(Matrix* ans, Matrix* a, float c);
void matrix_inverse(Matrix* ans, Matrix* a);
void matrix_add(Matrix* ans, Matrix* a, Matrix* b);
void matrix_sub(Matrix* ans, Matrix* a, Matrix* b);
void matrix_hadamard_product(Matrix *ans, Matrix *a, Matrix *b);
float matrix_trace(Matrix* a);
int matrix_rank(Matrix* a);
void matrix_eigenvalues(Vector* eigenvalues, Matrix* a);
float matrix_max(Matrix* a);
float matrix_min(Matrix* a);
void matrix_exp(Matrix* ans, Matrix *a);
float matrix_sum(Matrix* a);

void make_identity_matrix(Matrix* matrix);
void make_upper_triangular_matrix(Matrix* matrix);
void make_lower_triangular_matrix(Matrix* matrix);
void matrix_fill(Matrix* matrix, float num);

int matrix_convolution_output_height(Matrix* a, Matrix* kernel, int padding, int stride);
int matrix_convolution_output_width(Matrix* a, Matrix* kernel, int padding, int stride);
void matrix_convolution(Matrix* ans, Matrix* a, Matrix* kernel, int padding, int stride);

#endif
