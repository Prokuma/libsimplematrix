#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdarg.h>
#include <string.h>
#ifdef USE_MP
#include <omp.h>
#endif

#define EPS 1e-8
#define ITER_MAX 100
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

void print_compile_info(){
    printf("Compile Options: ");
    #ifdef USE_MP
    printf("OpenMP");
    #endif
    printf("\n");
}

void initialize_vector(Vector* vector, int size){
    int i;

    vector->size = size;

    vector->data = (float*) malloc(sizeof(float) * size);
    for(i = 0; i < size; i++){
        vector->data[i] = 0;
    }
}

void free_vector(Vector *vector){
    free(vector->data);
}

void print_vector(Vector *vector){
    int i;

    for(i = 0; i < vector->size; i++){
        printf("%f\t", vector->data[i]);
    }
    printf("\n");
}

void copy_vector_from_array(Vector *vector, float* array){
    int i;

    for(i = 0; i < vector->size; i++){
        vector->data[i] = array[i];
    }
}

void copy_vector(Vector *copy, Vector *org){
    int i;

    for(i = 0; i < org->size; i++){
        copy->data[i] = org->data[i];
    }
}

void row_vector_to_matrix(Matrix *dst, Vector *org){
    copy_matrix_from_array(dst, org->data);
}

void column_vector_to_matrix(Matrix *dst, Vector *org){
    copy_matrix_from_array(dst, org->data);
}

float vector_get(Vector *a, int n){
    return a->data[n];
}

void vector_add(Vector *ans, Vector *a, Vector *b){
    if(a->size != b->size){
        return;
    }
    if(a->size != ans->size){
        return;
    }

    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for(int i = 0; i < a->size; i++){
        ans->data[i] = a->data[i] + b->data[i];
    }
}

void vector_sub(Vector *ans, Vector *a, Vector *b){
    if(a->size != b->size){
        return;
    }
    if(a->size != ans->size){
        return;
    }

    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for(int i = 0; i < a->size; i++){
        ans->data[i] = a->data[i] - b->data[i];
    }
}

float vector_dot(Vector *a, Vector *b){
    int i;
    float ans = 0;

    if(a->size != b->size){
        return 0;
    }

    for(i = 0; i < a->size; i++){
        ans += a->data[i] * b->data[i];
    }

    return ans;
}

//Matrix Initialization
void initialize_matrix(Matrix* matrix, int row, int column){
    int i, j;

    matrix->row = row;
    matrix->column = column;

    matrix->data = (float**) malloc(sizeof(float*) * row);
    for(i = 0; i < row; i++){
        matrix->data[i] = (float*) malloc(sizeof(float) * column);
        for(j = 0; j < column; j++){
            matrix->data[i][j] = 0;
        }
    }
}

void free_matrix(Matrix* matrix){
    int i;

    for(i = 0; i < matrix->row; i++){
        free(matrix->data[i]);
    }
    free(matrix->data);
}

void print_matrix(Matrix* matrix){
    int i, j;

    for(i = 0; i < matrix->row; i++){
        for(j = 0; j < matrix->column; j++){
            printf("%f\t", matrix->data[i][j]);
        }
        printf("\n");
    }
}

void copy_matrix_from_array(Matrix *matrix, float *array){
    int i, j;

    for(i = 0; i < matrix->row; i++){
        for(j = 0; j < matrix->column; j++){
            matrix->data[i][j] = array[i*matrix->row + j];
        }
    }
}

void copy_matrix(Matrix* copy, Matrix *org){
    int i, j;

    for(i = 0; i < org->row; i++){
        for(j = 0; j < org->column; j++){
            copy->data[i][j] = org->data[i][j];
        }
    }
}

float matrix_get(Matrix* a, int n, int m){
    return a->data[n][m];
}

//Dot of Matrix
void matrix_dot(Matrix* ans, Matrix* a, Matrix* b){
    initialize_matrix(ans, a->row, b->column);

    if(a->column != b->row || a->row != ans->row || b->column != ans->column){
        return;
    }

    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for(int i = 0; i < a->row; i++){
        for(int j = 0; j < b->column; j++){
            for(int k = 0; k < a->column; k++){
                ans->data[i][j] += a->data[i][k] * b->data[k][j];
            }
        }
    }
}

//Determinant
float matrix_determinant(Matrix* a){
    Matrix mMatrix;
    float *tmp;
    int i, x, y;
    float det = 0;

    if(a->row != a->column){
        return 0;
    }


    if(a->row == 2){
        return a->data[0][0] * a->data[1][1] - a->data[0][1] * a->data[1][0];
    }else if(a->row == 1){
        return a->data[0][0];
    }else{
        tmp = (float*)malloc(sizeof(float) * a->column);
        initialize_matrix(&mMatrix, a->row - 1, a->column - 1);
        for(i = 0; i < a->column; i++){
            for(x = 0; x < a->row - 1; x++){
                for(y = 0; y < a->column -1; y++){
                    if(i > y){
                        mMatrix.data[x][y] = a->data[x+1][y];
                    }else{
                        mMatrix.data[x][y] = a->data[x+1][y+1];
                    }
                }
            }
            tmp[i] = pow(-1, i + 2) * matrix_determinant(&mMatrix);
        }

        for(i = 0; i < a->column; i++){
            det += a->data[0][i] * tmp[i];
        }

        free_matrix(&mMatrix);
        free(tmp);
        return det;
    }
}

void matrix_transpose(Matrix* ans, Matrix* a){
    if(a->row != ans->column || a->column != ans->row){
        return;
    }

    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for(int i = 0; i < a->row; i++){
        for(int j = 0; j < a->column; j++){
            ans->data[j][i] = a->data[i][j];
        }
    }
}

void matrix_cMul(Matrix* ans, Matrix* a, float c){
    if(a->row != ans->row || a->column != ans->column){
        return;
    }

    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for(int i = 0; i < a->row; i++){
        for(int j = 0; j < a->column; j++){
            ans->data[i][j] = c * a->data[i][j];
        }
    }
}

void matrix_inverse(Matrix *ans, Matrix* a){
    float det;
    Matrix mMatrix, tmp, tmp2;

    if(a->row != a->column){
        return;
    }
    if(a->row != ans->row || a->column != ans->column){
        return;
    }

    initialize_matrix(&tmp, a->row, a->column);
    initialize_matrix(&tmp2, a->row, a->column);
    initialize_matrix(&mMatrix, a->row - 1, a->column - 1);
    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for(int i = 0; i < a->row; i++){
        for(int j = 0; j < a->column; j++){
            for(int x = 0; x < a->row - 1; x++){
                for(int y = 0; y < a->row -1; y++){
                    if(i > x){
                        if(j > y){
                            mMatrix.data[x][y] = a->data[x][y];
                        }else{
                            mMatrix.data[x][y] = a->data[x][y+1];
                        }
                    }else{
                        if(j > y){
                            mMatrix.data[x][y] = a->data[x+1][y];
                        }else{
                            mMatrix.data[x][y] = a->data[x+1][y+1];
                        }
                    }
                }
            }
            tmp.data[i][j] = pow(-1, i + j) * matrix_determinant(&mMatrix);
        }
    }

    matrix_transpose(&tmp2, &tmp);
    det = matrix_determinant(a);
    matrix_cMul(ans, &tmp2, 1/det);		
}

//addition between matrix
void matrix_add(Matrix* ans, Matrix* a, Matrix* b){
    if(a->row != b->row || a->column != b->column || 
            ans->row != a->row || ans->column != a->column){
        return;
    }

    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for(int i = 0; i < a->row; i++){
        for(int j = 0; j < a->column; j++){
            ans->data[i][j] = a->data[i][j] + b->data[i][j];
        }
    }
}

//substraction between matrix
void matrix_sub(Matrix* ans, Matrix* a, Matrix* b){
    if(a->row != b->row || a->column != b->column || 
            ans->row != a->row || ans->column != a->column){
        return;
    }

    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for(int i = 0; i < a->row; i++){
        for(int j = 0; j < a->column; j++){
            ans->data[i][j] = a->data[i][j] - b->data[i][j];
        }
    }
}

void matrix_hadamard_product(Matrix *ans, Matrix *a, Matrix *b){
    if(a->row != b->row || a->column != b->column){
        return;
    }
    if(a->row != ans->row || a->column != ans->column){
        return;
    }

    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for(int i = 0; i < a->row; i++){
        for(int j = 0; j < a->column; j++){
            ans->data[i][j] = a->data[i][j] * b->data[i][j];
        }
    }
}

float matrix_trace(Matrix* a){
    int i;
    float ans = 0;

    if(a->row != a->column){
        return 0;
    }

    for(i = 0; i < a->row; i++){
        ans += a->data[i][i];
    }

    return ans;
}

float abs_array_sum(float* array, int size){
    float sum = 0;
    int i = 0;

    for(i = 0; i < size; i++){
        sum += fabs(array[i]);
    }

    return sum;
}

int matrix_row_smaller_than(float* a, float* b, int size){
    int i;

    for(i = 0; i < size; i++){
        if(a[i] == b[i]){
            continue;
        }else{
            if(a[i] == 0){
                return 0;
            } else if(b[i] == 0){
                return 1;
            } else if(a[i] < b[i]){
                return 1;
            }
        }
    }

    return 0;
}

void matrix_sort(Matrix* matrix){
    int i, j, min;
    float* tmp = (float*)malloc(sizeof(float) * matrix->column);

    for(i = 0; i < matrix->row; i++){
        min = i;
        for(j = i + 1; j < matrix->row; j++){
            if(matrix_row_smaller_than(matrix->data[j], matrix->data[min], matrix->column)){
                min = j;
            }
        }

        memcpy(tmp, matrix->data[i], sizeof(float) * matrix->column);
        memcpy(matrix->data[i], matrix->data[min], sizeof(float) * matrix->column);
        memcpy(matrix->data[min], tmp, sizeof(float) * matrix->column);
    }

    free(tmp);
}

void matrix_reduce(Matrix* matrix, int column){
    int i, j, start_row = 0, end_row = matrix->column - 1;
    float t;

    matrix_sort(matrix);
    for(i = 0; i < column; i++){
        for(j = 0; j < matrix->row; j++){
            if(matrix->data[j][i] == 0 && start_row < j){
                start_row = j;
                break;
            }
        } 
    }

    while(matrix->data[end_row][column] == 0){
        end_row--;
        if(start_row == end_row){
            return;
        }
    }

    for(i = start_row; i <= end_row; i++){
        t = matrix->data[i][column];
        for(j = column; j < matrix->column; j++){
            matrix->data[i][j] /= t;
        }
    }

    for(i = start_row+1; i <= end_row; i++){
        for(j = column; j < matrix->column; j++){
            matrix->data[i][j] -= matrix->data[start_row][j];
            if(0 < matrix->data[i][j] && matrix->data[i][j] < EPS){
                matrix->data[i][j] = 0;
            }
        }
    }
}

int matrix_rank(Matrix* a){
    int rank = a->column;
    int i;
    Matrix tmp;

    initialize_matrix(&tmp, a->row, a->column);
    copy_matrix(&tmp, a);
    for(i = 0; i < MIN(a->row,a->column); i++){
        matrix_reduce(&tmp, i);
    }

    for(i = 0; i < a->row; i++){
        if(abs_array_sum(tmp.data[i], tmp.column) == 0){
            rank--;
        }
    }

    return rank;
}

void matrix_householder_transform(Matrix *matrix){
    Matrix b, p, q;
    float *u, alpha, s;
    int i, j, k, m;

    initialize_matrix(&b, matrix->row, matrix->column);
    initialize_matrix(&p, matrix->row, matrix->column);
    initialize_matrix(&q, matrix->row, matrix->column);
    u = (float*)malloc(sizeof(float)*matrix->row);

    for(k = 0; k < matrix->row - 2; k++){
        s = 0;
        for(i = k+1; i < matrix->row; i++){
            s += matrix->data[i][k] * matrix->data[i][k];
        }
        s = -((matrix->data[k+1][k] >= 0) ? 1 : -1) * sqrt(s);

        alpha = sqrt(2.0 * s * (s - matrix->data[k+1][k]));
        if(fabs(alpha) < EPS) continue;

        u[k+1] = (matrix->data[k+1][k] - s) / alpha;
        for(i = k+2; i < matrix->row; i++){
            u[i] = matrix->data[i][k] / alpha;
        }

        for(i = k+1; i < matrix->row; i++){
            for(j = i; j < matrix->row; j++){
                if(j == i){
                    p.data[i][j] = 1.0 - 2.0 * u[i] * u[i];
                }else{
                    p.data[i][j] = -2.0 * u[i] * u[j];
                    p.data[j][i] = p.data[i][j];
                }
            }
        }

        matrix_dot(&q, &p, matrix);

        for(i = 0; i <= k; i++){
            b.data[i][k] = matrix->data[i][k];
        }
        b.data[k+1][k] = s;
        for(i = k+2; i < matrix->row; i++){
            b.data[i][k] = 0;
        }
        for(j = k+1; j < matrix->row; j++){
            for(i = 0; i <= k; i++){
                b.data[i][j] = 0;
                for(m = k+1; m < matrix->row; m++){
                    b.data[i][j] += matrix->data[i][m] * p.data[j][m];
                }
            }
            for(i = k+1; i < matrix->row; i++){
                b.data[i][j] = 0;
                for(m = k+1; m < matrix->row; m++){
                    b.data[i][j] += q.data[i][m] * p.data[j][m];
                }
            }
        }

        copy_matrix(matrix, &b);
    }

    free_matrix(&b);
    free_matrix(&p);
    free_matrix(&q);
    free(u);
}

void matrix_QR(Matrix* matrix){
    Matrix r, q, t;
    float *u, *v;
    float alpha, c, s, e, rq;
    int i, j, k, m, cnt = 0;

    initialize_matrix(&r, matrix->row, matrix->column);
    initialize_matrix(&q, matrix->row, matrix->column);
    initialize_matrix(&t, matrix->row, matrix->column);
    u = (float*)malloc(sizeof(float) * matrix->row);
    v = (float*)malloc(sizeof(float) * matrix->row);

    copy_matrix(&r, matrix);

    for(;;){
        make_identity_matrix(&q);

        for(k = 0; k < matrix->row-1; k++){
            alpha = sqrt(r.data[k][k] * r.data[k][k] + r.data[k+1][k] * r.data[k+1][k]);
            if(fabs(alpha) < EPS) continue;

            c = r.data[k][k]/alpha;
            s = -r.data[k+1][k]/alpha;

            for(j = k+1; j < matrix->column; j++){
                u[j] = c*r.data[k][j] - s*r.data[k+1][j];
                v[j] = s*r.data[k][j] + c*r.data[k+1][j];
            }
            r.data[k][k] = alpha;
            r.data[k+1][k] = 0;
            for(j = k+1; j < matrix->column; j++){
                r.data[k][j] = u[j];
                r.data[k+1][j] = v[j];
            }

            for(j = 0; j <= k; j++){
                u[j] = c*q.data[k][j];
                v[j] = s*q.data[k][j]; 
            }
            q.data[k][k+1] = -s;
            q.data[k+1][k+1] = c;
            for(j = 0; j <= k; j++){
                q.data[k][j] = u[j];
                q.data[k+1][j] = v[j];
            }
        }

        for(i = 0; i < matrix->row; i++){
            for(j = 0; j < matrix->row; j++){
                rq = 0.0;
                for(m = 0; m < matrix->row; m++){
                    rq += r.data[i][m] * q.data[j][m];
                }

                t.data[i][j] = rq;
            }
        }

        e = 0.0;
        for(i = 0; i < matrix->row; i++){
            e += fabs(t.data[i][i] - matrix->data[i][i]);
        }
        if(e < EPS) break;
        if(cnt > ITER_MAX) break;

        copy_matrix(&r, &t);
        copy_matrix(matrix, &t);

        cnt++;
    }

    free_matrix(&r);
    free_matrix(&q);
    free_matrix(&t);
    free(u);
    free(v);
}

float matrix_max(Matrix* a){
    int i, j, max;

    if(a->row == 0 || a->column == 0){
        return -1;
    }

    max = a->data[0][0];

    for(i = 0; i < a->row; i++){
        for(j = 0; j < a->column; j++){
            max = MAX(max, a->data[i][j]);
        }
    }

    return max;
}

float matrix_min(Matrix* a){
    int i, j, min;

    if(a->row == 0 || a->column == 0){
        return -1;
    }

    min = a->data[0][0];

    for(i = 0; i < a->row; i++){
        for(j = 0; j < a->column; j++){
            min = MIN(min, a->data[i][j]);
        }
    }

    return min;
}

void matrix_exp(Matrix* ans, Matrix *a){
    int i, j;
    
    if(ans->row != a->row || ans->column != a->column){
        return;
    }

    for(i = 0; i < a->row; i++){
        for(j = 0; j < a->column; j++){
            ans->data[i][j] = exp(a->data[i][j]);
        }
    }
}

float matrix_sum(Matrix* a){
    float sum = 0;
    int i, j;

    for(i = 0; i < a->row; i++){
        for(j = 0; j < a->column; j++){
            sum += a->data[i][j];
        }
    }

    return sum;
}

void matrix_eigenvalues(Vector* eigenvalues, Matrix* a){
    Matrix tmp;
    
    if(a->row != a->column){
        return;
    }
    if(eigenvalues->size != a->row){
        return;
    }

    initialize_matrix(&tmp, a->row, a->column);
    copy_matrix(&tmp, a);

    matrix_householder_transform(&tmp);
    matrix_QR(&tmp);
    for(int i = 0; i < eigenvalues->size; i++){
        eigenvalues->data[i] = tmp.data[i][i];
    }
}

void make_identity_matrix(Matrix* matrix){
    int i, j;

    if(matrix->row != matrix->column){
        return;
    }

    for(i = 0; i < matrix->row; i++){
        for(j = 0; j < matrix->column; j++){
            matrix->data[i][j] = (i == j) ? 1 : 0;
        }
    }
}

void make_upper_triangular_matrix(Matrix* matrix){
    int i, j;

    if(matrix->row != matrix->column){
        return;
    }

    for(i = 0; i < matrix->row; i++){
        for(j = 0; j < matrix->column; j++){
            matrix->data[i][j] = (i >= j) ? 1 : 0;
        }
    }
}

void make_lower_triangular_matrix(Matrix* matrix){
    int i, j;

    if(matrix->row != matrix->column){
        return;
    }

    for(i = 0; i < matrix->row; i++){
        for(j = 0; j < matrix->column; j++){
            matrix->data[i][j] = (i <= j) ? 1 : 0;
        }
    }
}

void matrix_fill(Matrix* matrix, float num){
    int i, j;

    for(i = 0; i < matrix->row; i++){
        for(j = 0; j < matrix->column; j++){
            matrix->data[i][j] = num;
        }
    }
}

int matrix_convolution_output_height(Matrix* a, Matrix* kernel, int padding, int stride){
    return ((a->row + 2 * padding - kernel->row) / stride) + 1;
}

int matrix_convolution_output_width(Matrix* a, Matrix* kernel, int padding, int stride){
    return ((a->column + 2 * padding - kernel->column) / stride) + 1;
}

void matrix_convolution(Matrix* ans, Matrix* a, Matrix* kernel, int padding, int stride){
    int sum, o_h, o_w;
    Matrix p, tmp, tmp2;

    if(stride < 1){
        return;
    }

    o_h = matrix_convolution_output_height(a, kernel, padding, stride); 
    o_w = matrix_convolution_output_width(a, kernel, padding, stride);

    if(ans->row != o_h || ans->column != o_w){
        return;
    }

    initialize_matrix(&p, a->row + padding, a->column + 2);

    for(int i = 0; i < a->row; i++){
        for(int j = 0; j < a->column; j++){
            p.data[i + padding][j + padding] = a->data[i][j];
        }
    }
    
    initialize_matrix(&tmp, kernel->row, kernel->column);
    initialize_matrix(&tmp2, kernel->row, kernel->column);

    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for(int i = 0; i < o_h; i += stride){
        for(int j = 0; j < o_w; j += stride){
            for(int k = 0; k < kernel->row; k++){
                for(int m = 0; m < kernel->column; m++){
                    tmp.data[k][m] = a->data[k + i][k + j];
                }
            } 

            matrix_hadamard_product(&tmp2, &tmp, kernel);

            sum = 0;
            for(int k = 0; k < kernel->row; k++){
                for(int m = 0; m < kernel->row; m++){
                    sum += tmp2.data[k][m];
                }
            }

            ans->data[i][j] = sum;
        }
    }

    free_matrix(&p);
    free_matrix(&tmp);
    free_matrix(&tmp2);
}
