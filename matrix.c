#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdarg.h>
#include <string.h>

#define EPS 0.00000001
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

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

void vector_add(Vector *ans, Vector *a, Vector *b){
    int i;

    if(a->size != b->size){
        return;
    }
    if(a->size != ans->size){
        return;
    }

    for(i = 0; i < a->size; i++){
        ans->data[i] = a->data[i] + b->data[i];
    }
}

void vector_sub(Vector *ans, Vector *a, Vector *b){
    int i;

    if(a->size != b->size){
        return;
    }
    if(a->size != ans->size){
        return;
    }

    for(i = 0; i < a->size; i++){
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

//Dot of Matrix
void matrix_dot(Matrix* ans, Matrix* a, Matrix* b){
    int i, j, k;
    initialize_matrix(ans, a->row, b->column);

    if(a->column != b->row || a->row != ans->row || b->column != ans->column){
        return;
    }

    for(i = 0; i < a->row; i++){
        for(j = 0; j < b->column; j++){
            for(k = 0; k < a->column; k++){
                ans->data[i][j] += a->data[i][k] * b->data[k][j];
            }
        }
    }
}

//Determinant
float matrix_determinant(Matrix* a){
    Matrix mMatrix;
    float tmp[a->column];
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
        return det;
    }
}

void matrix_transpose(Matrix* ans, Matrix* a){
    int i, j;

    if(a->row != ans->column || a->column != ans->row){
        return;
    }

    for(i = 0; i < a->row; i++){
        for(j = 0; j < a->column; j++){
            ans->data[j][i] = a->data[i][j];
        }
    }
}

void matrix_cMul(Matrix* ans, Matrix* a, float c){
    int i, j;
    
    if(a->row != ans->row || a->column != ans->column){
        return;
    }

    for(i = 0; i < a->row; i++){
        for(j = 0; j < a->column; j++){
            ans->data[i][j] = c * a->data[i][j];
        }
    }
}

void matrix_inverse(Matrix *ans, Matrix* a){
    float det;
    Matrix mMatrix, tmp, tmp2;
    int i, j, x, y;

    if(a->row != a->column){
        return;
    }
    if(a->row != ans->row || a->column != ans->column){
        return;
    }

    initialize_matrix(&tmp, a->row, a->column);
    initialize_matrix(&tmp2, a->row, a->column);
    initialize_matrix(&mMatrix, a->row - 1, a->column - 1);
    for(i = 0; i < a->row; i++){
        for(j = 0; j < a->column; j++){
            for(x = 0; x < a->row - 1; x++){
                for(y = 0; y < a->row -1; y++){
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
    int i,j;
    initialize_matrix(ans, a->row, a->column);

    if(a->row != b->row || a->column != b->column || 
            ans->row != a->row || ans->column != a->column){
        return;
    }

    for(i = 0; i < a->row; i++){
        for(j = 0; j < a->column; j++){
            ans->data[i][j] = a->data[i][j] + b->data[i][j];
        }
    }
}

//substraction between matrix
void matrix_sub(Matrix* ans, Matrix* a, Matrix* b){
    int i,j;
    initialize_matrix(ans, a->row, a->column);

    if(a->row != b->row || a->column != b->column || 
            ans->row != a->row || ans->column != a->column){
        return;
    }

    for(i = 0; i < a->row; i++){
        for(j = 0; j < a->column; j++){
            ans->data[i][j] = a->data[i][j] - b->data[i][j];
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
    float* tmp; 

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

void matrix_eigenvalues(Matrix* eigenvalues, Matrix* a);
