#include "matrix.h"
#include <stdlib.h>
#include <math.h>

//Matrix Initialization
void initialize_matrix(Matrix* matrix, int row, int column){
	int i, j;
	
	matrix->row = row;
	matrix->column = column;
	
	matrix->data = (float**) malloc(sizeof(float*) * row);
	for(i = 0; i < row; i++){
		matrix->data[i] = (float*) malloc(sizeof(float) * column);
		for(j = 0; j < row; j++){
			matrix->data[i][j] = 0;
		}
	}
}

//Dot of Matrix
void dot(Matrix* a, Matrix* b, Matrix* ans){
	int i, j, k;
	initialize_matrix(ans, a->row, b->column);

	for(i = 0; i < a->row; i++){
		for(j = 0; j < b->column; j++){
			for(k = 0; k < a->column; k++){
				ans->data[i][j] += a->data[i][k] * b->data[k][j];
			}
		}
	}
}

//Determinant
float determinant(Matrix* a){
	Matrix mMatrix;
	float tmp[a->row][a->column];
	int i, j, x, y;
	float det = 0;

	if(a->row != a->column){
		return 0;
	}

	if(a->row == 2){
		return a->data[0][0] * a->data[1][1] - a->data[0][1] * a->data[1][0];
	}else if(a->row == 1){
		return a->data[0][0];
	}else{
		for(i = 0; i < a->row; i++){
			for(j = 0; j < a->column; j++){
				initialize_matrix(&mMatrix, a->row - 1, a->column - 1);
				for(x = 0; x < a->row - 1; x++){
					for(y = 0; y < a->row -1; y++){
						if(i < x){
							if(j < y){
								mMatrix.data[x][y] = a->data[x][y];
							}else{
								mMatrix.data[x][y] = a->data[x][y+1];
							}
						}else{
							if(j < y){
								mMatrix.data[x][y] = a->data[x+1][y];
							}else{
								mMatrix.data[x][y] = a->data[x+1][y+1];
							}
						}
					}
				}
				tmp[i][j] = pow(-1, i + j) * determinant(&mMatrix);
			}
		}
		
		for(i = 0; i < a->row; i++){
			for(j = 0; j < a->column; j++){
				det += a->data[i][j] * tmp[i][j];
			}
		}
		return det;
	}
}

void transpose(Matrix* a, Matrix* ans){
	int i, j;
	for(i = 0; i < a->row; i++){
		for(j = 0; j < a->column; j++){
			ans->data[j][i] = a->data[i][j];
		}
	}
}

void cMul(Matrix* a, float c, Matrix* ans){
	int i, j;
	for(i = 0; i < a->row; i++){
		for(j = 0; j < a->column; j++){
			ans->data[i][j] = c * a->data[i][j];
		}
	}
}

void inverse(Matrix* a, Matrix* ans){
	float det;
	Matrix mMatrix, tmp, tmp2;
	int i, j, x, y;

	initialize_matrix(&tmp, a->row, a->column);
	for(i = 0; i < a->row; i++){
		for(j = 0; j < a->column; j++){
			initialize_matrix(&mMatrix, a->row - 1, a->column - 1);
			for(x = 0; x < a->row - 1; x++){
				for(y = 0; y < a->row -1; y++){
					if(i < x){
						if(j < y){
							mMatrix.data[x][y] = a->data[x][y];
						}else{
							mMatrix.data[x][y] = a->data[x][y+1];
						}
					}else{
						if(j < y){
							mMatrix.data[x][y] = a->data[x+1][y];
						}else{
							mMatrix.data[x][y] = a->data[x+1][y+1];
						}
					}
				}
			}
			tmp.data[i][j] = pow(-1, i + j) * determinant(&mMatrix);
		}
	}

	transpose(&tmp, &tmp2);
	det = determinant(a);
	cMul(&tmp2, 1/det, ans);		
}

//addition between matrix
void add(Matrix* a, Matrix* b, Matrix* ans){
	int i,j;
	initialize_matrix(ans, a->row, a->column);

	for(i = 0; i < a->row; i++){
		for(j = 0; j < a->column; j++){
			ans->data[i][j] = a->data[i][j] + b->data[i][j];
		}
	}
}

//substraction between matrix
void sub(Matrix* a, Matrix* b, Matrix* ans){
	int i,j;
	initialize_matrix(ans, a->row, a->column);

	for(i = 0; i < a->row; i++){
		for(j = 0; j < a->column; j++){
			ans->data[i][j] = a->data[i][j] - b->data[i][j];
		}
	}
}

int rank(Matrix* a);

