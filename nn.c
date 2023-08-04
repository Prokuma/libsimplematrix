#include "nn.h"
#include <math.h>
#ifdef USE_MP
#include <omp.h>
#endif

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

void nn_forward_sigmoid(Matrix *output, Matrix *input){
    if(input->row != output->row || input->column != output->column){
        return;
    }

    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for(int i = 0; i < input->row; i++){
        for(int j = 0; j < input->column; j++){
            output->data[i][j] = 1 / (1 + exp(-input->data[i][j]));
        }
    }
}

void nn_forward_lelu(Matrix *output, Matrix *input){
    if(input->row != output->row || input->column != output->column){
        return;
    }

    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for(int i = 0; i < input->row; i++){
        for(int j = 0; j < input->column; j++){
            output->data[i][j] = MAX(0, input->data[i][j]);
        }
    }
}

void nn_forward_softmax(Matrix *output, Matrix *input){
    float c, sum;
    Matrix tmp;
    
    if(input->row != output->row || input->column != output->column){
        return;
    }

    initialize_matrix(&tmp, input->row, input->column);

    c = matrix_max(input);
    matrix_fill(&tmp, c);
    matrix_sub(&tmp, input, &tmp);
    matrix_exp(&tmp, &tmp);
    sum = matrix_sum(&tmp);

    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for(int i = 0; i < input->row; i++){
        for(int j = 0; j < input->column; j++){
            output->data[i][j] = exp(tmp.data[i][j]) / sum;
        }
    }

    free_matrix(&tmp);
}

void nn_create_linear_layer(NNLinear *output, int inputSize, int outputSize){
    initialize_matrix(&(output->w), outputSize, inputSize);
    initialize_matrix(&(output->b), outputSize, 1);

    output->inputSize = inputSize;
    output->outputSize = outputSize;
}

void nn_forward_linear_layer(Matrix *output, NNLinear *layer, Matrix *input){
    Matrix wx;

    initialize_matrix(&wx, output->row, 1);
    matrix_dot(&wx, &(layer->w), input);
    matrix_add(output, &wx, &(layer->b));

    free_matrix(&wx);
}
