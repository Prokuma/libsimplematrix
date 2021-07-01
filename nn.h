#ifndef _NN_H_
#include "matrix.h"

typedef struct nnLinear{
    Matrix w;
    Matrix b;
    int inputSize;
    int outputSize;
} NNLinear;

void nn_forward_sigmoid(Matrix *output, Matrix *input);
void nn_forward_lelu(Matrix *output, Matrix *input);
void nn_forward_softmax(Matrix *output, Matrix *input);

void nn_create_linear_layer(NNLinear *output, int inputSize, int outputSize);
void nn_forward_linear_layer(Matrix *output, NNLinear *layer, Matrix *input);

#endif
