#ifndef NN_H
#define NN_H

#include "math/matrix.h"

typedef struct {
    Matrix* W;
    Matrix* b;
    Matrix* x;
    Matrix* dW;
    Matrix* db;
} Linear;

Linear* linear_create(size_t in_features, size_t out_features);

Matrix* linear_forward(Linear* layer, Matrix* x);

Matrix* linear_backward(Linear* layer, Matrix* grad_out);

#endif