#ifndef NN_H
#define NN_H

#include "math/matrix.h"
#include "math/functions.h"

// linear layer stuff

typedef struct {
    Matrix* W;
    Matrix* b;
    Matrix* x;
    Matrix* dW;
    Matrix* db;
    Activation* activation;
} Linear;

Linear* linear_create(size_t in_features, size_t out_features, Activation* activation);

Matrix* linear_forward(Linear* layer, Matrix* x);

Matrix* linear_backward(Linear* layer, Matrix* grad_out);

// mlp / nn stuff

typedef struct {
    Linear** layers; 
    Matrix** pre_activations; // cache each forward pass
    int n_layers;
} Mlp;

#endif