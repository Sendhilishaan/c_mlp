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
    Activation* activation; // NULL if end of net
} Linear;

Linear* linear_create(size_t in_features, size_t out_features, Activation* activation);

Matrix* linear_forward(Linear* layer, Matrix* x);

Matrix* linear_backward(Linear* layer, Matrix* grad_out);

void sgd_step(Linear** layers, int n_layers, double lr);

// mlp / nn stuff

typedef struct {
    Linear** layers; 
    Matrix** pre_activations; // cache each forward pass
    size_t n_layers;
} Mlp;

Mlp* mlp_create(Linear** layers, int n_layers);

Matrix* mlp_forward(Mlp* mlp, Matrix* x);

Matrix* mlp_backward(Mlp* mlp, Matrix* grad);

void mlp_fit(Mlp* mlp, Matrix* X, Matrix* y, int epochs, double lr);

void linear_free(Linear* layer);

void mlp_free(Mlp* mlp);

#endif