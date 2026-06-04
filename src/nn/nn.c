#include "nn.h"
#include <stdlib.h>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>

// linear layer

Linear* linear_create(size_t in_features, size_t out_features, Activation* activation) {
    Linear* layer = malloc(sizeof(Linear));

    layer->W = mat_create(in_features, out_features);

    for (size_t i = 0; i < in_features * out_features; i++) {
        // box muller transform
        double u1 = (double) rand() / RAND_MAX;
        double u2 = (double) rand() / RAND_MAX;
        layer->W->data[i] = sqrt(-2.0 * log(u1))  * sin(2.0 * M_PI * u2) / sqrt(in_features); 
    }

    layer->b = mat_create(1, out_features);
    layer->activation = activation; // NULL if end of net
    layer->x = NULL;
    layer->dW = NULL;
    layer->db = NULL;

    return layer;
}

void linear_free(Linear* layer) {
    mat_free(layer->W);
    mat_free(layer->b);

    if (layer->x != NULL) {mat_free(layer->x);}
    if (layer->dW != NULL) {mat_free(layer->dW);}
    if (layer->db != NULL) {mat_free(layer->db);}

    free(layer);
}

Matrix* linear_forward(Linear* layer, Matrix* x) {
    if (layer->x != NULL) {
        mat_free(layer->x);
    }

    // forward pass
    Matrix* temp = mat_mul(x, layer->W);
    Matrix* z = mat_add_broadcast(temp, layer->b);

    // cache (copy so layer owns it)
    layer->x = mat_create(x->rows, x->cols);
    for (size_t i = 0; i < x->rows * x->cols; i++) { layer->x->data[i] = x->data[i]; }

    mat_free(temp);
    return z;
}

Matrix* linear_backward(Linear* layer, Matrix* grad_out) {
    if (layer->dW != NULL) {
        mat_free(layer->dW);
    }

    Matrix* w_temp = mat_transpose(layer->x);
    layer->dW = mat_mul(w_temp, grad_out);
    mat_free(w_temp);

    if (layer->db != NULL) {
        mat_free(layer->db);
    }

    Matrix* db = mat_create(1, grad_out->cols);
    // column-wise sum
    for (size_t j = 0; j < grad_out->cols; j++) {

        double sum = 0;

        for(size_t i = 0; i < grad_out->rows; i++) {
            sum += AT(grad_out, i, j);
        }

        db->data[j] = sum;
    }

    layer->db = db;

    Matrix* x_temp = mat_transpose(layer->W);
    Matrix* dx = mat_mul(grad_out, x_temp);
    mat_free(x_temp);

    return dx;
}

// sgd step function

void sgd_step(Linear** layers, int n_layers, double lr) {
    for (int i = 0; i < n_layers; i++) {
        Matrix* W_temp = smat_mul(lr, layers[i]->dW);
        Matrix* W = mat_sub(layers[i]->W, W_temp);

        mat_free(W_temp);
        mat_free(layers[i]->W);
        layers[i]->W = W;


        Matrix* b_temp = smat_mul(lr, layers[i]->db);
        Matrix* b = mat_sub(layers[i]->b, b_temp);

        mat_free(b_temp);
        mat_free(layers[i]->b);
        layers[i]->b = b;
    }
}

// mlp funcs

Mlp* mlp_create(Linear** layers, size_t n_layers) {
    // create mlp with linear layers
    Mlp* mlp = malloc(sizeof(Mlp));
    mlp->layers = layers;
    mlp->n_layers = n_layers;
    mlp->pre_activations = calloc(n_layers, sizeof(Matrix*));

    return mlp;
}

void mlp_free(Mlp* mlp) {
    for (size_t i = 0; i < mlp->n_layers; i++) {
        linear_free(mlp->layers[i]);
        if (mlp->pre_activations[i] != NULL){mat_free(mlp->pre_activations[i]);}
    }
    free(mlp->pre_activations);
    free(mlp);
}

Matrix* mlp_forward(Mlp* mlp, Matrix* x) {
    // reset pre activations before pass
    if (mlp->pre_activations[0] != NULL) { // use calloc so correct
        for (int i = 0; i < mlp->n_layers; i++) {
            mat_free(mlp->pre_activations[i]);
        }
    }

    Matrix* x_curr = x;
    for (int i = 0; i < mlp->n_layers; i++) {
        Matrix* z = linear_forward(mlp->layers[i], x_curr);

        mlp->pre_activations[i] = z;

        if (mlp->layers[i]->activation != NULL) {
            x_curr = mlp->layers[i]->activation->forward(z);
        }
        else {
            x_curr = z;
        }
    }

    return x_curr;
}

Matrix* mlp_backward(Mlp* mlp, Matrix* grad) {
    Matrix* grad_cur = grad;

    for (int i = mlp->n_layers - 1; i >= 0; i--) {
        Matrix* prev = grad_cur;

        if (mlp->layers[i]->activation != NULL) {
            Matrix* z = mlp->layers[i]->activation->backward(mlp->pre_activations[i], grad_cur);
            grad_cur = linear_backward(mlp->layers[i], z);
            mat_free(z);
        } else {
            grad_cur = linear_backward(mlp->layers[i], grad_cur);
        }

        if (prev != grad) { mat_free(prev); }
    }

    return grad_cur;
}

void mlp_fit(Mlp* mlp, Matrix* X, Matrix* y, long epochs, double lr) {
    for (long i = 0; i < epochs; i++) {
        Matrix* y_pred = mlp_forward(mlp, X);
        double loss = mse(y_pred, y);

        if (i % 100 == 0) {
            printf("epoch %ld loss: %f\n", i, loss);
        }

        Matrix* grad = dmse(y_pred, y);
        Matrix* backward = mlp_backward(mlp, grad);

        sgd_step(mlp->layers,mlp->n_layers, lr);

        if (y_pred != mlp->pre_activations[mlp->n_layers - 1]) { mat_free(y_pred); }
        mat_free(grad);
        mat_free(backward);
    }
}