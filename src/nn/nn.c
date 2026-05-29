#include "nn.h"
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>

Linear* linear_create(size_t in_features, size_t out_features) {
    Linear* layer = malloc(sizeof(Linear));

    layer->W = mat_create(in_features, out_features);

    for (size_t i = 0; i < in_features * out_features; i++) {
        // box muller transform
        double u1 = (double) rand() / RAND_MAX;
        double u2 = (double) rand() / RAND_MAX;
        layer->W->data[i] = sqrt(-2.0 * log(u1))  * sin(2.0 * M_PI * u2) / sqrt(in_features); 
    }

    layer->b = mat_create(1, out_features);
    layer->x = NULL;
    layer->dW = NULL;
    layer->db = NULL;

    return layer;
}

Matrix* linear_forward(Linear* layer, Matrix* x) {
    if (layer->x != NULL) {
        mat_free(layer->x);
    }

    // forward pass
    Matrix* temp = mat_mul(x, layer->W);
    Matrix* z = mat_add_broadcast(temp, layer->b);

    // cache
    layer->x = x;

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

