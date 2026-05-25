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

}