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
    layer->x = mat_create(1, out_features);
    layer->dW = mat_create(1, out_features);
    layer->db = mat_create(1, out_features);
}