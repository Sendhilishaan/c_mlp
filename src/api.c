#include <string.h>
#include "api.h"
#include "nn.h"
#include "functions.h"

EXPORT void* api_linear_create(int in_features, int out_features, const char* activation) {
    Activation* act;

    if (activation == NULL) act = NULL;

    else if (strcmp(activation, "SIGMOID") == 0) {
        act = &SIGMOID;
    }
    else if (strcmp(activation, "RELU") == 0) {
        act = &RELU;
    }
    else if (strcmp(activation, "TANH") == 0) {
        act = &TANH;
    }
    else return NULL;

    return linear_create(in_features, out_features, act);
}

EXPORT void* api_mlp_create(void** layers, int n_layers) {
    return mlp_create( (Linear**) layers, n_layers);
}

EXPORT void api_mlp_run(void* mlp, double* X, double* y, int rows, int cols, int epochs, double lr) {
    // rows and cols for X y shape
    Matrix X_mat = {.data = X, .rows = rows, .cols = cols};
    Matrix y_mat = {.data = y, .rows = rows, .cols = 1};

    mlp_fit(mlp, &X_mat, &y_mat, epochs, lr);
}