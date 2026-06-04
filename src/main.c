#include <stdio.h>
#include "math/matrix.h"
#include "nn/nn.h"
#include "math/functions.h"

int main() {

    Matrix* X = mat_create(1000, 1);

    for (size_t i = 0; i < X->cols * X->rows; i++) {
        X->data[i] = (double) i / (999 / (2 * M_PI));
    }

    Matrix* y = mat_create(1000, 1);

    for (size_t i = 0; i < y->cols * y->rows; i++) {
        y->data[i] = sin(X->data[i]);
    }

    Linear* l1 = linear_create(1, 32, &RELU);
    Linear* l2 = linear_create(32, 32, &RELU);
    Linear* l3 = linear_create(32, 32, &RELU);
    Linear* l4 = linear_create(32, 1, NULL);

    Linear* layers[] = {l1, l2, l3, l4};

    Mlp* mlp = mlp_create(layers, 4);
    mlp_fit(mlp, X, y, 100001, 0.001);

    mat_free(X);
    mat_free(y);
    linear_free(l1);
    linear_free(l2);
    linear_free(l3);
    linear_free(l4);
    mlp_free(mlp);

   return 0; 
}