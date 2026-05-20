#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#include "matrix.h"
#include <math.h>

// activations

Matrix* sigmoid(Matrix* A);

Matrix* dsigmoid(Matrix* A, Matrix* grad_out);

Matrix* relu(Matrix* A);

Matrix* drelu(Matrix* A, Matrix* grad_out);

// loss

double mse(Matrix* y_pred, Matrix* y_true);

Matrix* dmse(Matrix* y_pred, Matrix* y_true);

#endif