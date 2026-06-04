#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#include "matrix.h"
#define _USE_MATH_DEFINES
#include <math.h>

// activations

typedef struct {
    Matrix* (*forward)(Matrix*);
    Matrix* (*backward)(Matrix*, Matrix*);
} Activation;

extern Activation SIGMOID;
extern Activation RELU;
extern Activation TANH;

Matrix* tanh_a(Matrix* A);

Matrix* dtanh_a(Matrix* A, Matrix* grad_out);

Matrix* sigmoid(Matrix* A);

Matrix* dsigmoid(Matrix* A, Matrix* grad_out);

Matrix* relu(Matrix* A);

Matrix* drelu(Matrix* A, Matrix* grad_out);

// loss

double mse(Matrix* y_pred, Matrix* y_true);

Matrix* dmse(Matrix* y_pred, Matrix* y_true);

#endif