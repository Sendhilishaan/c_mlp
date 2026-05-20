#include "functions.h"

Matrix* sigmoid(Matrix* A) {
    Matrix* result = mat_create(A->rows, A->cols);

    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j++) {
            AT(result, i, j) = 1 / (1 + exp(-(AT(A, i, j))));
        }
    }

    return result;
}

Matrix* dsigmoid(Matrix* A, Matrix* grad_out) {
    Matrix* s = sigmoid(A);
    Matrix* temp = smat_add(s, -1);
    Matrix* sum = smat_mul(-1, temp);

    Matrix* dz_dx = mat_hadamard(s, sum);

    Matrix* result = mat_hadamard(dz_dx, grad_out);

    mat_free(s);
    mat_free(temp);
    mat_free(sum);
    mat_free(dz_dx);

    return result;
}

Matrix* relu(Matrix* A) {
    Matrix* result = mat_create(A->rows, A->cols);

    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j++) {
            AT(result, i, j) = (AT(A, i, j) > 0) ? AT(A, i, j) : 0;
        }
    }

    return result;
}

Matrix* drelu(Matrix* A, Matrix* grad_out) {
    Matrix* temp = mat_create(A->rows, A->cols);

    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j++) {
            AT(temp, i, j) = (AT(A, i, j) > 0) ? 1 : 0;
        }
    }

    Matrix* result = mat_hadamard(temp, grad_out);
    mat_free(temp);

    return result;
}

// loss
double mse(Matrix* y_pred, Matrix* y_true) {
    double sum = 0.0;
    Matrix* diff = mat_sub(y_pred, y_true);

    for (size_t i = 0; i < diff->rows; i++) {
        sum += pow(AT(diff, i, 0), 2);
    }

    mat_free(diff);

    return sum / y_pred->rows;
}

Matrix* dmse(Matrix* y_pred, Matrix* y_true) {
    Matrix* diff = mat_sub(y_pred, y_true);
    Matrix* result = smat_mul(2.0 / (double) diff->rows, diff);

    mat_free(diff);
    return result;
}