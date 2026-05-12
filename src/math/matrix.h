#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>
#define AT(M, i, j) ((M)->data[(i) * (M)->cols + (j)])


typedef struct {
    size_t rows;
    size_t cols;
    double* data; // row-major order (row * num_cols + col)
} Matrix;

Matrix* mat_create(size_t rows, size_t cols); // 0 init matrix on heap

void mat_free(Matrix* M);

void mat_print(Matrix* M);

Matrix* mat_mul(Matrix* A, Matrix* B);

#endif