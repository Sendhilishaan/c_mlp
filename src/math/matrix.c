#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

Matrix* mat_create(size_t rows, size_t cols){
    // 0 init matrix on heap
    Matrix* mat = malloc(sizeof(Matrix));

    if (mat == NULL) perror("mat create");

    mat->data = calloc(rows * cols, sizeof(double));

    if (mat->data == NULL) perror("mat create data");

    mat->cols = cols;
    mat->rows = rows;

    return mat;
} 

void mat_free(Matrix* M) {
    free(M->data);
    free(M);
}

void mat_print(Matrix* M) {
    for (size_t i = 0; i < M->rows; i++) {
        for (size_t j = 0; j < M->cols; j++) {
            printf("%lf ", AT(M, i, j));

            if (j == M->cols - 1) printf("\n");
        }
    }
}

Matrix* mat_mul(Matrix* A, Matrix* B) {
    // AB

    if (A->cols != B->rows) {
        printf("wrong dim");
        return NULL;
    }

    Matrix* result = mat_create(A->rows, B->cols);

    // simple, slow triple loop (optimise later)
    for (size_t i = 0; i < result->rows; i++) {
        for (size_t j = 0; j < result->cols; j++) {
            for (size_t k = 0; k < A->cols; k++) { // or B->rows, same
                AT(result, i, j) += AT(A, i, k) * AT(B, k, j);
            }
        }
    }

    return result;
}