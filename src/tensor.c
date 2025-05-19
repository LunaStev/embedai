//
// Created by sobi1 on 2025-05-19.
//

#include <tensor.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

Tensor* tensor_create(u_int8_t ndim, const u_int16_t* dims) {
    if (ndim > 4) return NULL;

    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) return NULL;

    tensor->ndim = ndim;
    tensor->size = 1;

    for (int i = 0; i < ndim; ++i) {
        tensor->dims[i] = dims[i];
        tensor->size *= dims[i];
    }

    tensor->data = (float*)calloc(tensor->size, sizeof(float));
    if (!tensor->data) {
        free(tensor);
        return NULL;
    }

    return tensor;
}

void tensor_free(Tensor* tensor) {
    if (!tensor) return;
    free(tensor->data);
    free(tensor);
}

void tensor_print(const Tensor* tensor) {
    printf("Tensor [");
    for (int i = 0; i < tensor->ndim; ++i) {
        printf("%d", tensor->dims[i]);
        if (i < tensor->ndim - 1) printf("x");
    }
    printf("] = {");
    for (size_t i = 0; i < tensor->size; ++i) {
        printf("%.2f ", tensor->data[i]);
        if (i >= 9) {
            printf("..."); break;
        }
    }
    printf("}\n");
}