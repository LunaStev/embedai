//
// Created by sobi1 on 2025-05-19.
//

#ifndef EMBEDAI_TENSOR_H
#define EMBEDAI_TENSOR_H

#include <stdlib.h>
#include <stddef.h>

typedef struct {
    float* data;
    size_t size;
    u_int16_t dims[4];
    u_int8_t ndim;
} Tensor;

Tensor* tensor_create(u_int8_t ndim, const u_int16_t* dims);
void tensor_free(Tensor* tensor);

void tensor_print(const Tensor* tensor);

#endif //EMBEDAI_TENSOR_H
