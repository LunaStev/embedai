#ifndef EMBEDAI_TENSOR_H
#define EMBEDAI_TENSOR_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    float* data;
    size_t size;
    uint16_t dims[4];
    uint8_t ndim;
} Tensor;

Tensor* tensor_create(uint8_t ndim, const uint16_t* dims);
void tensor_free(Tensor* tensor);

void tensor_print(const Tensor* tensor);

#endif //EMBEDAI_TENSOR_H
