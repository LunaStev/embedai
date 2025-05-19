#ifndef EMBEDAI_LAYERS_H
#define EMBEDAI_LAYERS_H

#include <tensor.h>

Tensor* layer_dense_forward(const Tensor* input, const Tensor* weights, const Tensor* bias);

#endif //EMBEDAI_LAYERS_H
