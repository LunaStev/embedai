#include <layers.h>
#include <stdlib.h>

Tensor* layer_dense_forward(const Tensor* input, const Tensor* weights, const Tensor* bias) {
    if (input->ndim != 2 || weights->ndim != 2 || bias->ndim != 1) {
        return NULL;
    }

    uint16_t in_features = input->dims[1];
    uint16_t out_features = weights->dims[1];

    if (input->dims[1] != weights->dims[0] || bias->dims[0] != out_features) {
        return NULL;
    }

    uint16_t out_dims[2] = {1, out_features};
    Tensor* output = tensor_create(2, out_dims);

    for (int i = 0; i < out_features; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < in_features; ++j) {
            sum += input->data[j] * weights->data[j * out_features + i];
        }
        output->data[i] = sum + bias->data[i];
    }

    return output;
}