#include <tensor.h>
#include <layers.h>

int main() {
    // Input vector [1 x 3]
    u_int16_t input_dims[2] = {1, 3};
    Tensor* input = tensor_create(2, input_dims);
    input->data[0] = 1.0f;
    input->data[1] = 2.0f;
    input->data[2] = 3.0f;

    // Weight matrix [3 x 2]
    u_int16_t weight_dims[2] = {3, 2};
    Tensor* weights = tensor_create(2, weight_dims);
    weights->data[0] = 0.1f; weights->data[1] = 0.2f;
    weights->data[2] = 0.3f; weights->data[3] = 0.4f;
    weights->data[4] = 0.5f; weights->data[5] = 0.6f;

    // Bias vector [2]
    u_int16_t bias_dims[1] = {2};
    Tensor* bias = tensor_create(1, bias_dims);
    bias->data[0] = 0.5f;
    bias->data[1] = -0.5f;

    // Dense operation
    Tensor* output = layer_dense_forward(input, weights, bias);

    // Print result
    tensor_print(output);

    // Free memory
    tensor_free(input);
    tensor_free(weights);
    tensor_free(bias);
    tensor_free(output);

    return 0;
}
