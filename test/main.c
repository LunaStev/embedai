#include <tensor.h>
#include <layers.h>
#include <model.h>
#include <stdio.h>
#include <unistd.h>
#include <unistd.h>

int main() {
    char cwd[256];
    getcwd(cwd, sizeof(cwd));
    printf("Current working directory: %s\n", cwd);
    Model* model = model_load("model/sample.emodel");
    if (!model) return 1;

    uint16_t dims[2] = {1, 3};  // 입력: 1x3
    Tensor* input = tensor_create(2, dims);
    input->data[0] = 1.0f;
    input->data[1] = 2.0f;
    input->data[2] = 3.0f;

    Tensor* output = model_run(model, input);
    tensor_print(output);

    tensor_free(input);
    tensor_free(output);
    model_free(model);

    return 0;
}
