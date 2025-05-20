#include <model.h>
#include <layers.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Model* model_load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("Failed to open file %s\n", path);
        return NULL;
    }

    char magic[4];
    fread(magic, 1, 4, f);
    if (memcmp(magic, "EDL", 4)) {
        printf("Invalid model file. %s\n", path);
        fclose(f);
        return NULL;
    }

    Model* model = (Model*)calloc(1, sizeof(Model));
    fread(&model->version, 1, 1, f);
    fread(&model->layers, 1, 1, f);

    for (int i = 0; i < model->layer_count; ++i) {
        uint8_t type;
        uint16_t in_dim, out_dim, w_count, b_count;

        fread(&type, 1, 1, f);
        fread(&in_dim, 2, 1, f);
        fread(&out_dim, 2, 1, f);
        fread(&w_count, 2, 1, f);
        fread(&b_count, 2, 1, f);

        model->layers[i].type = type;

        if (type == LAYER_DENSE) {
            uint16_t w_dim[2] = {in_dim, out_dim};
            model->layers[i].weights = tensor_create(2, w_dim);
            fread(model->layers[i].weights->data, sizeof(float), w_count, f);

            uint16_t b_dim[1] = {out_dim};
            model->layers[i].bias = tensor_create(1, b_dim);
            fread(model->layers[i].bias->data, sizeof(float), b_count, f);
        } else if (type == LAYER_RELU) {
            model->layers[i].weights = NULL;
            model->layers[i].bias = NULL;
        } else {
            printf("Unknown layer type: %d\n", type);
            model_free(model);
            fclose(f);
            return NULL;
        }
    }

    fclose(f);
    return model;
}

void model_free(Model* model) {
    if (!model) return;
    for (int i = 0; i < model->layer_count; ++i) {
        tensor_free(model->layers[i].weights);
        tensor_free(model->layers[i].bias);
    }
    free(model);
}

Tensor* model_run(const Model* model, const Tensor* input) {
    Tensor* current = tensor_create(input->ndim, input->dims);
    memcpy(current->data, input->data, sizeof(float) * input->size);

    for (int i = 0; i < model->layer_count; ++i) {
        Layer layer = model->layers[i];
        Tensor* next = NULL;

        if (layer.type == LAYER_DENSE) {
            next = layer_dense_forward(current, layer.weights, layer.bias);
        } else if (layer.type == LAYER_RELU) {
            next = tensor_create(current->ndim, current->dims);
            for (size_t i = 0; i < current->size; ++i) {
                next->data[i] = current->data[i] > 0 ? current->data[i] : 0;
            }
        } else {
            printf("Unsupported layer type: %d\n", layer.type);
            tensor_free(current);
            return NULL;
        }

        tensor_free(current);
        current = next;
    }

    return current;
}