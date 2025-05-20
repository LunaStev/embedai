#ifndef EMBEDAI_MODEL_H
#define EMBEDAI_MODEL_H

#include <tensor.h>

#define MAX_LAYERS 8

typedef enum {
    LAYER_DENSE = 1,
    LAYER_RELU = 2,
} LayerType;

typedef struct {
    LayerType type;
    Tensor* weights;
    Tensor* bias;
} Layer;

typedef struct {
    uint8_t version;
    uint8_t layer_count;
    Layer layers[MAX_LAYERS];
} Model;

Model* model_load(const char* path);
Tensor* model_run(const Model* model, const Tensor* input);
void model_free(Model* model);

#endif //EMBEDAI_MODEL_H
