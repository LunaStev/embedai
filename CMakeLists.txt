cmake_minimum_required(VERSION 3.22)
project(EmbedAI C)

set(CMAKE_C_STANDARD 11)

include_directories(include)

add_library(embedai STATIC
        src/tensor.c
        src/layer_dense.c
        src/layer_relu.c
        src/model_loader.c
)

add_executable(embedai_test
        test/main.c
)

target_link_libraries(embedai_test embedai)