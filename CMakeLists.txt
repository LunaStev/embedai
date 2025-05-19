cmake_minimum_required(VERSION 3.22)
project(EmbedAI C)

set(CMAKE_C_STANDARD 11)

include_directories(include)

file(GLOB_RECURSE SRC_FILES
        src/*.c
)

add_library(embedai STATIC ${SRC_FILES})

add_executable(embedai_test
        test/main.c
)

target_link_libraries(embedai_test embedai)