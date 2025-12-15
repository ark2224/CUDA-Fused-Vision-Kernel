cmake_minimum_required(VERSION 3.18)
project(Cuda-Fused-Vision-Kernel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cuda_vision
    src/main.cpp
    src/cpu_baseline.cpp
    src/cuda_kernels.cu
    src/gpu_pipeline.cu
)

target_include_directories(cuda_vision PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/stb
)

target_compile_options(cuda_vision PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
)

if (MSVC)
    target_compile_definitions(cuda_vision PRIVATE /W4)
else()
    target_compile_options(cuda_vision PRIVATE -Wall -Wextra -Wpedantic)
endif()

set_target_properties(cuda_vision PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

set(CMAKE_CUDA_ARCHITECTURES 89)