#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

inline void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}