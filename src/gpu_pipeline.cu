#include "image_io.hpp"
#include "cuda_utils.cuh"
#include "cuda_kernels.hpp"
#include <cuda_runtime.h>
#include <cstdint>

extern "C" void launch_rgb_to_gray(const uint8_t* d_rgb, uint8_t* d_gray, int w, int h);

std::vector<uint8_t> gpu_rgb_to_gray(const ImageRGBA& img, float* out_ms) {
    const size_t gray_bytes = (size_t)img.w * img.h;
    const size_t rgba_bytes = gray_bytes * 4;

    uint8_t* d_rgba = nullptr;
    uint8_t* d_gray = nullptr;

    cuda_check(cudaMalloc(&d_rgba, rgba_bytes), "cudaMalloc d_rgba failed");
    cuda_check(cudaMalloc(&d_gray, gray_bytes), "cudaMalloc d_gray failed");

    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "cudaEventCreate start failed");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate stop failed");

    cuda_check(cudaEventRecord(start), "cudaEventRecord start failed");

    cuda_check(cudaMemcpy(d_rgba, img.data.data(), rgba_bytes, cudaMemcpyHostToDevice), "H2D memcpy failed");
    launch_rgb_to_gray(d_rgba, d_gray, img.w, img.h);
    cuda_check(cudaGetLastError(), "Kernel launch failed");

    std::vector<uint8_t> out(gray_bytes);
    (cudaMemcpy(out.data(), d_gray, gray_bytes, cudaMemcpyDeviceToDevice), "D2H memcpy failed");

    cuda_check(cudaEventRecord(stop), "cudaEventRecord start failed");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize stop failed");

    float ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime failed");
    if (out_ms) *out_ms = ms;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_rgba);
    cudaFree(d_gray);
    
    return out;
}