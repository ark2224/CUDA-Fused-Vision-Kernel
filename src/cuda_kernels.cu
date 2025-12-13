#include "cuda_kernels.hpp"
#include <cuda_runtime.h>
#include <stdint.h>

// __global__ void rgba_to_gray_u8(const uint8_t* __restrict__ rgba,
//                                 uint8_t* __restrict__ gray,
//                                 int w, int h)
// {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     if (x >= w || y >= h) return;

//     int idx = y * w + x;
//     int base = idx * 4;

//     uint8_t r = rgba[base + 0];
//     uint8_t g = rgba[base + 1];
//     uint8_t b = rgba[base + 2];

//     int y_val = (77 * (int)r + 150 * (int)g + 29 * (int)b) >> 8;
//     gray[idx] = (uint8_t)y_val;
// }

// extern "C" void launch_rgba_to_gray(const uint8_t* d_rgba, uint8_t* d_gray, int w, int h) {
//     dim3 block(16, 16);
//     dim3 grid((w + block.x - 1) / block.x,
//               (h + block.y - 1) / block.y);
//     rgba_to_gray_u8<<<grid, block>>>(d_rgba, d_gray, w, h);
// }


__global__ void rgb_to_grayscale_kernel(
    const uint8_t* rgba,
    uint8_t* __restrict__ gray,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int base = idx * 3;

    uint8_t r = rgba[base + 0];
    uint8_t g = rgba[base + 1];
    uint8_t b = rgba[base + 2];

    gray[y * width + x] = 
        static_cast<uint8_t>(0.299f * r +
                             0.587f * g + 
                             0.114f * b);
}

void launch_rgb_to_grayscale_cuda(
    const uint8_t* h_rgb,
    uint8_t* h_gray,
    int width,
    int height
) {
    uint8_t* d_rgb = nullptr;
    uint8_t* d_gray = nullptr;

    size_t rgb_bytes = width * height * 3;
    size_t gray_bytes = width * height;

    cudaMalloc(&d_rgb, rgb_bytes);
    cudaMalloc(&d_gray, gray_bytes);

    cudaMemcpy(d_rgb, h_rgb, rgb_bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    rgb_to_grayscale_kernel<<<grid, block>>>(
        d_rgb, d_gray, width, height
    );

    cudaMemcpy(h_gray, d_gray, gray_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_rgb);
    cudaFree(d_gray);
}