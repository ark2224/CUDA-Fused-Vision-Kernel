#include <cuda_runtime.h>
#include <stdint.h>

__global__ void rgba_to_gray_u8(const uint8_t* __restrict__ rgba,
                                uint8_t* __restrict__ gray,
                                int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * w + x;
    int base = idx * 4;

    uint8_t r = rgba[base + 0];
    uint8_t g = rgba[base + 1];
    uint8_t b = rgba[base + 2];

    int y_val = (77 * (int)r + 150 * (int)g + 29 * (int)b) >> 8;
    gray[idx] = (uint8_t)y_val;
}

extern "C" void launch_rgba_to_gray(const uint8_t* d_rgba, uint8_t* d_gray, int w, int h) {
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x,
              (h + block.y - 1) / block.y);
    rgba_to_gray_u8<<<grid, block>>>(d_rgba, d_gray, w, h);
}