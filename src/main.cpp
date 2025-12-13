#include "cpu_baseline.hpp"
#include "cuda_kernels.hpp"
#include <iostream>
#include <chrono>

void main() {

    // CPU / GPU comparison
    std::vector<uint8_t> cpu_gray(width * height);
    std::vector<uint8_t> gpu_gray(width * height);

    cpu_rgb_to_grayscale(
        image_rgb.data(),
        cpu_gray.data(),
        width,
        height
    );
    cudaMemcpy(
        gpu_gray.data(),
        d_gray,
        width * height * sizeof(uint8_t),
        cudaMemcpyDeviceToHost
    );

    int mismatch_count = 0;

    for (int i = 0; i < width * height; ++i) {
        int diff = std::abs(
            static_cast<int>(cpu_gray[i]) - 
            static_cast<int>(gpu_gray[i])
        );

        if (diff > 1) {
            ++mismatch_count;
            if (mismatch_count < 10) {
                std::cerr << "Mismatch at index " << i
                          << ": CPU=" << (int)cpu_gray[i]
                          << " GPU=" << (int)gpu_gray[i] << "\n";
            }
        }
    }

    if (mismatch_count > 0) {
        std::cerr << "Total mismatches: " << mismatch_count << "\n";
    } else {
        std::cout << "CPU/GPU outputs match.\n";
    }

}