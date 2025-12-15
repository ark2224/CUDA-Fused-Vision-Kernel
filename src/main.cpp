#include <vector>
#include <iostream>
#include <cstdint>
#include "cpu_baseline.hpp"
#include "cuda_kernels.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

bool load_image_rgba(
    const char* path,
    std::vector<uint8_t>& rgba,
    int& width,
    int& height
) {
    int channels = 0;
    
    unsigned char* data = stbi_load(
        path, &width,
        &height,
        &channels,
        4
    );

    if (!data) {
        std::cerr << "Failed to load image: " << path << "\n";
        return false;
    }

    size_t size = static_cast<size_t>(width) *
                  static_cast<size_t>(height) * 4;

    rgba.assign(data, data + size);

    stbi_image_free(data);
    return true;
}


void rgba_to_rgb(
    const std::vector<uint8_t>& rgba,
    std::vector<uint8_t>& rgb
) {
    size_t pixels = rgba.size() / 4;
    rgb.resize(pixels * 3);

    for (size_t i = 0; i < pixels; ++i) {
        rgb[i * 3] = rgba[i * 4];
        rgb[i * 3 + 1] = rgba[i * 4 + 1];
        rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }
}


int main(int argc, char** argv) {// TEMPORARY SCAFFOLDING
    if (argc < 2) {
        std::cerr << "Usage: cuda_vision <image_path>\n";
        return 1;
    }

    std::vector<uint8_t> rgba;
    int width = 0, height = 0;

    if (!load_image_rgba(argv[1], rgba, width, height)) {
        return 1;
    }

    std::vector<uint8_t> rgb;
    rgba_to_rgb(rgba, rgb);


    // load image 
    std::vector<uint8_t> cpu_gray(width * height);
    std::vector<uint8_t> gpu_gray(width * height);

    // call cpu_rgb_to_grayscale(...)
    cpu_rgb_to_grayscale(
        rgb.data(),
        cpu_gray.data(),
        width,
        height
    );
    // call launch_rgb_to_grayscale_cuda(...)
    rgb_to_gray_cuda(
        rgb.data(),
        gpu_gray.data(),
        width,
        height
    );

    // CPU / GPU comparison
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