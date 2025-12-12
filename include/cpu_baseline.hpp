#pragma once
#include <cstdint>

// std::vector<uint8_t> cpu_rgb_to_gray(const ImageRGBA& img) {

void cpu_rgb_to_grayscale(
    const uint8_t* rgb,
    uint8_t* gray,
    int width,
    int height
);

void cpu_box_blur(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height
);

void cpu_gaussian_blur(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height
);