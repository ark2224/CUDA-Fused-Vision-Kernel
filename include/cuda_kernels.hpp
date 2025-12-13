#pragma once
#include <cstdint>

void launch_rgb_to_grayscale_cuda(
    const uint8_t* h_rgb,
    uint8_t* h_gray,
    int width,
    int height
);

void rgba_to_gray_u8(
    const uint8_t* rgba,
    uint8_t* gray,
    int width, int height
);