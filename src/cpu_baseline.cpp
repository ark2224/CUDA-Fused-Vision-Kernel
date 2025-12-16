#include "image_io.hpp"
#include "cpu_baseline.hpp"
#include <vector>
#include <cmath>
#include <algorithm>


// Flattened Vector Version
// ITU-R BT.601 luma approx: 0.299 R + 0.587 G + 0.114 B
// std::vector<uint8_t> cpu_rgb_to_gray(const ImageRGBA& img) {
    // std::vector<uint8_t> out ((size_t)img.w + img.h);

    // const uint8_t* p = img.data.data();
    // for (int i = 0; i < img.w * img.h; ++i) {
    //     uint8_t r = p[4*i + 0];
    //     uint8_t g = p[4*i + 1];
    //     uint8_t b = p[4*i + 2];
    //     int y = (int)(0.299*r + 0.587*g + 0.114*b + 0.5f);
    //     out[i] = (uint8_t)std::clamp(y, 0, 255);
    // }
    // return out;
// }

void cpu_rgb_to_grayscale(
    const uint8_t* rgb,
    uint8_t* gray,
    int width,
    int height
) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;

            uint8_t r = rgb[idx];
            uint8_t g = rgb[idx + 1];
            uint8_t b = rgb[idx + 2];
            
            gray[y * width + x] = 
                static_cast<uint8_t>(0.299f * r +
                                     0.587f * g + 
                                     0.114f * b);
        }
    }
}


// void cpu_box_blur(
//     const uint8_t* input,
//     uint8_t* output,
//     int width,
//     int height
// ) {
//     const int kernel_radius = 1;

//     for (int y = 0; y < height; ++y) {
//         for (int x = 0; x <  width; ++x) {
//             int sum = 0;

//             for (int ky = -kernel_radius; ky <= kernel_radius; ++ky) {
//                 for (int kx = -kernel_radius; kx <= kernel_radius; ++kx) {
//                     int nx = x + kx;
//                     int ny = y + ky;

//                     if (nx >= 0 && nx < width &&
//                         ny >= 0 && ny < height) {
//                         sum += input[ny * width + nx];
//                     }
//                 }
//             }
            
//             int area = pow((kernel_radius * 2 + 1), 2);
//             output[y * width + x] = static_cast<uint8_t>(sum / area);
//         }
//     }
// }

static const float GAUSS[3][3] = {
    {1, 2, 1},
    {2, 4, 2},
    {1, 2, 1}
};

void cpu_gaussian_blur(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height
) {
    int kernel_radius = 1;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x <  width; ++x) {
            float sum = 0.0f;
            int area = 0;

            for (int ky = -kernel_radius; ky <= kernel_radius; ++ky) {
                for (int kx = -kernel_radius; kx <= kernel_radius; ++kx) {
                    if (y + ky < 0 || height <= y + ky || x + kx < 0 || width <= x + kx) continue;
                    sum += input[(y + ky) * width + (x + kx)] *
                           GAUSS[ky+kernel_radius][kx+kernel_radius];
                    area += GAUSS[ky+kernel_radius][kx+kernel_radius];
                }
            }
            
            output[y * width + x] = static_cast<uint8_t>(sum / area);
        }
    }
}